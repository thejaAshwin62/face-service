import "dotenv/config";
import express from "express";
import multer from "multer";
import fs from "fs-extra";
import faceapi from "face-api.js";
import canvas from "canvas";
import { Pinecone } from "@pinecone-database/pinecone";
import path from "path";
import cors from "cors";
import winston from "winston";
import sharp from "sharp";
import NodeCache from "node-cache";
import { join } from 'path';
import http from 'http';

// Memory management for Render
const memoryManagement = () => {
  if (global.gc) {
    global.gc();
    logger.info("Manual garbage collection triggered");
  }
};

// Schedule memory cleanup every 5 minutes
setInterval(memoryManagement, 5 * 60 * 1000);

// Improved agent configuration specifically for Render
const agent = new http.Agent({
  keepAlive: true,
  maxSockets: 25,  // Further reduced to prevent Render connection limits
  maxFreeSockets: 5,
  timeout: 120000,  // 2 minutes for Render's longer cold starts
});

// Circuit breaker for external services
class CircuitBreaker {
  constructor(name, timeout = 10000) {
    this.name = name;
    this.state = 'CLOSED';
    this.failureCount = 0;
    this.failureThreshold = 5;
    this.resetTimeout = 30000;
    this.timeout = timeout;
  }

  async exec(fn) {
    if (this.state === 'OPEN') {
      logger.warn(`Circuit breaker for ${this.name} is OPEN`);
      throw new Error(`Service ${this.name} is unavailable`);
    }

    try {
      const result = await Promise.race([
        fn(),
        new Promise((_, reject) => setTimeout(() => 
          reject(new Error(`${this.name} timeout`)), this.timeout))
      ]);

      this.onSuccess();
      return result;
    } catch (error) {
      this.onFailure();
      throw error;
    }
  }

  onSuccess() {
    this.failureCount = 0;
    this.state = 'CLOSED';
  }

  onFailure() {
    this.failureCount++;
    if (this.failureCount >= this.failureThreshold) {
      this.state = 'OPEN';
      logger.warn(`Circuit breaker for ${this.name} opened`);
      setTimeout(() => {
        this.state = 'HALF-OPEN';
        this.failureCount = 0;
        logger.info(`Circuit breaker for ${this.name} half-open`);
      }, this.resetTimeout);
    }
  }
}

// Create circuit breakers for key operations
const pineconeBreaker = new CircuitBreaker('pinecone', 15000);
const faceDetectionBreaker = new CircuitBreaker('faceDetection', 8000);

// Initialize request queue and constants
const requestQueue = new Map();
const MAX_CONCURRENT_REQUESTS = 10;
const REQUEST_TIMEOUT = 30000; // 30 seconds

// Request queue cleanup utility
const cleanupQueue = () => {
  const now = Date.now();
  for (const [id, request] of requestQueue.entries()) {
    if (now - parseInt(id) > REQUEST_TIMEOUT) {
      requestQueue.delete(id);
    }
  }
};

// Schedule queue cleanup every minute
setInterval(cleanupQueue, 60000);

// Setup face-api.js with node-canvas
const { Image, Canvas, ImageData } = canvas;
faceapi.env.monkeyPatch({ Image, Canvas, ImageData });

// Express App with improved settings
const app = express();
app.use(express.json({ limit: "2mb" })); // Reduced JSON size limit
app.use(cors({
  origin: "*",
  methods: "GET,HEAD,PUT,PATCH,POST,DELETE",
  preflightContinue: false,
  optionsSuccessStatus: 204
}));

// Add log cleanup function
const cleanOldLogs = async () => {
  try {
    const logsDir = join(process.cwd(), 'logs');
    const files = await fs.readdir(logsDir);
    const currentLog = 'app.log';
    
    for (const file of files) {
      if (file !== currentLog) {
        await fs.remove(join(logsDir, file));
        logger.info(`Cleaned up old log file: ${file}`);
      }
    }
  } catch (error) {
    console.error('Log cleanup failed:', error);
  }
};

// Configure Winston Logger with rotation
const logDir = 'logs';
fs.ensureDirSync(logDir);

const logger = winston.createLogger({
  level: "info",
  format: winston.format.combine(
    winston.format.timestamp(),
    winston.format.json()
  ),
  transports: [
    new winston.transports.Console(),
    new winston.transports.File({
      filename: join(logDir, 'app.log'),
      maxsize: 5242880, // 5MB
      maxFiles: 1, // Keep only one file
      tailable: true,
      options: { flags: 'w' } // Overwrite file on restart
    })
  ]
});

// Streamlined logging middleware
app.use((req, res, next) => {
  logger.info({
    message: "Request",
    method: req.method,
    url: req.url,
  });

  const originalSend = res.send;
  res.send = function (body) {
    logger.info({
      message: "Response",
      method: req.method,
      url: req.url,
      statusCode: res.statusCode,
    });
    return originalSend.call(this, body);
  };

  next();
});


// Pinecone Initialization
const pinecone = new Pinecone({ apiKey: process.env.PINECONE_API_KEY });
const index = pinecone.index(process.env.PINECONE_INDEX);

// Configure multer with reduced file size
const upload = multer({
  storage: multer.memoryStorage(),
  limits: {
    fileSize: 2 * 1024 * 1024, // 2MB limit (reduced from 5MB)
  },
});

// Improved cache with longer TTL
const imageCache = new NodeCache({
  stdTTL: 1800, // 30 minutes (increased from 10 minutes)
  checkperiod: 600, // Check for expired keys every 10 minutes
  useClones: false, // Don't clone objects (better performance)
  maxKeys: 100, // Limit cache size
});

// More aggressive image optimization for Render's constraints
const optimizeImage = async (buffer) => {
  return sharp(buffer)
    .resize(240, 240, {  // Further reduced size for faster processing
      fit: "inside",
      withoutEnlargement: true,
    })
    .jpeg({
      quality: 65,  // Further reduced quality for better performance
      progressive: true,
      optimizeScans: true,
    })
    .toBuffer();
};

// Cleanup middleware (more efficient)
const cleanup = async (req, res, next) => {
  if (req.file && req.file.path) {
    res.on("finish", () => {
      fs.remove(req.file.path).catch((err) =>
        logger.error("Cleanup error:", err)
      );
    });
  }
  next();
};

app.use(cleanup);

// Load Face Detection Models
const modelPath = path.resolve("models");
const loadModels = async () => {
  try {
    // Load all required models
    await faceapi.nets.ssdMobilenetv1.loadFromDisk(modelPath);
    await faceapi.nets.faceLandmark68Net.loadFromDisk(modelPath); // Added back
    await faceapi.nets.faceRecognitionNet.loadFromDisk(modelPath);
    logger.info("✅ Models loaded successfully");
  } catch (error) {
    logger.error("Failed to load models:", error);
    process.exit(1);
  }
};

// Optimized face detection for Render
const detectFaces = async (buffer) => {
  try {
    return await faceDetectionBreaker.exec(async () => {
      const img = await canvas.loadImage(buffer);
      
      // Simplified detection options
      const detectionOptions = new faceapi.SsdMobilenetv1Options({
        minConfidence: 0.15,  // Lower threshold to improve success rate
        maxResults: 1,
      });

      // Improved detection pipeline with better error handling
      try {
        const detections = await faceapi
          .detectAllFaces(img, detectionOptions)
          .withFaceLandmarks()
          .withFaceDescriptors();

        if (detections.length === 0) {
          logger.info("No faces detected in image");
          return [];
        }

        return [detections[0].descriptor.slice(0, 128)];
      } catch (innerError) {
        logger.error("Detection pipeline error:", innerError);
        // Attempt fallback with just face detection
        const simpleFaces = await faceapi.detectAllFaces(img, detectionOptions);
        if (simpleFaces.length > 0) {
          logger.info("Face detected but descriptor extraction failed");
        }
        return [];
      }
    });
  } catch (error) {
    logger.error("Face detection error:", error);
    return [];
  }
};

// Highly optimized upload endpoint with improved timeout handling
app.post("/upload", upload.single("image"), async (req, res) => {
  // Set timeout to prevent socket hang up
  res.setTimeout(60000, () => {
    logger.warn("Request timeout reached in /upload endpoint");
    if (!res.headersSent) {
      res.status(408).json({ message: "Request timeout", status: 'failed' });
    }
  });

  if (!req.file) {
    return res.status(400).json({ message: "No image provided" });
  }

  const requestId = Date.now().toString();
  requestQueue.set(requestId, { status: 'processing' });

  // Enhanced headers for Render's proxy settings
  res.setHeader('Connection', 'keep-alive');
  res.setHeader('Keep-Alive', 'timeout=120');
  res.setHeader('X-Request-ID', requestId);
  res.setHeader('Cache-Control', 'no-transform');

  try {
    // Check file size early to fail fast
    if (req.file.size > 2 * 1024 * 1024) {
      requestQueue.delete(requestId);
      return res.status(413).json({ message: "Image too large", status: 'failed' });
    }

    const cacheKey = Buffer.from(req.file.buffer).toString('base64').substring(0, 20);
    const cachedResult = imageCache.get(cacheKey);

    if (cachedResult) {
      requestQueue.delete(requestId);
      return res.json({ embedding: cachedResult, status: 'completed' });
    }

    // Process image with shorter timeouts
    let optimizedBuffer, faceEmbeddings;
    
    try {
      // First optimize image with 5 second timeout
      optimizedBuffer = await Promise.race([
        optimizeImage(req.file.buffer),
        new Promise((_, reject) => setTimeout(() => reject(new Error('Image optimization timeout')), 5000))
      ]);
      
      // Then detect faces with 7 second timeout
      faceEmbeddings = await Promise.race([
        detectFaces(req.file.buffer),
        new Promise((_, reject) => setTimeout(() => reject(new Error('Face detection timeout')), 7000))
      ]);
    } catch (processingError) {
      logger.error("Processing error:", processingError);
      if (!res.headersSent) {
        requestQueue.delete(requestId);
        return res.status(422).json({ 
          message: "Image processing failed", 
          error: processingError.message,
          status: 'failed' 
        });
      }
      return;
    }

    // Clean response handling to prevent double responses
    if (!faceEmbeddings || faceEmbeddings.length === 0) {
      requestQueue.delete(requestId);
      if (!res.headersSent) {
        return res.status(400).json({ message: "No face detected", status: 'completed' });
      }
      return;
    }

    const result = Array.from(faceEmbeddings[0]);
    imageCache.set(cacheKey, result);
    
    requestQueue.delete(requestId);
    if (!res.headersSent) {
      return res.json({ embedding: result, status: 'completed' });
    }

  } catch (error) {
    requestQueue.delete(requestId);
    logger.error("Error processing image:", error);
    
    // Only send response if headers haven't been sent yet
    if (!res.headersSent) {
      return res.status(500).json({
        message: "Error processing image",
        error: error.message,
        status: 'failed'
      });
    }
  }
});

// Root Endpoint
app.get("/", (req, res) => {
  res.json({ message: "Welcome to the Face Service API" });
});

// Store Face in Pinecone with Name (optimized)
app.post("/save-face", async (req, res) => {
  const { name, embedding } = req.body;

  if (!name || !embedding || embedding.length !== 128) {
    return res.status(400).json({ message: "Invalid input or embedding size" });
  }

  try {
    await index.upsert([{ id: name, values: embedding, metadata: { name } }]);
    res.json({ message: "✅ Face saved in Pinecone!" });
  } catch (error) {
    logger.error("Error saving face:", error);
    res.status(500).json({ message: "Error saving face to database" });
  }
});

// Helper function to retry async operations (optimized)
const retryAsync = async (fn, retries = 2, delay = 500) => {
  for (let i = 0; i < retries; i++) {
    try {
      return await fn();
    } catch (error) {
      if (i === retries - 1) throw error;
      await new Promise((resolve) => setTimeout(resolve, delay));
    }
  }
};

// Enhanced match-face endpoint with reliable response handling
app.post("/match-face", async (req, res) => {
  const { embedding } = req.body;
  const requestId = Date.now().toString();

  if (!embedding || embedding.length !== 128) {
    return res.status(400).json({ 
      message: "Invalid embedding dimension", 
      requestId,
      status: 'failed' 
    });
  }

  requestQueue.set(requestId, { status: 'processing' });

  try {
    const matchResult = await pineconeBreaker.exec(async () => {
      const queryResponse = await index.query({
        vector: embedding,
        topK: 1,
        includeMetadata: true,
      });
      
      if (!queryResponse || !queryResponse.matches) {
        throw new Error('Invalid query response');
      }
      
      return queryResponse;
    });

    // Process match results
    if (matchResult.matches.length > 0 && matchResult.matches[0].score >= 0.85) {
      requestQueue.delete(requestId);
      return res.json({
        match: matchResult.matches[0].metadata.name,
        score: matchResult.matches[0].score,
        requestId,
        status: 'completed'
      });
    }

    requestQueue.delete(requestId);
    return res.json({ 
      message: "No match found", 
      requestId,
      status: 'completed'
    });

  } catch (error) {
    requestQueue.delete(requestId);
    logger.error("Face matching error:", error);
    
    return res.status(500).json({ 
      message: "Face matching failed", 
      error: error.message,
      requestId,
      status: 'failed'
    });
  }
});

// Add status check endpoint
app.get("/status/:requestId", (req, res) => {
  const { requestId } = req.params;
  const request = requestQueue.get(requestId);
  
  if (!request) {
    return res.json({ status: 'not_found' });
  }
  
  res.json({ status: request.status });
});

// Add a keepalive endpoint for preventing cold starts
app.get("/ping", (req, res) => {
  res.status(200).send("pong");
});

// Warmup function for Render
const warmup = async () => {
  try {
    logger.info("Warming up service...");
    
    // Preload models in memory
    await loadModels();
    
    // Pre-initialize Sharp
    await sharp(Buffer.from([0])).resize(1, 1).toBuffer();
    
    // Ping Pinecone to establish connection
    await index.describeIndexStats();
    
    logger.info("Warmup complete");
  } catch (error) {
    logger.error("Warmup failed:", error);
  }
};

// Add a self-ping to prevent Render's free tier from sleeping
const keepAlive = () => {
  const interval = 14 * 60 * 1000; // 14 minutes (just under Render's 15-min sleep time)
  setInterval(() => {
    http.get(`http://localhost:${process.env.PORT || 5000}/ping`, res => {
      logger.debug("Keep-alive ping sent");
    }).on('error', err => {
      logger.error("Keep-alive error:", err);
    });
  }, interval);
};

// Error handling middleware (improved)
app.use((err, req, res, next) => {
  logger.error("Unhandled error:", err);
  res.status(500).json({ message: "Internal server error" });
});

// Initialize server with Render-specific optimizations
const startServer = async () => {
  try {
    // Run memory optimization before startup
    memoryManagement();
    
    // Clean old logs before starting
    await cleanOldLogs();
    
    // Warm up the service
    await warmup();
    
    const PORT = process.env.PORT || 5000;
    
    // Configure periodic maintenance tasks
    setInterval(cleanOldLogs, 24 * 60 * 60 * 1000);
    setInterval(memoryManagement, 30 * 60 * 1000);
    
    const server = app.listen(PORT, () => {
      logger.info(`✅ Server running on port ${PORT}`);
      // Start keep-alive mechanism for Render free tier
      if (process.env.ENVIRONMENT === 'render') {
        keepAlive();
      }
    });
    
    // Configure server timeouts specifically for Render to prevent socket hang ups
    server.keepAliveTimeout = 65000;  // Increased from 120000
    server.headersTimeout = 66000;    // Just above keepAliveTimeout
    server.timeout = 70000;           // Reduced from 180000 to fail faster
    
    // Handle server-level connection errors
    server.on('error', (error) => {
      logger.error("Server error:", error);
    });
    
    // Handle graceful shutdown
    process.on('SIGTERM', () => {
      logger.info('SIGTERM received, shutting down gracefully');
      server.close(() => {
        logger.info('Server closed');
        process.exit(0);
      });
    });
    
  } catch (error) {
    logger.error("Failed to start server:", error);
    process.exit(1);
  }
};

startServer();

// Export for production (Render)
export default app;
