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

// Add request queue and agent configuration
const agent = new http.Agent({
  keepAlive: true,
  maxSockets: 100,
  maxFreeSockets: 10,
  timeout: 30000,
});

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

// Express App
const app = express();
app.use(express.json({ limit: "2mb" })); // Reduced JSON size limit
app.use(cors({ origin: "*" }));

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

// More aggressive image optimization
const optimizeImage = async (buffer) => {
  return sharp(buffer)
    .resize(400, 400, {
      // Increased from 320x320 to preserve more details
      fit: "inside",
      withoutEnlargement: true,
    })
    .jpeg({
      quality: 80, // Increased quality to preserve details
      progressive: true,
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

// Optimized face detection
const detectFaces = async (buffer) => {
  try {
    const img = await canvas.loadImage(buffer);

    // More sensitive detection options
    const detectionOptions = new faceapi.SsdMobilenetv1Options({
      minConfidence: 0.1, // Much lower threshold to detect more faces
      maxResults: 3, // Look for multiple faces and take the best one
    });

    // Use all faces detection to find any possible faces
    const detections = await faceapi
      .detectAllFaces(img, detectionOptions)
      .withFaceLandmarks() // Re-enable landmarks to help with difficult faces
      .withFaceDescriptors();

    if (detections.length === 0) {
      logger.info("No faces detected in image");
      return [];
    }

    // Sort by detection confidence and take the highest
    detections.sort((a, b) => b.detection.score - a.detection.score);
    return [detections[0].descriptor.slice(0, 128)];
  } catch (error) {
    logger.error("Face detection error:", error);
    return [];
  }
};

// Highly optimized upload endpoint
app.post("/upload", upload.single("image"), async (req, res) => {
  if (!req.file) {
    return res.status(400).json({ message: "No image provided" });
  }

  const requestId = Date.now().toString();
  requestQueue.set(requestId, { status: 'processing' });

  // Set response headers for faster client processing
  res.setHeader('Connection', 'keep-alive');
  res.setHeader('X-Request-ID', requestId);

  try {
    const cacheKey = Buffer.from(req.file.buffer).toString('base64').substring(0, 20);
    const cachedResult = imageCache.get(cacheKey);

    if (cachedResult) {
      requestQueue.delete(requestId);
      return res.json({ embedding: cachedResult, status: 'completed' });
    }

    // Parallel processing
    const [optimizedBuffer, faceDetectionPromise] = await Promise.all([
      optimizeImage(req.file.buffer),
      Promise.race([
        detectFaces(req.file.buffer),
        new Promise((_, reject) => 
          setTimeout(() => reject(new Error('Face detection timeout')), 5000)
        )
      ])
    ]);

    const faceEmbeddings = await faceDetectionPromise;

    if (!faceEmbeddings || faceEmbeddings.length === 0) {
      requestQueue.delete(requestId);
      return res.json({ message: "No face detected", status: 'completed' });
    }

    const result = Array.from(faceEmbeddings[0]);
    imageCache.set(cacheKey, result);
    
    requestQueue.delete(requestId);
    return res.json({ embedding: result, status: 'completed' });

  } catch (error) {
    requestQueue.delete(requestId);
    logger.error("Error processing image:", error);
    return res.status(500).json({
      message: "Error processing image",
      error: error.message,
      status: 'failed'
    });
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
    // Set longer timeout for face matching
    const matchPromise = retryAsync(
      async () => {
        const queryResponse = await index.query({
          vector: embedding,
          topK: 1,
          includeMetadata: true,
        });
        
        // Validate query response
        if (!queryResponse || !queryResponse.matches) {
          throw new Error('Invalid query response');
        }
        
        return queryResponse;
      },
      3, // number of retries
      1000 // delay between retries
    );

    // Wait for response with timeout
    const matchResult = await Promise.race([
      matchPromise,
      new Promise((_, reject) => 
        setTimeout(() => reject(new Error('Match operation timeout')), 10000)
      )
    ]);

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

// Error handling middleware (improved)
app.use((err, req, res, next) => {
  logger.error("Unhandled error:", err);
  res.status(500).json({ message: "Internal server error" });
});

// Initialize server
const startServer = async () => {
  try {
    // Clean old logs before starting
    await cleanOldLogs();
    
    await loadModels();
    const PORT = process.env.PORT || 5000;
    
    // Schedule periodic log cleanup (every 24 hours)
    setInterval(cleanOldLogs, 24 * 60 * 60 * 1000);
    
    const server = app.listen(PORT, () => logger.info(`✅ Server running on port ${PORT}`));
    
    // Configure server timeouts
    server.keepAliveTimeout = 30000;
    server.headersTimeout = 35000;
    
  } catch (error) {
    logger.error("Failed to start server:", error);
    process.exit(1);
  }
};

startServer();

// Export for production (Render)
export default app;
