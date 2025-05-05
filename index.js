import "dotenv/config";
import express from "express";
import multer from "multer";
import fs from "fs-extra";
import faceapi from "face-api.js";
import canvas from "canvas";
import { HfInference } from "@huggingface/inference";
import { Pinecone } from "@pinecone-database/pinecone";
import path from "path";
import cors from "cors";
import winston from "winston";
import sharp from "sharp";
import NodeCache from "node-cache";

// Setup face-api.js with node-canvas
const { Image, Canvas, ImageData } = canvas;
faceapi.env.monkeyPatch({ Image, Canvas, ImageData });

// Express App
const app = express();
app.use(express.json({ limit: "2mb" })); // Reduced JSON size limit
app.use(cors({ origin: "*" }));

// Custom timestamp format for Indian timezone
const indianTimestamp = winston.format((info) => {
  const now = new Date();
  const indianTime = new Date(now.toLocaleString("en-US", { timeZone: "Asia/Kolkata" }));
  
  // Format: DD/MM/YYYY HH:mm:ss
  const formattedDate = indianTime.toLocaleDateString("en-GB", { timeZone: "Asia/Kolkata" });
  const formattedTime = indianTime.toLocaleTimeString("en-GB", { 
    timeZone: "Asia/Kolkata",
    hour12: false,
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit"
  });
  
  info.timestamp = `${formattedDate} ${formattedTime}`;
  return info;
});

// Configure Winston Logger with reduced verbosity
const logger = winston.createLogger({
  level: "info",
  format: winston.format.combine(
    indianTimestamp(),
    winston.format.json()
  ),
  transports: [
    new winston.transports.Console({
      format: winston.format.combine(
        winston.format.colorize(),
        winston.format.printf(({ timestamp, level, message, ...meta }) => {
          return `${timestamp} [${level}]: ${message} ${
            Object.keys(meta).length ? JSON.stringify(meta, null, 2) : ''
          }`;
        })
      ),
    }),
    new winston.transports.File({
      filename: "logs/app.log",
      maxsize: 5242880, // 5MB
      maxFiles: 5,
      format: winston.format.combine(
        winston.format.printf(({ timestamp, level, message, ...meta }) => {
          return `${timestamp} [${level}]: ${message} ${
            Object.keys(meta).length ? JSON.stringify(meta, null, 2) : ''
          }`;
        })
      ),
    }),
  ],
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

// Hugging Face Inference API
const hf = new HfInference(process.env.HF_ACCESS_TOKEN);

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

// Optimized embedding function
const getEmbedding = async (text) => {
  try {
    const cacheKey = `text_${text.substring(0, 20)}`;
    const cached = imageCache.get(cacheKey);
    if (cached) return cached;

    const output = await hf.featureExtraction({
      model: "intfloat/multilingual-e5-large-instruct",
      inputs: text,
      provider: "hf-inference",
    });

    const result = output.slice(0, 128);
    imageCache.set(cacheKey, result);
    return result;
  } catch (error) {
    logger.error("Error getting embedding:", error);
    throw error;
  }
};

app.post("/upload", upload.single("image"), async (req, res) => {
  if (!req.file) {
    return res.status(400).json({ 
      success: false,
      message: "No image provided" 
    });
  }

  try {
    // Set timeout for the operation
    const timeout = setTimeout(() => {
      return res.status(408).json({ 
        success: false,
        message: "Request timeout" 
      });
    }, 10000);

    // Validate file size
    if (req.file.size > 2 * 1024 * 1024) {
      clearTimeout(timeout);
      return res
        .status(400)
        .json({ 
          success: false,
          message: "Image too large, max 2MB allowed" 
        });
    }

    // Always process new image
    const optimizedBuffer = await optimizeImage(req.file.buffer);
    
    // Get fresh face embeddings
    const faceEmbeddings = await detectFaces(optimizedBuffer);

    clearTimeout(timeout);

    if (!faceEmbeddings || faceEmbeddings.length === 0) {
      return res.json({ 
        success: false,
        message: "No face detected in image" 
      });
    }

    // Convert to array and ensure we have the correct format
    const freshEmbeddings = Array.from(faceEmbeddings[0]);

    res.json({ 
      success: true,
      embedding: freshEmbeddings 
    });
  } catch (error) {
    logger.error("Error processing image:", error);
    res.status(500).json({
      success: false,
      message: "Error processing image",
      error: error.message,
    });
  }
});

// Face Registration Endpoint (Combined upload and save)
app.post("/face-register", upload.single("image"), async (req, res) => {
  if (!req.file) {
    return res.status(400).json({ message: "No image provided" });
  }

  const { name } = req.body;

  if (!name) {
    return res.status(400).json({ message: "Name is required" });
  }

  try {
    // Set timeout for the operation
    const timeout = setTimeout(() => {
      return res.status(408).json({ message: "Request timeout" });
    }, 10000);

    // Validate file size
    if (req.file.size > 2 * 1024 * 1024) {
      clearTimeout(timeout);
      return res
        .status(400)
        .json({ message: "Image too large, max 2MB allowed" });
    }

    // Always process new image
    const optimizedBuffer = await optimizeImage(req.file.buffer);
    
    // Get fresh face embeddings
    const faceEmbeddings = await detectFaces(optimizedBuffer);

    clearTimeout(timeout);

    if (!faceEmbeddings || faceEmbeddings.length === 0) {
      return res.json({ 
        success: false,
        message: "No face detected in image" 
      });
    }

    // Convert to array and ensure we have the correct format
    const freshEmbeddings = Array.from(faceEmbeddings[0]);

    // Save to Pinecone with fresh embeddings
    await index.upsert([{ 
      id: name, 
      values: freshEmbeddings, 
      metadata: { 
        name,
        timestamp: new Date().toISOString() // Add timestamp to track when the face was registered
      } 
    }]);

    res.json({ 
      success: true,
      message: "✅ Face registered successfully!", 
      name: name,
      embedding: freshEmbeddings 
    });
  } catch (error) {
    logger.error("Error processing face registration:", error);
    res.status(500).json({
      success: false,
      message: "Error processing face registration",
      error: error.message,
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

// Match Face from Pinecone (optimized)
app.post("/match-face", async (req, res) => {
  const { embedding } = req.body;

  if (!embedding || embedding.length !== 128) {
    return res.status(400).json({ message: "Embedding dimension must be 128" });
  }

  try {
    const results = await retryAsync(() =>
      index.query({
        vector: embedding,
        topK: 1,
        includeMetadata: true,
      })
    );

    if (results.matches.length > 0 && results.matches[0].score >= 0.90) {
      res.json({
        match: results.matches[0].metadata.name,
        score: results.matches[0].score,
      });
    } else {
      res.json({ message: "No match found" });
    }
  } catch (error) {
    logger.error("Error querying Pinecone:", error);
    res.status(500).json({ message: "Internal server error" });
  }
});

// Error handling middleware (improved)
app.use((err, req, res, next) => {
  logger.error("Unhandled error:", err);
  res.status(500).json({ message: "Internal server error" });
});

// Initialize server
const startServer = async () => {
  try {
    await loadModels();
    const PORT = process.env.PORT || 5000;
    app.listen(PORT, () => logger.info(`✅ Server running on port ${PORT}`));
  } catch (error) {
    logger.error("Failed to start server:", error);
    process.exit(1);
  }
};

startServer();

// Export for production (Render)
export default app;