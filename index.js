require("dotenv").config();
const express = require("express");
const multer = require("multer");
const fs = require("fs-extra");
const faceapi = require("face-api.js");
const canvas = require("canvas");
const { HfInference } = require("@huggingface/inference");
const { Pinecone } = require("@pinecone-database/pinecone");
const path = require("path");
const cors = require("cors"); // Import cors

// Setup face-api.js with node-canvas
const { Image, Canvas, ImageData } = canvas;
faceapi.env.monkeyPatch({ Image, Canvas, ImageData });

// Express App
const app = express();
app.use(express.json());
app.use(cors({ origin: "*" })); // Allow all origins (for testing))

// Add request logging middleware
app.use((req, res, next) => {
  console.log(`[${new Date().toISOString()}] ${req.method} ${req.path}`);
  next();
});

// Hugging Face Inference API
const hf = new HfInference(process.env.HF_ACCESS_TOKEN);

// Pinecone Initialization
const pinecone = new Pinecone({ apiKey: process.env.PINECONE_API_KEY });
const index = pinecone.index(process.env.PINECONE_INDEX); // Fixed reference

// Image Upload Configuration - Change to memory storage for Vercel compatibility
const upload = multer({ 
  storage: multer.memoryStorage(),
  limits: { fileSize: 5 * 1024 * 1024 } // 5MB limit
});

// Load Face Detection Models
const modelPath = path.join(process.cwd(), "models");
const loadModels = async () => {
  try {
    console.log("Loading face-api.js models from:", modelPath);
    console.log("Files in models directory:", fs.existsSync(modelPath) ? fs.readdirSync(modelPath) : "Directory not found");
    
    await faceapi.nets.ssdMobilenetv1.loadFromDisk(modelPath);
    await faceapi.nets.faceLandmark68Net.loadFromDisk(modelPath);
    await faceapi.nets.faceRecognitionNet.loadFromDisk(modelPath);
    console.log("✅ Models loaded successfully");
  } catch (error) {
    console.error("❌ Error loading models:", error);
    // Continue execution even if models fail to load
  }
};

// Load models but don't block server startup
loadModels();

// Function to Extract Face Embeddings (Reduce to 128 Dimensions)
const detectFaces = async (imagePath) => {
  const img = await canvas.loadImage(imagePath);
  const detections = await faceapi
    .detectAllFaces(img)
    .withFaceLandmarks()
    .withFaceDescriptors();

  return detections.map((d) => d.descriptor.slice(0, 128)); // Reduce to 128D
};

// Modified function to detect faces from buffer
const detectFacesFromBuffer = async (buffer) => {
  const img = await canvas.loadImage(buffer);
  const detections = await faceapi
    .detectAllFaces(img)
    .withFaceLandmarks()
    .withFaceDescriptors();

  return detections.map((d) => d.descriptor.slice(0, 128)); // Reduce to 128D
};

// Convert Name to Vector (Ensure 128D)
const getEmbedding = async (text) => {
  const output = await hf.featureExtraction({
    model: "intfloat/multilingual-e5-large-instruct",
    inputs: text,
    provider: "hf-inference",
  });

  return output.slice(0, 128); // Ensure 128D
};

// Enhanced retry mechanism with exponential backoff
const retryAsync = async (fn, options = {}) => {
  const {
    retries = 3,
    initialDelay = 1000,
    maxDelay = 10000,
    factor = 2,
    timeout = 30000
  } = options;

  const timeoutPromise = (ms, message) =>
    new Promise((_, reject) => 
      setTimeout(() => reject(new Error(message)), ms)
    );

  const executeWithTimeout = async () => {
    return Promise.race([
      fn(),
      timeoutPromise(timeout, 'Operation timed out')
    ]);
  };

  for (let attempt = 0; attempt < retries; attempt++) {
    try {
      return await executeWithTimeout();
    } catch (error) {
      if (attempt === retries - 1) throw error;
      
      const delay = Math.min(initialDelay * Math.pow(factor, attempt), maxDelay);
      console.log(`Retry attempt ${attempt + 1}/${retries} after ${delay}ms`);
      await new Promise(resolve => setTimeout(resolve, delay));
    }
  }
};

// Request timeout middleware
app.use((req, res, next) => {
  req.setTimeout(30000, () => {
    res.status(408).json({ message: 'Request timeout' });
  });
  next();
});

// Upload & Extract Face Embeddings - Updated for memory storage
app.post("/upload", upload.single("image"), async (req, res) => {
  if (!req.file) {
    return res.status(400).json({ message: "No image provided" });
  }

  try {
    const result = await retryAsync(
      async () => {
        console.log("Processing image upload, size:", req.file.size);
        
        if (!faceapi.nets.ssdMobilenetv1.isLoaded) {
          console.log("Models not loaded, attempting to load...");
          await loadModels();
        }
        
        const faceEmbeddings = await detectFacesFromBuffer(req.file.buffer);
        console.log(`Detected ${faceEmbeddings.length} faces`);

        if (faceEmbeddings.length === 0) {
          throw new Error('No face detected');
        }

        return { embedding: Array.from(faceEmbeddings[0]) };
      },
      { 
        retries: 3,
        timeout: 45000,
        initialDelay: 2000
      }
    );

    res.json(result);
  } catch (error) {
    console.error("Error processing image:", error);
    res.status(error.message === 'No face detected' ? 400 : 500).json({
      message: error.message === 'No face detected' ? error.message : "Error processing image",
      error: error.message
    });
  }
});

// Root Endpoint
app.get("/", (req, res) => {
  res.json({ message: "Welcome to the Face Service API" });
});

// Store Face in Pinecone with Name
app.post("/save-face", async (req, res) => {
  const { name, embedding } = req.body;

  if (!name || !embedding) {
    return res.status(400).json({ message: "Name and embedding are required" });
  }
  
  if (embedding.length !== 128) {
    console.log(`Warning: Embedding dimension is ${embedding.length}, expected 128`);
    return res.status(400).json({ message: "Embedding must be 128 dimensions" });
  }

  try {
    console.log(`Saving face for: ${name}`);
    await index.upsert([{ id: name, values: embedding, metadata: { name } }]);
    console.log(`Successfully saved face for: ${name}`);
    res.json({ message: "✅ Face saved in Pinecone!" });
  } catch (error) {
    console.error("Error saving face to Pinecone:", error);
    res.status(500).json({ 
      message: "Error saving face", 
      error: error.message 
    });
  }
});

// Match Face from Pinecone
app.post("/match-face", async (req, res) => {
  const { embedding } = req.body;

  if (!embedding || embedding.length !== 128) {
    return res.status(400).json({ 
      message: "Valid 128-dimension embedding is required" 
    });
  }

  try {
    const result = await retryAsync(
      async () => {
        console.log("Querying Pinecone for face match");
        const results = await index.query({
          vector: embedding,
          topK: 1,
          includeMetadata: true,
        });
        
        if (results.matches.length === 0) {
          throw new Error('No matches found');
        }
        
        return results;
      },
      {
        retries: 5,
        initialDelay: 1000,
        maxDelay: 8000,
        timeout: 20000
      }
    );

    if (result.matches[0].score >= 0.85) {
      res.json({
        match: result.matches[0].metadata.name,
        score: result.matches[0].score,
      });
    } else {
      res.json({ message: "No confident match found" });
    }
  } catch (error) {
    console.error("Error querying Pinecone:", error);
    res.status(error.message === 'No matches found' ? 404 : 500).json({
      message: error.message === 'No matches found' ? error.message : "Error matching face",
      error: error.message
    });
  }
});

// Add a /debug endpoint to check environment and configurations
app.get("/debug", (req, res) => {
  const safeEnvVars = {
    NODE_ENV: process.env.NODE_ENV,
    PORT: process.env.PORT,
    HF_TOKEN_SET: !!process.env.HF_ACCESS_TOKEN,
    PINECONE_KEY_SET: !!process.env.PINECONE_API_KEY,
    PINECONE_INDEX: process.env.PINECONE_INDEX,
    PINECONE_REGION: process.env.PINECONE_REGION,
    CWD: process.cwd(),
    MODEL_PATH: modelPath,
    MODEL_PATH_EXISTS: fs.existsSync(modelPath)
  };
  
  res.json({
    environment: safeEnvVars,
    serverInfo: {
      platform: process.platform,
      nodeVersion: process.version,
      memoryUsage: process.memoryUsage()
    }
  });
});

// Error handling middleware
app.use((err, req, res, next) => {
  console.error("Unhandled error:", err);
  res.status(500).json({ message: "Internal server error" });
});

// Make sure port is set properly for Vercel
const PORT = process.env.PORT || 3000;

// Update how your server listens for requests
if (process.env.NODE_ENV !== 'production') {
  app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
  });
}

// Export your Express app for Vercel serverless function
module.exports = app;
