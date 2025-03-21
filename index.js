import "dotenv/config";
import express from "express";
import multer from "multer";
import fs from "fs-extra";
import faceapi from "face-api.js";
import canvas from "canvas";
import { HfInference } from "@huggingface/inference";
import { Pinecone } from "@pinecone-database/pinecone";
import path from "path";
import cors from "cors"; // Import cors

// Setup face-api.js with node-canvas
const { Image, Canvas, ImageData } = canvas;
faceapi.env.monkeyPatch({ Image, Canvas, ImageData });

// Express App
const app = express();
app.use(express.json());
app.use(cors({ origin: "*" })); // Allow all origins (for testing)

// Hugging Face Inference API
const hf = new HfInference(process.env.HF_ACCESS_TOKEN);

// Pinecone Initialization
const pinecone = new Pinecone({ apiKey: process.env.PINECONE_API_KEY });
const index = pinecone.index(process.env.PINECONE_INDEX); // Fixed reference

// Configure multer for memory storage instead of disk
const upload = multer({
  storage: multer.memoryStorage(),
  limits: {
    fileSize: 5 * 1024 * 1024 // 5MB limit
  }
});

// Cleanup middleware
const cleanup = async (req, res, next) => {
  res.on('finish', async () => {
    if (req.file && req.file.path) {
      try {
        await fs.remove(req.file.path);
        console.log('Cleaned up:', req.file.path);
      } catch (error) {
        console.error('Cleanup error:', error);
      }
    }
  });
  next();
};

app.use(cleanup);

// Load Face Detection Models
const modelPath = path.resolve("models");
const loadModels = async () => {
  await faceapi.nets.ssdMobilenetv1.loadFromDisk(modelPath);
  await faceapi.nets.faceLandmark68Net.loadFromDisk(modelPath);
  await faceapi.nets.faceRecognitionNet.loadFromDisk(modelPath);
};

loadModels().then(() => console.log("✅ Models loaded successfully"));

// Function to Extract Face Embeddings (Reduce to 128 Dimensions)
const detectFaces = async (buffer) => {
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

// Upload & Extract Face Embeddings
app.post("/upload", upload.single("image"), async (req, res) => {
  if (!req.file) {
    return res.status(400).json({ message: "No image provided" });
  }

  try {
    const faceEmbeddings = await detectFaces(req.file.buffer);

    if (faceEmbeddings.length === 0) {
      return res.json({ message: "No face detected" });
    }

    res.json({ embedding: Array.from(faceEmbeddings[0]) });
  } catch (error) {
    console.error("Error processing image:", error);
    res.status(500).json({ 
      message: "Error processing image", 
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

  if (!name || !embedding || embedding.length !== 128) {
    return res.status(400).json({ message: "Invalid input or embedding size" });
  }

  await index.upsert([{ id: name, values: embedding, metadata: { name } }]);
  res.json({ message: "✅ Face saved in Pinecone!" });
});

// Helper function to retry async operations
const retryAsync = async (fn, retries = 3, delay = 1000) => {
  for (let i = 0; i < retries; i++) {
    try {
      return await fn();
    } catch (error) {
      if (i === retries - 1) throw error;
      await new Promise((resolve) => setTimeout(resolve, delay));
    }
  }
};

// Match Face from Pinecone
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

    if (results.matches.length > 0 && results.matches[0].score >= 0.85) {
      res.json({
        match: results.matches[0].metadata.name,
        score: results.matches[0].score,
      });
    } else {
      res.json({ message: "No match found" });
    }
  } catch (error) {
    console.error("Error querying Pinecone:", error);
    res.status(500).json({ message: "Internal server error" });
  }
});

// Error handling middleware
app.use((err, req, res, next) => {
  console.error("Unhandled error:", err);
  res.status(500).json({ message: "Internal server error" });
});

// Start the Server
const PORT = process.env.PORT || 5000;

if (process.env.NODE_ENV !== 'production') {
  app.listen(PORT, () => {
    console.log(`⚡ Development server running on port ${PORT}`);
  });
}

// Export for production (Render)
export default app;
