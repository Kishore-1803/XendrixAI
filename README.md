<p align="center">
  <img src="frontend/public/logo.png" alt="Xendrix Logo" width="180"/>
</p>

<h1 align="center">Xendrix AI Assistant</h1>

<p align="center">
  An intelligent, multimodal AI assistant built using <strong>FastAPI</strong> (backend) and <strong>Next.js</strong> (frontend). 
  Xendrix combines conversational AI, document analysis, multilingual capabilities, data visualization, and image generation to deliver a powerful interactive experience.
</p>

---

## Features

### Chat Interface
- Context-aware conversation with persistent history
- Multilingual support (English, Hindi, Tamil, Telugu, French, and more)
- AI introduction detection across languages
- Mathematical rendering using KaTeX
- Code highlighting via Prism.js
- Typing animation for AI responses

### File Upload + Retrieval-Augmented Generation (RAG)
- Supports `.pdf`, `.docx`, and `.csv` uploads
- Text is chunked, embedded using `all-MiniLM-L6-v2`, and indexed with FAISS
- AI answers based on contextually retrieved document content

### Data Visualization
- Parses structured/tabular data from chat or uploaded files
- Generates bar, pie, line, and scatter plots
- Backend rendering with `matplotlib`, frontend with Recharts

### Image Generation
- Generates images based on user prompts
- Uses Stable Diffusion v1.5 with DPM++ 2M Karras scheduler
- Compatible with CPU and CUDA-enabled GPUs

### Additional Smart Features
- Weather queries using Weatherstack API
- Web search with SerpAPI (key rotation support)
- Math problem solving via SymPy
- Multilingual name and entity recognition

---

## Tech Stack

| Layer        | Technologies & Libraries |
|--------------|---------------------------|
| **Backend**  | FastAPI, FAISS, SentenceTransformers, Matplotlib, SymPy, Ollama (Mistral), Stable Diffusion |
| **Frontend** | Next.js (React), Recharts, Prism.js, KaTeX |
| **Models**   | `all-MiniLM-L6-v2`, `Stable Diffusion v1.5`, `MistralAI`, `Gemma` |
| **APIs**     | Weatherstack, SerpAPI, Deep Translator |
| **Storage**  | JSON files, Pickle, FAISS vector store |

---
