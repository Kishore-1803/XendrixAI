# 🧠 Xendrix AI Assistant

An intelligent, multimodal AI assistant built using **FastAPI** for the backend and **Next.js (React)** for the frontend. Xendrix supports:

- Conversational AI with context
- File-based RAG (PDF, DOCX, CSV)
- Multilingual support and translation
- Math problem solving with symbolic computation
- Chart/Graph visualization from structured/unstructured data
- Image generation using Stable Diffusion
- Weather queries, SerpAPI integration, and more!

---

## 🚀 Features

### 💬 Chat Interface
- Multilingual chat (English, Hindi, Tamil, etc.)
- Name detection & AI intro in multiple languages
- Typing animation, KaTeX for math, Prism.js for code highlighting

### 📄 File Upload & RAG
- Supports `.pdf`, `.docx`, `.csv` files
- Vector embedding via `sentence-transformers`
- Context-aware Q&A using FAISS similarity search

### 📊 Data Visualization
- Extracts structured data from text or tables
- Generates bar, pie, line, and scatter charts
- Frontend visualizations via Recharts + static image from Matplotlib

### 🎨 Image Generation
- Prompts generate images using Stable Diffusion v1.5
- Enhanced with DPM++ 2M Karras sampling

### 🌐 Translation & Search
- Text translated using Deep Translator
- SerpAPI integration for factual queries
- Weatherstack integration for real-time weather info

---

## 🧰 Tech Stack

| Layer      | Tools/Libs Used |
|------------|-----------------|
| **Backend**    | FastAPI, FAISS, SentenceTransformers, Matplotlib, SymPy, Ollama (Mistral), Stable Diffusion |
| **Frontend**   | Next.js (React), Recharts, PrismJS, KaTeX |
| **ML Models**  | `all-MiniLM-L6-v2`, Stable Diffusion v1.5 |
| **APIs**       | Weatherstack, SerpAPI, Deep Translator |
| **Storage**    | Local filesystem, Pickle, JSON |

---

## 🛠️ Setup Instructions

### Backend (FastAPI)

```bash
# Install Python packages
pip install -r requirements.txt

# Run the API server
uvicorn app:app --reload
