# 🧠 Xendrix AI Assistant

An intelligent, multimodal AI assistant built using **FastAPI** (backend) and **Next.js** (frontend). Xendrix combines conversational AI, document analysis, multilingual capabilities, data visualization, and image generation to provide a powerful interactive assistant.

---

## 🚀 Features

### 💬 Chat Interface
- Context-aware conversation history
- Multilingual support (English, Hindi, Tamil, Telugu, French, etc.)
- AI introduction detection in various languages
- KaTeX for math rendering
- Prism.js for code syntax highlighting
- Typing animation for AI responses

### 📄 File Upload + RAG (Retrieval-Augmented Generation)
- Upload `.pdf`, `.docx`, `.csv` files
- Extracted text is chunked, embedded using `all-MiniLM-L6-v2`, and indexed with FAISS
- AI answers contextually using relevant document chunks

### 📊 Data Visualization
- Parses tabular and structured data from chat or document content
- Supports **Bar**, **Pie**, **Line**, and **Scatter** charts
- Charts rendered with `matplotlib` (backend) and Recharts (frontend)

### 🎨 Image Generation
- Generates images from user prompts
- Uses Stable Diffusion v1.5 + DPM++ 2M Karras scheduler
- CUDA or CPU supported

### 🌍 Other Smart Features
- 🌦️ Weather queries using Weatherstack API
- 🔍 Web search using SerpAPI with fallback key rotation
- 📈 Math problem solving using SymPy
- 📚 Name recognition across languages

---

## 🧰 Tech Stack

| Layer        | Tools/Libs Used |
|--------------|-----------------|
| **Backend**  | FastAPI, FAISS, SentenceTransformers, Matplotlib, SymPy, Ollama (Mistral), Stable Diffusion |
| **Frontend** | Next.js (React), Recharts, PrismJS, KaTeX |
| **Models**   | `all-MiniLM-L6-v2`, `Stable Diffusion v1.5` ,`MistralAI` ,`gemma` |
| **APIs**     | Weatherstack, SerpAPI, Deep Translator |
| **Storage**  | JSON files, Pickle, FAISS vector store |

---

## 🛠️ Setup Instructions

### ✅ Prerequisites
- Python 3.8+
- Node.js 18+
- CUDA-enabled GPU (optional, for faster image generation)
