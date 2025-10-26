<p align="center">
  <img src="frontend/public/logo.png" alt="Xendrix Logo" width="180"/>
</p>

# Xendrix AI Assistant

An intelligent, multimodal AI assistant with a FastAPI backend and a Next.js frontend. Xendrix combines conversational AI, retrieval-augmented document analysis, multilingual support, data visualization, and image generation to provide an interactive assistant experience.

This README summarizes the project structure, features, installation steps, configuration, and how to run the app based on the repository code.

---

## Key Features

- Conversation chat interface with persistent chat history
- Multilingual support (English, Hindi, Tamil, Telugu, French, Spanish, German, Japanese, Chinese, Russian, Arabic, Portuguese)
- Introduction/name detection across multiple languages
- File upload + Retrieval-Augmented Generation (RAG) for `.pdf`, `.docx`, and `.csv`
  - Text chunking, embedding (SentenceTransformers `all-MiniLM-L6-v2`) and FAISS indexing
- Data visualization: backend-rendered charts (matplotlib) + frontend charts (Recharts)
- Image generation using Stable Diffusion v1.5 (CPU/CUDA compatible)
- Math problem solving via SymPy and KaTeX rendering support in the frontend
- Code highlighting (Prism.js), typing animation for AI responses
- Utility features: Weather queries (Weatherstack), web search (SerpAPI with key rotation), translations (deep-translator)

---

## Tech Stack

- Backend: FastAPI, Python
  - Core libs: sentence-transformers, faiss, matplotlib, sympy, pandas, PyPDF2, python-docx
  - Models / Engines: `all-MiniLM-L6-v2` (embeddings), Stable Diffusion v1.5 (image gen), Ollama-hosted models (Mistral/Gemma) for chat (configurable)
  - Storage: JSON files (chats.json), Pickle, FAISS vector store
- Frontend: Next.js (React), Recharts, Prism.js, KaTeX
- Other: SerpAPI, Weatherstack, Deep Translator
- Containerization: optional Docker workflow (Dockerfile not guaranteed)

---

## Repo layout (high level)

- frontend/ — Next.js application (UI, chat, upload UI, visualization)
- backend/ — FastAPI service (chat endpoints, file ingestion, RAG, image generation)
- chats.json — persistent chat history (written by backend)
- vector_db/ — FAISS indices and pickled metadata (created at runtime)
- examples/ (may include sample inputs)

---

## Installation (development)

Prerequisites:
- Python 3.9+ (recommended)
- Node.js 16+ / pnpm or npm
- (Optional) CUDA-enabled GPU + NVIDIA drivers for faster Stable Diffusion
- FFmpeg (depending on any media processing in future)

1. Clone the repo
   ```
   git clone https://github.com/Kishore-1803/XendrixAI.git
   cd XendrixAI
   ```

2. Backend: create and activate a virtual environment, install dependencies
   ```
   cd backend
   python -m venv .venv
   # macOS / Linux
   source .venv/bin/activate
   # Windows
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Frontend: install dependencies and start dev server
   ```
   cd ../frontend
   npm install
   npm run dev
   # or
   # pnpm install && pnpm dev
   ```

---

## Configuration / Environment Variables

Add a `.env` or set environment variables for the following (examples):

- SERPAPI_KEY — SerpAPI key (for web search)
- WEATHERSTACK_KEY — Weatherstack API key (for weather queries)
- OLLAMA_HOST — (if using a local/remote Ollama service)
- VECTOR_DB_DIR — directory for FAISS indices (defaults to `vector_db`)
- CHAT_HISTORY_FILE — path to chat history JSON (defaults to `chats.json`)
- Any other API keys (Deep Translator if required by setup)

Note: Backend code uses several constants (e.g., TOP_K_RESULTS, CHUNK_SIZE) that can be adjusted in backend/app.py.

---

## Running the services

Backend (FastAPI)
```
cd backend
# recommended: from virtualenv with dependencies installed
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Frontend (Next.js)
```
cd frontend
npm run dev
# Open http://localhost:3000
```

The frontend expects the backend API at http://localhost:8000 by default (fetch calls to endpoints such as /chats, /documents, /languages, /upload_file, /new_chat).

---

## Important Endpoints (backend)

- GET  /chats           — list chat histories
- POST /new_chat?document_id=<id>&language=<lang> — create a new chat optionally tied to an uploaded document
- GET  /documents       — list uploaded documents
- POST /upload_file     — upload files (.pdf, .docx, .csv) for ingestion and indexing
- GET  /languages       — supported languages list
- Additional endpoints: image generation, visualization generation, search/RAG endpoints

Refer to backend/app.py for exact route signatures, expected request bodies and response payloads.

---

## File Upload & RAG flow

- Files accepted: .pdf, .docx, .csv
- Text is extracted then chunked (CHUNK_SIZE, CHUNK_OVERLAP in backend)
- Embeddings are computed with SentenceTransformers (e.g., `all-MiniLM-L6-v2`)
- FAISS index stores embeddings and metadata; query-time retrieval supplies context to the chat model

---

## Image Generation

- Stable Diffusion v1.5 pipeline is loaded at backend startup.
- The app uses a DPM++ style scheduler for higher-quality results.
- If CUDA is available and torch detects it, the model is moved to GPU automatically for faster generation. Otherwise CPU fallback is used (much slower).

---

## Data Visualization

- Backend can render matplotlib charts (bar, pie, line, scatter) and return them to the frontend.
- Frontend provides interactive visualizations using Recharts for client-side rendering.

---

## Development

- Linting, tests and type checks can be added to the repo. Suggested tools:
  - flake8 / black / isort for Python
  - eslint / prettier for frontend
  - pytest for tests
- Use feature branches, open pull requests and run CI prior to merging.

---

## Troubleshooting / Tips

- If the frontend shows fetch errors, ensure the backend is running on port 8000 and CORS is configured to allow http://localhost:3000.
- For large PDFs or many documents, FAISS index building can be memory intensive. Consider batching or increasing system resources.
- Stable Diffusion on CPU can be very slow and memory heavy — use GPU if available.
- Keep API keys out of version control (.env + .gitignore).

---

## Contributing

Contributions welcome. Suggested workflow:
- Fork the repo, create feature branches from `main`: `git checkout -b feat/your-feature`
- Add tests for new functionality
- Open a pull request describing your changes

Please include a CONTRIBUTING.md if you plan to accept outside contributions.

---
