<p align="center">
  <img src="frontend/public/logo.png" alt="Xendrix Logo" width="180"/>
</p>

<h1 align="center">Xendrix AI Assistant</h1>

<p align="center">
  <strong>An intelligent, multimodal AI assistant with chat, RAG, visualization, and image generation.</strong>
</p>

<p align="center">
  <a href="#"><img src="https://img.shields.io/badge/Python-3.9%2B-blue.svg"/></a>
  <a href="#"><img src="https://img.shields.io/badge/FastAPI-Backend-green.svg"/></a>
  <a href="#"><img src="https://img.shields.io/badge/Next.js-Frontend-black.svg"/></a>
  <a href="#"><img src="https://img.shields.io/badge/AI-Multimodal-purple.svg"/></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg"/></a>
  <a href="#"><img src="https://img.shields.io/badge/Status-Active-success.svg"/></a>
</p>

---

## 📌 Overview

**Xendrix** is a **multimodal AI assistant** combining conversational AI, Retrieval-Augmented Generation (RAG), multilingual understanding, data visualization, mathematical reasoning, and image generation.

It uses a **FastAPI backend** and a **Next.js frontend**, making it suitable for **research, productivity tools, and real-world AI assistant deployments**.

---

## ✨ Key Features

- 💬 Conversational chat with **persistent chat history**
- 🌍 Multilingual support  
  *(English, Hindi, Tamil, Telugu, French, Spanish, German, Japanese, Chinese, Russian, Arabic, Portuguese)*
- 🧠 Introduction & name detection across languages
- 📄 File upload + **Retrieval-Augmented Generation (RAG)**
  - Supported formats: `.pdf`, `.docx`, `.csv`
  - Chunking, embeddings (`all-MiniLM-L6-v2`), FAISS indexing
- 📊 Data visualization
  - Backend-rendered charts (Matplotlib)
  - Frontend interactive charts (Recharts)
- 🎨 Image generation using **Stable Diffusion v1.5** (CPU / CUDA)
- ➗ Math problem solving using **SymPy** with KaTeX rendering
- 💻 Code highlighting (Prism.js) + typing animation
- 🌦️ Utility integrations
  - Weather queries (Weatherstack)
  - Web search (SerpAPI with key rotation)
  - Translations (Deep Translator)

---

## 🛠️ Tech Stack

### Backend
- **Framework**: FastAPI (Python)
- **Core Libraries**:
  - sentence-transformers
  - FAISS
  - matplotlib
  - sympy
  - pandas
  - PyPDF2
  - python-docx
- **Models / Engines**:
  - Embeddings: `all-MiniLM-L6-v2`
  - Image Generation: Stable Diffusion v1.5
  - Chat Models: Ollama-hosted models (Mistral / Gemma – configurable)
- **Storage**:
  - JSON (chat history)
  - Pickle
  - FAISS vector store

### Frontend
- **Framework**: Next.js (React)
- **Visualization**: Recharts
- **Rendering**: Prism.js, KaTeX

### Other
- SerpAPI
- Weatherstack
- Deep Translator
- Optional Docker workflow

---

## 📂 Project Structure

```text
XendrixAI/
├── frontend/                 # Next.js frontend (UI, chat, visualizations)
├── backend/                  # FastAPI backend (AI logic, RAG, image gen)
├── chats.json                # Persistent chat history (auto-generated)
├── vector_db/                # FAISS indices & metadata (runtime)
├── examples/                 # Sample inputs (if present)
├── LICENSE                   # MIT License
└── README.md                 # Project documentation
````

---

## 🚀 Installation (Development)

### Prerequisites

* Python **3.9+**
* Node.js **16+**
* npm or pnpm
* *(Optional)* CUDA-enabled GPU for faster image generation

---

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Kishore-1803/XendrixAI.git
cd XendrixAI
```

---

### 2️⃣ Backend Setup

```bash
cd backend
python -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate

pip install -r requirements.txt
```

---

### 3️⃣ Frontend Setup

```bash
cd ../frontend
npm install
npm run dev
# or: pnpm install && pnpm dev
```

---

## ⚙️ Configuration (Environment Variables)

Create a `.env` file or export variables:

```env
SERPAPI_KEY=your_serpapi_key
WEATHERSTACK_KEY=your_weatherstack_key
OLLAMA_HOST=http://localhost:11434
VECTOR_DB_DIR=vector_db
CHAT_HISTORY_FILE=chats.json
```

Backend constants such as `CHUNK_SIZE`, `TOP_K_RESULTS`, etc., can be adjusted in `backend/app.py`.

---

## ▶️ Running the Application

### Backend

```bash
cd backend
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### Frontend

```bash
cd frontend
npm run dev
```

📍 Frontend: [http://localhost:3000](http://localhost:3000)
📍 Backend API: [http://localhost:8000](http://localhost:8000)

---

## 🔌 Important Backend Endpoints

* `GET  /chats` — List chat histories
* `POST /new_chat` — Create a new chat session
* `GET  /documents` — List uploaded documents
* `POST /upload_file` — Upload files for RAG
* `GET  /languages` — Supported languages
* Additional endpoints for image generation, visualization, and search

Refer to `backend/app.py` for full details.

---

## 📄 File Upload & RAG Pipeline

1. File ingestion (`.pdf`, `.docx`, `.csv`)
2. Text extraction and chunking
3. Embedding generation (`SentenceTransformers`)
4. FAISS indexing
5. Context retrieval at query time

---

## 🎨 Image Generation

* Uses **Stable Diffusion v1.5**
* Automatically switches to **GPU** if CUDA is available
* CPU fallback supported (slower)

---

## 📊 Data Visualization

* Backend generates Matplotlib plots
* Frontend renders interactive charts via Recharts

---

## 🧪 Development Notes

Recommended tooling:

* Python: `black`, `flake8`, `isort`, `pytest`
* Frontend: `eslint`, `prettier`

---

## 🛠️ Troubleshooting

* Ensure backend runs on port **8000**
* Enable CORS for `http://localhost:3000`
* Large PDFs may require higher memory
* Keep `.env` files out of version control

---

## 🤝 Contributing

Contributions are welcome!

1. Fork the repository
2. Create a feature branch

   ```bash
   git checkout -b feat/your-feature
   ```
3. Commit changes and open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License**.
See the [LICENSE](LICENSE) file for details.

---

⭐ **If you like this project, consider starring the repository!** ⭐

Just tell me 🚀
```
