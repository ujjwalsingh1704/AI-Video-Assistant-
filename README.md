# 🎬 AI Video Assistant (Meeting Intelligence)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io)
[![LangChain](https://img.shields.io/badge/LangChain-0.2%2B-1C3C3C.svg?logo=langchain&logoColor=white)](https://langchain.com)
[![faster--whisper](https://img.shields.io/badge/faster--whisper-CTranslate2-purple.svg)](https://github.com/SYSTRAN/faster-whisper)
[![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-orange.svg)](https://trychroma.com)
[![Groq](https://img.shields.io/badge/Inference-Groq_LPU-F55036.svg)](https://groq.com)

An enterprise-ready AI meeting intelligence assistant designed for **code-mixed (Hinglish / English)** and multi-speaker meetings. Converts YouTube videos or local audio/video uploads into structured notes, speaker-attributed action items, timestamped chapters, and an interactive **Hybrid RAG (BM25 + Dense Vector Search)** chat.

---

## 🌟 Key Highlights & Innovations

- ⚡ **4× Faster Transcription with VAD**: Powered by `faster-whisper` (CTranslate2 `int8` quantization) with Voice Activity Detection to skip silence.
- 🗣️ **Speaker Diarization**: Multi-speaker attribution (`SPEAKER_00`, `SPEAKER_01`) via WhisperX integration.
- 🇮🇳 **Code-Mixed Hinglish Support**: First-class support for Indian team meetings with Sarvam AI STT & translation fallback.
- 📦 **Type-Safe Pydantic Extraction**: Action items, key decisions, open questions, and chapters validated with Pydantic and self-healing LLM retry loops.
- 🧠 **Hybrid RAG Engine (BM25 + Chroma)**: Combines keyword search (for project names, numbers, acronyms) and semantic embeddings (`all-MiniLM-L6-v2`) via `EnsembleRetriever`.
- ⏱️ **Timestamp Citations**: RAG responses cite exact moments (`[@ 04:32]`), with expandable snippet sources in chat.
- 📄 **1-Click Markdown Export**: Download fully formatted executive meeting summaries and transcripts.

---

## 🏗️ System Architecture & Workflow

```mermaid
flowchart TD
    subgraph Ingestion["1. Audio Ingestion & Preprocessing"]
        A["YouTube URL / Video File / Audio Upload"] --> B["utils/audio_processor.py"]
        B --> C["Convert to 16kHz Mono WAV"]
        C --> D["Split into 10-min Chunks with Time Offsets"]
    end

    subgraph Transcription["2. ASR & Diarization Engine"]
        D --> E{"Language Mode"}
        E -- "Auto / English" --> F["faster-whisper (VAD enabled)"]
        E -- "Hinglish" --> G["Sarvam AI STT Engine"]
        F --> H["WhisperX Diarization Pipeline\n(Speaker Recognition)"]
        G --> H
        H --> I["Structured Segments\n[{text, speaker, start, end}]"]
    end

    subgraph Extraction["3. Structured Intelligence (Pydantic)"]
        I --> J["core/extractor.py & core/summarize.py"]
        J --> K["Session Title & Executive Summary"]
        J --> L["Action Items (with Owners & Deadlines)"]
        J --> M["Key Decisions (with Decision Maker)"]
        J --> N["Open Follow-up Questions"]
        J --> O["Chapters & Timestamp Segmentation"]
    end

    subgraph RAG["4. Hybrid RAG Pipeline"]
        I --> P["core/vector_store.py"]
        P --> Q["Speaker-Turn Chunks (with Metadata)"]
        Q --> R1["BM25 Keyword Retriever"]
        Q --> R2["HuggingFace Embeddings + ChromaDB"]
        R1 & R2 --> S["LangChain EnsembleRetriever\n(Weights: 0.4 BM25 + 0.6 Chroma)"]
        S --> T["core/rag_engine.py"]
        T --> U["Groq LLM (gpt-oss-20b)\nContext Grounding + Citation Prompt"]
    end

    subgraph UI["5. Cyber-Dark Streamlit UI"]
        K & L & M & N & O --> V["Interactive Dashboard"]
        U --> W["Chat with Meeting + Citation Sources"]
        V --> X["1-Click Markdown Export (.md)"]
    end
```

---

## 🔍 Retrieval & Citation Flow (RAG Deep Dive)

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant Streamlit as Streamlit UI (app.py)
    participant Hybrid as Hybrid Retriever (BM25 + Chroma)
    participant Engine as RAG Engine (rag_engine.py)
    participant LLM as Groq LLM (gpt-oss-20b)

    User->>Streamlit: "Who is responsible for the budget deadline?"
    Streamlit->>Engine: ask_question_with_citations(question)
    Engine->>Hybrid: Query top-6 relevant chunks
    Hybrid-->>Engine: [SPEAKER_00 @ 03:15]: "Rahul handles budget by Friday"
    Engine->>LLM: Formatted prompt with attributed context & citation instructions
    LLM-->>Engine: "Rahul is responsible for the budget by Friday [@ 03:15]."
    Engine-->>Streamlit: {answer, sources: [{speaker, timestamp, text}]}
    Streamlit-->>User: Display answer bubble with expandable [📎 Sources]
```

---

## 📁 Repository Structure

```text
AI-Video-Assistant/
├── app.py                     # Streamlit frontend with cyber-dark theme & chat
├── requirements.txt           # Python dependencies
├── .env.example               # Template for environment variables
├── packages.txt               # System-level dependencies (ffmpeg)
│
├── core/
│   ├── schemas.py             # Pydantic models (ActionItem, Decision, Chapter, etc.)
│   ├── transcriber.py         # faster-whisper + WhisperX diarization + Sarvam
│   ├── extractor.py           # Structured extraction with retry logic & speaker context
│   ├── summarize.py           # Map-reduce summarization & meeting title generator
│   ├── vector_store.py        # Speaker-turn chunking & BM25 + Chroma hybrid retriever
│   └── rag_engine.py          # Grounded RAG chain with timestamp citation formatting
│
└── utils/
    └── audio_processor.py     # yt-dlp downloader, FFmpeg converter, audio chunker
```

---

## 🛠️ Installation & Setup

### Prerequisites
1. **Python 3.10+**
2. **FFmpeg**:
   - **Windows**: `winget install Gyan.FFmpeg` or `choco install ffmpeg`
   - **macOS**: `brew install ffmpeg`
   - **Ubuntu/Debian**: `sudo apt install ffmpeg`

### 1. Clone the Repository
```bash
git clone https://github.com/ujjwalsingh1704/AI-Video-Assistant-.git
cd AI-Video-Assistant-
```

### 2. Set Up Virtual Environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create a `.env` file in the project root:

```env
# Required for fast LLM inference & extraction
GROQ_API_KEY=your_groq_api_key

# Optional: Hugging Face Token for WhisperX speaker diarization
HF_TOKEN=your_huggingface_token

# Optional: Sarvam AI Key for Hinglish translation/STT
SARVAM_API_KEY=your_sarvam_api_key

# Model Configurations
WHISPER_MODEL=base
GROQ_MODEL=openai/gpt-oss-20b
SARVAM_STT_MODEL=saaras:v2.5
```

> **Note on Diarization**: To enable automatic speaker recognition, obtain a free Hugging Face token and accept the license on [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1). If `HF_TOKEN` is omitted, the app gracefully defaults to single-speaker mode (`SPEAKER_00`).

---

## 🚀 Running the Application

```bash
streamlit run app.py
```

1. Open your browser at `http://localhost:8501`.
2. Paste a YouTube URL or upload a local video/audio file (`.mp4`, `.mp3`, `.wav`, `.mkv`, `.webm`).
3. Select your language preference (`auto`, `english`, or `hinglish`).
4. Click **⚡ Analyse** to trigger the end-to-end pipeline.
5. Review the **Summary**, **Chapters**, **Action Items**, **Key Decisions**, and **Open Questions**.
6. Use the **💬 Chat with your Meeting** section to ask questions with grounded citations.
7. Click **📄 Export Markdown** to save the complete session report.

---

## 📊 Feature Comparison

| Feature | Vanilla Whisper Setup | This Assistant |
|---|---|---|
| **Transcription Speed** | Baseline | **~4× faster** (CTranslate2 + VAD) |
| **Silence Handling** | Transcribes background noise | **Auto-skipped** via Voice Activity Detection |
| **Speaker Identification** | None (Single text block) | **WhisperX Diarization** (`SPEAKER_00`, `SPEAKER_01`) |
| **Extraction Reliability** | Unstructured plain strings | **Pydantic Validation** with automated JSON retry |
| **Information Retrieval** | Pure Dense Vector Search | **Hybrid Search** (BM25 keyword + Chroma dense) |
| **Answer Grounding** | Generic LLM responses | **Exact Timestamp Citations** (`[@ MM:SS]`) |
| **Export Formats** | Copy-paste from UI | **Structured Markdown Export** |

---

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request:
1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.
