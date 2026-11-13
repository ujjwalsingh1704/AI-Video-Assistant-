"""
core/vector_store.py

Enhanced vector store with:
  - Speaker-aware chunking when segments are provided
  - Hybrid BM25 + Chroma ensemble retriever (falls back to pure Chroma if rank_bm25 is unavailable)
  - Backward-compatible character-based chunking when no segments are given
"""

import os
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# ── Constants ────────────────────────────────────────────────────────────────
CHROMA_DIR = "vector_db"
COLLECTION_NAME = "meeting_transcript"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# ── Embeddings helper ─────────────────────────────────────────────────────────
def get_embeddings() -> HuggingFaceEmbeddings:
    """Return the shared HuggingFace embedding model (CPU)."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"}
    )


# ── Build vector store ────────────────────────────────────────────────────────
def build_vector_store(
    transcript: str,
    segments: list[dict] = None
) -> Chroma:
    """
    Build (or rebuild) the Chroma vector store.

    Parameters
    ----------
    transcript : str
        Full plain-text transcript.  Always required as a fallback.
    segments   : list[dict], optional
        Structured segments from the transcriber, each with keys:
            text     – spoken text for this turn
            speaker  – e.g. "SPEAKER_00"
            start    – float, seconds from start of recording
            end      – float, seconds from start of recording
        When provided, one Document is created per segment, carrying
        rich metadata for citation and hybrid search.

    Returns
    -------
    Chroma
        A persisted Chroma collection ready for retrieval.
    """
    print("Building vector store …")
    embeddings = get_embeddings()

    if segments:
        # ── Speaker-aware chunking ─────────────────────────────────────────
        # Each speaker segment becomes its own Document so we preserve
        # speaker identity and exact timestamps in the vector index.
        docs = [
            Document(
                page_content=seg["text"].strip(),
                metadata={
                    "speaker":     seg.get("speaker", "UNKNOWN"),
                    "start":       seg.get("start", 0.0),
                    "end":         seg.get("end", 0.0),
                    "chunk_index": i,
                }
            )
            for i, seg in enumerate(segments)
            if seg.get("text", "").strip()          # skip empty segments
        ]
        print(f"  → {len(docs)} speaker-aware chunks created from {len(segments)} segments.")
    else:
        # ── Backward-compatible character-based chunking ───────────────────
        # Used when segments are unavailable (e.g. old transcriber output).
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )
        chunks = splitter.split_text(transcript)
        docs = [
            Document(
                page_content=chunk,
                metadata={"chunk_index": i}
            )
            for i, chunk in enumerate(chunks)
        ]
        print(f"  → {len(docs)} character-based chunks created (no segments provided).")

    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR
    )

    return vector_store


# ── Load existing vector store ────────────────────────────────────────────────
def load_vector_store() -> Chroma:
    """
    Load a previously persisted Chroma collection from disk.
    Raises if the collection does not exist yet.
    """
    embeddings = get_embeddings()

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR
    )

    return vector_store


# ── Build retriever ───────────────────────────────────────────────────────────
def get_retriever(vector_store: Chroma, k: int = 6):
    """
    Build the best available retriever.

    Strategy
    --------
    1. Try to construct a hybrid retriever:
         BM25Retriever (sparse / keyword)  weight=0.4
       + Chroma similarity retriever       weight=0.6
       using langchain's EnsembleRetriever.
    2. If rank_bm25 / BM25Retriever is not installed, fall back to a
       pure Chroma similarity retriever — identical to the original behaviour.

    Parameters
    ----------
    vector_store : Chroma
        An already-built or loaded Chroma instance.
    k            : int
        Number of documents to retrieve.

    Returns
    -------
    A LangChain retriever (EnsembleRetriever or Chroma retriever).
    """
    # Chroma similarity retriever — always available
    chroma_retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )

    try:
        from langchain_community.retrievers import BM25Retriever
        from langchain.retrievers import EnsembleRetriever

        # Fetch all stored documents so BM25 can build its index
        all_docs = vector_store.get(include=["documents", "metadatas"])
        bm25_docs = [
            Document(page_content=text, metadata=meta)
            for text, meta in zip(
                all_docs["documents"],
                all_docs["metadatas"]
            )
        ]

        if not bm25_docs:
            # Empty store — nothing to index with BM25
            raise ValueError("No documents in the store; cannot build BM25 index.")

        bm25_retriever = BM25Retriever.from_documents(bm25_docs)
        bm25_retriever.k = k

        hybrid_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, chroma_retriever],
            weights=[0.4, 0.6]       # sparse (keyword) + dense (semantic)
        )

        print("  → Hybrid BM25 + Chroma retriever created.")
        return hybrid_retriever

    except ImportError:
        # rank_bm25 not installed — degrade gracefully
        print("  → rank_bm25 not installed; using pure Chroma similarity retriever.")
        return chroma_retriever

    except Exception as exc:
        # Any other error (empty store, etc.) — degrade gracefully
        print(f"  → Hybrid retriever failed ({exc}); using pure Chroma similarity retriever.")
        return chroma_retriever
