"""
core/rag_engine.py

Enhanced RAG engine with:
  - Speaker + timestamp citations in context and answers
  - ask_question_with_citations() returning structured sources
  - Hybrid retriever support (via updated vector_store.get_retriever)
  - Full backward compatibility (ask_question / load_rag_chain unchanged API)
"""

import os
from typing import Optional

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.documents import Document

from core.vector_store import (
    build_vector_store,
    load_vector_store,
    get_retriever
)


# ── LLM factory ───────────────────────────────────────────────────────────────
def get_llm() -> ChatGroq:
    """Return a Groq-hosted LLM instance."""
    return ChatGroq(
        model="openai/gpt-oss-20b",
        groq_api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.3
    )


# ── Document formatters ───────────────────────────────────────────────────────
def format_docs(docs: list[Document]) -> str:
    """
    Basic formatter — joins documents with double newlines.
    Kept for internal use; external callers should prefer
    format_docs_with_citations().
    """
    return "\n\n".join(doc.page_content for doc in docs)


def format_docs_with_citations(docs: list[Document]) -> str:
    """
    Format retrieved documents as attributed speaker quotes so the LLM
    can echo timestamps back to the user.

    Output format (when metadata is present):
        [SPEAKER_A @ 04:32]: "text content here"
        [SPEAKER_B @ 06:10]: "text content here"

    If a document carries no speaker metadata, the raw text is included
    without attribution so backward compatibility is preserved.
    """
    parts = []
    for doc in docs:
        speaker = doc.metadata.get("speaker")
        start   = doc.metadata.get("start")

        if speaker and start is not None:
            # Convert raw seconds → MM:SS for readability
            minutes, seconds = divmod(int(start), 60)
            timestamp = f"{minutes:02d}:{seconds:02d}"
            parts.append(f'[{speaker} @ {timestamp}]: "{doc.page_content}"')
        else:
            # Fallback: no attribution metadata available
            parts.append(doc.page_content)

    return "\n\n".join(parts)


# ── System prompt ─────────────────────────────────────────────────────────────
_SYSTEM_PROMPT = """\
You are an expert meeting assistant.

The context below is extracted from a meeting transcript.  Each passage is
labelled with the speaker name and the timestamp at which they spoke, using the
format:

    [SPEAKER_NAME @ MM:SS]: "spoken text"

Answer the user's question based ONLY on the context provided.

Rules:
- If the answer is present, give a clear and concise response.
- When you reference a specific moment or statement, include its timestamp in
  your answer using the format [@ MM:SS].
  Example: "The budget was set at 50 lakhs [@ 04:32]."
- If the answer is NOT found in the context, say:
  "I could not find this information in the meeting transcript."
- Never invent information not present in the context.

Context from meeting transcript:
{context}"""


# ── Internal helper: build the prompt + chain ─────────────────────────────────
def _build_chain_from_retriever(retriever):
    """
    Given a retriever, wire up the full RAG chain:
      retriever → format_docs_with_citations → prompt → LLM → StrOutputParser
    """
    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("system", _SYSTEM_PROMPT),
        ("human", "{question}"),
    ])

    rag_chain = (
        {
            "context":  retriever | RunnableLambda(format_docs_with_citations),
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


# ── Public build / load API ───────────────────────────────────────────────────
def build_rag_chain(
    transcript: str,
    segments: list[dict] = None
):
    """
    Build the RAG chain from scratch.

    Parameters
    ----------
    transcript : str
        Full plain-text transcript.
    segments   : list[dict], optional
        Structured segments (text, speaker, start, end) for speaker-aware
        chunking.  Passed through to build_vector_store().

    Returns
    -------
    A LangChain runnable chain.
    """
    vector_store = build_vector_store(transcript, segments=segments)
    retriever    = get_retriever(vector_store, k=6)
    return _build_chain_from_retriever(retriever)


def load_rag_chain(segments: list[dict] = None):
    """
    Load a RAG chain from a previously persisted vector store.

    Parameters
    ----------
    segments : list[dict], optional
        Passed to get_retriever — currently used only for informational
        purposes; the retriever reads from the persisted Chroma store.

    Returns
    -------
    A LangChain runnable chain.
    """
    vector_store = load_vector_store()
    retriever    = get_retriever(vector_store, k=6)
    return _build_chain_from_retriever(retriever)


# ── Question-answering helpers ─────────────────────────────────────────────────
def ask_question(rag_chain, question: str) -> str:
    """
    Ask a question and return a plain string answer.

    Backward-compatible wrapper — existing callers that only need the text
    response can continue using this function unchanged.
    """
    print(f"Question: {question}")
    answer = rag_chain.invoke(question)
    print(f"Answer: {answer}")
    return answer


def ask_question_with_citations(
    rag_chain,
    question: str
) -> dict:
    """
    Ask a question and return both the answer and the source documents.

    The chain is re-invoked with intermediate document capture so we can
    surface the retrieved passages alongside the final answer.

    Returns
    -------
    dict with keys:
        "answer"  : str            – the LLM's answer (may contain [@ MM:SS] refs)
        "sources" : list[dict]     – retrieved passages, each containing:
                        "speaker" : str   (or "" if unavailable)
                        "start"   : float (seconds, or 0.0 if unavailable)
                        "text"    : str   – the passage content
    """
    # ── 1. Retrieve relevant documents directly from the retriever ────────
    # The retriever is embedded inside the chain; we extract it by walking
    # the chain's first step (the dict's "context" branch).
    try:
        # LangChain Runnable graph: first step is a RunnableParallel dict.
        # We pull the retriever out so we can get docs AND run the chain.
        retriever = rag_chain.first.steps["context"].first
        retrieved_docs: list[Document] = retriever.invoke(question)
    except Exception:
        # Fallback: if introspection fails, run the chain normally and
        # return an empty sources list.
        answer = rag_chain.invoke(question)
        return {"answer": answer, "sources": []}

    # ── 2. Build structured source list ──────────────────────────────────
    sources = []
    for doc in retrieved_docs:
        sources.append({
            "speaker": doc.metadata.get("speaker", ""),
            "start":   doc.metadata.get("start", 0.0),
            "text":    doc.page_content,
        })

    # ── 3. Run the full chain (retriever → prompt → LLM) ─────────────────
    answer = rag_chain.invoke(question)

    return {"answer": answer, "sources": sources}
