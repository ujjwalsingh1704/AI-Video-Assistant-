"""
Extractor module — structured extraction of action items, key decisions,
open questions, and chapters from meeting transcripts.

Accepts both a plain transcript string and an optional list of speaker segments
(each with keys: text, speaker, start, end).  When segments are provided the
prompts are enriched with speaker-aware context and source timestamps are
populated in the returned Pydantic objects.

Return convention (backward-compatible with app.py):
  extract_action_items  -> tuple[str, list[ActionItem]]
  extract_key_decisions -> tuple[str, list[KeyDecision]]
  extract_questions     -> tuple[str, list[OpenQuestion]]
  extract_chapters      -> list[Chapter]          (no display-string needed)
  extract_all           -> MeetingInsights
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from core.schemas import (
    ActionItem,
    KeyDecision,
    OpenQuestion,
    Chapter,
    MeetingInsights,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LLM / chain helpers
# ---------------------------------------------------------------------------

def get_llm() -> ChatGroq:
    """Return a configured ChatGroq instance."""
    return ChatGroq(
        model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
        groq_api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.2,
    )


def build_chain(system_prompt: str):
    """Build a simple LangChain chain: passthrough → prompt → LLM → str."""
    llm = get_llm()
    return (
        RunnablePassthrough()
        | RunnableLambda(lambda x: {"text": x})
        | ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{text}"),
        ])
        | llm
        | StrOutputParser()
    )


# ---------------------------------------------------------------------------
# Segment formatting helper
# ---------------------------------------------------------------------------

def format_segments_for_prompt(segments: list[dict]) -> str:
    """
    Convert a list of segment dicts into a human-readable, speaker-aware
    transcript for inclusion in LLM prompts.

    Each segment must have keys: text, speaker, start (float seconds).
    Output format per line:  [MM:SS SPEAKER]: "text"
    """
    lines: list[str] = []
    for seg in segments:
        start = seg.get("start", 0.0)
        minutes = int(start) // 60
        seconds = int(start) % 60
        speaker = seg.get("speaker", "UNKNOWN")
        text = seg.get("text", "").strip()
        lines.append(f"[{minutes:02d}:{seconds:02d} {speaker}]: \"{text}\"")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal: JSON extraction with retry
# ---------------------------------------------------------------------------

def _extract_json_with_retry(
    system_prompt: str,
    user_text: str,
    pydantic_class,
) -> tuple[str, list]:
    """
    Invoke LLM with *system_prompt* and *user_text*, ask for a JSON array
    matching *pydantic_class*, parse it, and return (raw_text, objects).

    On first parse failure retries once with a correction prompt.
    On second failure falls back to (raw_text, []).
    """
    llm = get_llm()
    parser = JsonOutputParser(pydantic_object=pydantic_class)

    # --- First attempt ---
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{text}"),
    ])
    chain = prompt | llm | StrOutputParser()

    raw_text: str = ""
    try:
        raw_text = chain.invoke({"text": user_text})
        parsed = parser.parse(raw_text)
        # JsonOutputParser may return a list or a dict with a list inside
        if isinstance(parsed, dict):
            # Try common wrapper keys
            for key in ("items", "action_items", "decisions", "questions", "chapters", "data"):
                if key in parsed:
                    parsed = parsed[key]
                    break
            else:
                # Wrap single object in a list
                parsed = [parsed]
        objects = [pydantic_class(**item) if isinstance(item, dict) else item for item in parsed]
        return raw_text, objects

    except Exception as first_err:
        logger.warning("First JSON parse attempt failed: %s — retrying with correction prompt.", first_err)

    # --- Retry attempt ---
    correction_system = (
        "You previously returned text that was not valid JSON. "
        "Return ONLY a JSON array matching this schema, with no explanation, "
        "no markdown, no code fences — just the raw JSON array.\n\n"
        f"Schema: {pydantic_class.model_json_schema()}"
    )
    retry_prompt = ChatPromptTemplate.from_messages([
        ("system", correction_system),
        ("human", "Previous (broken) response:\n{prev}\n\nOriginal context:\n{text}"),
    ])
    retry_chain = retry_prompt | llm | StrOutputParser()

    try:
        raw_retry = retry_chain.invoke({"prev": raw_text, "text": user_text})
        parsed = parser.parse(raw_retry)
        if isinstance(parsed, dict):
            for key in ("items", "action_items", "decisions", "questions", "chapters", "data"):
                if key in parsed:
                    parsed = parsed[key]
                    break
            else:
                parsed = [parsed]
        objects = [pydantic_class(**item) if isinstance(item, dict) else item for item in parsed]
        return raw_retry, objects

    except Exception as retry_err:
        logger.error("Retry JSON parse also failed: %s — falling back to empty list.", retry_err)
        return raw_text, []


# ---------------------------------------------------------------------------
# Public extraction functions
# ---------------------------------------------------------------------------

def extract_action_items(
    transcript: str,
    segments: list[dict] | None = None,
) -> tuple[str, list[ActionItem]]:
    """
    Extract action items from *transcript*.

    Returns (formatted_display_string, list[ActionItem]).
    The formatted string is used by app.py for backward compatibility.
    When *segments* are provided, speaker attribution and source timestamps
    are populated.
    """
    schema = ActionItem.model_json_schema()
    has_segments = bool(segments)

    if has_segments:
        speaker_context = format_segments_for_prompt(segments)
        system_prompt = (
            "You are an expert meeting analyst.\n\n"
            "Below is a speaker-annotated transcript.  "
            "Extract every action item — a concrete task assigned to or taken on by someone.\n\n"
            "Return ONLY a JSON array where each element matches this schema:\n"
            f"{json.dumps(schema, indent=2)}\n\n"
            "Rules:\n"
            "- 'owner': use the speaker label (e.g. SPEAKER_A) if a real name is not known.\n"
            "- 'source_timestamp': the 'start' time in seconds of the segment where the task appears.\n"
            "- 'confidence': your confidence (0.0-1.0) that this is a genuine action item.\n"
            "- Return an empty array [] if no action items are found.\n"
            "- Return ONLY raw JSON — no markdown, no code fences, no explanation."
        )
        user_text = f"Speaker-annotated transcript:\n{speaker_context}\n\nFull transcript:\n{transcript}"
    else:
        system_prompt = (
            "You are an expert meeting analyst.\n\n"
            "From the meeting transcript, extract every action item — a concrete task "
            "assigned to or taken on by someone.\n\n"
            "Return ONLY a JSON array where each element matches this schema:\n"
            f"{json.dumps(schema, indent=2)}\n\n"
            "Rules:\n"
            "- 'owner': use 'Unknown' if no person is mentioned.\n"
            "- 'source_timestamp': set to null (no timestamps available).\n"
            "- 'confidence': your confidence (0.0-1.0) that this is a genuine action item.\n"
            "- Return an empty array [] if no action items are found.\n"
            "- Return ONLY raw JSON — no markdown, no code fences, no explanation."
        )
        user_text = transcript

    raw_text, items = _extract_json_with_retry(system_prompt, user_text, ActionItem)

    # Build a human-readable display string
    if items:
        lines = ["**Action Items:**\n"]
        for i, item in enumerate(items, 1):
            ts = f" (at {item.source_timestamp:.0f}s)" if item.source_timestamp is not None else ""
            deadline = f" — deadline: {item.deadline}" if item.deadline else ""
            lines.append(f"{i}. [{item.owner}]{ts} {item.task}{deadline}")
        display = "\n".join(lines)
    else:
        display = raw_text or "No action items found."

    return display, items


def extract_key_decisions(
    transcript: str,
    segments: list[dict] | None = None,
) -> tuple[str, list[KeyDecision]]:
    """
    Extract key decisions from *transcript*.

    Returns (formatted_display_string, list[KeyDecision]).
    """
    schema = KeyDecision.model_json_schema()
    has_segments = bool(segments)

    if has_segments:
        speaker_context = format_segments_for_prompt(segments)
        system_prompt = (
            "You are an expert meeting analyst.\n\n"
            "Below is a speaker-annotated transcript.  "
            "Extract every key decision made during the meeting.\n\n"
            "Return ONLY a JSON array where each element matches this schema:\n"
            f"{json.dumps(schema, indent=2)}\n\n"
            "Rules:\n"
            "- 'made_by': speaker label or name of who made/announced the decision.\n"
            "- 'source_timestamp': the 'start' time in seconds of the segment.\n"
            "- Return an empty array [] if no decisions are found.\n"
            "- Return ONLY raw JSON — no markdown, no code fences, no explanation."
        )
        user_text = f"Speaker-annotated transcript:\n{speaker_context}\n\nFull transcript:\n{transcript}"
    else:
        system_prompt = (
            "You are an expert meeting analyst.\n\n"
            "From the meeting transcript, extract every key decision made.\n\n"
            "Return ONLY a JSON array where each element matches this schema:\n"
            f"{json.dumps(schema, indent=2)}\n\n"
            "Rules:\n"
            "- 'made_by': set to null if unknown.\n"
            "- 'source_timestamp': set to null.\n"
            "- Return an empty array [] if no decisions are found.\n"
            "- Return ONLY raw JSON — no markdown, no code fences, no explanation."
        )
        user_text = transcript

    raw_text, items = _extract_json_with_retry(system_prompt, user_text, KeyDecision)

    if items:
        lines = ["**Key Decisions:**\n"]
        for i, item in enumerate(items, 1):
            ts = f" (at {item.source_timestamp:.0f}s)" if item.source_timestamp is not None else ""
            by = f" — by {item.made_by}" if item.made_by else ""
            lines.append(f"{i}.{ts}{by} {item.decision}")
        display = "\n".join(lines)
    else:
        display = raw_text or "No key decisions found."

    return display, items


def extract_questions(
    transcript: str,
    segments: list[dict] | None = None,
) -> tuple[str, list[OpenQuestion]]:
    """
    Extract open/unresolved questions from *transcript*.

    Returns (formatted_display_string, list[OpenQuestion]).
    """
    schema = OpenQuestion.model_json_schema()
    has_segments = bool(segments)

    if has_segments:
        speaker_context = format_segments_for_prompt(segments)
        system_prompt = (
            "You are an expert meeting analyst.\n\n"
            "Below is a speaker-annotated transcript.  "
            "Extract every unresolved question or topic needing follow-up.\n\n"
            "Return ONLY a JSON array where each element matches this schema:\n"
            f"{json.dumps(schema, indent=2)}\n\n"
            "Rules:\n"
            "- 'raised_by': speaker label or name who raised the question.\n"
            "- 'source_timestamp': the 'start' time in seconds of the segment.\n"
            "- Return an empty array [] if no open questions are found.\n"
            "- Return ONLY raw JSON — no markdown, no code fences, no explanation."
        )
        user_text = f"Speaker-annotated transcript:\n{speaker_context}\n\nFull transcript:\n{transcript}"
    else:
        system_prompt = (
            "You are an expert meeting analyst.\n\n"
            "From the meeting transcript, extract every unresolved question or "
            "topic that needs follow-up.\n\n"
            "Return ONLY a JSON array where each element matches this schema:\n"
            f"{json.dumps(schema, indent=2)}\n\n"
            "Rules:\n"
            "- 'raised_by': set to null if unknown.\n"
            "- 'source_timestamp': set to null.\n"
            "- Return an empty array [] if no open questions are found.\n"
            "- Return ONLY raw JSON — no markdown, no code fences, no explanation."
        )
        user_text = transcript

    raw_text, items = _extract_json_with_retry(system_prompt, user_text, OpenQuestion)

    if items:
        lines = ["**Open Questions:**\n"]
        for i, item in enumerate(items, 1):
            ts = f" (at {item.source_timestamp:.0f}s)" if item.source_timestamp is not None else ""
            by = f" — raised by {item.raised_by}" if item.raised_by else ""
            lines.append(f"{i}.{ts}{by} {item.question}")
        display = "\n".join(lines)
    else:
        display = raw_text or "No open questions found."

    return display, items


def extract_chapters(
    transcript: str,
    segments: list[dict] | None = None,
) -> list[Chapter]:
    """
    Divide *transcript* into logical chapters/topics with timestamps.

    Returns list[Chapter].  No display string — callers format as needed.
    """
    schema = Chapter.model_json_schema()
    has_segments = bool(segments)

    if has_segments:
        speaker_context = format_segments_for_prompt(segments)
        system_prompt = (
            "You are an expert meeting analyst.\n\n"
            "Below is a speaker-annotated transcript.  "
            "Divide the meeting into logical chapters or topic sections.\n\n"
            "Return ONLY a JSON array where each element matches this schema:\n"
            f"{json.dumps(schema, indent=2)}\n\n"
            "Rules:\n"
            "- 'start_timestamp': the start time in seconds (from the segment timestamps) "
            "where this topic begins.\n"
            "- 'title': a short, descriptive title for the section.\n"
            "- 'summary': 1-2 sentence summary of what was discussed.\n"
            "- Return ONLY raw JSON — no markdown, no code fences, no explanation."
        )
        user_text = f"Speaker-annotated transcript:\n{speaker_context}\n\nFull transcript:\n{transcript}"
    else:
        system_prompt = (
            "You are an expert meeting analyst.\n\n"
            "From the meeting transcript, divide the conversation into logical chapters "
            "or topic sections.\n\n"
            "Return ONLY a JSON array where each element matches this schema:\n"
            f"{json.dumps(schema, indent=2)}\n\n"
            "Rules:\n"
            "- 'start_timestamp': estimate a time in seconds (start at 0.0 for the first chapter).\n"
            "- 'title': a short, descriptive title.\n"
            "- 'summary': 1-2 sentence summary.\n"
            "- Return ONLY raw JSON — no markdown, no code fences, no explanation."
        )
        user_text = transcript

    _raw_text, items = _extract_json_with_retry(system_prompt, user_text, Chapter)
    return items


# ---------------------------------------------------------------------------
# Aggregate extractor
# ---------------------------------------------------------------------------

def extract_all(
    transcript: str,
    segments: list[dict] | None = None,
) -> MeetingInsights:
    """
    Run all four extractors and return a combined MeetingInsights object.
    """
    _action_display, action_items = extract_action_items(transcript, segments)
    _decision_display, decisions = extract_key_decisions(transcript, segments)
    _question_display, questions = extract_questions(transcript, segments)
    chapters = extract_chapters(transcript, segments)

    return MeetingInsights(
        action_items=action_items,
        key_decisions=decisions,
        open_questions=questions,
        chapters=chapters,
    )
