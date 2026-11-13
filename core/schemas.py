"""
Pydantic schemas for structured extraction of meeting insights.
"""

from pydantic import BaseModel, Field
from typing import Optional


class ActionItem(BaseModel):
    task: str = Field(description="Description of the task to be done")
    owner: str = Field(description="Person responsible. Use speaker label if name unknown.")
    deadline: Optional[str] = Field(default=None, description="Deadline if mentioned, else None")
    source_timestamp: Optional[float] = Field(default=None, description="Start time in seconds")
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)


class KeyDecision(BaseModel):
    decision: str
    made_by: Optional[str] = None
    source_timestamp: Optional[float] = None


class OpenQuestion(BaseModel):
    question: str
    raised_by: Optional[str] = None
    source_timestamp: Optional[float] = None


class Chapter(BaseModel):
    title: str
    start_timestamp: float = Field(default=0.0)
    summary: str = ""

    @property
    def start_time(self) -> float:
        return self.start_timestamp


class MeetingInsights(BaseModel):
    action_items: list[ActionItem] = []
    key_decisions: list[KeyDecision] = []
    open_questions: list[OpenQuestion] = []
    chapters: list[Chapter] = []
