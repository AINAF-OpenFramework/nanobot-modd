"""Data structures for Fractal Memory and Active Learning State."""

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class FractalNode(BaseModel):
    """
    A fractal memory node representing a lesson or knowledge piece.
    
    Stored as lesson_X.json in archives directory.
    """
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.now)
    tags: list[str] = Field(default_factory=list)
    content: str  # The core lesson/fact
    context_summary: str = ""  # Brief summary for the index
    embedding: list[float] | None = None  # Placeholder for vector embeddings
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ActiveLearningState(BaseModel):
    """
    Active Learning State tracking the agent's evolution and focus areas.
    
    Stored as ALS.json in memory directory.
    """
    
    current_focus: str = "General Assistance"
    sparring_partners: list[str] = Field(default_factory=list)  # User IDs or personas
    evolution_stage: int = 1
    recent_reflections: list[str] = Field(default_factory=list)
    last_updated: datetime = Field(default_factory=datetime.now)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ContextBlock(BaseModel):
    """
    A block in the context-engineered workflow.
    
    Used for structured prompt building.
    """
    
    name: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
