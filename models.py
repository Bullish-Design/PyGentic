"""Pydantic domain models for LLM archive analysis."""

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field


class ModelUsage(BaseModel):
    """Token usage for a specific model."""

    model_name: str
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


class Message(BaseModel):
    """A single message in a conversation."""

    id: str
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    created_at: Optional[datetime] = None
    model_usage: Optional[ModelUsage] = None


class Conversation(BaseModel):
    """A conversation containing multiple messages."""

    id: str
    title: Optional[str] = None
    started_at: Optional[datetime] = None
    messages: list[Message] = Field(default_factory=list)


class Archive(BaseModel):
    """Complete archive from a provider."""

    source: Literal["chatgpt", "claude"]
    ingested_at: datetime
    conversations: list[Conversation] = Field(default_factory=list)


class CostEstimate(BaseModel):
    """Cost estimate for model usage."""

    model_name: str
    currency: str = "USD"
    input_cost: float = 0.0
    output_cost: float = 0.0
    total_cost: float = 0.0
    pricing_source: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
