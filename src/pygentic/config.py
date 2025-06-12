#!/usr/bin/env uv run
# /// script
# dependencies = [
#     "mirascope",
#     "pydantic",
#     "python-dotenv"
# ]
# ///

from __future__ import annotations

import os
from typing import Any, Literal
from pathlib import Path
from dotenv import load_dotenv

from pydantic import BaseModel, Field, validator
from mirascope import llm

# Get LLM API Keys:

load_dotenv()


class LLMConfig(BaseModel):
    """Configuration for LLM providers and models."""

    provider: llm.Provider = Field(default="openai", description="LLM provider to use")
    model: str = Field(default="gpt-4o-mini", description="Model name to use")
    temperature: float = Field(
        default=0.7, ge=0.0, le=2.0, description="Sampling temperature"
    )
    max_tokens: int | None = Field(
        default=None, gt=0, description="Maximum tokens to generate"
    )
    stream: bool = Field(default=False, description="Whether to stream responses")
    json_mode: bool = Field(default=False, description="Whether to use JSON mode")
    system_prompt: str | None = Field(default=None, description="System prompt to use")
    context_files: list[Path] = Field(
        default_factory=list, description="Files to load as context"
    )
    call_params: dict[str, Any] = Field(
        default_factory=dict, description="Additional call parameters"
    )

    @validator("context_files")
    def validate_context_files(cls, v: list[Path]) -> list[Path]:
        """Validate that context files exist."""
        for file_path in v:
            if not file_path.exists():
                raise ValueError(f"Context file not found: {file_path}")
        return v

    def get_call_params(self) -> dict[str, Any]:
        """Get call parameters for Mirascope."""
        params = {"temperature": self.temperature, **self.call_params}
        if self.max_tokens is not None:
            params["max_tokens"] = self.max_tokens
        return params

    def load_context(self) -> str:
        """Load content from context files."""
        if not self.context_files:
            return ""

        context_parts = []
        for file_path in self.context_files:
            try:
                content = file_path.read_text(encoding="utf-8")
                context_parts.append(f"=== {file_path.name} ===\n{content}\n")
            except Exception as e:
                context_parts.append(
                    f"=== {file_path.name} (Error) ===\nCould not read file: {e}\n"
                )

        return "\n".join(context_parts)

    class Config:
        """Pydantic configuration."""

        arbitrary_types_allowed = True
