#!/usr/bin/env uv run
# /// script
# dependencies = [
#     "mirascope",
#     "pydantic",
# ]
# ///

"""
PyGentic - Self-generating Pydantic models powered by Mirascope.

Provides easy-to-use interfaces for single calls, stateful conversations,
and self-generating Pydantic models with LLM-powered field population.
"""

from __future__ import annotations

from .config import LLMConfig
from .base import BaseLLM, LLMResponse
from .simple import SimpleLLM
from .conversation import ConversationLLM, ConversationMessage, ConversationState
from .genmodel import GenModel, generated_property, GenField

__version__ = "0.1.0"
__all__ = [
    "LLMConfig",
    "BaseLLM", 
    "LLMResponse",
    "SimpleLLM",
    "ConversationLLM",
    "ConversationMessage",
    "ConversationState",
    "GenModel",
    "generated_property",
    "GenField",
]


def create_simple_llm(
    provider: str = "openai",
    model: str = "gpt-4o-mini",
    temperature: float = 0.7,
    **kwargs
) -> SimpleLLM:
    """Factory function to create a SimpleLLM with common settings."""
    config = LLMConfig(
        provider=provider,
        model=model,
        temperature=temperature,
        **kwargs
    )
    return SimpleLLM(config=config)


def create_conversation(
    provider: str = "openai",
    model: str = "gpt-4o-mini",
    temperature: float = 0.7,
    state_file: str | None = None,
    **kwargs
) -> ConversationLLM:
    """Factory function to create a ConversationLLM with common settings."""
    config = LLMConfig(
        provider=provider,
        model=model,
        temperature=temperature,
        **kwargs
    )
    return ConversationLLM(config=config, state_file=state_file)

