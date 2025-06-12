#!/usr/bin/env uv run
# /// script
# dependencies = [
#     "mirascope",
#     "pydantic",
# ]
# ///

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, TypeVar, Generic

from pydantic import BaseModel, Field
from mirascope import Messages, llm, BaseDynamicConfig, BaseMessageParam

from .config import LLMConfig

T = TypeVar('T')


class LLMResponse(BaseModel, Generic[T]):
    """Wrapper for LLM responses with metadata."""
    
    content: str = Field(description="Response content")
    raw_response: Any = Field(description="Raw Mirascope response")
    config: LLMConfig = Field(description="Configuration used")
    parsed_response: T | None = Field(
        default=None, 
        description="Parsed response if response_model used"
    )
    
    class Config:
        arbitrary_types_allowed = True


class BaseLLM(BaseModel, ABC):
    """Base class for LLM interactions."""
    
    config: LLMConfig = Field(description="LLM configuration")
    
    def __init__(self, config: LLMConfig | None = None, **kwargs):
        """Initialize with optional config override."""
        if config is None:
            config = LLMConfig()
        super().__init__(config=config, **kwargs)
    
    def _prepare_prompt(
        self, 
        prompt: str, 
        additional_context: str = ""
    ) -> str:
        """Prepare the complete prompt with context."""
        context_parts = []
        
        # Add file context
        file_context = self.config.load_context()
        if file_context:
            context_parts.append(f"Context:\n{file_context}")
        
        # Add additional context
        if additional_context:
            context_parts.append(f"Additional Context:\n{additional_context}")
        
        # Add main prompt
        context_parts.append(f"Request:\n{prompt}")
        
        return "\n\n".join(context_parts)
    
    def _build_messages(
        self, 
        prompt: str, 
        additional_context: str = ""
    ) -> list[BaseMessageParam]:
        """Build message list for Mirascope call."""
        messages = []
        
        # System message
        if self.config.system_prompt:
            messages.append(Messages.System(self.config.system_prompt))
        
        # User message with context and prompt
        full_prompt = self._prepare_prompt(prompt, additional_context)
        messages.append(Messages.User(full_prompt))
        
        return messages
    
    def _create_dynamic_config(
        self, 
        messages: list[BaseMessageParam],
        response_model: type[T] | None = None,
        tools: list[Any] | None = None
    ) -> BaseDynamicConfig:
        """Create dynamic configuration for Mirascope call."""
        config_dict = {
            "messages": messages,
            "call_params": self.config.get_call_params()
        }
        
        if response_model:
            config_dict["response_model"] = response_model
        
        if tools:
            config_dict["tools"] = tools
            
        if self.config.json_mode:
            config_dict["json_mode"] = True
            
        return config_dict
    
    def load_file_content(self, file_path: Path | str) -> str:
        """Load content from a file."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        try:
            return path.read_text(encoding='utf-8')
        except Exception as e:
            raise ValueError(f"Could not read file {path}: {e}")
    
    def add_context_file(self, file_path: Path | str) -> None:
        """Add a file to the context."""
        path = Path(file_path)
        if path not in self.config.context_files:
            self.config.context_files.append(path)
    
    def clear_context_files(self) -> None:
        """Clear all context files."""
        self.config.context_files.clear()
    
    @abstractmethod
    def call(
        self, 
        prompt: str | Path, 
        **kwargs
    ) -> LLMResponse[Any]:
        """Make an LLM call. Must be implemented by subclasses."""
        pass
    
    class Config:
        arbitrary_types_allowed = True
