#!/usr/bin/env uv run
# /// script
# dependencies = [
#     "mirascope",
#     "pydantic",
# ]
# ///

from __future__ import annotations

from pathlib import Path
from typing import Any, TypeVar, Type

from pydantic import Field
from mirascope import llm

from .base import BaseLLM, LLMResponse
from .config import LLMConfig

T = TypeVar('T')


class SimpleLLM(BaseLLM):
    """Simple single-call LLM implementation."""
    
    def call(
        self,
        prompt: str | Path,
        additional_context: str = "",
        response_model: Type[T] | None = None,
        tools: list[Any] | None = None,
        **kwargs
    ) -> LLMResponse[T]:
        """Make a single LLM call."""
        
        # Handle file input
        if isinstance(prompt, (str, Path)) and Path(prompt).exists():
            prompt = self.load_file_content(prompt)
        
        # Build messages
        messages = self._build_messages(
            str(prompt), 
            additional_context
        )
        
        # Create dynamic config
        dynamic_config = self._create_dynamic_config(
            messages, 
            response_model, 
            tools
        )
        
        # Define the actual call function
        @llm.call(
            provider=self.config.provider,
            model=self.config.model,
            stream=self.config.stream,
            **kwargs
        )
        def _make_call() -> dict[str, Any]:
            return dynamic_config
        
        # Make the call
        response = _make_call()
        
        # Handle streaming vs non-streaming
        if self.config.stream:
            content_parts = []
            for chunk, _ in response:
                content_parts.append(chunk.content)
            content = "".join(content_parts)
            raw_response = response
        else:
            content = response.content
            raw_response = response
        
        # Extract parsed response if response_model was used
        parsed_response = None
        if response_model and hasattr(response, 'response_model'):
            parsed_response = response
        elif response_model and not self.config.stream:
            # For streaming, the response model is the final response
            parsed_response = response if isinstance(
                response, response_model
            ) else None
        
        return LLMResponse[T](
            content=content,
            raw_response=raw_response,
            config=self.config,
            parsed_response=parsed_response
        )
    
    def quick_call(
        self, 
        prompt: str, 
        model: str | None = None,
        temperature: float | None = None,
        **kwargs
    ) -> str:
        """Quick call with optional parameter overrides."""
        
        # Create temporary config with overrides
        config_dict = self.config.dict()
        if model:
            config_dict["model"] = model
        if temperature is not None:
            config_dict["temperature"] = temperature
        
        temp_config = LLMConfig(**config_dict)
        temp_llm = SimpleLLM(config=temp_config)
        
        response = temp_llm.call(prompt, **kwargs)
        return response.content
    
    def extract_structured(
        self, 
        prompt: str, 
        response_model: Type[T],
        **kwargs
    ) -> T:
        """Extract structured data using a response model."""
        response = self.call(
            prompt=prompt, 
            response_model=response_model,
            **kwargs
        )
        
        if response.parsed_response is None:
            raise ValueError(
                "Failed to parse response into structured format"
            )
        
        return response.parsed_response
