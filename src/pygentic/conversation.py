#!/usr/bin/env uv run
# /// script
# dependencies = [
#     "mirascope",
#     "pydantic",
# ]
# ///

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, TypeVar, Type

from pydantic import BaseModel, Field
from mirascope import Messages, llm, BaseMessageParam

from .base import BaseLLM, LLMResponse
from .config import LLMConfig

T = TypeVar('T')


class ConversationMessage(BaseModel):
    """A single message in a conversation."""
    
    role: str = Field(description="Message role (system/user/assistant)")
    content: str = Field(description="Message content")
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When message was created"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional message metadata"
    )


class ConversationState(BaseModel):
    """State of a conversation."""
    
    messages: list[ConversationMessage] = Field(
        default_factory=list,
        description="Conversation messages"
    )
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="When conversation was created"
    )
    updated_at: datetime = Field(
        default_factory=datetime.now,
        description="Last update time"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Conversation metadata"
    )


class ConversationLLM(BaseLLM):
    """Stateful conversation LLM implementation."""
    
    state: ConversationState = Field(
        default_factory=ConversationState,
        description="Conversation state"
    )
    state_file: Path | None = Field(
        default=None,
        description="File to persist conversation state"
    )
    auto_save: bool = Field(
        default=True,
        description="Auto-save state after each call"
    )
    
    def __init__(
        self, 
        config: LLMConfig | None = None,
        state_file: Path | str | None = None,
        auto_save: bool = True,
        **kwargs
    ):
        """Initialize conversation with optional state file."""
        super().__init__(config=config, **kwargs)
        
        self.state_file = Path(state_file) if state_file else None
        self.auto_save = auto_save
        
        # Load existing state if file exists
        if self.state_file and self.state_file.exists():
            self.load_state()
    
    def add_message(
        self, 
        role: str, 
        content: str,
        metadata: dict[str, Any] | None = None
    ) -> None:
        """Add a message to the conversation."""
        message = ConversationMessage(
            role=role,
            content=content,
            metadata=metadata or {}
        )
        self.state.messages.append(message)
        self.state.updated_at = datetime.now()
    
    def get_messages_for_llm(self) -> list[BaseMessageParam]:
        """Convert conversation messages to Mirascope format."""
        messages = []
        
        # Add system prompt if configured
        if self.config.system_prompt:
            messages.append(Messages.System(self.config.system_prompt))
        
        # Add conversation history
        for msg in self.state.messages:
            if msg.role == "user":
                messages.append(Messages.User(msg.content))
            elif msg.role == "assistant":
                messages.append(Messages.Assistant(msg.content))
            elif msg.role == "system":
                messages.append(Messages.System(msg.content))
        
        return messages
    
    def call(
        self,
        prompt: str | Path,
        additional_context: str = "",
        response_model: Type[T] | None = None,
        tools: list[Any] | None = None,
        save_to_history: bool = True,
        **kwargs
    ) -> LLMResponse[T]:
        """Make a conversational LLM call."""
        
        # Handle file input
        if isinstance(prompt, (str, Path)) and Path(prompt).exists():
            prompt = self.load_file_content(prompt)
        
        # Prepare full prompt with context
        full_prompt = self._prepare_prompt(str(prompt), additional_context)
        
        # Add user message to conversation
        if save_to_history:
            self.add_message("user", full_prompt)
        
        # Get all messages for LLM
        messages = self.get_messages_for_llm()
        
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
        
        # Add assistant response to conversation
        if save_to_history:
            self.add_message("assistant", content)
        
        # Auto-save if enabled
        if self.auto_save:
            self.save_state()
        
        # Extract parsed response if response_model was used
        parsed_response = None
        if response_model and hasattr(response, 'response_model'):
            parsed_response = response
        elif response_model and not self.config.stream:
            parsed_response = response if isinstance(
                response, response_model
            ) else None
        
        return LLMResponse[T](
            content=content,
            raw_response=raw_response,
            config=self.config,
            parsed_response=parsed_response
        )
    
    def save_state(self, file_path: Path | str | None = None) -> None:
        """Save conversation state to file."""
        target_file = Path(file_path) if file_path else self.state_file
        
        if not target_file:
            raise ValueError("No state file specified")
        
        # Ensure parent directory exists
        target_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Save state as JSON
        with open(target_file, 'w', encoding='utf-8') as f:
            json.dump(
                self.state.dict(),
                f,
                indent=2,
                default=str,
                ensure_ascii=False
            )
    
    def load_state(self, file_path: Path | str | None = None) -> None:
        """Load conversation state from file."""
        source_file = Path(file_path) if file_path else self.state_file
        
        if not source_file or not source_file.exists():
            raise FileNotFoundError(f"State file not found: {source_file}")
        
        with open(source_file, 'r', encoding='utf-8') as f:
            state_data = json.load(f)
        
        self.state = ConversationState(**state_data)
    
    def clear_conversation(self) -> None:
        """Clear the conversation history."""
        self.state.messages.clear()
        self.state.updated_at = datetime.now()
        
        if self.auto_save:
            self.save_state()
    
    def get_conversation_summary(self) -> str:
        """Get a summary of the conversation."""
        if not self.state.messages:
            return "Empty conversation"
        
        message_count = len(self.state.messages)
        user_messages = sum(
            1 for msg in self.state.messages if msg.role == "user"
        )
        assistant_messages = sum(
            1 for msg in self.state.messages if msg.role == "assistant"
        )
        
        return (
            f"Conversation: {message_count} total messages "
            f"({user_messages} user, {assistant_messages} assistant)\n"
            f"Created: {self.state.created_at}\n"
            f"Updated: {self.state.updated_at}"
        )
