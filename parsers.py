"""Parsers for ChatGPT and Claude archive exports."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from models import Archive, Conversation, Message, ModelUsage


def parse_chatgpt_export(file_path: Path) -> Archive:
    """Parse ChatGPT conversations.json export."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    conversations = []

    for conv_data in data:
        conv_id = conv_data.get("id", conv_data.get("conversation_id", "unknown"))
        title = conv_data.get("title")

        created_time = conv_data.get("create_time")
        started_at = datetime.fromtimestamp(created_time) if created_time else None

        messages = []
        mapping = conv_data.get("mapping", {})

        for node_id, node_data in mapping.items():
            message_data = node_data.get("message")
            if not message_data:
                continue

            msg_id = message_data.get("id", node_id)
            author = message_data.get("author", {})
            role = author.get("role", "unknown")

            if role not in ["system", "user", "assistant", "tool"]:
                role = "assistant" if role == "assistant" else "user"

            content_data = message_data.get("content", {})
            if isinstance(content_data, dict):
                parts = content_data.get("parts", [])
                content = " ".join(str(part) for part in parts if part)
            else:
                content = str(content_data)

            created_time = message_data.get("create_time")
            created_at = datetime.fromtimestamp(created_time) if created_time else None

            metadata = message_data.get("metadata", {})
            model_slug = metadata.get("model_slug")
            model_usage = None

            if model_slug:
                model_usage = ModelUsage(
                    model_name=model_slug,
                    input_tokens=0,
                    output_tokens=0,
                    total_tokens=0
                )

            messages.append(Message(
                id=msg_id,
                role=role,
                content=content,
                created_at=created_at,
                model_usage=model_usage
            ))

        if messages:
            conversations.append(Conversation(
                id=conv_id,
                title=title,
                started_at=started_at,
                messages=messages
            ))

    return Archive(
        source="chatgpt",
        ingested_at=datetime.now(),
        conversations=conversations
    )


def parse_claude_export(file_path: Path) -> Archive:
    """Parse Claude export (conversations.json or similar)."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    conversations = []

    if isinstance(data, list):
        conv_list = data
    else:
        conv_list = [data]

    for conv_data in conv_list:
        conv_id = conv_data.get("uuid", conv_data.get("id", "unknown"))
        title = conv_data.get("name", conv_data.get("title"))

        created_at_str = conv_data.get("created_at")
        started_at = None
        if created_at_str:
            try:
                started_at = datetime.fromisoformat(created_at_str.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                pass

        messages = []
        chat_messages = conv_data.get("chat_messages", [])

        for msg_data in chat_messages:
            msg_id = msg_data.get("uuid", msg_data.get("id", "unknown"))
            sender = msg_data.get("sender", "user")

            role = "user" if sender == "human" else "assistant"
            if sender in ["system"]:
                role = "system"

            content = msg_data.get("text", "")

            created_at_str = msg_data.get("created_at")
            created_at = None
            if created_at_str:
                try:
                    created_at = datetime.fromisoformat(created_at_str.replace("Z", "+00:00"))
                except (ValueError, AttributeError):
                    pass

            model_usage = None
            if "model" in msg_data:
                model_name = msg_data["model"]
                model_usage = ModelUsage(
                    model_name=model_name,
                    input_tokens=0,
                    output_tokens=0,
                    total_tokens=0
                )

            messages.append(Message(
                id=msg_id,
                role=role,
                content=content,
                created_at=created_at,
                model_usage=model_usage
            ))

        if messages:
            conversations.append(Conversation(
                id=conv_id,
                title=title,
                started_at=started_at,
                messages=messages
            ))

    return Archive(
        source="claude",
        ingested_at=datetime.now(),
        conversations=conversations
    )


def parse_archive(file_path: Path, provider: Literal["chatgpt", "claude"]) -> Archive:
    """Parse an archive export based on provider."""
    if provider == "chatgpt":
        return parse_chatgpt_export(file_path)
    elif provider == "claude":
        return parse_claude_export(file_path)
    else:
        raise ValueError(f"Unknown provider: {provider}")
