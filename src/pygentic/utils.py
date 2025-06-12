#!/usr/bin/env uv run
# /// script
# dependencies = [
#     "mirascope",
#     "pydantic",
# ]
# ///

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

T = TypeVar('T', bound=BaseModel)


def clean_text(text: str) -> str:
    """Clean and normalize text content."""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove leading/trailing whitespace
    text = text.strip()
    return text


def load_multiple_files(
    file_paths: list[Path | str],
    separator: str = "\n\n---\n\n"
) -> str:
    """Load and combine content from multiple files."""
    contents = []
    
    for file_path in file_paths:
        path = Path(file_path)
        if path.exists():
            try:
                content = path.read_text(encoding='utf-8')
                contents.append(f"=== {path.name} ===\n{content}")
            except Exception as e:
                contents.append(f"=== {path.name} (Error) ===\n{e}")
        else:
            contents.append(f"=== {path.name} (Not Found) ===")
    
    return separator.join(contents)


def extract_code_blocks(text: str) -> list[str]:
    """Extract code blocks from markdown-formatted text."""
    pattern = r'```(?:\w+)?\n(.*?)\n```'
    matches = re.findall(pattern, text, re.DOTALL)
    return [match.strip() for match in matches]


def create_context_from_directory(
    directory: Path | str,
    extensions: list[str] | None = None,
    recursive: bool = True
) -> str:
    """Create context from all files in a directory."""
    directory = Path(directory)
    
    if not directory.is_dir():
        raise ValueError(f"Not a directory: {directory}")
    
    if extensions is None:
        extensions = ['.txt', '.md', '.py', '.json', '.yaml', '.yml']
    
    files = []
    if recursive:
        for ext in extensions:
            files.extend(directory.rglob(f'*{ext}'))
    else:
        for ext in extensions:
            files.extend(directory.glob(f'*{ext}'))
    
    return load_multiple_files(files)


def truncate_text(
    text: str, 
    max_length: int = 1000,
    suffix: str = "..."
) -> str:
    """Truncate text to maximum length with suffix."""
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix


def validate_response_model(
    response_data: Any, 
    model_class: type[T]
) -> T:
    """Validate and parse response data into a Pydantic model."""
    if isinstance(response_data, model_class):
        return response_data
    
    if isinstance(response_data, dict):
        return model_class(**response_data)
    
    raise ValueError(
        f"Cannot convert {type(response_data)} to {model_class.__name__}"
    )


def safe_filename(filename: str) -> str:
    """Create a safe filename by removing invalid characters."""
    # Remove invalid characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Remove leading/trailing dots and spaces
    filename = filename.strip('. ')
    # Limit length
    return filename[:200] if len(filename) > 200 else filename


def format_conversation_export(
    messages: list[dict[str, Any]],
    format_type: str = "markdown"
) -> str:
    """Format conversation messages for export."""
    if format_type == "markdown":
        lines = []
        for msg in messages:
            role = msg.get('role', 'unknown').title()
            content = msg.get('content', '')
            timestamp = msg.get('timestamp', '')
            
            lines.append(f"## {role}")
            if timestamp:
                lines.append(f"*{timestamp}*")
            lines.append(f"\n{content}\n")
        
        return "\n".join(lines)
    
    elif format_type == "text":
        lines = []
        for msg in messages:
            role = msg.get('role', 'unknown').upper()
            content = msg.get('content', '')
            lines.append(f"{role}: {content}")
        
        return "\n\n".join(lines)
    
    else:
        raise ValueError(f"Unsupported format type: {format_type}")
