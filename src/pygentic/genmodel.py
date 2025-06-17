#!/usr/bin/env uv run
# /// script
# dependencies = [
#     "mirascope",
#     "pydantic",
# ]
# ///

"""
PyGentic - Self-generating Pydantic models powered by Mirascope.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import Any, Callable, TypeVar, get_type_hints, get_origin, get_args, List

from pydantic import BaseModel, Field, ConfigDict
from mirascope import llm, Messages

from .config import LLMConfig

T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])


def GenField(prompt: str, **kwargs) -> Any:
    """Create a generative field that auto-populates via LLM."""
    return Field(default=None, description=prompt, **kwargs)


def generated_property(
    provider: str | None = None,
    model: str | None = None,
    response_model: type[T] | None = None,
    temperature: float | None = None,
    **kwargs,
) -> Callable[[F], property]:
    """Decorator for LLM-generated cached properties."""

    def decorator(func: F) -> property:
        @functools.wraps(func)
        def wrapper(self: GenModel) -> Any:
            # Check if already cached
            cache_key = f"_cached_{func.__name__}"
            if hasattr(self, cache_key):
                return getattr(self, cache_key)

            # Get base config and create overrides
            config = self._get_llm_config()
            print(f"\n\nUsing LLM config:\n    {config}\n")
            if provider:
                config.provider = provider
            if model:
                config.model = model
            if temperature is not None:
                # if provider == "openai":
                config.temperature = temperature
                # else:
                #    del config.temperature
                #    config.call_params = {"temperature": temperature}
            print(f"Updated LLM config:\n    {config}\n\n")

            # Format the docstring template
            prompt = func.__doc__ or f"Generate {func.__name__}"
            formatted_prompt = self._format_template(prompt)

            # Create messages
            messages = []
            if hasattr(self, "__doc__") and self.__doc__:
                messages.append(Messages.System(self.__doc__.strip()))
            messages.append(Messages.User(formatted_prompt))

            # Use type annotation as response model if not explicitly provided
            final_response_model = response_model
            if not final_response_model:
                hints = get_type_hints(func)
                if "return" in hints:
                    return_type = hints["return"]
                    # Use return type if it's not a basic type
                    if (
                        return_type not in (str, int, float, bool)
                        and return_type is not Any
                    ):
                        final_response_model = return_type

            llm_kwargs = {k: v for k, v in kwargs.items() if k != "depends_on"}
            # print(f"\n\nOG Kwargs:\n    {kwargs}")
            # print(f"\nLLM Kwargs:\n    {llm_kwargs}\n")
            # for key, val in llm_kwargs.items():
            #    print(f"    LLM call parameter: {key} = {val}")
            print(f"Config Call Params:\n    {config.get_call_params()}\n\n")
            # print(f"\n\n")

            # Make LLM call
            @llm.call(
                provider=config.provider,
                model=config.model,
                response_model=final_response_model,
                **llm_kwargs,
            )
            def _generate() -> dict[str, Any]:
                return {"messages": messages, "call_params": config.get_call_params()}

            result = _generate()

            # Extract content based on response type
            if final_response_model:
                content = result
            else:
                content = result.content

            # Cache and return
            setattr(self, cache_key, content)
            return content

        return property(wrapper)

    return decorator


class GenModel(BaseModel):
    """PyGentic base class for self-generating Pydantic models."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="allow",
        # json_encoders={Path: lambda p: str(p)},
    )

    _llm_config: LLMConfig | None = None
    output_file: Path | str | None = Field(
        default=None, description="File to save generated data", exclude=True
    )

    def __init__(self, output_file: Path | str | None = None, **data):
        """Initialize and auto-populate missing fields."""
        super().__init__(**data)

        if output_file:
            self.output_file = Path(output_file)
        self._populate_missing_fields()

    @classmethod
    def set_llm_config(cls, config: LLMConfig) -> None:
        """Set the default LLM configuration for this class."""
        cls._llm_config = config

    def _get_llm_config(self) -> LLMConfig:
        """Get LLM configuration, using class default or creating new."""
        if self._llm_config:
            return self._llm_config
        return LLMConfig()

    def _format_template(self, template: str) -> str:
        """Format template string with current field values."""
        format_dict = {}

        # Get all current field values
        for field_name, field_value in self.__dict__.items():
            if not field_name.startswith("_"):
                format_dict[field_name] = field_value

        try:
            return template.format(**format_dict)
        except KeyError as e:
            # If template references missing field, return as-is
            return template

    def _get_field_info(self) -> list[tuple[str, Any, str | None]]:
        """Get ordered field information with descriptions."""
        field_info = []

        # Get field order from model definition
        for field_name, field in self.__fields__.items():
            field_type = field.annotation
            description = None

            # Extract description from Field()
            if hasattr(field, "field_info") and field.field_info:
                description = field.field_info.description

            field_info.append((field_name, field_type, description))

        return field_info

    def _populate_missing_fields(self) -> None:
        """Populate fields that are None and have descriptions."""
        field_info = self._get_field_info()
        print(f"\n\nField Info:\n{field_info}\n\n")

        for field_name, field_type, description in field_info:
            # Skip if field already has value
            current_value = getattr(self, field_name, None)
            if current_value is not None:
                continue

            # Skip if no description (no prompt template)
            if not description:
                continue

            # Generate value using LLM
            generated_value = self._generate_field_value(
                field_name, field_type, description
            )

            if generated_value is not None:
                setattr(self, field_name, generated_value)

    def _generate_field_value(
        self, field_name: str, field_type: type, description: str
    ) -> Any:
        """Generate a single field value using LLM."""
        config = self._get_llm_config()

        # Format the description template
        formatted_prompt = self._format_template(description)

        # Build messages
        messages = []
        if hasattr(self, "__doc__") and self.__doc__:
            messages.append(Messages.System(self.__doc__.strip()))

        # Add context about current state
        current_fields = {
            k: v
            for k, v in self.__dict__.items()
            if not k.startswith("_") and v is not None
        }

        if current_fields:
            context = "Current information:\n"
            for k, v in current_fields.items():
                context += f"- {k}: {v}\n"
            messages.append(Messages.User(context))

        messages.append(Messages.User(f"Task: {formatted_prompt}"))

        # Use field type as response model if it's complex enough
        response_model = None
        origin = get_origin(field_type)

        if origin is list or origin is List:
            # For List[T], use the full type
            response_model = field_type
        elif hasattr(field_type, "__bases__") and BaseModel in field_type.__bases__:
            # For Pydantic models
            response_model = field_type
        elif field_type not in (str, int, float, bool, Any):
            # For other complex types
            response_model = field_type

        # Make LLM call
        @llm.call(
            provider=config.provider, model=config.model, response_model=response_model
        )
        def _generate() -> dict[str, Any]:
            return {"messages": messages, "call_params": config.get_call_params()}

        try:
            result = _generate()

            # Extract content based on response type
            if response_model:
                return result
            else:
                content = result.content.strip()

                # Try to convert to appropriate type
                if field_type == int:
                    try:
                        return int(content)
                    except ValueError:
                        # Extract number from text
                        import re

                        numbers = re.findall(r"\d+", content)
                        return int(numbers[0]) if numbers else None

                elif field_type == float:
                    try:
                        return float(content)
                    except ValueError:
                        import re

                        numbers = re.findall(r"\d+\.?\d*", content)
                        return float(numbers[0]) if numbers else None

                elif field_type == bool:
                    return content.lower() in ("true", "yes", "1", "on")

                else:
                    return content

        except Exception as e:
            print(f"Failed to generate {field_name}: {e}")
            return None

    def _serialize_state(self) -> dict[str, Any]:
        """Serialize complete object state including cached properties."""
        state = {}

        # Serialize regular fields
        for field_name in self.__fields__:
            value = getattr(self, field_name, None)
            if value is not None:
                # Handle complex types
                if isinstance(value, BaseModel):
                    state[field_name] = value.dict()
                elif (
                    isinstance(value, list)
                    and value
                    and isinstance(value[0], BaseModel)
                ):
                    state[field_name] = [item.dict() for item in value]
                else:
                    state[field_name] = value

        # Serialize cached properties
        cached_props = {}
        for attr_name in dir(self):
            if attr_name.startswith("_cached_"):
                prop_name = attr_name[8:]  # Remove '_cached_' prefix
                cached_props[prop_name] = getattr(self, attr_name)

        if cached_props:
            state["_cached_properties"] = cached_props

        # Add metadata
        state["_class_name"] = self.__class__.__name__
        state["_module_name"] = self.__class__.__module__

        return state

    def _deserialize_state(self, state: dict[str, Any]) -> None:
        """Restore object state including cached properties."""
        # Restore cached properties
        if "_cached_properties" in state:
            for prop_name, value in state["_cached_properties"].items():
                setattr(self, f"_cached_{prop_name}", value)

    def output(self, file_path: Path | str | None = None) -> None:
        """Write object state to JSONL file."""
        target_file = Path(file_path) if file_path else self.output_file

        if not target_file:
            raise ValueError("No output file specified")

        target_file = Path(target_file)

        # Ensure parent directory exists
        target_file.parent.mkdir(parents=True, exist_ok=True)

        # Serialize state
        state = self._serialize_state()

        # Write to JSONL (append mode for multiple objects)
        with open(target_file, "a", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, default=str)
            f.write("\n")

    @classmethod
    def from_jsonl(cls, file_path: Path | str, index: int = -1) -> "GenModel":
        """Load object from JSONL file without making LLM calls."""
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"JSONL file not found: {file_path}")

        # Read all lines
        lines = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    lines.append(json.loads(line.strip()))

        if not lines:
            raise ValueError("No data found in JSONL file")

        # Get the specified entry (default to last)
        if index < 0:
            index = len(lines) + index

        if index >= len(lines) or index < 0:
            raise IndexError(f"Index {index} out of range for {len(lines)} entries")

        state = lines[index]

        # Verify class match
        if state.get("_class_name") != cls.__name__:
            raise ValueError(
                f"Class mismatch: expected {cls.__name__}, "
                f"got {state.get('_class_name')}"
            )

        # Extract regular fields
        field_data = {
            k: v
            for k, v in state.items()
            if not k.startswith("_") and k in cls.__fields__
        }

        # Create instance without auto-population
        instance = cls.__new__(cls)
        BaseModel.__init__(instance, **field_data)

        # Restore cached properties
        instance._deserialize_state(state)

        return instance

    def __str__(self) -> str:
        """String representation showing all populated fields."""
        lines = [f"{self.__class__.__name__}:"]

        # Show regular fields
        for field_name in self.__fields__:
            value = getattr(self, field_name, None)
            if value is not None:
                lines.append(f"  {field_name}: {value}")

        # Show cached properties if any
        cached_props = []
        for attr_name in dir(self):
            if attr_name.startswith("_cached_"):
                prop_name = attr_name[8:]  # Remove '_cached_' prefix
                cached_props.append(prop_name)

        if cached_props:
            lines.append("  Cached Properties:")
            for prop_name in sorted(cached_props):
                lines.append(f"    {prop_name}: <cached>")

        return "\n".join(lines)
