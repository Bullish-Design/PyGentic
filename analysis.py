"""Cost and usage analysis for LLM archives."""

from collections import defaultdict
from typing import Dict

from models import Archive, CostEstimate


PRICING_TABLE = {
    "gpt-4": {
        "input": 0.03 / 1000,
        "output": 0.06 / 1000,
    },
    "gpt-4-turbo": {
        "input": 0.01 / 1000,
        "output": 0.03 / 1000,
    },
    "gpt-4o": {
        "input": 0.0025 / 1000,
        "output": 0.01 / 1000,
    },
    "gpt-3.5-turbo": {
        "input": 0.0005 / 1000,
        "output": 0.0015 / 1000,
    },
    "claude-3-opus": {
        "input": 0.015 / 1000,
        "output": 0.075 / 1000,
    },
    "claude-3-sonnet": {
        "input": 0.003 / 1000,
        "output": 0.015 / 1000,
    },
    "claude-3-haiku": {
        "input": 0.00025 / 1000,
        "output": 0.00125 / 1000,
    },
    "claude-3-5-sonnet": {
        "input": 0.003 / 1000,
        "output": 0.015 / 1000,
    },
}


def normalize_model_name(model_name: str) -> str:
    """Normalize model name to match pricing table keys."""
    model_lower = model_name.lower()

    if "gpt-4o" in model_lower:
        return "gpt-4o"
    elif "gpt-4-turbo" in model_lower:
        return "gpt-4-turbo"
    elif "gpt-4" in model_lower:
        return "gpt-4"
    elif "gpt-3.5" in model_lower:
        return "gpt-3.5-turbo"
    elif "claude-3-5-sonnet" in model_lower or "claude-3.5-sonnet" in model_lower:
        return "claude-3-5-sonnet"
    elif "claude-3-opus" in model_lower:
        return "claude-3-opus"
    elif "claude-3-sonnet" in model_lower:
        return "claude-3-sonnet"
    elif "claude-3-haiku" in model_lower:
        return "claude-3-haiku"

    return model_name


def analyze_archive(archive: Archive) -> list[CostEstimate]:
    """Analyze archive and compute cost estimates per model."""
    usage_by_model: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"input": 0, "output": 0, "total": 0}
    )

    for conversation in archive.conversations:
        for message in conversation.messages:
            if message.model_usage:
                model = normalize_model_name(message.model_usage.model_name)
                usage_by_model[model]["input"] += message.model_usage.input_tokens
                usage_by_model[model]["output"] += message.model_usage.output_tokens
                usage_by_model[model]["total"] += message.model_usage.total_tokens

    estimates = []
    for model_name, usage in usage_by_model.items():
        pricing = PRICING_TABLE.get(model_name, {"input": 0.0, "output": 0.0})

        input_cost = usage["input"] * pricing["input"]
        output_cost = usage["output"] * pricing["output"]
        total_cost = input_cost + output_cost

        estimates.append(CostEstimate(
            model_name=model_name,
            currency="USD",
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=total_cost,
            pricing_source="built-in",
            input_tokens=usage["input"],
            output_tokens=usage["output"]
        ))

    return estimates
