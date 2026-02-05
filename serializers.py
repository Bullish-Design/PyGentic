"""Serialization utilities for outputting analysis results."""

import json
from pathlib import Path

from models import Archive, CostEstimate


def write_jsonl(items: list, output_path: Path) -> None:
    """Write a list of Pydantic models to JSONL format."""
    with open(output_path, "w", encoding="utf-8") as f:
        for item in items:
            json_str = item.model_dump_json()
            f.write(json_str + "\n")


def write_summary_report(
    archive: Archive,
    cost_estimates: list[CostEstimate],
    output_path: Path
) -> None:
    """Write a human-readable summary report."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# LLM Archive Analysis Report\n\n")
        f.write(f"**Provider:** {archive.source}\n")
        f.write(f"**Ingested at:** {archive.ingested_at.isoformat()}\n")
        f.write(f"**Total conversations:** {len(archive.conversations)}\n")

        total_messages = sum(len(conv.messages) for conv in archive.conversations)
        f.write(f"**Total messages:** {total_messages}\n\n")

        f.write("## Cost Estimates\n\n")

        if cost_estimates:
            total_cost = sum(est.total_cost for est in cost_estimates)
            f.write(f"**Total estimated cost:** ${total_cost:.4f} USD\n\n")

            f.write("| Model | Input Tokens | Output Tokens | Input Cost | Output Cost | Total Cost |\n")
            f.write("|-------|--------------|---------------|------------|-------------|------------|\n")

            for est in cost_estimates:
                f.write(
                    f"| {est.model_name} | "
                    f"{est.input_tokens:,} | "
                    f"{est.output_tokens:,} | "
                    f"${est.input_cost:.4f} | "
                    f"${est.output_cost:.4f} | "
                    f"${est.total_cost:.4f} |\n"
                )
        else:
            f.write("No usage data available (archives may not include token counts).\n")

        f.write("\n## Conversations\n\n")
        for i, conv in enumerate(archive.conversations[:10], 1):
            f.write(f"{i}. **{conv.title or conv.id}** - {len(conv.messages)} messages\n")

        if len(archive.conversations) > 10:
            f.write(f"\n... and {len(archive.conversations) - 10} more conversations\n")
