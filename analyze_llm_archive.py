#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pydantic>=2.0.0",
# ]
# ///

"""
LLM Archive Analysis Script

Analyze ChatGPT and Claude archive exports to compute usage and cost estimates.

Usage:
    uv run analyze_llm_archive.py --provider chatgpt --file path/to/conversations.json
    uv run analyze_llm_archive.py --provider claude --file path/to/export.json

Output:
    - archive.jsonl: Complete normalized archive
    - conversations.jsonl: All conversations
    - messages.jsonl: All messages
    - cost_estimates.jsonl: Cost analysis per model
    - summary.md: Human-readable summary report
"""

import argparse
import sys
from pathlib import Path

from analysis import analyze_archive
from models import Archive
from parsers import parse_archive
from serializers import write_jsonl, write_summary_report


def main():
    parser = argparse.ArgumentParser(
        description="Analyze LLM chat archive exports (ChatGPT or Claude)"
    )
    parser.add_argument(
        "--provider",
        choices=["chatgpt", "claude"],
        required=True,
        help="Provider type (chatgpt or claude)"
    )
    parser.add_argument(
        "--file",
        type=Path,
        required=True,
        help="Path to the unzipped JSON export file"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./output"),
        help="Output directory for analysis results (default: ./output)"
    )

    args = parser.parse_args()

    if not args.file.exists():
        print(f"Error: File not found: {args.file}", file=sys.stderr)
        sys.exit(1)

    print(f"Parsing {args.provider} archive from {args.file}...")
    archive = parse_archive(args.file, args.provider)

    print(f"Found {len(archive.conversations)} conversations")
    total_messages = sum(len(conv.messages) for conv in archive.conversations)
    print(f"Found {total_messages} total messages")

    print("\nAnalyzing usage and costs...")
    cost_estimates = analyze_archive(archive)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nWriting results to {args.output_dir}/")

    archive_path = args.output_dir / "archive.jsonl"
    write_jsonl([archive], archive_path)
    print(f"  - {archive_path.name}")

    conversations_path = args.output_dir / "conversations.jsonl"
    write_jsonl(archive.conversations, conversations_path)
    print(f"  - {conversations_path.name}")

    all_messages = []
    for conv in archive.conversations:
        all_messages.extend(conv.messages)
    messages_path = args.output_dir / "messages.jsonl"
    write_jsonl(all_messages, messages_path)
    print(f"  - {messages_path.name}")

    if cost_estimates:
        cost_path = args.output_dir / "cost_estimates.jsonl"
        write_jsonl(cost_estimates, cost_path)
        print(f"  - {cost_path.name}")

        total_cost = sum(est.total_cost for est in cost_estimates)
        print(f"\nEstimated total cost: ${total_cost:.4f} USD")
    else:
        print("\nNo cost estimates available (archives may not include token counts)")

    summary_path = args.output_dir / "summary.md"
    write_summary_report(archive, cost_estimates, summary_path)
    print(f"  - {summary_path.name}")

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
