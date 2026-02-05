# LLM Archive Analysis Script

A UV-runnable Python script for analyzing ChatGPT and Claude chat archive exports. Parses the exports into normalized Pydantic models, computes usage statistics, estimates costs, and outputs JSONL files plus summary reports.

## Quick Start

```bash
# Analyze a ChatGPT export
uv run analyze_llm_archive.py --provider chatgpt --file path/to/conversations.json

# Analyze a Claude export
uv run analyze_llm_archive.py --provider claude --file path/to/export.json

# Specify custom output directory
uv run analyze_llm_archive.py --provider chatgpt --file conversations.json --output-dir ./results
```

## Requirements

- Python 3.11+
- UV (https://github.com/astral-sh/uv)
- Pydantic 2.0+

The script uses UV's inline script metadata, so dependencies are automatically installed when you run it with `uv run`.

## Input Format

### ChatGPT Export
The script expects a `conversations.json` file from ChatGPT's data export feature. The file should contain an array of conversation objects with a `mapping` structure.

### Claude Export
The script expects a JSON file with conversation data, typically containing `chat_messages` arrays. Format may vary depending on the Claude export method used.

## Output Files

All outputs are written to the `--output-dir` (default: `./output/`):

- **archive.jsonl**: Complete normalized archive with all conversations
- **conversations.jsonl**: All conversations as separate JSONL entries
- **messages.jsonl**: All messages across all conversations
- **cost_estimates.jsonl**: Cost analysis breakdown by model
- **summary.md**: Human-readable markdown summary report

## Domain Models

The script normalizes data into these Pydantic models:

- **Archive**: Top-level container with source provider and conversations
- **Conversation**: A conversation with messages
- **Message**: Individual message with role, content, and optional usage data
- **ModelUsage**: Token counts for a specific model
- **CostEstimate**: Cost breakdown with input/output costs

## Cost Estimation

Cost estimates use current API pricing (as of 2025) for:

### OpenAI Models
- GPT-4
- GPT-4 Turbo
- GPT-4o
- GPT-3.5 Turbo

### Anthropic Models
- Claude 3 Opus
- Claude 3 Sonnet
- Claude 3.5 Sonnet
- Claude 3 Haiku

**Note:** Archives may not always include token counts. When unavailable, cost estimates will show zero.

## Project Structure

```
analyze_llm_archive.py  # Main UV runnable script
models.py               # Pydantic domain models
parsers.py              # ChatGPT and Claude parsers
analysis.py             # Cost/usage analysis with pricing
serializers.py          # JSONL and summary output
```

## Example Output

```
$ uv run analyze_llm_archive.py --provider chatgpt --file conversations.json

Parsing chatgpt archive from conversations.json...
Found 47 conversations
Found 892 total messages

Analyzing usage and costs...

Writing results to output/
  - archive.jsonl
  - conversations.jsonl
  - messages.jsonl
  - cost_estimates.jsonl
  - summary.md

Estimated total cost: $12.3456 USD

Analysis complete!
```

## Limitations

- No ZIP file handling (unzip archives before running)
- Token counts may not be available in all export formats
- Pricing data is hardcoded (not fetched from APIs)
- No database storage (filesystem-based outputs only)
