# SpatialChain Judge — LLM-as-Judge Evaluation Package

This package implements the scene-graph-aware **LLM-as-judge** evaluation protocol described in §4.3 of the SpatialChain paper. It scores VLM reasoning chains on three axes: **answer correctness**, **reasoning faithfulness**, and **reasoning completeness**.

For full usage instructions see the [main README](../README.md#7-evaluation--axis-2-llm-as-judge).

## Quick install

```bash
pip install -e .
# or with uv:
uv venv && source .venv/bin/activate && uv pip sync requirements.in
```

## CLI commands

| Command | Description |
|---------|-------------|
| `spatial-eval generate` | Generate answers + reasoning for a dataset |
| `spatial-eval generate-iq` | Single image+question inference |
| `spatial-eval generate-iq-batch` | Batch inference from CSV/JSONL with `imageId` |
| `spatial-eval judge` | Score existing `generation_records.jsonl` |
| `spatial-eval evaluate` | Full generate → judge pipeline |
| `spatial-eval convert-dataset` | Convert external format to canonical |

## Supported LLM backends

`--judge-backend` / `--generation-backend`: `anthropic` | `openai` | `gemini` | `ollama`

## Output schema (`eval_records.jsonl`)

Each line contains:
- `answer_correctness` — float in [0, 1]  
- `reasoning_faithfulness` — float in [0, 1]  
- `reasoning_completeness` — float in [0, 1]  
- `verdict` — `"pass"` if AC ≥ 0.5 and (RF+RC)/2 ≥ 0.5, else `"fail"`  
- `baseline_match` — exact-match against normalized ground truth  
- `raw_response` — full judge output for auditing  

## Shortcut rate

The shortcut rate reported in the paper is computed as:

```
shortcut_rate = |{correct answers with RF < 0.5}| / |{correct answers}|
```

This metric detects models that reach the right answer without faithful spatial inference.
