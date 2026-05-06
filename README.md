# SpatialChain — A Benchmark for Auditing Spatial Reasoning Faithfulness in VLMs

> **Anonymous submission** — code and data released for peer review.

**SpatialChain** is a dataset of **28,350 training** and **899 test** examples pairing spatially-oriented GQA questions with scene-graph-grounded reasoning chains. It enables a two-axis evaluation of thinking-enabled VLMs: standard VQA accuracy *and* faithfulness of the reasoning chain against the symbolic ground truth — surfacing shortcut behavior even when the final answer is correct.

---

## Table of Contents

1. [Dataset](#1-dataset)
2. [Repository Structure](#2-repository-structure)
3. [Setup](#3-setup)
4. [Dataset Construction](#4-dataset-construction)
5. [Training](#5-training)
6. [Evaluation — Axis 1: VQA Accuracy](#6-evaluation--axis-1-vqa-accuracy)
   - [6.1 Baseline Models](#61-baseline-models)
   - [6.2 Fine-tuned Checkpoints](#62-fine-tuned-checkpoints)
   - [6.3 External Benchmarks](#63-external-benchmarks)
7. [Evaluation — Axis 2: LLM-as-Judge](#7-evaluation--axis-2-llm-as-judge)
   - [7.1 Setup](#71-setup)
   - [7.2 Convert predictions to judge format](#72-convert-predictions-to-judge-format)
   - [7.3 Run the judge](#73-run-the-judge)
   - [7.4 Full pipeline](#74-full-pipeline)
8. [Citation](#8-citation)

---

## 1. Dataset

| Resource | Link |
|----------|------|
| 🤗 Dataset | [spatialchain/SpatialChain-Benchmark](https://huggingface.co/datasets/spatialchain/SpatialChain-Benchmark) |
| 🤗 Fine-tuned model | [spatialchain/Qwen3-VL-8B-Thinking-SpatialChain](https://huggingface.co/spatialchain/Qwen3-VL-8B-Thinking-SpatialChain) |

| Split | Examples | Images | Yes-rate | Avg. chain (words) |
|-------|----------|--------|----------|--------------------|
| `train` | 28,350 | 19,263 | 0.58 | 115 |
| `test` | 899 | 719 | 0.63 | 145 |

Projective relations (`left_of` / `right_of`) account for 62.7% of training examples. Test examples are biased toward longer, compositional multi-step chains.

**Schema** (Parquet / CSV):

| Column | Type | Description |
|--------|------|-------------|
| `imageId` | str | GQA image identifier |
| `question` | str | Natural-language spatial question |
| `fullAnswer` | str | Ground-truth answer (`yes` / `no` / short phrase) |
| `thinking` | str | Scene-graph-grounded reasoning chain (reference) |
| `scene_graph` | str/JSON | Relevant GQA scene-graph subgraph |
| `relation` | str | Primary spatial relation (`left_of`, `above`, …) |
| `spatial` | bool | Spatial-relevance flag (Stage 2 filter) |

**Format** — training target stored as:
```
<think>
[reasoning chain]
</think>
<answer>
[yes / no / phrase]
</answer>
```

Images come from [GQA](https://cs.stanford.edu/people/dorarad/gqa/download.html) — download them separately and pass `--images_dir` to all scripts.

---

## 2. Repository Structure

```
SpatialChainBenchmark/
├── README.md
├── requirements.txt          # GPU training & vLLM evaluation
├── .gitignore
│
├── notebooks/
│   └── GQA_subset_selection.ipynb   # Stage 1: type-based question filtering
│
├── scripts/
│   ├── filter_spatial_questions.py  # Stage 2: LLM spatial-relevance classifier
│   └── finetune_qwen3vl4b.py        # LoRA SFT on A100 80 GB
│
├── eval/                            # Axis 1 — VQA accuracy
│   ├── eval_hf_models_vllm.py       # Any HuggingFace VLM via vLLM
│   ├── eval_checkpoints_vllm.py     # Fine-tuned adapter sweep via vLLM
│   ├── eval_erqa.py                 # FlagEval/ERQA benchmark
│   └── eval_multi.py                # ERQA + EmbSpatial + VSI-Bench + MMMU-Pro
│
└── judge/                           # Axis 2 — LLM-as-judge reasoning evaluation
    ├── pyproject.toml
    ├── requirements.in
    └── src/spatial_eval/
        ├── cli.py                   # CLI entry-point: generate / judge / evaluate
        ├── judge.py                 # Judge prompt builder & response parser
        ├── pipeline.py              # Full generate→judge pipeline
        ├── metrics.py               # Aggregate metrics (faithfulness, completeness, pass rate)
        ├── prompts/
        │   ├── judge.md             # Scene-graph-aware judge rubric
        │   └── generator.md        # Reference-chain generator prompt
        ├── datasets/                # Dataset adapters (current, image-qa, testset-image-qa)
        └── providers/               # LLM backends (Anthropic, OpenAI, Gemini, Ollama)
```

---

## 3. Setup

```bash
git clone <repo_url>
cd SpatialChainBenchmark

# --- Axis 1: VQA evaluation & training ---
pip install -r requirements.txt       # torch, transformers, vllm, peft, trl, …

# --- Axis 2: LLM-as-judge ---
pip install -e judge/                 # installs the spatial_eval CLI
# or with uv:
cd judge && uv venv && source .venv/bin/activate && uv pip sync requirements.in
```

**Hardware:** Training requires a single **A100 80 GB** GPU. vLLM evaluation works on any GPU with ≥ 24 GB VRAM. The judge runs CPU-side via any LLM API.

---

## 4. Dataset Construction

The dataset was constructed in three stages.

### Stage 1 — Type-based filtering (`notebooks/GQA_subset_selection.ipynb`)

Reproduces the initial selection of spatially-relevant questions from GQA's balanced training split. Expects the following GQA files in the working directory (download from [GQA](https://cs.stanford.edu/people/dorarad/gqa/download.html)):

- `train_balanced_questions.json`
- `train_sceneGraphs.json`

The notebook selects questions whose GQA type falls into one of 17 relation-centric categories (e.g. `relVerify`, `existRelS`, `placeVerify`, `twoDifferent`). Applied to the full training split this yields 135,061 candidates.

### Stage 2 — LLM spatial-relevance classifier (`scripts/filter_spatial_questions.py`)

Applies a binary LLM classifier to remove questions answerable by object recognition alone (e.g. "Is the dog sitting?"), retaining only those that require relative position, proximity, containment, or directional reasoning.

```bash
export OPENAI_API_KEY="<your-key>"

python scripts/filter_spatial_questions.py \
  --input_csv  data/gqa_type_filtered.csv \
  --output_csv data/gqa_spatial_filtered.csv
```

The script classifies each (question, answer) pair in parallel (`--max_workers 20`) and retries failed rows automatically. Any OpenAI chat-completion model works; the default is `gpt-4o-mini`.

After both stages: **86,773 training** and **900 test** candidates, each paired with the corresponding scene-graph subgraph.

### Stage 3 — Reasoning chain generation

Chains were generated with **Claude Haiku 4.5** (extended thinking, budget 3,000 tokens, max output 4,096 tokens) via the Anthropic Message Batches API. Each request provides the full scene-graph JSON as context; the system prompt instructs the model to reason in natural visual language without referencing scene-graph fields directly. Chains are retained only when the generated answer matches the GQA ground truth, yielding the final **28,350 training** and **899 test** examples. The generation prompt is reproduced in Appendix C of the paper. The final dataset with chains is released on HuggingFace.

---

## 5. Training

Fine-tune **Qwen3-VL-4B-Thinking** (or 8B) with LoRA on the SpatialChain training split. The script saves checkpoints at 1k / 5k / 10k steps and every ½ epoch, and logs generative metrics (answer exact match, token F1, ROUGE-L) per checkpoint.

```bash
python scripts/finetune_qwen3vl4b.py \
  --trainset_csv   trainset.csv \
  --valset_csv     valset.csv \
  --images_dir     images \
  --output_dir     outputs/qwen3vl4b_run1 \
  --max_train_samples 30000 \
  --max_val_samples   1000 \
  --num_train_epochs  2 \
  --time_budget_hours 5 \
  --per_device_train_batch_size 6 \
  --gradient_accumulation_steps 2
```

**Key training arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_name` | `Qwen/Qwen3-VL-4B-Thinking` | Base model |
| `--lora_r` | 16 | LoRA rank |
| `--lora_alpha` | 32 | LoRA alpha |
| `--learning_rate` | 1e-4 | AdamW LR |
| `--max_seq_length` | 4096 | Max token length |
| `--image_max_size` | 640 | Resize longest image side (px) |
| `--include_thinking` | True | Train on `<think>…</think>` traces |
| `--time_budget_hours` | — | Soft wall-clock stop |
| `--wandb_project` | — | W&B project name (optional) |

Checkpoints are saved under `--output_dir/checkpoint-<step>/`. The final adapter lands at `--output_dir/lora_adapter_final/`.

> For the 8B model or H100 hardware, see the comments inside `finetune_qwen3vl4b.py` for recommended batch size and sequence length adjustments.

---

## 6. Evaluation — Axis 1: VQA Accuracy

All scripts produce per-example JSONL predictions that feed directly into the judge (Axis 2).

### 6.1 Baseline Models

Evaluate any HuggingFace VLM on the SpatialChain test set:

```bash
# Standard instruct model
python eval/eval_hf_models_vllm.py \
    --models Qwen/Qwen3-VL-4B-Instruct \
    --csv_path  data/testset.csv \
    --images_dir images \
    --output_dir eval_results/baselines

# Thinking-mode model — use sampling + larger token budget
python eval/eval_hf_models_vllm.py \
    --models OpenGVLab/InternVL3-8B \
    --csv_path   data/testset.csv \
    --images_dir images \
    --enable_thinking \
    --temperature 0.6 --top_p 0.95 --top_k 20 \
    --max_new_tokens 16384 \
    --output_dir eval_results/baselines
```

Predictions land in `<output_dir>/<model_name>_predictions.jsonl`.

### 6.2 Fine-tuned Checkpoints

Evaluate all saved adapters in a single loop with vLLM LoRA hot-swap:

```bash
python eval/eval_checkpoints_vllm.py \
    --model_name     Qwen/Qwen3-VL-4B-Thinking \
    --checkpoint_dir outputs/qwen3vl4b_run1 \
    --strip_modules_to_save \
    --max_lora_rank  16 \
    --csv_path       data/testset.csv \
    --images_dir     images \
    --output_dir     eval_results/checkpoints
```

> **Note:** Pass `--strip_modules_to_save` if you trained with `modules_to_save=["lm_head","embed_tokens"]` (default). The flag creates vLLM-compatible stripped copies automatically.

### 6.3 External Benchmarks

Evaluate on the four external benchmarks used in the paper:

```bash
python eval/eval_multi.py \
    --model_name     Qwen/Qwen3-VL-4B-Thinking \
    --checkpoint_dir outputs/qwen3vl4b_run1 \
    --datasets       erqa embspatial mmmupro_vision \
    --output_dir     eval_results/external \
    --max_new_tokens 512
```

| Key | Dataset | Metric |
|-----|---------|--------|
| `erqa` | [FlagEval/ERQA](https://huggingface.co/datasets/FlagEval/ERQA) | Accuracy (multi-image MCQ) |
| `embspatial` | [FlagEval/EmbSpatial-Bench](https://huggingface.co/datasets/FlagEval/EmbSpatial-Bench) | Accuracy |
| `vsibench` | [nyu-visionx/VSI-Bench](https://huggingface.co/datasets/nyu-visionx/VSI-Bench) | Accuracy + MRA |
| `mmmupro_vision` | [MMMU/MMMU-Pro](https://huggingface.co/datasets/MMMU/MMMU_Pro) vision | Accuracy |

The script is resumable: already-computed checkpoints are skipped. Results accumulate in `eval_results/external/summary.csv`.

---

## 7. Evaluation — Axis 2: LLM-as-Judge

The `judge/` package implements the scene-graph-aware LLM judge described in §4.3 of the paper. It scores each model output on three dimensions in `[0, 1]`:

| Score | Description |
|-------|-------------|
| `answer_correctness` (AC) | Does the final answer match the ground truth? |
| `reasoning_faithfulness` (RF) | Is the chain grounded in spatial evidence from the scene graph? |
| `reasoning_completeness` (RC) | Does the chain cover all key inference steps? |

A prediction **passes** when `AC ≥ 0.5` and `(RF + RC) / 2 ≥ 0.5`. The **shortcut rate** is the fraction of *correct* answers with `RF < 0.5`.

The judge accepts evidence from the scene graph (`img_graph`), the image (`image`), or both (`both`). Scene-graph mode (`img_graph`) is recommended for reproducibility since it does not require GPU inference.

### 7.1 Setup

```bash
pip install -e judge/

# API keys (add to .env or export)
export ANTHROPIC_API_KEY="<your-key>"   # recommended judge backend
# or
export OPENAI_API_KEY="<your-key>"
```

### 7.2 Convert predictions to judge format

The Axis 1 scripts produce vLLM-style JSONL. Convert them to the judge's canonical format:

```bash
PYTHONPATH=judge/src python -m spatial_eval.datasets.external_to_current \
  --input      eval_results/baselines/Qwen3-VL-4B-Thinking_predictions.jsonl \
  --output     judge_inputs/qwen3vl4b_generation_records.jsonl \
  --images-dir images
```

### 7.3 Run the judge

```bash
PYTHONPATH=judge/src python -m spatial_eval.cli judge \
  --generation-records judge_inputs/qwen3vl4b_generation_records.jsonl \
  --output-dir         judge_results/qwen3vl4b \
  --judge-provider     llm \
  --judge-backend      anthropic \
  --judge-model        claude-haiku-4-5 \
  --judge-evidence     img_graph \
  --judge-img-graph-file data/val_sceneGraphs.json \
  --env-file           .env
```

**Outputs** in `--output-dir`:

| File | Description |
|------|-------------|
| `eval_records.jsonl` | Per-example scores (AC, RF, RC, verdict, raw output) |
| `eval_metrics.json` | Aggregate metrics (mean faithfulness, pass rate, shortcut rate) |
| `eval_report.md` | Human-readable summary |

### 7.4 Full pipeline

Generate predictions *and* judge them in one call (useful for models accessible via API):

```bash
PYTHONPATH=judge/src python -m spatial_eval.cli evaluate \
  --dataset       data/testset.jsonl \
  --images-dir    images \
  --dataset-adapter testset-image-qa \
  --output-dir    judge_results/full_pipeline \
  --generation-provider llm \
  --generation-backend  anthropic \
  --generation-model    claude-haiku-4-5 \
  --judge-provider      llm \
  --judge-backend       anthropic \
  --judge-model         claude-haiku-4-5 \
  --judge-evidence      img_graph \
  --judge-img-graph-file data/val_sceneGraphs.json \
  --env-file .env
```

**Iterative workflow (recommended for large evaluations):**

1. Run `generate` → produces `generation_records.jsonl`
2. Inspect a few rows to validate format
3. Run `judge` → produces `eval_records.jsonl` + `eval_metrics.json`
4. Compare `eval_metrics.json` across models/checkpoints

---

## 8. Citation

```bibtex
@article{spatialchain2026,
  title   = {SpatialChain: A Benchmark for Auditing Spatial Reasoning Faithfulness in VLMs},
  author  = {Anonymous},
  journal = {Under review at NeurIPS 2026},
  year    = {2026}
}
```

---

## License

The code in this repository is released under the **MIT License**.  
The GQA images and annotations are subject to the original [GQA Terms of Use](https://cs.stanford.edu/people/dorarad/gqa/about.html).  
The SpatialChain reasoning chains were generated by Claude Haiku 4.5 and follow [Anthropic's usage policies](https://www.anthropic.com/legal/usage-policy).
