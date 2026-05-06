#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Avaliação multi-checkpoint × multi-dataset para o paper.

Datasets suportados:
  - erqa          : FlagEval/ERQA             (MCQ, múltiplas imagens PIL)
  - embspatial    : FlagEval/EmbSpatial-Bench (MCQ, imagem PIL)
  - vsibench      : nyu-visionx/VSI-Bench     (MCQ + numérico, frames de vídeo)
  - mmmupro_vision: MMMU/MMMU_Pro "vision"    (MCQ, questão embutida na imagem)

Saídas por (checkpoint, dataset):
  {output_dir}/{run_name}/{dataset}/predictions.jsonl  ← uma linha por exemplo
  {output_dir}/{run_name}/{dataset}/metrics.json       ← métricas agregadas
  {output_dir}/summary.csv                             ← tabela comparativa

Uso:
python eval_multi.py \
  --model_name  Qwen/Qwen3-VL-8B-Instruct \
  --checkpoint_dir  /workspace/outputs/qwen3vl8b_paper \
  --output_dir  /workspace/eval_results \
  --datasets    erqa embspatial mmmupro_vision \
  --vsi_frames_dir  /workspace/data/vsi_frames  # só para vsibench
"""

import base64
import csv
import gc
import io
import json
import math
import os
import re
import string
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128")

import argparse

import torch
from datasets import load_dataset
from PIL import Image, ImageFile
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig
from peft import PeftModel

ImageFile.LOAD_TRUNCATED_IMAGES = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

VALID_MCQ = {"A", "B", "C", "D", "E", "F", "G", "H", "I", "J"}

# ══════════════════════════════════════════════════════════════════════════════
# Extração de resposta
# ══════════════════════════════════════════════════════════════════════════════
_ANSWER_TAG = re.compile(r"<answer>\s*([A-Ja-j0-9][^\n<]*?)\s*</answer>", re.IGNORECASE)
_LETTER_RE  = re.compile(r"\b([A-Ja-j])\b")
_NUMBER_RE  = re.compile(r"[-+]?\d+(?:\.\d+)?")


def extract_letter(text: str, n_opts: int = 4) -> str:
    t = text.strip()
    m = _ANSWER_TAG.search(t)
    if m:
        c = m.group(1).strip().upper()[:1]
        if c in VALID_MCQ:
            return c
    letters = [l.upper() for l in _LETTER_RE.findall(t)
               if l.upper() in VALID_MCQ and ord(l.upper()) - ord("A") < n_opts]
    if letters:
        return letters[0]
    c = t[:1].upper()
    return c if c in VALID_MCQ else ""


def extract_number(text: str) -> Optional[float]:
    t = text.strip()
    m = _ANSWER_TAG.search(t)
    if m:
        nums = _NUMBER_RE.findall(m.group(1))
        if nums:
            return float(nums[0])
    nums = _NUMBER_RE.findall(t)
    return float(nums[0]) if nums else None


def mra(pred: Optional[float], gt: float) -> float:
    """Mean Relative Accuracy — métrica numérica do VSI-Bench."""
    if pred is None or gt == 0:
        return 0.0
    return max(0.0, 1.0 - abs(pred - gt) / abs(gt))


# ══════════════════════════════════════════════════════════════════════════════
# Helpers de imagem
# ══════════════════════════════════════════════════════════════════════════════
def pil_from_field(field: Any) -> Image.Image:
    """Aceita PIL, bytes ou base64."""
    if isinstance(field, Image.Image):
        return field.convert("RGB")
    if isinstance(field, bytes):
        return Image.open(io.BytesIO(field)).convert("RGB")
    data = base64.b64decode(field)
    return Image.open(io.BytesIO(data)).convert("RGB")


def resize_img(img: Image.Image, max_size: int) -> Image.Image:
    img = img.convert("RGB")
    w, h = img.size
    s = max(w, h)
    if s <= max_size:
        return img
    r = max_size / s
    return img.resize((max(1, int(round(w * r))), max(1, int(round(h * r)))),
                      Image.Resampling.LANCZOS)


def sample_frames(frames_dir: Path, scene_name: str,
                  n_frames: int = 8) -> List[Image.Image]:
    """Carrega N frames uniformemente espaçados de um diretório de cena."""
    scene_dir = frames_dir / str(scene_name)
    if not scene_dir.exists():
        # tenta busca recursiva
        matches = sorted(frames_dir.rglob(f"{scene_name}*"))
        scene_dir = matches[0] if matches else scene_dir

    exts = {".jpg", ".jpeg", ".png", ".webp"}
    files = sorted(p for p in scene_dir.iterdir() if p.suffix.lower() in exts)
    if not files:
        raise FileNotFoundError(f"Nenhum frame em {scene_dir}")

    if len(files) <= n_frames:
        chosen = files
    else:
        idx = [int(round(i * (len(files) - 1) / (n_frames - 1)))
               for i in range(n_frames)]
        chosen = [files[i] for i in idx]

    return [Image.open(p).convert("RGB") for p in chosen]


# ══════════════════════════════════════════════════════════════════════════════
# Carregamento dos datasets
# ══════════════════════════════════════════════════════════════════════════════
def load_erqa() -> List[Dict]:
    ds = load_dataset("FlagEval/ERQA", split="test", trust_remote_code=True)
    return [dict(s) for s in ds]


def load_embspatial() -> List[Dict]:
    ds = load_dataset(
        "json",
        data_files={"test": "hf://datasets/FlagEval/EmbSpatial-Bench/embspatial_bench.json"},
        split="test",
    )
    return [dict(s) for s in ds]


def load_vsibench() -> List[Dict]:
    ds = load_dataset("nyu-visionx/VSI-Bench", "full", split="test",
                      trust_remote_code=True)
    return [dict(s) for s in ds]


def load_mmmupro_vision() -> List[Dict]:
    ds = load_dataset("MMMU/MMMU_Pro", "vision", split="test",
                      trust_remote_code=True)
    return [dict(s) for s in ds]


DATASET_LOADERS = {
    "erqa"          : load_erqa,
    "embspatial"    : load_embspatial,
    "vsibench"      : load_vsibench,
    "mmmupro_vision": load_mmmupro_vision,
}

# Tipos de questão numérica no VSI-Bench
VSI_NUMERICAL_TYPES = {
    "object_counting", "object_size_estimation",
    "object_distance_estimation", "room_size_estimation",
}


# ══════════════════════════════════════════════════════════════════════════════
# Builders de mensagem por dataset
# ══════════════════════════════════════════════════════════════════════════════
def build_erqa(sample: Dict, cfg: Dict) -> Tuple[List[Dict], str]:
    """ERQA: questão já tem as choices embutidas. Múltiplas imagens PIL."""
    imgs = [resize_img(img, cfg["image_max_size"])
            for img in sample.get("images", []) if img is not None]
    if not imgs:
        raise ValueError("Sem imagens")

    content = [{"type": "image", "image": img} for img in imgs]
    content.append({"type": "text", "text": sample["question"]})

    gt = str(sample.get("answer", "")).strip().upper()
    return _wrap(content, cfg), gt


def build_embspatial(sample: Dict, cfg: Dict) -> Tuple[List[Dict], str]:
    """EmbSpatial: MCQ com answer_options e answer como índice int."""
    img    = pil_from_field(sample["image"])
    img    = resize_img(img, cfg["image_max_size"])
    opts   = sample["answer_options"]
    gt_idx = int(sample["answer"])
    letters = "ABCDEFGHIJ"

    opts_text = "\n".join(f"{letters[i]}) {o}" for i, o in enumerate(opts))
    prompt = (
        f"{sample['question']}\n\n"
        f"Options:\n{opts_text}\n\n"
        "Reply with the letter of the correct option inside <answer> tags. "
        "Example: <answer>A</answer>"
    )
    content = [{"type": "image", "image": img}, {"type": "text", "text": prompt}]
    gt = letters[gt_idx]
    return _wrap(content, cfg), gt


def build_vsibench(sample: Dict, cfg: Dict,
                   frames_dir: Optional[Path]) -> Tuple[List[Dict], str]:
    """
    VSI-Bench: frames de vídeo de uma cena.
    - MCQ: tem campo 'options'
    - Numérico: não tem 'options' (None ou lista vazia)
    """
    if frames_dir is None:
        raise ValueError("--vsi_frames_dir é obrigatório para vsibench")

    scene  = str(sample["scene_name"])
    frames = sample_frames(frames_dir, scene, cfg.get("vsi_n_frames", 8))
    frames = [resize_img(f, cfg["image_max_size"]) for f in frames]

    opts = sample.get("options")
    gt   = str(sample.get("ground_truth", "")).strip()

    if opts:                              # MCQ
        letters  = "ABCDEFGHIJ"
        opts_text = "\n".join(f"{letters[i]}) {o}" for i, o in enumerate(opts))
        prompt = (
            f"{sample['question']}\n\n"
            f"Options:\n{opts_text}\n\n"
            "Reply with the letter of the correct option inside <answer> tags. "
            "Example: <answer>A</answer>"
        )
    else:                                 # numérico
        prompt = (
            f"{sample['question']}\n\n"
            "Provide only the numerical value inside <answer> tags. "
            "Example: <answer>42</answer>"
        )

    content = [{"type": "image", "image": f} for f in frames]
    content.append({"type": "text", "text": prompt})
    return _wrap(content, cfg), gt


def build_mmmupro_vision(sample: Dict, cfg: Dict) -> Tuple[List[Dict], str]:
    """
    MMMU-Pro vision: a questão está EMBUTIDA na imagem.
    Não há texto de questão — o modelo precisa ler a imagem.
    """
    img = pil_from_field(sample["image"])
    img = resize_img(img, cfg["image_max_size"])

    prompt = (
        "The question and options are embedded in the image above. "
        "Read the image carefully and answer the multiple-choice question. "
        "Reply with the letter of the correct option inside <answer> tags. "
        "Example: <answer>A</answer>"
    )
    content = [{"type": "image", "image": img}, {"type": "text", "text": prompt}]
    gt = str(sample.get("answer", "")).strip().upper()
    return _wrap(content, cfg), gt


def _wrap(user_content: List[Dict], cfg: Dict) -> List[Dict]:
    msgs: List[Dict] = []
    if cfg.get("system_prompt"):
        msgs.append({"role": "system",
                     "content": [{"type": "text", "text": cfg["system_prompt"]}]})
    msgs.append({"role": "user", "content": user_content})
    return msgs


# ══════════════════════════════════════════════════════════════════════════════
# Modelo
# ══════════════════════════════════════════════════════════════════════════════
def load_base_model(model_name: str):
    torch.cuda.init()
    torch.cuda.empty_cache()
    print(f"[INFO] VRAM livre: {torch.cuda.mem_get_info()[0] / 1e9:.1f} GB")

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForVision2Seq.from_pretrained(
        model_name, quantization_config=bnb,
        torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True, low_cpu_mem_usage=True,
    )
    model.config.use_cache = True
    return model, processor


def swap_adapter(model, adapter_path: Optional[Path]):
    if isinstance(model, PeftModel):
        try:
            model = model.merge_and_unload()
        except Exception:
            try:
                model = model.unload()
            except Exception:
                pass

    if adapter_path is not None:
        print(f"[INFO] Carregando adapter: {adapter_path}")
        model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=False)

    model.eval()
    gc.collect()
    torch.cuda.empty_cache()
    return model


def discover_checkpoints(checkpoint_dir: Path) -> List[Tuple[int, Optional[Path], str]]:
    """
    Retorna lista de (step, adapter_path_or_None, run_name).
    Inclui step=0 (modelo base) e todos os checkpoint-stepN.
    """
    results: List[Tuple[int, Optional[Path], str]] = []

    # Modelo base (step 0)
    results.append((0, None, "base_model"))

    # checkpoint-stepN (formato do nosso script de treino)
    for p in sorted(checkpoint_dir.glob("checkpoint-step*")):
        if (p / "adapter_config.json").exists():
            try:
                step = int(p.name.split("step")[1])
                results.append((step, p, p.name))
            except (ValueError, IndexError):
                pass

    # checkpoint-N (formato padrão HF Trainer)
    for p in sorted(checkpoint_dir.glob("checkpoint-[0-9]*")):
        if (p / "adapter_config.json").exists() and "step" not in p.name:
            try:
                step = int(p.name.split("-")[1])
                results.append((step, p, p.name))
            except (ValueError, IndexError):
                pass

    # adapter final
    for name in ("lora_adapter_final", "lora_adapter", "adapter"):
        final = checkpoint_dir / name
        if final.exists() and (final / "adapter_config.json").exists():
            results.append((999999, final, name))
            break

    results.sort(key=lambda x: x[0])
    seen = set()
    deduped = []
    for item in results:
        if item[2] not in seen:
            seen.add(item[2])
            deduped.append(item)
    return deduped


# ══════════════════════════════════════════════════════════════════════════════
# Geração
# ══════════════════════════════════════════════════════════════════════════════
def encode(processor, msgs: List[Dict], max_length: int) -> Dict[str, torch.Tensor]:
    images, clean = [], []
    for msg in msgs:
        content = msg.get("content", [])
        if not isinstance(content, list):
            clean.append(msg)
            continue
        nc = []
        for part in content:
            if part.get("type") == "image":
                images.append(part["image"])
                nc.append({"type": "image"})
            else:
                nc.append(part)
        clean.append({**msg, "content": nc})

    return processor(
        text=processor.apply_chat_template(
            clean, tokenize=False, add_generation_prompt=True,
            enable_thinking=True,
        ),
        images=images if images else None,
        padding=True, truncation=True,
        max_length=max_length, return_tensors="pt",
    )


def generate_one(model, processor, msgs: List[Dict], cfg: Dict) -> str:
    pad_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id or 0
    enc    = encode(processor, msgs, cfg["max_seq_length"])
    ilen   = enc["input_ids"].shape[1]
    enc    = {k: v.to(model.device) for k, v in enc.items()}

    with torch.inference_mode():
        out = model.generate(
            **enc,
            max_new_tokens = cfg["max_new_tokens"],
            do_sample      = True,
            temperature    = 0.6,
            top_p          = 0.95,
            top_k          = 20,
            use_cache      = True,
            pad_token_id   = pad_id,
        )
    return processor.tokenizer.decode(
        out[0, ilen:], skip_special_tokens=True
    ).strip()


# ══════════════════════════════════════════════════════════════════════════════
# Avaliação de um dataset
# ══════════════════════════════════════════════════════════════════════════════
def run_dataset(
    dataset_name: str,
    samples     : List[Dict],
    model       : Any,
    processor   : Any,
    cfg         : Dict,
    out_dir     : Path,
    frames_dir  : Optional[Path],
) -> Dict[str, Any]:

    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "predictions.jsonl"

    # Métricas acumuladas
    acc_correct = acc_total = 0
    mra_sum = mra_total = 0
    cat_correct: Dict[str, int] = {}
    cat_total:   Dict[str, int] = {}

    with jsonl_path.open("w", encoding="utf-8") as fout:
        for idx, sample in enumerate(tqdm(samples,
                                          desc=f"{dataset_name}", unit="ex")):
            try:
                # ── Constrói mensagens ────────────────────────────────────────
                if dataset_name == "erqa":
                    msgs, gt = build_erqa(sample, cfg)
                    cat = sample.get("question_type", "unknown")
                    task_type = "mcq"

                elif dataset_name == "embspatial":
                    msgs, gt = build_embspatial(sample, cfg)
                    cat = sample.get("relation", "unknown")
                    task_type = "mcq"

                elif dataset_name == "vsibench":
                    msgs, gt = build_vsibench(sample, cfg, frames_dir)
                    cat = sample.get("question_type", "unknown")
                    task_type = ("numerical"
                                 if cat in VSI_NUMERICAL_TYPES else "mcq")

                elif dataset_name == "mmmupro_vision":
                    msgs, gt = build_mmmupro_vision(sample, cfg)
                    cat = sample.get("subject", sample.get("category", "unknown"))
                    task_type = "mcq"

                else:
                    raise ValueError(f"Dataset desconhecido: {dataset_name}")

                # ── Gera resposta ─────────────────────────────────────────────
                pred_raw = generate_one(model, processor, msgs, cfg)

                # ── Avalia ───────────────────────────────────────────────────
                if task_type == "mcq":
                    opts = sample.get("options") or sample.get("answer_options") or []
                    n = len(opts) if opts else 4
                    pred_val  = extract_letter(pred_raw, n)
                    correct   = int(pred_val == gt and pred_val != "")
                    score_val = float(correct)
                    acc_correct += correct
                    acc_total   += 1
                else:
                    pred_num  = extract_number(pred_raw)
                    try:
                        gt_num = float(gt)
                    except ValueError:
                        gt_num = None
                    score_val = mra(pred_num, gt_num) if gt_num is not None else 0.0
                    pred_val  = str(pred_num)
                    mra_sum  += score_val
                    mra_total += 1

                cat_correct[cat] = cat_correct.get(cat, 0) + int(score_val == 1.0
                                   if task_type == "mcq" else score_val > 0)
                cat_total[cat]   = cat_total.get(cat, 0) + 1

            except Exception as e:
                pred_raw  = f"ERROR: {e}"
                pred_val  = ""
                score_val = 0.0
                gt        = ""
                cat       = "error"
                task_type = "mcq"
                print(f"[WARN] {dataset_name}[{idx}]: {e}")

            # ── Salva no JSONL ────────────────────────────────────────────────
            row = {
                "idx"        : idx,
                "question_id": sample.get("question_id",
                               sample.get("id", str(idx))),
                "category"   : cat,
                "task_type"  : task_type,
                "ground_truth": gt,
                "prediction" : pred_val,
                "score"      : score_val,
                "raw_output" : pred_raw,
            }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

            torch.cuda.empty_cache()

    # ── Métricas finais ───────────────────────────────────────────────────────
    metrics: Dict[str, Any] = {"dataset": dataset_name}

    if acc_total > 0:
        metrics["accuracy"]        = round(acc_correct / acc_total, 4)
        metrics["accuracy_correct"] = acc_correct
        metrics["accuracy_total"]  = acc_total

    if mra_total > 0:
        metrics["mra"]       = round(mra_sum / mra_total, 4)
        metrics["mra_total"] = mra_total

    # Per-category
    per_cat: Dict[str, Any] = {}
    for c in sorted(cat_total):
        n = cat_total[c]
        k = cat_correct.get(c, 0)
        per_cat[c] = {"score": round(k / n, 4), "n": n, "correct": k}
    metrics["per_category"] = per_cat

    metrics_path = out_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    return metrics


# ══════════════════════════════════════════════════════════════════════════════
# Summary CSV
# ══════════════════════════════════════════════════════════════════════════════
def append_summary(summary_path: Path, run_name: str,
                   dataset_name: str, metrics: Dict[str, Any]) -> None:
    row = {
        "run"     : run_name,
        "dataset" : dataset_name,
        "accuracy": metrics.get("accuracy", ""),
        "mra"     : metrics.get("mra", ""),
    }
    write_header = not summary_path.exists()
    with summary_path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Avaliação multi-checkpoint × multi-dataset"
    )

    p.add_argument("--model_name",    type=str,
                   default="Qwen/Qwen3-VL-8B-Instruct")
    p.add_argument("--checkpoint_dir",type=str, default=None,
                   help="Pasta com todos os checkpoints. "
                        "Descobre automaticamente.")
    p.add_argument("--adapter_paths", type=str, nargs="*",
                   help="Caminhos explícitos de adapters (alternativa ao --checkpoint_dir).")
    p.add_argument("--eval_base",     action="store_true", default=True,
                   help="Inclui o modelo base (sem adapter) na avaliação.")

    p.add_argument("--datasets", type=str, nargs="+",
                   default=["erqa", "embspatial", "mmmupro_vision"],
                   choices=list(DATASET_LOADERS.keys()),
                   help="Datasets a avaliar.")
    p.add_argument("--max_samples",   type=int, default=0,
                   help="0 = tudo. Útil para smoke tests.")

    p.add_argument("--output_dir",    type=str, default="eval_results")

    # Geração
    p.add_argument("--max_seq_length",type=int, default=2048)
    p.add_argument("--image_max_size",type=int, default=448)
    p.add_argument("--max_new_tokens",type=int, default=512)

    # VSI-Bench
    p.add_argument("--vsi_frames_dir",type=str, default=None,
                   help="Diretório com frames extraídos dos vídeos do VSI-Bench. "
                        "Estrutura: {vsi_frames_dir}/{scene_name}/*.jpg")
    p.add_argument("--vsi_n_frames",  type=int, default=8,
                   help="Número de frames por cena para o VSI-Bench.")

    p.add_argument("--system_prompt", type=str,
                   default=(
                       "You are an expert in visual and spatial reasoning. "
                       "Analyze the image(s) carefully and answer the question. "
                       "For multiple-choice questions, reply with the letter of "
                       "the correct option inside <answer> tags. "
                       "For numerical questions, reply with the number inside "
                       "<answer> tags."
                   ))

    return p.parse_args()


def main() -> None:
    args = parse_args()

    output_dir   = Path(args.output_dir)
    summary_path = output_dir / "summary.csv"
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = {
        "max_seq_length": args.max_seq_length,
        "image_max_size": args.image_max_size,
        "max_new_tokens": args.max_new_tokens,
        "vsi_n_frames"  : args.vsi_n_frames,
        "system_prompt" : args.system_prompt,
    }

    frames_dir = Path(args.vsi_frames_dir) if args.vsi_frames_dir else None

    # ── Constrói lista de checkpoints ─────────────────────────────────────────
    runs: List[Tuple[Optional[Path], str]] = []

    if args.checkpoint_dir:
        discovered = discover_checkpoints(Path(args.checkpoint_dir))
        if not args.eval_base:
            discovered = [(s, p, n) for s, p, n in discovered if p is not None]
        runs = [(p, n) for _, p, n in discovered]
        print(f"[INFO] Checkpoints descobertos: {[n for _, n in runs]}")
    elif args.adapter_paths:
        if args.eval_base:
            runs.append((None, "base_model"))
        for ap in args.adapter_paths:
            path = Path(ap)
            runs.append((path, path.name))
    else:
        runs = [(None, "base_model")]

    # ── Carrega datasets ──────────────────────────────────────────────────────
    print("[INFO] Carregando datasets...")
    datasets: Dict[str, List[Dict]] = {}
    for ds_name in args.datasets:
        print(f"  {ds_name} ...")
        data = DATASET_LOADERS[ds_name]()
        if args.max_samples and args.max_samples > 0:
            data = data[: args.max_samples]
        datasets[ds_name] = data
        print(f"    {len(data):,} exemplos")

    # ── Carrega modelo base ───────────────────────────────────────────────────
    print(f"[INFO] Carregando modelo base: {args.model_name}")
    base_model, processor = load_base_model(args.model_name)
    current_model = base_model

    all_metrics: List[Dict] = []

    # ── Loop principal: checkpoints × datasets ────────────────────────────────
    for adapter_path, run_name in runs:
        print(f"\n{'='*65}")
        print(f"RUN: {run_name}")
        print(f"{'='*65}")

        current_model = swap_adapter(base_model, adapter_path)
        run_dir = output_dir / run_name

        for ds_name, samples in datasets.items():
            print(f"\n  Dataset: {ds_name} ({len(samples):,} exemplos)")
            ds_out = run_dir / ds_name

            # Verifica se já foi avaliado (permite retomar)
            if (ds_out / "metrics.json").exists():
                print(f"  [SKIP] Já avaliado → {ds_out}/metrics.json")
                with (ds_out / "metrics.json").open() as f:
                    m = json.load(f)
                all_metrics.append({"run": run_name, **m})
                continue

            m = run_dataset(
                dataset_name = ds_name,
                samples      = samples,
                model        = current_model,
                processor    = processor,
                cfg          = cfg,
                out_dir      = ds_out,
                frames_dir   = frames_dir,
            )
            m["run"] = run_name
            all_metrics.append(m)
            append_summary(summary_path, run_name, ds_name, m)

            # Imprime resultado
            if "accuracy" in m:
                print(f"  accuracy = {m['accuracy']:.4f}  "
                      f"({m.get('accuracy_correct',0)}/{m.get('accuracy_total',0)})")
            if "mra" in m:
                print(f"  MRA      = {m['mra']:.4f}  "
                      f"(n={m.get('mra_total',0)})")

        gc.collect()
        torch.cuda.empty_cache()

    # ── Resumo final ──────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("RESUMO FINAL")
    print(f"{'='*65}")

    header = f"{'run':<35} {'dataset':<18} {'accuracy':>10} {'mra':>8}"
    print(header)
    print("-" * 65)
    for m in all_metrics:
        acc = f"{m.get('accuracy', ''):.4f}" if m.get("accuracy") != "" else "-"
        mra_v = f"{m.get('mra', ''):.4f}" if m.get("mra") != "" else "-"
        print(f"{m['run']:<35} {m['dataset']:<18} {acc:>10} {mra_v:>8}")

    print(f"\n[INFO] Resultados → {output_dir}")
    print(f"[INFO] Tabela     → {summary_path}")


if __name__ == "__main__":
    main()
