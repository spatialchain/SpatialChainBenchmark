#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Avaliação do checkpoint no FlagEval/ERQA.

Dataset: FlagEval/ERQA (HuggingFace) — 400 exemplos, split "test"
Formato:
  - question      : string com questão + choices já embutidas
  - answer        : string letra ("A", "B", "C" ou "D")
  - question_type : categoria (Trajectory Reasoning, Action Reasoning, etc.)
  - images        : lista de PIL Images (1 a 16 imagens por exemplo)
  - visual_indices: índices das imagens relevantes
Métrica: Accuracy overall + por question_type

Uso — modelo base:
python eval_erqa.py \
  --model_name Qwen/Qwen3-VL-8B-Instruct \
  --output_file results/erqa_base.json

Uso — com adapter finetuned:
python eval_erqa.py \
  --model_name Qwen/Qwen3-VL-8B-Instruct \
  --adapter_path outputs/qwen3vl8b_hf/lora_adapter \
  --output_file results/erqa_finetuned.json
"""

import gc
import json
import os
import re
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
torch.backends.cudnn.allow_tf32       = True

# Letras válidas de resposta
VALID_LETTERS = {"A", "B", "C", "D"}

_ANSWER_TAG_RE = re.compile(r"<answer>\s*([A-Da-d])\s*</answer>", re.IGNORECASE)
_LETTER_RE     = re.compile(r"\b([A-Da-d])\b")


# ══════════════════════════════════════════════════════════════════════════════
# Extração de resposta
# ══════════════════════════════════════════════════════════════════════════════
def extract_letter(pred_text: str) -> str:
    """
    Extrai a letra (A/B/C/D) da resposta do modelo.
    Retorna "" se não conseguir parsear.
    """
    text = pred_text.strip()

    # 1. Tag <answer>A</answer>
    m = _ANSWER_TAG_RE.search(text)
    if m:
        return m.group(1).upper()

    # 2. Primeira letra isolada A-D
    letters = [l.upper() for l in _LETTER_RE.findall(text)
               if l.upper() in VALID_LETTERS]
    if letters:
        return letters[0]

    # 3. Primeira letra do texto (o modelo às vezes responde só "A")
    first = text.strip()[:1].upper()
    if first in VALID_LETTERS:
        return first

    return ""


# ══════════════════════════════════════════════════════════════════════════════
# Prompt builder
# ══════════════════════════════════════════════════════════════════════════════
def build_messages(
    question     : str,
    images       : List[Image.Image],
    system_prompt: Optional[str],
) -> List[Dict]:
    """
    ERQA já inclui as opções no campo question e pede resposta com letra.
    As imagens são inseridas antes do texto da questão.
    """
    # Conteúdo do turno user: imagens + questão
    user_content: List[Dict] = []
    for img in images:
        user_content.append({"type": "image", "image": img})
    user_content.append({"type": "text", "text": question})

    msgs: List[Dict] = []
    if system_prompt:
        msgs.append({"role": "system",
                     "content": [{"type": "text", "text": system_prompt}]})
    msgs.append({"role": "user", "content": user_content})
    return msgs


def resize_image(img: Image.Image, max_size: int) -> Image.Image:
    img = img.convert("RGB")
    w, h = img.size
    s = max(w, h)
    if s <= max_size:
        return img
    r = max_size / s
    return img.resize((max(1, int(w * r)), max(1, int(h * r))),
                      Image.Resampling.LANCZOS)


# ══════════════════════════════════════════════════════════════════════════════
# Modelo
# ══════════════════════════════════════════════════════════════════════════════
def load_model(model_name: str, adapter_path: Optional[Path]):
    torch.cuda.init()
    torch.cuda.empty_cache()
    print(f"[INFO] VRAM livre: {torch.cuda.mem_get_info()[0] / 1e9:.1f} GB")

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit              = True,
        bnb_4bit_quant_type       = "nf4",
        bnb_4bit_compute_dtype    = torch.float16,
        bnb_4bit_use_double_quant = True,
    )
    print(f"[INFO] Carregando modelo: {model_name}")
    model = AutoModelForVision2Seq.from_pretrained(
        model_name,
        quantization_config = bnb_config,
        torch_dtype         = torch.float16,
        device_map          = "cuda:0",
        low_cpu_mem_usage   = True,
        trust_remote_code   = True,
    )
    model.config.use_cache = True

    if adapter_path is not None:
        print(f"[INFO] Carregando adapter: {adapter_path}")
        model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=False)

    model.eval()
    return model, processor


# ══════════════════════════════════════════════════════════════════════════════
# Geração
# ══════════════════════════════════════════════════════════════════════════════
def _encode(
    processor  : Any,
    msgs       : List[Dict],
    max_length : int,
) -> Dict[str, torch.Tensor]:
    images, clean_msgs = [], []
    for msg in msgs:
        content = msg.get("content", [])
        if not isinstance(content, list):
            clean_msgs.append(msg)
            continue
        new_content = []
        for part in content:
            if part.get("type") == "image":
                images.append(part["image"])
                new_content.append({"type": "image"})
            else:
                new_content.append(part)
        clean_msgs.append({**msg, "content": new_content})

    return processor(
        text       = processor.apply_chat_template(
            clean_msgs, tokenize=False, add_generation_prompt=True,
        ),
        images     = images if images else None,
        padding    = True,
        truncation = True,
        max_length = max_length,
        return_tensors = "pt",
    )


def generate_one(
    model         : Any,
    processor     : Any,
    msgs          : List[Dict],
    max_new_tokens: int,
    max_length    : int,
) -> str:
    pad_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id or 0
    enc    = _encode(processor, msgs, max_length)
    ilen   = enc["input_ids"].shape[1]
    enc    = {k: v.to(model.device) for k, v in enc.items()}

    with torch.inference_mode():
        out = model.generate(
            **enc,
            max_new_tokens = max_new_tokens,
            do_sample      = False,
            use_cache      = True,
            pad_token_id   = pad_id,
        )
    return processor.tokenizer.decode(out[0, ilen:], skip_special_tokens=True).strip()


# ══════════════════════════════════════════════════════════════════════════════
# Avaliação
# ══════════════════════════════════════════════════════════════════════════════
def run_eval(
    model         : Any,
    processor     : Any,
    dataset       : Any,
    system_prompt : Optional[str],
    max_new_tokens: int,
    max_seq_length: int,
    image_max_size: int,
) -> Tuple[List[Dict], Dict[str, Any]]:

    results: List[Dict] = []
    type_correct: Dict[str, int] = {}
    type_total  : Dict[str, int] = {}
    n_correct = n_total = n_skipped = n_unparsed = 0

    for sample in tqdm(dataset, desc="Evaluating ERQA", unit="sample"):
        q_id    = sample.get("question_id", "")
        question= sample.get("question", "")
        gt      = str(sample.get("answer", "")).strip().upper()
        qtype   = sample.get("question_type", "unknown")
        raw_imgs= sample.get("images", [])

        try:
            # Resize imagens
            imgs = [resize_image(img, image_max_size)
                    for img in raw_imgs if img is not None]
            if not imgs:
                raise ValueError("Nenhuma imagem válida.")

            msgs = build_messages(question, imgs, system_prompt)
            pred = generate_one(model, processor, msgs, max_new_tokens, max_seq_length)

            pred_letter = extract_letter(pred)
            if not pred_letter:
                n_unparsed += 1

            correct = int(pred_letter == gt and pred_letter != "")

        except Exception as e:
            print(f"[WARN] {q_id}: {e}")
            pred        = f"ERROR: {e}"
            pred_letter = ""
            correct     = 0
            n_skipped  += 1

        results.append({
            "question_id"   : q_id,
            "question_type" : qtype,
            "question"      : question[:200],   # trunca para não engordar o JSON
            "gt_answer"     : gt,
            "pred_letter"   : pred_letter,
            "raw_prediction": pred,
            "correct"       : correct,
        })

        n_correct += correct
        n_total   += 1
        type_correct[qtype] = type_correct.get(qtype, 0) + correct
        type_total[qtype]   = type_total.get(qtype,   0) + 1

        torch.cuda.empty_cache()

    d = max(n_total, 1)
    per_type = {
        t: round(type_correct[t] / max(type_total[t], 1), 4)
        for t in sorted(type_total)
    }

    metrics: Dict[str, Any] = {
        "overall_accuracy": round(n_correct / d, 4),
        "n_total"         : n_total,
        "n_correct"       : n_correct,
        "n_skipped"       : n_skipped,
        "n_unparsed"      : n_unparsed,
        "per_question_type": per_type,
    }
    return results, metrics


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Avaliação no FlagEval/ERQA | HF model + LoRA adapter"
    )

    p.add_argument("--model_name",    type=str,
                   default="Qwen/Qwen3-VL-8B-Instruct")
    p.add_argument("--adapter_path",  type=str, default=None,
                   help="Caminho para o adapter LoRA. Omita para usar o modelo base.")
    p.add_argument("--dataset_name",  type=str, default="FlagEval/ERQA")
    p.add_argument("--dataset_split", type=str, default="test")
    p.add_argument("--max_samples",   type=int, default=0,
                   help="0 = usar tudo. Use para testes rápidos.")

    # Parâmetros para 3090
    p.add_argument("--max_seq_length", type=int, default=1024)
    p.add_argument("--image_max_size", type=int, default=336,
                   help="Menor que EmbSpatial pois ERQA pode ter múltiplas imagens.")
    p.add_argument("--max_new_tokens", type=int, default=16,
                   help="ERQA pede só a letra — 16 tokens é suficiente.")

    p.add_argument("--system_prompt", type=str,
                   default=(
                       "You are an expert in embodied spatial reasoning and robotics. "
                       "Analyze the image(s) carefully and answer the multiple-choice question. "
                       "Reply with only the letter of the correct option (A, B, C, or D)."
                   ))
    p.add_argument("--output_file",   type=str, default="erqa_results.json")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # ── Carrega dataset ───────────────────────────────────────────────────────
    print(f"[INFO] Carregando {args.dataset_name} ({args.dataset_split})...")
    ds = load_dataset(
        args.dataset_name,
        split             = args.dataset_split,
        trust_remote_code = True,
    )
    if args.max_samples and args.max_samples > 0:
        ds = ds.select(range(min(args.max_samples, len(ds))))
    print(f"[INFO] {len(ds):,} exemplos")
    print(f"[INFO] Colunas: {ds.column_names}")

    # ── Carrega modelo ────────────────────────────────────────────────────────
    adapter_path = Path(args.adapter_path) if args.adapter_path else None
    model, processor = load_model(args.model_name, adapter_path)

    run_name = "base_model" if adapter_path is None else Path(args.adapter_path).name

    # ── Avalia ────────────────────────────────────────────────────────────────
    results, metrics = run_eval(
        model          = model,
        processor      = processor,
        dataset        = ds,
        system_prompt  = args.system_prompt,
        max_new_tokens = args.max_new_tokens,
        max_seq_length = args.max_seq_length,
        image_max_size = args.image_max_size,
    )

    metrics["run"] = run_name

    # ── Imprime resumo ────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"[RESULTS] {run_name} — FlagEval/ERQA")
    print(f"  Overall Accuracy : {metrics['overall_accuracy']:.4f}  "
          f"({metrics['n_correct']}/{metrics['n_total']})")
    print(f"  Skipped          : {metrics['n_skipped']}")
    print(f"  Unparsed         : {metrics['n_unparsed']}")
    print(f"\n  Per question type:")
    for qtype, acc in metrics["per_question_type"].items():
        n = sum(1 for r in results if r["question_type"] == qtype)
        c = sum(1 for r in results if r["question_type"] == qtype and r["correct"])
        print(f"    {qtype:<25}: {acc:.4f}  ({c}/{n})")
    print(f"{'='*65}\n")

    # ── Salva ─────────────────────────────────────────────────────────────────
    with output_file.open("w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "results": results}, f,
                  indent=2, ensure_ascii=False)

    print(f"[INFO] Resultados salvos → {output_file}")


if __name__ == "__main__":
    main()
