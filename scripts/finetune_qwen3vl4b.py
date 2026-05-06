#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fine-tuning do Qwen3-VL-8B-Thinking | A100 80GB | RunPod
Dataset: trainset.csv (30k) + valset.csv (1k) — pré-separados

Checkpoints salvos em:
  1k, 5k, 10k amostras → saves irregulares iniciais
  1 época (30k)         → save normal
  Após isso, a cada 15k amostras (½ época)

W&B:
  - Loss de treino a cada 50 steps
  - eval_loss + métricas gerativas a cada checkpoint salvo

Exemplo de uso no RunPod:
python finetune_qwen3vl8b_paper.py \
  --trainset_csv /workspace/data/trainset.csv \
  --valset_csv   /workspace/data/valset.csv \
  --images_dir   /workspace/data/gqa/images \
  --output_dir   /workspace/outputs/qwen3vl8b_paper \
  --model_name   Qwen/Qwen3-VL-8B-Thinking \
  --max_train_samples 30000 \
  --max_val_samples   1000 \
  --wandb_project spatial-reasoning \
  --wandb_run_name    qwen3vl8b_paper_v1 \
  --time_budget_hours 22
"""

import gc
import json
import logging
import math
import os
import random
import re
import string
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse

import pandas as pd
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    BitsAndBytesConfig,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType

ImageFile.LOAD_TRUNCATED_IMAGES = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Regex para extração de métricas gerativas ──────────────────────────────────
_THINK_RE  = re.compile(r"<think>\s*(.*?)\s*</think>",   re.IGNORECASE | re.DOTALL)
_ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
_TAG_RE    = re.compile(r"</?(think|answer)>",           re.IGNORECASE)
_PUNCT     = str.maketrans("", "", string.punctuation)
_ARTICLES  = {"a", "an", "the"}


# ══════════════════════════════════════════════════════════════════════════════
# Métricas de texto
# ══════════════════════════════════════════════════════════════════════════════
def clean(x: Any) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    return str(x).strip()


def normalize(text: str) -> str:
    t = clean(text).lower()
    t = _TAG_RE.sub(" ", t).translate(_PUNCT)
    return " ".join(w for w in t.split() if w and w not in _ARTICLES)


def _tokens(t: str) -> List[str]:
    n = normalize(t)
    return n.split() if n else []


def _lcs(a: List[str], b: List[str]) -> int:
    if not a or not b:
        return 0
    s, l = (a, b) if len(a) <= len(b) else (b, a)
    prev = [0] * (len(s) + 1)
    for tok in l:
        curr = [0]
        for j, oth in enumerate(s, 1):
            curr.append(prev[j - 1] + 1 if tok == oth else max(prev[j], curr[-1]))
        prev = curr
    return prev[-1]


def rouge_l(pred: str, ref: str) -> float:
    p, r = _tokens(pred), _tokens(ref)
    if not p or not r:
        return 0.0
    lcs = _lcs(p, r)
    pr, rc = lcs / len(p), lcs / len(r)
    return 2 * pr * rc / (pr + rc) if pr + rc else 0.0


def token_f1(pred: str, ref: str) -> float:
    p, r = _tokens(pred), _tokens(ref)
    if not p or not r:
        return 0.0
    rc: Dict[str, int] = {}
    for t in r:
        rc[t] = rc.get(t, 0) + 1
    ov = 0
    for t in p:
        if rc.get(t, 0) > 0:
            ov += 1
            rc[t] -= 1
    if not ov:
        return 0.0
    pr, rec = ov / len(p), ov / len(r)
    return 2 * pr * rec / (pr + rec)


def exact_match(pred: str, ref: str) -> float:
    return float(normalize(pred) == normalize(ref))


def extract_parts(text: str) -> Tuple[str, str]:
    raw = clean(text)
    tm  = _THINK_RE.search(raw)
    am  = _ANSWER_RE.search(raw)
    thinking = clean(tm.group(1)) if tm else ""
    answer   = clean(am.group(1)) if am else re.sub(r"\s+", " ", _TAG_RE.sub(" ", raw)).strip()
    return thinking, answer


# ══════════════════════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════════════════════
def _candidate_paths(images_dir: Path, image_id: str) -> Sequence[Path]:
    cands = [
        images_dir / f"{image_id}.jpg",
        images_dir / f"{image_id}.jpeg",
        images_dir / f"{image_id}.png",
    ]
    if image_id.isdigit():
        cands.append(images_dir / f"{int(image_id):012d}.jpg")
    return tuple(cands)


def resolve_image(images_dir: Path, image_id: str, pattern: Optional[str] = None) -> Path:
    if pattern:
        p = images_dir / pattern.format(imageId=image_id)
        if p.exists():
            return p
        raise FileNotFoundError(p)
    for p in _candidate_paths(images_dir, image_id):
        if p.exists():
            return p
    raise FileNotFoundError(f"imageId={image_id} não encontrado em {images_dir}")


def resize_keep_aspect(img: Image.Image, max_size: int) -> Image.Image:
    img = img.convert("RGB")
    w, h = img.size
    s = max(w, h)
    if s <= max_size:
        return img
    r = max_size / s
    return img.resize((max(1, int(round(w * r))), max(1, int(round(h * r)))),
                      Image.Resampling.LANCZOS)


class SpatialDataset(Dataset):
    """Dataset lazy — imagem aberta só no __getitem__."""

    def __init__(
        self,
        records         : List[Dict[str, Any]],
        images_dir      : Path,
        image_max_size  : int,
        system_prompt   : Optional[str],
        image_pattern   : Optional[str] = None,
        include_thinking: bool = True,
        strict          : bool = False,
    ):
        if not images_dir.exists():
            raise FileNotFoundError(f"images_dir não encontrado: {images_dir}")
        self.records          = records
        self.images_dir       = images_dir
        self.image_max_size   = image_max_size
        self.system_prompt    = system_prompt
        self.image_pattern    = image_pattern
        self.include_thinking = include_thinking
        self.strict           = strict
        log.info(f"Dataset: {len(records):,} registros")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.records[idx]
        try:
            return self._build(row)
        except Exception as e:
            if self.strict:
                raise
            iid = clean(row.get("imageId", f"idx_{idx}"))
            q   = clean(row.get("question_x")) or clean(row.get("question", "Describe the image."))
            log.warning(f"idx={idx} imageId={iid}: {e}")
            blank = Image.new("RGB", (self.image_max_size, self.image_max_size))
            return self._wrap(blank, q, f"<answer>\nSkipped: {e}\n</answer>")

    def _build(self, row: Dict[str, Any]) -> Dict[str, Any]:
        iid     = clean(row.get("imageId", ""))
        q       = clean(row.get("question_x")) or clean(row.get("question", ""))
        thinking= clean(row.get("thinking", ""))
        answer  = clean(row.get("fullAnswer")) or clean(row.get("answer", ""))

        if not iid:    raise ValueError("imageId vazio")
        if not q:      raise ValueError("question vazio")
        if not answer: raise ValueError("answer vazio")

        if self.include_thinking and thinking:
            target = f"<think>\n{thinking}\n</think>\n<answer>\n{answer}\n</answer>"
        else:
            target = f"<answer>\n{answer}\n</answer>"

        with Image.open(resolve_image(self.images_dir, iid, self.image_pattern)) as raw:
            img = resize_keep_aspect(raw.copy(), self.image_max_size)

        return self._wrap(img, q, target)

    def _wrap(self, img, question, target) -> Dict[str, Any]:
        msgs: List[Dict] = []
        if self.system_prompt:
            msgs.append({"role": "system",
                         "content": [{"type": "text", "text": self.system_prompt}]})
        msgs.append({"role": "user",
                     "content": [{"type": "image", "image": img},
                                  {"type": "text",  "text": question}]})
        msgs.append({"role": "assistant",
                     "content": [{"type": "text", "text": target}]})
        return {"messages": msgs, "question": question, "target": target}


# ══════════════════════════════════════════════════════════════════════════════
# Collator
# ══════════════════════════════════════════════════════════════════════════════
class MultimodalCollator:
    def __init__(self, processor: Any, max_length: int):
        self.processor  = processor
        self.max_length = max_length

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        texts:  List[str]         = []
        images: List[Image.Image] = []

        for sample in batch:
            sample_images, clean_msgs = [], []
            for msg in sample["messages"]:
                content = msg.get("content", [])
                if not isinstance(content, list):
                    clean_msgs.append(msg)
                    continue
                new_content = []
                for part in content:
                    if part.get("type") == "image":
                        sample_images.append(part["image"])
                        new_content.append({"type": "image"})
                    else:
                        new_content.append(part)
                clean_msgs.append({**msg, "content": new_content})

            texts.append(self.processor.apply_chat_template(
                clean_msgs, tokenize=False, add_generation_prompt=False, enable_thinking=True,
            ))
            images.extend(sample_images)

        encoding = self.processor(
            text=texts, images=images if images else None,
            padding=True, truncation=True,
            max_length=self.max_length, return_tensors="pt",
        )

        labels = encoding["input_ids"].clone()

        # Mascara tokens do prompt — só calcula loss na resposta do assistant
        try:
            assistant_ids = self.processor.tokenizer(
                "<|im_start|>assistant", add_special_tokens=False
            )["input_ids"]
        except Exception:
            assistant_ids = []

        if assistant_ids:
            for i in range(labels.shape[0]):
                seq      = labels[i].tolist()
                last_pos = -1
                for pos in range(len(seq) - len(assistant_ids)):
                    if seq[pos: pos + len(assistant_ids)] == assistant_ids:
                        last_pos = pos
                if last_pos >= 0:
                    labels[i, : last_pos + len(assistant_ids)] = -100
                else:
                    labels[i, :] = -100

        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is not None:
            labels[encoding["input_ids"] == pad_id] = -100

        encoding["labels"] = labels
        return encoding


# ══════════════════════════════════════════════════════════════════════════════
# Callback de checkpoints nos milestones definidos para o paper
# ══════════════════════════════════════════════════════════════════════════════
class MilestoneCheckpointCallback(TrainerCallback):
    """
    Salva checkpoints e roda avaliação nos marcos definidos:
      - 1k, 5k, 10k amostras  (saves iniciais irregulares)
      - 1 época (30k amostras)
      - Depois: a cada 15k amostras (½ época)
    Loga tudo no W&B.
    """

    def __init__(
        self,
        trainer_ref          : Any,         # referência ao Trainer (setada depois)
        max_train_samples    : int,
        effective_batch_size : int,
        val_dataset          : Dataset,
        processor            : Any,
        output_dir           : Path,
        gen_eval_samples     : int = 50,
        max_new_tokens       : int = 512,
        use_wandb            : bool = True,
    ):
        self.trainer_ref          = trainer_ref
        self.max_train_samples    = max_train_samples
        self.effective_batch      = effective_batch_size
        self.val_dataset          = val_dataset
        self.processor            = processor
        self.output_dir           = output_dir
        self.gen_eval_samples     = gen_eval_samples
        self.max_new_tokens       = max_new_tokens
        self.use_wandb            = use_wandb
        self.start_time           = time.time()

        steps_per_epoch   = math.ceil(max_train_samples / effective_batch_size)
        half_epoch_steps  = math.ceil(steps_per_epoch / 2)

        # Milestones em steps
        early = [
            math.ceil(1_000  / effective_batch_size),
            math.ceil(5_000  / effective_batch_size),
            math.ceil(10_000 / effective_batch_size),
        ]
        self.milestone_steps = set(early + [steps_per_epoch, 0])
        self.half_epoch_steps = half_epoch_steps
        self.steps_per_epoch  = steps_per_epoch

        self.fired: set = set()

        log.info(f"MilestoneCheckpointCallback:")
        log.info(f"  Steps/época         : {steps_per_epoch}")
        log.info(f"  Steps/meia época    : {half_epoch_steps}")
        log.info(f"  Milestones iniciais : {sorted(self.milestone_steps)}")
        log.info(f"  Depois do 1º epoch  : a cada {half_epoch_steps} steps")

    def _should_fire(self, step: int) -> bool:
        if step in self.milestone_steps:
            return True
        # Depois de 1 epoch, a cada half_epoch_steps
        if step > self.steps_per_epoch:
            steps_after = step - self.steps_per_epoch
            if steps_after % self.half_epoch_steps == 0:
                return True
        return False

    def on_step_end(
        self,
        args : TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> TrainerControl:
        step = state.global_step
        if not self._should_fire(step):
            return control
        if step in self.fired:
            return control
        self.fired.add(step)

        trainer = self.trainer_ref
        samples = step * self.effective_batch
        elapsed = (time.time() - self.start_time) / 3600

        log.info(f"{'='*65}")
        log.info(f"CHECKPOINT  step={step}  samples≈{samples:,}  elapsed={elapsed:.1f}h")

        # ── Salva checkpoint ──────────────────────────────────────────────────
        ckpt_dir = self.output_dir / f"checkpoint-step{step}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(ckpt_dir))
        self.processor.save_pretrained(str(ckpt_dir))
        log.info(f"  Saved → {ckpt_dir}")

        # ── Eval loss (forward pass, rápido) ──────────────────────────────────
        eval_metrics = trainer.evaluate()
        eval_loss    = eval_metrics.get("eval_loss", float("nan"))
        log.info(f"  eval_loss = {eval_loss:.4f}")

        # ── Métricas gerativas (sample de gen_eval_samples) ───────────────────
        gen_metrics = self._gen_eval(trainer.model)
        log.info(f"  gen/answer_f1   = {gen_metrics.get('gen/answer_f1', 0):.4f}")
        log.info(f"  gen/answer_em   = {gen_metrics.get('gen/answer_em', 0):.4f}")

        # ── Loga no W&B ───────────────────────────────────────────────────────
        if self.use_wandb:
            try:
                import wandb
                log_dict = {
                    "milestone/step"    : step,
                    "milestone/samples" : samples,
                    "milestone/epoch"   : round(step / self.steps_per_epoch, 3),
                    "milestone/eval_loss": eval_loss,
                    "milestone/elapsed_h": elapsed,
                    **{f"milestone/{k}": v for k, v in gen_metrics.items()},
                }
                wandb.log(log_dict, step=step)
            except Exception as e:
                log.warning(f"W&B log falhou: {e}")

        # Volta ao modo treino
        trainer.model.train()
        gc.collect()
        torch.cuda.empty_cache()

        return control

    def _gen_eval(self, model) -> Dict[str, float]:
        """Gera respostas para uma amostra do val set e calcula EM/F1/ROUGE."""
        model.eval()
        pad_id = (self.processor.tokenizer.pad_token_id
                  or self.processor.tokenizer.eos_token_id or 0)

        total = len(self.val_dataset)
        indices = random.sample(range(total), min(self.gen_eval_samples, total))

        ans_em = ans_f1 = ans_rouge = thi_f1 = thi_rouge = 0.0
        n = 0

        for idx in indices:
            try:
                sample  = self.val_dataset[idx]
                ref_t, ref_a = extract_parts(sample["target"])

                msgs    = sample["messages"][:-1]   # sem assistant turn
                images, clean_msgs = [], []
                for msg in msgs:
                    content = msg.get("content", [])
                    if not isinstance(content, list):
                        clean_msgs.append(msg)
                        continue
                    nc = []
                    for part in content:
                        if part.get("type") == "image":
                            images.append(part["image"])
                            nc.append({"type": "image"})
                        else:
                            nc.append(part)
                    clean_msgs.append({**msg, "content": nc})

                enc = self.processor(
                    text=self.processor.apply_chat_template(
                        clean_msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True,
                    ),
                    images=images if images else None,
                    return_tensors="pt",
                ).to(model.device)

                with torch.inference_mode():
                    out = model.generate(
                        **enc,
                        max_new_tokens=self.max_new_tokens,
                        temperature     = 0.6,
                        top_p           = 0.95,
                        top_k           = 20,
                        use_cache       = True,
                        pad_token_id    = pad_id,
                    )

                pred_text = clean(self.processor.tokenizer.decode(
                    out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True
                ))
                if(n < 3):
                    log.info(f"Referencia: {ref_t} - {ref_a}\n")
                    log.info(f"saida: {pred_text}\n")
                    log.info("=" * 65)
                pred_t, pred_a = extract_parts(pred_text)

                ans_em    += exact_match(pred_a, ref_a)
                ans_f1    += token_f1(pred_a, ref_a)
                ans_rouge += rouge_l(pred_a, ref_a)

                thi_f1    += token_f1(pred_t, ref_t)
                thi_rouge += rouge_l(pred_t, ref_t)
                n += 1

            except Exception as e:
                log.warning(f"gen_eval idx={idx}: {e}")
            finally:
                if "enc" in dir():
                    del enc
                if "out" in dir():
                    del out
                torch.cuda.empty_cache()

        d = max(n, 1)
        return {
            "gen/answer_em"   : round(ans_em    / d, 4),
            "gen/answer_f1"   : round(ans_f1    / d, 4),
            "gen/answer_rouge": round(ans_rouge / d, 4),
            "gen/thinking_f1"   : round(thi_f1    / d, 4),
            "gen/thinking_rouge": round(thi_rouge / d, 4),
            "gen/n_samples"   : n,
        }
        
    def on_train_begin(self, args, state, control, **kwargs):
        """Avalia o modelo base antes de qualquer step de treino."""
        trainer = self.trainer_ref
        self._current_step = 0
        self.fired.add(0)
    
        log.info("=" * 65)
        log.info("EVAL INICIAL (step=0 — modelo base / checkpoint resumido)")
    
        ckpt_dir = self.output_dir / "checkpoint-step0"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(ckpt_dir))
        self.processor.save_pretrained(str(ckpt_dir))
    
        eval_metrics = trainer.evaluate()
        eval_loss    = eval_metrics.get("eval_loss", float("nan"))
        log.info(f"  eval_loss = {eval_loss:.4f}")
    
        gen_metrics = self._gen_eval(trainer.model)
        log.info(f"  gen/answer_f1 = {gen_metrics.get('gen/answer_f1', 0):.4f}")
        log.info(f"  gen/thinking_f1 = {gen_metrics.get('gen/thinking_f1', 0):.4f}")
    
        if self.use_wandb:
            try:
                import wandb
                wandb.log({
                    "milestone/step"     : 0,
                    "milestone/samples"  : 0,
                    "milestone/epoch"    : 0.0,
                    "milestone/eval_loss": eval_loss,
                    **{f"milestone/{k}": v for k, v in gen_metrics.items()},
                }, step=0)
            except Exception as e:
                log.warning(f"W&B log falhou: {e}")
    
        trainer.model.train()
        gc.collect()
        torch.cuda.empty_cache()
        return control


# ══════════════════════════════════════════════════════════════════════════════
# Helpers de dados
# ══════════════════════════════════════════════════════════════════════════════
def load_csv(path: Path, max_samples: int, seed: int) -> List[Dict[str, Any]]:
    log.info(f"Lendo {path} ...")
    df = pd.read_csv(path, low_memory=False)
    log.info(f"  Linhas totais: {len(df):,}")

    if "question_x" not in df.columns and "question" in df.columns:
        df = df.rename(columns={"question": "question_x"})
    if "fullAnswer" not in df.columns and "answer" in df.columns:
        df = df.rename(columns={"answer": "fullAnswer"})

    before = len(df)
    df = df.dropna(subset=["imageId", "question_x", "fullAnswer"])
    if len(df) < before:
        log.warning(f"  {before - len(df):,} linhas removidas por campos nulos")

    records = df.to_dict(orient="records")
    random.Random(seed).shuffle(records)

    if max_samples and max_samples > 0:
        records = records[:max_samples]
        log.info(f"  Usando {len(records):,} amostras")

    return records


def print_gpu_info() -> None:
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        log.info(f"GPU: {props.name} | VRAM: {props.total_memory / 1e9:.1f} GB")
    else:
        log.warning("CUDA não disponível!")


# ══════════════════════════════════════════════════════════════════════════════
# Args
# ══════════════════════════════════════════════════════════════════════════════
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Qwen3-VL-8B-Thinking | A100 80GB | Paper training"
    )

    # Dados
    p.add_argument("--trainset_csv",    type=str, required=True)
    p.add_argument("--valset_csv",      type=str, required=True)
    p.add_argument("--images_dir",      type=str, required=True)
    p.add_argument("--image_pattern",   type=str, default=None)
    p.add_argument("--max_train_samples", type=int, default=30_000)
    p.add_argument("--max_val_samples",   type=int, default=1_000)
    p.add_argument("--seed",            type=int, default=3407)

    # Modelo
    p.add_argument("--model_name",      type=str,
                   default="Qwen/Qwen3-VL-8B-Thinking")
    p.add_argument("--max_seq_length",  type=int, default=32768)
    p.add_argument("--image_max_size",  type=int, default=640)

    # LoRA
    p.add_argument("--lora_r",          type=int,   default=16)
    p.add_argument("--lora_alpha",      type=int,   default=32)
    p.add_argument("--lora_dropout",    type=float, default=0.05)

    # Treino
    p.add_argument("--output_dir",      type=str, default="outputs/qwen3vl8b_paper")
    p.add_argument("--num_train_epochs",type=float, default=2.0)
    p.add_argument("--per_device_train_batch_size", type=int, default=4)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--per_device_eval_batch_size",  type=int, default=4)
    p.add_argument("--learning_rate",   type=float, default=3e-5)
    p.add_argument("--weight_decay",    type=float, default=0.01)
    p.add_argument("--warmup_ratio",    type=float, default=0.05)
    p.add_argument("--max_grad_norm",   type=float, default=1.0)
    p.add_argument("--logging_steps",   type=int,   default=25)
    p.add_argument("--time_budget_hours", type=float, default=22.0,
                   help="Para o treino se ultrapassar este limite de horas.")

    # Eval gerativa no checkpoint
    p.add_argument("--gen_eval_samples",    type=int, default=0,
                   help="Amostras para eval gerativo em cada checkpoint.")
    p.add_argument("--gen_eval_max_new_tokens", type=int, default=32768)

    # Thinking
    p.add_argument("--system_prompt",   type=str,
                   default=(
                        "You are a spatial reasoning expert. "
                        "Analyze the image carefully, reason step-by-step about "
                        "the spatial relationships between objects. "
                        "Wrap all your reasoning in <think> and </think>. "
                        "Then provide your final answer wrapped in <answer> and </answer>. "
                        "Example: <answer>yes</answer>"
                   ))

    # W&B
    p.add_argument("--wandb_project",   type=str, default=None)
    p.add_argument("--wandb_run_name",  type=str, default=None)
    p.add_argument("--wandb_tags",      type=str, default=None)

    # Misc
    p.add_argument("--resume_from_checkpoint", type=str, default=None)
    p.add_argument("--save_final_merged",      action="store_true")

    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# Callback para parar treino pelo budget de horas
# ══════════════════════════════════════════════════════════════════════════════
class TimeBudgetCallback(TrainerCallback):
    def __init__(self, max_hours: float):
        self.deadline = time.time() + max_hours * 3600

    def on_step_end(self, args, state, control, **kwargs):
        if time.time() >= self.deadline:
            log.info("⏰ Time budget atingido. Parando treino...")
            control.should_training_stop = True
        return control


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    args = parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print_gpu_info()

    # ── W&B ───────────────────────────────────────────────────────────────────
    use_wandb = bool(args.wandb_project)
    if use_wandb:
        import wandb
        eff_batch = args.per_device_train_batch_size * args.gradient_accumulation_steps
        steps_per_epoch = math.ceil(args.max_train_samples / eff_batch)
        tags = args.wandb_tags.split(",") if args.wandb_tags else []
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            tags=tags,
            config={
                "model"           : args.model_name,
                "max_train_samples": args.max_train_samples,
                "max_val_samples" : args.max_val_samples,
                "lora_r"          : args.lora_r,
                "lora_alpha"      : args.lora_alpha,
                "batch_effective" : eff_batch,
                "steps_per_epoch" : steps_per_epoch,
                "learning_rate"   : args.learning_rate,
                "seq_len"         : args.max_seq_length,
                "image_max_size"  : args.image_max_size,
                "epochs"          : args.num_train_epochs,
                "time_budget_h"   : args.time_budget_hours,
            },
        )
        log.info(f"W&B run: {wandb.run.url}")

    # ── Dados ─────────────────────────────────────────────────────────────────
    log.info("=" * 65)
    train_records = load_csv(Path(args.trainset_csv), args.max_train_samples, args.seed)
    val_records   = load_csv(Path(args.valset_csv),   args.max_val_samples,   args.seed)
    log.info(f"Train: {len(train_records):,}  |  Val: {len(val_records):,}")

    images_dir = Path(args.images_dir)
    ds_kwargs  = dict(
        images_dir      = images_dir,
        image_max_size  = args.image_max_size,
        system_prompt   = args.system_prompt,
        image_pattern   = args.image_pattern,
        include_thinking= True,
    )
    train_ds = SpatialDataset(records=train_records, **ds_kwargs)
    val_ds   = SpatialDataset(records=val_records,   **ds_kwargs)

    # ── Processor ─────────────────────────────────────────────────────────────
    log.info("=" * 65)
    log.info(f"Carregando processor: {args.model_name}")
    processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    # ── Modelo ────────────────────────────────────────────────────────────────
    log.info(f"Carregando modelo: {args.model_name}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit              = True,
        bnb_4bit_quant_type       = "nf4",
        bnb_4bit_compute_dtype    = torch.bfloat16,
        bnb_4bit_use_double_quant = True,
    )
    model = AutoModelForVision2Seq.from_pretrained(
        args.model_name,
        quantization_config = bnb_config,
        torch_dtype         = torch.bfloat16,
        device_map          = "auto",
        trust_remote_code   = True,
        attn_implementation = "flash_attention_2",
    )
    model.config.use_cache = False   # necessário com gradient checkpointing

    # ── LoRA ──────────────────────────────────────────────────────────────────
    log.info(f"LoRA r={args.lora_r} alpha={args.lora_alpha}")
    linear_names = {
        name.split(".")[-1]
        for name, mod in model.named_modules()
        if isinstance(mod, torch.nn.Linear)
    } - {"lm_head"}

    lora_config = LoraConfig(
        task_type      = TaskType.CAUSAL_LM,
        r              = args.lora_r,
        lora_alpha     = args.lora_alpha,
        lora_dropout   = args.lora_dropout,
        bias           = "none",
        target_modules = sorted(linear_names),
        modules_to_save= ["lm_head", "embed_tokens"],
        use_rslora     = True,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if use_wandb:
        import wandb
        wandb.config.update({"trainable_params_M": n_trainable / 1e6},
                            allow_val_change=True)

    # ── Collator ──────────────────────────────────────────────────────────────
    collator = MultimodalCollator(processor, args.max_seq_length)

    # ── TrainingArguments ─────────────────────────────────────────────────────
    # eval_strategy="steps" com eval_steps muito alto para que apenas
    # o MilestoneCheckpointCallback controle quando avaliar.
    # O Trainer ainda calcula eval_loss quando chamado pelo callback.
    training_args = TrainingArguments(
        output_dir                  = str(output_dir),
        per_device_train_batch_size = args.per_device_train_batch_size,
        per_device_eval_batch_size  = args.per_device_eval_batch_size,
        gradient_accumulation_steps = args.gradient_accumulation_steps,
        learning_rate               = args.learning_rate,
        weight_decay                = args.weight_decay,
        warmup_ratio                = args.warmup_ratio,
        lr_scheduler_type           = "cosine",
        optim                       = "adamw_torch_fused",
        max_grad_norm               = args.max_grad_norm,
        num_train_epochs            = args.num_train_epochs,
        # Precisão — A100 80GB
        bf16                        = True,
        fp16                        = False,
        tf32                        = True,
        # Logging no W&B a cada 50 steps
        logging_steps               = args.logging_steps,
        report_to                   = "wandb" if use_wandb else "none",
        # Salvar desabilitado — o MilestoneCallback controla os saves
        save_strategy               = "no",
        # Eval controlado pelo callback, mas habilitado para trainer.evaluate()
        eval_strategy               = "no",
        # DataLoader — A100/Linux
        dataloader_num_workers      = 8,
        dataloader_pin_memory       = True,
        dataloader_prefetch_factor  = 2,
        # Misc
        gradient_checkpointing      = False,
        gradient_checkpointing_kwargs = {"use_reentrant": False},
        seed                        = args.seed,
        remove_unused_columns       = False,
    )

    trainer = Trainer(
        model            = model,
        args             = training_args,
        train_dataset    = train_ds,
        eval_dataset     = val_ds,
        data_collator    = collator,
        processing_class = processor.tokenizer,
    )

    # ── Callbacks ─────────────────────────────────────────────────────────────
    eff_batch = args.per_device_train_batch_size * args.gradient_accumulation_steps

    milestone_cb = MilestoneCheckpointCallback(
        trainer_ref          = trainer,
        max_train_samples    = args.max_train_samples,
        effective_batch_size = eff_batch,
        val_dataset          = val_ds,
        processor            = processor,
        output_dir           = output_dir,
        gen_eval_samples     = args.gen_eval_samples,
        max_new_tokens       = args.gen_eval_max_new_tokens,
        use_wandb            = use_wandb,
    )
    trainer.add_callback(milestone_cb)
    trainer.add_callback(TimeBudgetCallback(args.time_budget_hours))

    # ── Info pré-treino ───────────────────────────────────────────────────────
    steps_per_epoch  = math.ceil(args.max_train_samples / eff_batch)
    half_epoch_steps = math.ceil(steps_per_epoch / 2)
    total_steps      = math.ceil(steps_per_epoch * args.num_train_epochs)

    log.info("=" * 65)
    log.info("CONFIGURAÇÃO FINAL")
    log.info(f"  Modelo             : {args.model_name}")
    log.info(f"  Train / Val        : {len(train_ds):,} / {len(val_ds):,}")
    log.info(f"  Batch efetivo      : {eff_batch}")
    log.info(f"  Steps por época    : {steps_per_epoch:,}")
    log.info(f"  Steps meia época   : {half_epoch_steps:,}")
    log.info(f"  Épocas             : {args.num_train_epochs}")
    log.info(f"  Total steps        : ~{total_steps:,}")
    log.info(f"  LoRA r             : {args.lora_r}")
    log.info(f"  Seq length         : {args.max_seq_length}")
    log.info(f"  Image max size     : {args.image_max_size}px")
    log.info(f"  Time budget        : {args.time_budget_hours}h")
    log.info(f"  Output             : {output_dir}")
    if use_wandb:
        import wandb
        log.info(f"  W&B                : {wandb.run.url}")
    log.info("=" * 65)

    # ── Treino ────────────────────────────────────────────────────────────────
    gc.collect()
    torch.cuda.empty_cache()

    log.info("Iniciando treino...")
    stats = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint or None)
    log.info("Treino finalizado.")
    log.info(stats.metrics)

    # ── Salva adapter final ───────────────────────────────────────────────────
    final_dir = output_dir / "lora_adapter_final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(final_dir))
    processor.save_pretrained(str(final_dir))
    log.info(f"Adapter final salvo → {final_dir}")

    if args.save_final_merged:
        merged_dir = output_dir / "merged_bf16_final"
        merged_dir.mkdir(parents=True, exist_ok=True)
        log.info(f"Fazendo merge BF16 → {merged_dir} ...")
        merged = model.merge_and_unload()
        merged.save_pretrained(str(merged_dir), safe_serialization=True)
        processor.save_pretrained(str(merged_dir))

    if use_wandb:
        import wandb
        wandb.finish()

    log.info("Tudo pronto.")


if __name__ == "__main__":
    main()
