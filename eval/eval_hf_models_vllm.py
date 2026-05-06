#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Avaliação de múltiplos modelos open-source do HuggingFace no dataset de
validação, usando vLLM. Adaptado do eval_checkpoints_vllm.py.

Diferenças vs eval_checkpoints_vllm.py:
  - Sem lógica de LoRA: nada de strip_modules_to_save, max_lora_rank, etc.
  - Em vez de iterar por adapters em cima de um modelo base, itera por
    modelos completos do HF, carregando cada um do zero.
  - Cada modelo tem seu próprio chat template — aplicado via processor.
  - Predições saem em <output_dir>/<model_safe_name>_predictions.jsonl
    com schema idêntico ao eval_checkpoints_vllm.py para reaproveitar
    o notebook de análise.

Uso típico — vários modelos em sequência (best-effort):
  python eval_hf_models_vllm.py \\
      --csv_path reasoning_verified.csv \\
      --images_dir /workspace/gqa/images \\
      --models \\
          Qwen/Qwen2.5-VL-7B-Instruct \\
          Qwen/Qwen3-VL-8B-Instruct \\
          OpenGVLab/InternVL3-8B \\
      --output_dir /workspace/eval_results/hf_models \\
      --val_split 0.02

Recomendado — UM modelo por execução (mais robusto contra OOM):
  for m in Qwen/Qwen2.5-VL-7B-Instruct Qwen/Qwen3-VL-8B-Instruct ; do
    python eval_hf_models_vllm.py \\
        --csv_path reasoning_verified.csv \\
        --images_dir /workspace/gqa/images \\
        --models "$m" \\
        --output_dir /workspace/eval_results/hf_models \\
        --val_split 0.02
  done

Para modelos "thinking" (Qwen3-VL-Thinking, etc.):
  python eval_hf_models_vllm.py \\
      --models Qwen/Qwen3-VL-8B-Thinking \\
      --enable_thinking \\
      --temperature 0.6 --top_p 0.95 --top_k 20 \\
      --max_new_tokens 2048 \\
      [...]

Modelos suportados pelo vLLM (lista parcial, abril/2026):
  Qwen2-VL, Qwen2.5-VL, Qwen3-VL (Instruct e Thinking)
  LLaVA-1.5, LLaVA-1.6 (Next), LLaVA-OneVision
  InternVL2, InternVL3
  MiniCPM-V-2.6
  Pixtral
  Llama-3.2-Vision (com cuidado, suporte parcial)
  Phi-3.5-Vision, Phi-4-multimodal

Verifique sempre em https://docs.vllm.ai/en/latest/models/supported_models.html
"""

import argparse
import gc
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import pandas as pd
from PIL import Image, ImageFile
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True


# ══════════════════════════════════════════════════════════════════════════════
# I/O helpers (idênticos ao eval_checkpoints_vllm.py)
# ══════════════════════════════════════════════════════════════════════════════
def clean_text(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and pd.isna(x):
        return ""
    return str(x).strip()


def _candidate_paths(images_dir: Path, image_id: str) -> Sequence[Path]:
    cands = [
        images_dir / f"{image_id}.jpg",
        images_dir / f"{image_id}.png",
        images_dir / f"{image_id}.jpeg",
        images_dir / f"image{image_id}.jpg",
        images_dir / f"image{image_id}.png",
    ]
    if image_id.isdigit():
        cands.append(images_dir / f"{int(image_id):012d}.jpg")
    return tuple(cands)


def resolve_image(images_dir: Path, image_id: str,
                  pattern: Optional[str] = None) -> Path:
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
    return img.resize(
        (max(1, int(round(w * r))), max(1, int(round(h * r)))),
        Image.Resampling.LANCZOS,
    )


def load_val_records(csv_path: Path, val_split: float,
                     max_samples: int, seed: int) -> List[Dict[str, Any]]:
    df = pd.read_csv(csv_path, low_memory=False)
    if "question_x" not in df.columns and "question" in df.columns:
        df = df.rename(columns={"question": "question_x"})
    if "fullAnswer" not in df.columns and "answer" in df.columns:
        df = df.rename(columns={"answer": "fullAnswer"})
    df = df.dropna(subset=["imageId", "question_x", "fullAnswer"])
    records = df.to_dict(orient="records")
    random.Random(seed).shuffle(records)
    if max_samples and max_samples > 0:
        records = records[:max_samples]
    n_val = max(1, int(round(len(records) * val_split))) if val_split > 0 else len(records)
    val_records = records[:n_val]
    print(f"[INFO] Val records: {len(val_records):,}  "
          f"(val_split={val_split}, total={len(records):,})")
    return val_records


def safe_model_name(name: str) -> str:
    """Converte 'Qwen/Qwen2.5-VL-7B-Instruct' -> 'Qwen__Qwen2.5-VL-7B-Instruct'."""
    return re.sub(r"[^A-Za-z0-9._-]", "__", name)


# ══════════════════════════════════════════════════════════════════════════════
# Construção de prompts — depende do processor de cada modelo
# ══════════════════════════════════════════════════════════════════════════════
def build_prompt_text(processor: Any, system_prompt: Optional[str],
                      question: str, enable_thinking: bool = False) -> str:
    """
    Aplica o chat template do modelo. Funciona para Qwen-VL, LLaVA, InternVL,
    MiniCPM-V, Pixtral — qualquer modelo cujo processor expõe apply_chat_template.
    """
    msgs: List[Dict] = []
    if system_prompt:
        msgs.append({"role": "system",
                     "content": [{"type": "text", "text": system_prompt}]})
    msgs.append({"role": "user", "content": [
        {"type": "image"},   # placeholder — vLLM substitui pela imagem real
        {"type": "text", "text": question},
    ]})

    kwargs = dict(tokenize=False, add_generation_prompt=True)
    # Modelos da família Qwen3 expõem enable_thinking; ignorado pelos outros.
    try:
        return processor.apply_chat_template(msgs, enable_thinking=enable_thinking, **kwargs)
    except TypeError:
        return processor.apply_chat_template(msgs, **kwargs)


def build_inputs(records: List[Dict[str, Any]],
                 images_dir: Path,
                 image_pattern: Optional[str],
                 image_max_size: int,
                 system_prompt: Optional[str],
                 processor: Any,
                 enable_thinking: bool) -> Tuple[List[Dict], List[Dict]]:
    """Retorna (vllm_inputs, metadata) — uma entrada por exemplo."""
    vllm_inputs: List[Dict] = []
    metadata:    List[Dict] = []
    n_skipped = 0

    for idx, row in enumerate(tqdm(records, desc="Carregando imagens", unit="img")):
        image_id = clean_text(row.get("imageId", ""))
        question = clean_text(row.get("question_x")) or clean_text(row.get("question", ""))
        thinking = clean_text(row.get("thinking", ""))
        answer   = clean_text(row.get("fullAnswer")) or clean_text(row.get("answer", ""))

        try:
            img_path = resolve_image(images_dir, image_id, image_pattern)
            with Image.open(img_path) as raw:
                img = resize_keep_aspect(raw.copy(), image_max_size)
        except Exception:
            n_skipped += 1
            continue

        prompt_text = build_prompt_text(
            processor, system_prompt, question, enable_thinking,
        )

        vllm_inputs.append({
            "prompt": prompt_text,
            "multi_modal_data": {"image": img},
        })
        metadata.append({
            "idx": idx,
            "imageId": image_id,
            "question": question,
            "ref_thinking": thinking,
            "ref_answer": answer,
        })

    print(f"[INFO] Inputs prontos: {len(vllm_inputs):,} (skipped={n_skipped})")
    return vllm_inputs, metadata


# ══════════════════════════════════════════════════════════════════════════════
# Inferência por modelo
# ══════════════════════════════════════════════════════════════════════════════
def run_one_model(model_name: str,
                  records: List[Dict[str, Any]],
                  images_dir: Path,
                  image_pattern: Optional[str],
                  output_dir: Path,
                  args: argparse.Namespace) -> bool:
    """
    Carrega o modelo, gera predições, salva em JSONL e libera memória.
    Retorna True se concluiu com sucesso.
    """
    safe = safe_model_name(model_name)
    out_path = output_dir / f"{safe}_predictions.jsonl"
    if out_path.exists() and not args.overwrite:
        print(f"[SKIP] {model_name}: já existe ({out_path}). Use --overwrite para refazer.")
        return True

    print(f"\n{'='*70}")
    print(f"[MODEL] {model_name}")
    print(f"{'='*70}")
    t0 = time.time()

    # ── Processor — necessário ANTES do vLLM para apply_chat_template ────────
    from transformers import AutoProcessor
    print(f"[INFO] Carregando processor: {model_name}")
    try:
        processor = AutoProcessor.from_pretrained(
            model_name, trust_remote_code=True,
        )
    except Exception as e:
        print(f"[ERROR] Falha ao carregar processor de {model_name}: {e}")
        return False

    # ── Constrói os inputs uma vez para ESTE modelo ──────────────────────────
    vllm_inputs, metadata = build_inputs(
        records         = records,
        images_dir      = images_dir,
        image_pattern   = image_pattern,
        image_max_size  = args.image_max_size,
        system_prompt   = args.system_prompt,
        processor       = processor,
        enable_thinking = args.enable_thinking,
    )
    if not vllm_inputs:
        print(f"[ERROR] Nenhum input válido para {model_name}.")
        return False

    # Libera o processor antes de carregar o LLM (economiza um pouco de RAM)
    del processor
    gc.collect()

    # ── Carrega vLLM ─────────────────────────────────────────────────────────
    from vllm import LLM, SamplingParams
    print(f"[INFO] Inicializando vLLM com {model_name}")
    try:
        llm = LLM(
            model                  = model_name,
            dtype                  = args.dtype,
            max_model_len          = args.max_seq_length,
            gpu_memory_utilization = args.gpu_mem_util,
            tensor_parallel_size   = args.tensor_parallel_size,
            trust_remote_code      = True,
            limit_mm_per_prompt    = {"image": args.limit_mm_per_prompt_image},
            enable_prefix_caching  = True,
            # NOTA: sem enable_lora aqui — modelos full do HF, não adapters.
        )
    except Exception as e:
        print(f"[ERROR] Falha ao inicializar vLLM com {model_name}: {e}")
        return False

    sampling = SamplingParams(
        temperature = args.temperature,
        top_p       = args.top_p,
        top_k       = args.top_k,
        max_tokens  = args.max_new_tokens,
        seed        = args.seed,
    )

    # ── Gera ────────────────────────────────────────────────────────────────
    print(f"[INFO] Gerando {len(vllm_inputs):,} predições...")
    try:
        outputs = llm.generate(
            vllm_inputs,
            sampling_params=sampling,
            use_tqdm=True,
        )
    except Exception as e:
        print(f"[ERROR] Falha durante geração: {e}")
        # tenta limpar antes de retornar
        del llm
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except ImportError:
            pass
        return False

    # ── Salva (mesmo schema do eval_checkpoints_vllm.py) ─────────────────────
    with out_path.open("w", encoding="utf-8") as f:
        for meta, out in zip(metadata, outputs):
            pred = out.outputs[0].text if out.outputs else ""
            row = {**meta, "run": safe, "model_name": model_name, "prediction": pred}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    elapsed = time.time() - t0
    print(f"[INFO] {model_name}: salvo → {out_path}  "
          f"({len(metadata):,} predições em {elapsed/60:.1f} min)")

    # ── Libera memória — best effort ────────────────────────────────────────
    del llm, vllm_inputs, outputs
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
        # destrói o process group do vLLM se existir (ajuda em multi-modelo)
        try:
            from vllm.distributed.parallel_state import destroy_model_parallel
            destroy_model_parallel()
        except Exception:
            pass
    except ImportError:
        pass

    return True


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Avaliação de modelos open do HF no conjunto de validação (vLLM)"
    )

    # ── Dados ──────────────────────────────────────────────────────────────
    p.add_argument("--csv_path",      type=str, required=True)
    p.add_argument("--images_dir",    type=str, required=True)
    p.add_argument("--image_pattern", type=str, default=None)
    p.add_argument("--val_split",     type=float, default=0.00,
                   help="Mesmo do treino para usar o mesmo subset.")
    p.add_argument("--max_samples",   type=int,   default=0)
    p.add_argument("--seed",          type=int,   default=3407)

    # ── Modelos ────────────────────────────────────────────────────────────
    p.add_argument("--models", type=str, nargs="+", required=True,
                   help="Lista de model names do HuggingFace (ex: Qwen/Qwen2.5-VL-7B-Instruct).")

    # ── Configuração de geração ────────────────────────────────────────────
    p.add_argument("--max_seq_length", type=int, default=4096*4,
                   help="max_model_len no vLLM. Aumente para modelos thinking.")
    p.add_argument("--image_max_size", type=int, default=448)
    p.add_argument("--gpu_mem_util",   type=float, default=0.90)
    p.add_argument("--dtype",          type=str, default="bfloat16",
                   choices=["bfloat16", "float16", "auto"])
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--limit_mm_per_prompt_image", type=int, default=1,
                   help="1 para datasets com 1 imagem (GQA). Aumente p/ ERQA, etc.")

    # ── Sampling ───────────────────────────────────────────────────────────
    p.add_argument("--max_new_tokens", type=int, default=4096*8)
    p.add_argument("--temperature",    type=float, default=0.0,
                   help="0.0 = greedy. Use 0.6 para Qwen3 thinking.")
    p.add_argument("--top_p",          type=float, default=1.0)
    p.add_argument("--top_k",          type=int,   default=-1,
                   help="-1 = desabilitado. Use 20 para Qwen3 thinking.")
    p.add_argument("--enable_thinking", action="store_true",
                   help="Ativa thinking mode em modelos da família Qwen3 (e ignora silenciosamente nos outros).")

    # ── Prompt ─────────────────────────────────────────────────────────────
    p.add_argument("--system_prompt", type=str,
                   default=("Your task:\n"
                            "1. Analyze the image carefully.\n"
                            "2. Provide concise reasoning grounded in visible evidence from the image.\n"
                            "3. End your response with 'Answer: <one short sentence>'."))

    # ── Output ─────────────────────────────────────────────────────────────
    p.add_argument("--output_dir", type=str, default="eval_results/hf_models")
    p.add_argument("--overwrite",  action="store_true",
                   help="Refaz inferência mesmo se o arquivo já existir.")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Carrega o conjunto de validação ──────────────────────────────────────
    val_records = load_val_records(
        csv_path    = Path(args.csv_path),
        val_split   = args.val_split,
        max_samples = args.max_samples,
        seed        = args.seed,
    )

    if not val_records:
        print("[ERROR] Nenhum exemplo no conjunto de validação. Abortando.")
        return

    print(f"\n[INFO] Modelos a avaliar ({len(args.models)}):")
    for m in args.models:
        print(f"       - {m}")
    if len(args.models) > 1:
        print("[WARN] Múltiplos modelos numa execução — vLLM pode segurar VRAM "
              "entre modelos. Se der OOM, rode um modelo por execução.")

    # ── Loop pelos modelos ──────────────────────────────────────────────────
    results: List[Tuple[str, bool, float]] = []
    grand_t0 = time.time()
    for model_name in args.models:
        t0 = time.time()
        ok = run_one_model(
            model_name    = model_name,
            records       = val_records,
            images_dir    = Path(args.images_dir),
            image_pattern = args.image_pattern,
            output_dir    = output_dir,
            args          = args,
        )
        results.append((model_name, ok, time.time() - t0))

    # ── Resumo ───────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"RESUMO  (total: {(time.time() - grand_t0)/60:.1f} min)")
    print(f"{'='*70}")
    for name, ok, elapsed in results:
        status = "✓" if ok else "✗"
        print(f"  {status}  {name:<55} {elapsed/60:>6.1f} min")

    print(f"\n[DONE] Predições em {output_dir}/")
    print(f"[NEXT] Use o checkpoint_analysis.ipynb para analisar — basta")
    print(f"       adicionar os JSONL gerados à lista CHECKPOINTS.")


if __name__ == "__main__":
    main()
