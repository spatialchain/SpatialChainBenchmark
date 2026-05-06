#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Avaliação rápida de múltiplos checkpoints LoRA com vLLM + hot-swap em A100.

Estratégia:
  1. Carrega o modelo base UMA vez no vLLM com enable_lora=True.
  2. Para cada checkpoint (incluindo base), faz UMA chamada llm.generate()
     com todos os prompts. O vLLM cuida do batching com continuous batching.
  3. Salva apenas as predições raw em JSONL — métricas calculadas offline.

Pré-requisito (uma vez):
  pip install "vllm>=0.7" peft

CAVEAT IMPORTANTE — modules_to_save:
  Se teus adapters foram treinados com modules_to_save=["lm_head","embed_tokens"]
  (como no teu finetune), eles contêm pesos full-precision desses módulos que
  o vLLM NÃO carrega. Use --strip_modules_to_save para criar cópias "limpas"
  dos adapters automaticamente antes da avaliação.

Uso típico:
  python eval_checkpoints_vllm.py \
      --csv_path reasoning_verified.csv \
      --images_dir /workspace/gqa/images \
      --model_name Qwen/Qwen3-VL-8B-Instruct \
      --checkpoint_dir /workspace/outputs/qwen3vl8b_paper \
      --output_dir /workspace/eval_results \
      --strip_modules_to_save \
      --val_split 0.02 \
      --max_lora_rank 32

Para gerar predições só do modelo base (sem adapter):
  python eval_checkpoints_vllm.py \
      --csv_path reasoning_verified.csv \
      --images_dir /workspace/gqa/images \
      --model_name Qwen/Qwen3-VL-8B-Instruct \
      --output_dir /workspace/eval_results \
      --eval_base \
      --no_eval_checkpoints
"""

import argparse
import gc
import json
import os
import random
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# vLLM gosta dessas envs para multi-modal
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import pandas as pd
from PIL import Image, ImageFile
from tqdm import tqdm

# vLLM imports — só importamos no momento de uso para permitir --help sem GPU
ImageFile.LOAD_TRUNCATED_IMAGES = True


# ══════════════════════════════════════════════════════════════════════════════
# Helpers de I/O — copiados/adaptados do teu eval_checkpoints_hf.py
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


def discover_checkpoints(checkpoint_dir: Path) -> List[Path]:
    found: List[Tuple[int, Path]] = []
    for ckpt in sorted(checkpoint_dir.glob("checkpoint-*")):
        if (ckpt / "adapter_config.json").exists():
            try:
                step = int(ckpt.name.split("-")[-1].replace("step", ""))
                found.append((step, ckpt))
            except ValueError:
                pass
        elif (ckpt / "lora_adapter" / "adapter_config.json").exists():
            try:
                step = int(ckpt.name.split("-")[-1].replace("step", ""))
                found.append((step, ckpt / "lora_adapter"))
            except ValueError:
                pass
    for final_name in ("lora_adapter", "adapter", "lora_adapter_final"):
        final = checkpoint_dir / final_name
        if final.exists() and (final / "adapter_config.json").exists():
            found.append((10**9, final))
            break
    found.sort(key=lambda x: x[0])
    return [p for _, p in found]


# ══════════════════════════════════════════════════════════════════════════════
# Strip de modules_to_save dos adapters — pré-processamento para vLLM
# ══════════════════════════════════════════════════════════════════════════════
def strip_adapter_for_vllm(src: Path, dst: Path) -> Path:
    """
    Cria uma cópia do adapter sem os tensores full-precision de
    modules_to_save (lm_head, embed_tokens). Mantém só os pesos LoRA.
    """
    import safetensors.torch as st
    import torch

    dst.mkdir(parents=True, exist_ok=True)

    # Copia tudo exceto os pesos
    for f in src.iterdir():
        if f.name in ("adapter_model.safetensors", "adapter_model.bin"):
            continue
        if f.is_file():
            shutil.copy2(f, dst / f.name)

    # Filtra pesos: mantém apenas lora_A / lora_B
    src_weights = src / "adapter_model.safetensors"
    if src_weights.exists():
        full = st.load_file(str(src_weights))
        clean = {k: v for k, v in full.items()
                 if "lora_" in k and "modules_to_save" not in k}
        n_removed = len(full) - len(clean)
        st.save_file(clean, str(dst / "adapter_model.safetensors"))
        print(f"[INFO]   {src.name}: {n_removed} tensores removidos "
              f"(de {len(full)} → {len(clean)})")
    else:
        # fallback .bin
        bin_path = src / "adapter_model.bin"
        if bin_path.exists():
            full = torch.load(str(bin_path), map_location="cpu")
            clean = {k: v for k, v in full.items()
                     if "lora_" in k and "modules_to_save" not in k}
            torch.save(clean, str(dst / "adapter_model.bin"))

    # Atualiza adapter_config.json para remover modules_to_save
    cfg_path = dst / "adapter_config.json"
    if cfg_path.exists():
        with cfg_path.open() as f:
            cfg = json.load(f)
        cfg["modules_to_save"] = None
        with cfg_path.open("w") as f:
            json.dump(cfg, f, indent=2)

    return dst


def prepare_adapters(adapter_paths: List[Path], strip_dir: Path) -> List[Path]:
    """Strip todos os adapters para um diretório limpo."""
    strip_dir.mkdir(parents=True, exist_ok=True)
    stripped: List[Path] = []
    print(f"[INFO] Strip de modules_to_save → {strip_dir}")
    for src in adapter_paths:
        dst = strip_dir / f"{src.parent.name}__{src.name}"
        if (dst / "adapter_model.safetensors").exists() or \
           (dst / "adapter_model.bin").exists():
            print(f"[INFO]   {src.name}: já existe, pulando")
        else:
            strip_adapter_for_vllm(src, dst)
        stripped.append(dst)
    return stripped


# ══════════════════════════════════════════════════════════════════════════════
# Construção de prompts para Qwen3-VL via apply_chat_template
# ══════════════════════════════════════════════════════════════════════════════
def build_prompt_text(processor: Any, system_prompt: Optional[str],
                      question: str) -> str:
    """
    Aplica o chat template do Qwen3-VL com placeholder de imagem.
    Retorna a string com tokens <|vision_start|><|image_pad|><|vision_end|>
    nos lugares certos — o vLLM substitui pelo embedding da imagem real.
    """
    msgs: List[Dict] = []
    if system_prompt:
        msgs.append({"role": "system",
                     "content": [{"type": "text", "text": system_prompt}]})
    msgs.append({"role": "user", "content": [
        {"type": "image"},                  # placeholder
        {"type": "text", "text": question},
    ]})
    return processor.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True,
    )


def build_inputs(records: List[Dict[str, Any]],
                 images_dir: Path,
                 image_pattern: Optional[str],
                 image_max_size: int,
                 system_prompt: Optional[str],
                 processor: Any) -> Tuple[List[Dict], List[Dict]]:
    """
    Constrói a lista de inputs para vLLM e a lista paralela de metadados.
    Retorna (vllm_inputs, metadata).
    """
    vllm_inputs: List[Dict] = []
    metadata:    List[Dict] = []
    n_skipped = 0

    for idx, row in enumerate(tqdm(records, desc="Carregando imagens",
                                   unit="img")):
        image_id = clean_text(row.get("imageId", ""))
        question = clean_text(row.get("question_x")) or clean_text(row.get("question", ""))
        thinking = clean_text(row.get("thinking", ""))
        answer   = clean_text(row.get("fullAnswer")) or clean_text(row.get("answer", ""))

        try:
            img_path = resolve_image(images_dir, image_id, image_pattern)
            with Image.open(img_path) as raw:
                img = resize_keep_aspect(raw.copy(), image_max_size)
        except Exception as e:
            n_skipped += 1
            continue

        prompt_text = build_prompt_text(processor, system_prompt, question)

        vllm_inputs.append({
            "prompt": prompt_text,
            "multi_modal_data": {"image": img},
        })
        metadata.append({
            "idx"      : idx,
            "imageId"  : image_id,
            "question" : question,
            "ref_thinking": thinking,
            "ref_answer"  : answer,
        })

    print(f"[INFO] Inputs prontos: {len(vllm_inputs):,} (skipped={n_skipped})")
    return vllm_inputs, metadata


# ══════════════════════════════════════════════════════════════════════════════
# Inferência por checkpoint
# ══════════════════════════════════════════════════════════════════════════════
def run_one_checkpoint(llm,
                       sampling_params,
                       vllm_inputs: List[Dict],
                       metadata:    List[Dict],
                       lora_request,
                       run_name:    str,
                       output_dir:  Path) -> None:
    """
    Roda llm.generate() para todos os inputs em uma única chamada.
    O vLLM faz continuous batching — não precisa de loop manual.
    """
    out_path = output_dir / f"{run_name}_predictions.jsonl"
    if out_path.exists():
        print(f"[INFO] {run_name}: já existe, pulando ({out_path})")
        return

    print(f"\n[RUN] {run_name} ({len(vllm_inputs):,} prompts)")
    outputs = llm.generate(
        vllm_inputs,
        sampling_params = sampling_params,
        lora_request    = lora_request,
        use_tqdm        = True,
    )

    # outputs vem na MESMA ordem dos inputs
    with out_path.open("w", encoding="utf-8") as f:
        for meta, out in zip(metadata, outputs):
            pred = out.outputs[0].text if out.outputs else ""
            row = {**meta, "run": run_name, "prediction": pred}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[INFO] {run_name}: salvo → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="vLLM eval multi-checkpoint A100")

    # Dados
    p.add_argument("--csv_path",      type=str, required=True)
    p.add_argument("--images_dir",    type=str, required=True)
    p.add_argument("--image_pattern", type=str, default=None)
    p.add_argument("--val_split",     type=float, default=0.00,
                   help="Mesmo do treino para usar o mesmo subset.")
    p.add_argument("--max_samples",   type=int,   default=0)
    p.add_argument("--seed",          type=int,   default=3407)

    # Modelo
    p.add_argument("--model_name",    type=str,
                   default="Qwen/Qwen3-VL-4B-Thinking")
    p.add_argument("--max_seq_length",type=int, default=4096*4,
                   help="max_model_len no vLLM. 4096 é confortável p/ 8B em A100.")
    p.add_argument("--image_max_size",type=int, default=650)
    p.add_argument("--gpu_mem_util",  type=float, default=0.90,
                   help="Fração da VRAM para o vLLM. 0.90 ok em A100 80GB.")
    p.add_argument("--dtype",         type=str, default="bfloat16",
                   choices=["bfloat16", "float16"])
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--limit_mm_per_prompt_image", type=int, default=1,
                   help="1 para datasets com 1 imagem (GQA). Aumenta p/ ERQA etc.")

    # LoRA
    p.add_argument("--max_lora_rank", type=int, default=32,
                   help="Deve ser >= rank usado no treino.")
    p.add_argument("--strip_modules_to_save", action="store_true",
                   help="Necessário se treinaste com modules_to_save.")
    p.add_argument("--strip_dir",     type=str,
                   default="/tmp/vllm_stripped_adapters")

    # Checkpoints
    p.add_argument("--adapter_paths", type=str, nargs="*", default=None)
    p.add_argument("--checkpoint_dir",type=str, default=None)
    p.add_argument("--eval_base",     action="store_true", default=True)
    p.add_argument("--no_eval_checkpoints", action="store_true",
                   help="Avalia só o base (debug).")

    # Geração
    p.add_argument("--max_new_tokens",type=int, default=4096*8)
    p.add_argument("--temperature",   type=float, default=0.6,
                   help="0.0 = greedy (equivale a do_sample=False).")
    p.add_argument("--top_p",         type=float, default=0.95)

    # Prompt
    p.add_argument("--system_prompt", type=str,
                   default=("Your task:\n"
                            "1. Analyze the image carefully.\n"
                            "2. Provide concise reasoning grounded in visible evidence from the image.\n"
                            "3. End your response with 'Answer: <one short sentence>'."))

    p.add_argument("--output_dir", type=str, default="eval_results")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Lista de adapters a avaliar ────────────────────────────────────────
    adapter_paths: List[Path] = []
    if args.adapter_paths:
        adapter_paths.extend(Path(p) for p in args.adapter_paths)
    if args.checkpoint_dir:
        found = discover_checkpoints(Path(args.checkpoint_dir))
        if found:
            print(f"[INFO] Checkpoints encontrados em {args.checkpoint_dir}:")
            for p in found:
                print(f"       {p}")
            adapter_paths.extend(found)
        else:
            print(f"[WARN] Nenhum checkpoint em {args.checkpoint_dir}")

    if args.no_eval_checkpoints:
        adapter_paths = []

    # ── 2. Strip de modules_to_save se necessário ─────────────────────────────
    if adapter_paths and args.strip_modules_to_save:
        adapter_paths = prepare_adapters(adapter_paths, Path(args.strip_dir))

    # ── 3. Records de validação ───────────────────────────────────────────────
    val_records = load_val_records(
        csv_path    = Path(args.csv_path),
        val_split   = args.val_split,
        max_samples = args.max_samples,
        seed        = args.seed,
    )

    # ── 4. Processor (só para apply_chat_template) ────────────────────────────
    from transformers import AutoProcessor
    print(f"[INFO] Carregando processor: {args.model_name}")
    processor = AutoProcessor.from_pretrained(
        args.model_name, trust_remote_code=True,
    )

    # ── 5. Constrói os inputs UMA vez (compartilhado por todos os runs) ───────
    vllm_inputs, metadata = build_inputs(
        records        = val_records,
        images_dir     = Path(args.images_dir),
        image_pattern  = args.image_pattern,
        image_max_size = args.image_max_size,
        system_prompt  = args.system_prompt,
        processor      = processor,
    )

    if not vllm_inputs:
        print("[ERROR] Nenhum input válido. Abortando.")
        return

    # ── 6. Inicializa vLLM ────────────────────────────────────────────────────
    print(f"[INFO] Inicializando vLLM com {args.model_name}")
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    llm = LLM(
        model                  = args.model_name,
        dtype                  = args.dtype,
        max_model_len          = args.max_seq_length,
        gpu_memory_utilization = args.gpu_mem_util,
        tensor_parallel_size   = args.tensor_parallel_size,
        trust_remote_code      = True,
        enable_lora            = bool(adapter_paths),
        max_loras              = 1 if adapter_paths else 0,
        max_lora_rank          = args.max_lora_rank,
        limit_mm_per_prompt    = {"image": args.limit_mm_per_prompt_image},
        # Para multi-image (ERQA), considera enable_prefix_caching=False
        enable_prefix_caching  = True,
    )

    sampling_params = SamplingParams(
        temperature  = args.temperature,
        max_tokens   = args.max_new_tokens,
        top_p=0.95,
        top_k=20,
        #do_sample=True,
        seed         = 123,

        # Greedy quando temperature=0
    )

    # ── 7. Loop pelos runs ────────────────────────────────────────────────────
    runs: List[Tuple[str, Optional[Any]]] = []
    if args.eval_base:
        runs.append(("base_model", None))
    for i, ap in enumerate(adapter_paths, start=1):
        nice = ap.name
        if nice in ("lora_adapter", "adapter", "lora_adapter_final"):
            nice = f"{ap.parent.name}_final"
        runs.append((nice.replace("/", "_"),
                     LoRARequest(nice, i, str(ap))))

    print(f"[INFO] Total de runs: {len(runs)}")
    for name, _ in runs:
        print(f"       - {name}")

    for name, lora_req in runs:
        run_one_checkpoint(
            llm             = llm,
            sampling_params = sampling_params,
            vllm_inputs     = vllm_inputs,
            metadata        = metadata,
            lora_request    = lora_req,
            run_name        = name,
            output_dir      = output_dir,
        )

    print(f"\n[DONE] Predições em {output_dir}/")
    print(f"[NEXT] Calcule métricas offline com um script separado lendo")
    print(f"       os JSONL — assim podes iterar sem re-rodar inferência.")


if __name__ == "__main__":
    main()
