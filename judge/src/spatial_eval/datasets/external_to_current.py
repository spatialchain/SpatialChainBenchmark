#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from spatial_eval.generator import parse_generation_response


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Converte formato externo de predições para generation_records do projeto."
    )
    parser.add_argument("--input", required=True, help="Arquivo de entrada (.jsonl ou .json).")
    parser.add_argument("--output", required=True, help="Arquivo de saída (.jsonl).")
    parser.add_argument(
        "--images-dir",
        required=True,
        help="Diretório base das imagens; image_path será <images-dir>/<imageId>.jpg.",
    )
    parser.add_argument(
        "--provider-name",
        default="",
        help="Provider salvo em generation.provider_name (default: vazio).",
    )
    return parser.parse_args()


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                rows.append(json.loads(stripped))
        return rows

    with path.open("r", encoding="utf-8") as f:
        parsed = json.load(f)
    if isinstance(parsed, list):
        return [row for row in parsed if isinstance(row, dict)]
    if isinstance(parsed, dict):
        return [parsed]
    raise ValueError(f"Formato de entrada não suportado: {path}")


def _clean_reasoning(reasoning: str) -> str:
    cleaned = re.sub(r"</?think>", "", reasoning or "", flags=re.IGNORECASE)
    return cleaned.strip()


def _as_int(value: Any, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _to_current_row(row: dict[str, Any], idx: int, images_dir: Path, provider_name: str) -> dict[str, Any]:
    row_idx = _as_int(row.get("idx"), idx)
    custom_id = str(row.get("custom_id") or f"row_{row_idx}")
    question = str(row.get("question", "")).strip()
    ref_thinking = str(row.get("ref_thinking", "")).strip()
    ref_answer = str(row.get("ref_answer", "")).strip()
    prediction = str(row.get("prediction", "")).strip()
    image_id = str(row.get("imageId", "")).strip()
    image_path = str(images_dir / f"{image_id}.jpg") if image_id else ""

    final_answer, reasoning = parse_generation_response(prediction)
    reasoning = _clean_reasoning(reasoning)

    return {
        "custom_id": custom_id,
        "row_idx": row_idx,
        "question_type": "image-qa",
        "question": question,
        "ground_truth": ref_answer,
        "thinking": ref_thinking,
        "image_path": image_path,
        "generation": {
            "final_answer": final_answer,
            "reasoning": reasoning,
            "raw_response": prediction,
            "provider_name": provider_name,
        },
    }


def convert_rows(input_rows: list[dict[str, Any]], images_dir: Path, provider_name: str) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for idx, row in enumerate(input_rows):
        converted.append(_to_current_row(row, idx, images_dir, provider_name))
    return converted


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    images_dir = Path(args.images_dir)

    if not input_path.exists():
        raise SystemExit(f"Arquivo de entrada não encontrado: {input_path}")
    if not images_dir.exists():
        raise SystemExit(f"Diretório de imagens não encontrado: {images_dir}")

    input_rows = _load_rows(input_path)
    converted_rows = convert_rows(input_rows, images_dir=images_dir, provider_name=args.provider_name)
    write_jsonl(output_path, converted_rows)
    print(f"Conversão concluída: {len(converted_rows)} linhas escritas em {output_path}")


if __name__ == "__main__":
    main()
