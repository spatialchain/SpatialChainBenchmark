#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

## script para converter um arquivo CSV para JSONL (1 objeto JSON por linha)
## chamar: python csv_to_jsonl.py   --input-csv data/testset.csv   --output-jsonl data/testset.jsonl   --skip-empty-rows

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Converte um arquivo CSV para JSONL (1 objeto JSON por linha)."
    )
    parser.add_argument(
        "--input-csv",
        required=True,
        help="Caminho do arquivo CSV de entrada.",
    )
    parser.add_argument(
        "--output-jsonl",
        required=False,
        help="Caminho do arquivo JSONL de saída. Se omitido, usa o mesmo nome do CSV com extensão .jsonl.",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8",
        help="Encoding do CSV de entrada (default: utf-8).",
    )
    parser.add_argument(
        "--skip-empty-rows",
        action="store_true",
        help="Ignora linhas totalmente vazias.",
    )
    return parser.parse_args()


def infer_output_path(input_csv: Path, output_jsonl: str | None) -> Path:
    if output_jsonl:
        return Path(output_jsonl)
    return input_csv.with_suffix(".jsonl")


def normalize_row(row: dict[str, str | None]) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for key, value in row.items():
        normalized_key = (key or "").lstrip("\ufeff").strip()
        normalized[normalized_key] = "" if value is None else str(value)
    return normalized


def is_row_empty(row: dict[str, str]) -> bool:
    return all(not value.strip() for value in row.values())


def convert_csv_to_jsonl(
    input_csv: Path,
    output_jsonl: Path,
    encoding: str = "utf-8",
    skip_empty_rows: bool = False,
) -> int:
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with input_csv.open("r", encoding=encoding, newline="") as f_in, output_jsonl.open(
        "w", encoding="utf-8", newline=""
    ) as f_out:
        reader = csv.DictReader(f_in)
        for row in reader:
            normalized = normalize_row(row)
            if skip_empty_rows and is_row_empty(normalized):
                continue
            f_out.write(json.dumps(normalized, ensure_ascii=False) + "\n")
            written += 1

    return written


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input_csv)
    if not input_csv.exists():
        raise SystemExit(f"Arquivo de entrada não encontrado: {input_csv}")

    output_jsonl = infer_output_path(input_csv, args.output_jsonl)
    count = convert_csv_to_jsonl(
        input_csv=input_csv,
        output_jsonl=output_jsonl,
        encoding=args.encoding,
        skip_empty_rows=args.skip_empty_rows,
    )
    print(f"Conversão concluída: {count} linhas escritas em {output_jsonl}")


if __name__ == "__main__":
    main()
