from __future__ import annotations

import json
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

from spatial_eval.datasets import adapt_records
from spatial_eval.generator import build_generator_prompt, build_image_question_prompt
from spatial_eval.judge import build_judge_prompt
from spatial_eval.metrics import compute_metrics
from spatial_eval.normalization import normalize_answer
from spatial_eval.prompt_store import load_default_prompt
from spatial_eval.providers.base import GenerationProvider, JudgeProvider
from spatial_eval.schemas import EvalRecord, GenerationOutput, InputSample, JudgeOutput


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            rows.append(json.loads(stripped))
    return rows


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def to_input_sample(record: dict[str, Any]) -> InputSample:
    return InputSample(
        row_idx=record.get("row_idx"),
        custom_id=str(record.get("custom_id", f"row_{record.get('row_idx', 'unknown')}")),
        question=str(record.get("question", "")),
        ground_truth=(str(record["ground_truth"]) if record.get("ground_truth") is not None else None),
        question_type=record.get("type"),
        scene_graph=record.get("scene_graph"),
        image_path=record.get("image_path"),
        raw_record=record,
    )


def stratified_sample(records: list[dict[str, Any]], per_type: int, max_samples: int | None, seed: int) -> list[dict[str, Any]]:
    if per_type <= 0:
        selected = records[:]
    else:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in records:
            grouped[str(row.get("type", "unknown"))].append(row)

        rng = random.Random(seed)
        selected = []
        for qtype, rows in grouped.items():
            del qtype
            rows_copy = rows[:]
            rng.shuffle(rows_copy)
            selected.extend(rows_copy[:per_type])

    if max_samples is not None:
        return selected[:max_samples]
    return selected


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    _ensure_parent_dir(path)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _append_jsonl_row(path: Path, row: dict[str, Any]) -> None:
    _ensure_parent_dir(path)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_jsonl_if_exists(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return load_jsonl(path)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    _ensure_parent_dir(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def run_dataset_conversion(
    dataset_path: str | Path,
    output_path: str | Path,
    dataset_adapter: str = "current",
) -> dict[str, Any]:
    dataset_path = Path(dataset_path)
    output_path = Path(output_path)
    raw_rows = _load_dataset_rows(dataset_path)
    converted_rows = adapt_records(raw_rows, adapter_name=dataset_adapter)
    _write_jsonl(output_path, converted_rows)
    return {
        "input_path": str(dataset_path),
        "output_path": str(output_path),
        "dataset_adapter": dataset_adapter,
        "total": len(converted_rows),
    }


def _write_markdown_report(path: Path, metrics: dict[str, Any], output_jsonl: Path, output_json: Path) -> None:
    _ensure_parent_dir(path)
    lines = [
        "# Relatório de Avaliação (MVP)",
        "",
        f"- total: **{metrics.get('total')}**",
        f"- baseline_accuracy: **{metrics.get('baseline_accuracy')}**",
        f"- judge_pass_rate: **{metrics.get('judge_pass_rate')}**",
        f"- avg_answer_correctness: **{metrics.get('avg_answer_correctness')}**",
        f"- avg_reasoning_faithfulness: **{metrics.get('avg_reasoning_faithfulness')}**",
        f"- avg_reasoning_completeness: **{metrics.get('avg_reasoning_completeness')}**",
        "",
        "## Métricas por tipo",
    ]
    by_type = metrics.get("by_type", {})
    if not by_type:
        lines.append("- Sem dados por tipo.")
    else:
        for qtype, stats in sorted(by_type.items()):
            lines.append(
                f"- `{qtype}`: total={stats.get('total')}, "
                f"baseline_accuracy={stats.get('baseline_accuracy')}, "
                f"judge_pass_rate={stats.get('judge_pass_rate')}"
            )

    lines.extend(
        [
            "",
            "## Artefatos",
            f"- records: `{output_jsonl}`",
            f"- metrics: `{output_json}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_dataset_rows(dataset_path: Path) -> list[dict[str, Any]]:
    if dataset_path.suffix == ".jsonl":
        return load_jsonl(dataset_path)
    return [load_json(dataset_path)]


def _select_samples(
    dataset_path: str | Path,
    dataset_adapter: str,
    sample_per_type: int,
    max_samples: int | None,
    seed: int,
    images_dir: str | Path | None = None,
) -> list[InputSample]:
    path = Path(dataset_path)
    raw_rows = _load_dataset_rows(path)
    if images_dir is not None:
        images_dir_value = str(images_dir)
        for row in raw_rows:
            row["__images_dir"] = images_dir_value
    rows = adapt_records(raw_rows, adapter_name=dataset_adapter)
    selected_rows = stratified_sample(rows, per_type=sample_per_type, max_samples=max_samples, seed=seed)
    return [to_input_sample(row) for row in selected_rows]


def _eval_record_from_sample(
    sample: InputSample,
    generation: GenerationOutput,
    judge_output,
) -> EvalRecord:
    normalized_pred = normalize_answer(generation.final_answer, sample.question_type, sample.question)
    normalized_gt = normalize_answer(sample.ground_truth or "", sample.question_type, sample.question)
    baseline_match = None
    if normalized_pred is not None and normalized_gt is not None:
        baseline_match = normalized_pred == normalized_gt

    return EvalRecord(
        custom_id=sample.custom_id,
        row_idx=sample.row_idx,
        question_type=sample.question_type,
        question=sample.question,
        ground_truth=sample.ground_truth,
        reference_thinking=(str(sample.raw_record.get("thinking")).strip() if sample.raw_record.get("thinking") is not None else None),
        generation=generation,
        judge=judge_output,
        normalized_prediction=normalized_pred,
        normalized_ground_truth=normalized_gt,
        baseline_match=baseline_match,
    )


def _persist_eval_artifacts(
    eval_records: list[EvalRecord],
    records_path: Path,
    metrics_path: Path,
    report_md_path: Path,
) -> dict[str, Any]:
    metrics = compute_metrics(eval_records)
    _write_json(metrics_path, metrics)
    _write_markdown_report(report_md_path, metrics, records_path, metrics_path)
    return metrics


def _eval_record_from_dict(row: dict[str, Any]) -> EvalRecord:
    generation_raw = row.get("generation") or {}
    judge_raw = row.get("judge") or {}
    generation = GenerationOutput(
        final_answer=str(generation_raw.get("final_answer", "")),
        reasoning=str(generation_raw.get("reasoning", "")),
        raw_response=str(generation_raw.get("raw_response", "")),
        provider_name=str(generation_raw.get("provider_name", "unknown-generation-provider")),
    )
    judge = JudgeOutput(
        answer_correctness=float(judge_raw.get("answer_correctness", 0.0)),
        reasoning_faithfulness=float(judge_raw.get("reasoning_faithfulness", 0.0)),
        reasoning_completeness=float(judge_raw.get("reasoning_completeness", 0.0)),
        verdict=str(judge_raw.get("verdict", "fail")),
        justification=str(judge_raw.get("justification", "")),
        raw_response=str(judge_raw.get("raw_response", "")),
        provider_name=str(judge_raw.get("provider_name", "unknown-judge-provider")),
    )
    baseline_match_raw = row.get("baseline_match")
    baseline_match: bool | None
    if isinstance(baseline_match_raw, bool):
        baseline_match = baseline_match_raw
    else:
        baseline_match = None
    return EvalRecord(
        custom_id=str(row.get("custom_id", "")),
        row_idx=row.get("row_idx"),
        question_type=row.get("question_type"),
        question=str(row.get("question", "")),
        ground_truth=(str(row["ground_truth"]) if row.get("ground_truth") is not None else None),
        reference_thinking=(
            str(row.get("reference_thinking")).strip()
            if row.get("reference_thinking") is not None
            else (str(row.get("thinking")).strip() if row.get("thinking") is not None else None)
        ),
        generation=generation,
        judge=judge,
        normalized_prediction=row.get("normalized_prediction"),
        normalized_ground_truth=row.get("normalized_ground_truth"),
        baseline_match=baseline_match,
    )


def run_generation(
    dataset_path: str | Path,
    output_dir: str | Path,
    generation_provider: GenerationProvider,
    dataset_adapter: str = "current",
    sample_per_type: int = 0,
    max_samples: int | None = None,
    seed: int = 42,
    generator_prompt_template: str | None = None,
    images_dir: str | Path | None = None,
    resume: bool = True,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    samples = _select_samples(
        dataset_path,
        dataset_adapter=dataset_adapter,
        sample_per_type=sample_per_type,
        max_samples=max_samples,
        seed=seed,
        images_dir=images_dir,
    )
    generator_prompt_template = generator_prompt_template or load_default_prompt("generator")

    records_path = output_dir / "generation_records.jsonl"
    existing_rows: list[dict[str, Any]] = []
    existing_custom_ids: set[str] = set()
    if resume:
        existing_rows = _load_jsonl_if_exists(records_path)
        existing_custom_ids = {str(row.get("custom_id", "")) for row in existing_rows}
    if not resume or not records_path.exists():
        _ensure_parent_dir(records_path)
        records_path.write_text("", encoding="utf-8")
    total_generated = len(existing_rows)
    try:
        for sample in samples:
            if sample.custom_id in existing_custom_ids:
                continue
            generation_prompt = build_generator_prompt(sample, generator_prompt_template)
            generation = generation_provider.generate(sample, generation_prompt)
            row = {
                "custom_id": sample.custom_id,
                "row_idx": sample.row_idx,
                "question_type": sample.question_type,
                "question": sample.question,
                "ground_truth": sample.ground_truth,
                "thinking": sample.raw_record.get("thinking"),
                "scene_graph": sample.scene_graph,
                "image_path": sample.image_path,
                "generation": generation.to_dict(),
            }
            _append_jsonl_row(records_path, row)
            total_generated += 1
            existing_custom_ids.add(sample.custom_id)
    finally:
        summary = {"total": total_generated, "records_path": str(records_path)}
        _write_json(output_dir / "generation_summary.json", summary)
    return summary


def _sample_and_generation_from_row(row: dict[str, Any]) -> tuple[InputSample, GenerationOutput]:
    sample = InputSample(
        row_idx=row.get("row_idx"),
        custom_id=str(row.get("custom_id", f"row_{row.get('row_idx', 'unknown')}")),
        question=str(row.get("question", "")),
        ground_truth=(str(row["ground_truth"]) if row.get("ground_truth") is not None else None),
        question_type=row.get("question_type") or row.get("type"),
        scene_graph=row.get("scene_graph"),
        image_path=row.get("image_path"),
        raw_record=row,
    )
    generation_raw = row.get("generation") or {}
    generation = GenerationOutput(
        final_answer=str(generation_raw.get("final_answer", "")),
        reasoning=str(generation_raw.get("reasoning", "")),
        raw_response=str(generation_raw.get("raw_response", "")),
        provider_name=str(generation_raw.get("provider_name", "unknown-generation-provider")),
    )
    return sample, generation


def _normalize_judge_evidence(judge_evidence: str) -> str:
    normalized = (judge_evidence or "").strip().lower()
    if normalized not in {"image", "img_graph", "both"}:
        raise ValueError("judge_evidence must be one of: image, img_graph, both")
    return normalized


def _extract_image_id(sample: InputSample) -> str | None:
    if sample.image_path:
        stem = Path(str(sample.image_path)).stem.strip()
        if stem:
            return stem

    for key in ("imageId", "imageID", "image_id"):
        value = sample.raw_record.get(key)
        if value is None:
            continue
        value_str = str(value).strip()
        if value_str:
            return value_str
    return None


def _load_scene_graph_lookup(img_graph_file: str | Path | None) -> dict[str, Any] | None:
    if img_graph_file is None:
        return None
    path = Path(img_graph_file)
    if not path.exists():
        raise ValueError(f"judge img-graph-file not found: {path}")
    loaded = load_json(path)
    if not isinstance(loaded, dict):
        raise ValueError(f"judge img-graph-file must be a JSON object keyed by imageId: {path}")
    return {str(k): v for k, v in loaded.items()}


def _resolve_image_from_dir(image_id: str, images_dir: Path) -> str | None:
    preferred = images_dir / f"{image_id}.jpg"
    if preferred.exists():
        return str(preferred)
    matches = sorted(images_dir.glob(f"{image_id}.*"))
    if not matches:
        return None
    return str(matches[0])


def _enrich_sample_for_judge(
    sample: InputSample,
    judge_evidence: str,
    judge_images_dir: str | Path | None,
    scene_graph_lookup: dict[str, Any] | None,
) -> InputSample:
    evidence = _normalize_judge_evidence(judge_evidence)
    image_id = _extract_image_id(sample)

    if judge_images_dir is not None and image_id:
        resolved = _resolve_image_from_dir(image_id, Path(judge_images_dir))
        if resolved is not None:
            sample.image_path = resolved
            sample.raw_record["image_path"] = resolved

    if scene_graph_lookup is not None and image_id:
        graph_obj = scene_graph_lookup.get(image_id)
        if graph_obj is not None:
            graph_text = graph_obj if isinstance(graph_obj, str) else json.dumps(graph_obj, ensure_ascii=False)
            sample.scene_graph = graph_text
            sample.raw_record["scene_graph"] = graph_text

    has_image = bool(sample.image_path and Path(str(sample.image_path)).exists())
    has_graph = bool(sample.scene_graph or sample.raw_record.get("scene_graph"))

    if evidence in {"image", "both"} and not has_image:
        raise ValueError(
            f"Missing required image for judge sample custom_id={sample.custom_id}, image_id={image_id or 'unknown'} "
            f"(judge_evidence={evidence})"
        )
    if evidence in {"img_graph", "both"} and not has_graph:
        raise ValueError(
            f"Missing required scene graph for judge sample custom_id={sample.custom_id}, "
            f"image_id={image_id or 'unknown'} (judge_evidence={evidence})"
        )
    return sample


def run_judging(
    generation_records_path: str | Path,
    output_dir: str | Path,
    judge_provider: JudgeProvider,
    judge_prompt_template: str | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
    score_justification: bool = True,
    judge_evidence: str = "img_graph",
    max_samples: int | None = None,
    judge_images_dir: str | Path | None = None,
    judge_img_graph_file: str | Path | None = None,
    resume: bool = True,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    rows = load_jsonl(generation_records_path)
    if max_samples is not None:
        rows = rows[:max_samples]
    if judge_images_dir is not None and not Path(judge_images_dir).exists():
        raise ValueError(f"judge images-dir not found: {judge_images_dir}")
    scene_graph_lookup = _load_scene_graph_lookup(judge_img_graph_file)
    judge_prompt_template = judge_prompt_template or load_default_prompt("judge")
    eval_records: list[EvalRecord] = []
    records_path = output_dir / "eval_records.jsonl"
    metrics_path = output_dir / "eval_metrics.json"
    report_md_path = output_dir / "eval_report.md"
    existing_records = [_eval_record_from_dict(row) for row in (_load_jsonl_if_exists(records_path) if resume else [])]
    existing_custom_ids = {record.custom_id for record in existing_records}
    if not resume or not records_path.exists():
        _ensure_parent_dir(records_path)
        records_path.write_text("", encoding="utf-8")
    eval_records.extend(existing_records)
    total_rows = len(rows)
    try:
        for idx, row in enumerate(rows, start=1):
            sample, generation = _sample_and_generation_from_row(row)
            if sample.custom_id in existing_custom_ids:
                continue
            sample = _enrich_sample_for_judge(
                sample=sample,
                judge_evidence=judge_evidence,
                judge_images_dir=judge_images_dir,
                scene_graph_lookup=scene_graph_lookup,
            )
            if progress_callback is not None:
                progress_callback(idx, total_rows, sample.custom_id)
            judge_prompt = build_judge_prompt(
                sample,
                generation,
                judge_prompt_template,
                score_justification=score_justification,
                judge_evidence=judge_evidence,
            )
            judge_output = judge_provider.judge(sample, generation, judge_prompt)
            record = _eval_record_from_sample(sample, generation, judge_output)
            eval_records.append(record)
            _append_jsonl_row(records_path, record.to_dict())
            existing_custom_ids.add(sample.custom_id)
    finally:
        metrics = _persist_eval_artifacts(eval_records, records_path, metrics_path, report_md_path)

    return {
        "metrics": metrics,
        "records_path": str(records_path),
        "metrics_path": str(metrics_path),
        "report_path": str(report_md_path),
    }


def run_image_question_generation(
    image_path: str | Path,
    question: str,
    generation_provider: GenerationProvider,
    prompt_template: str | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    prompt_template = prompt_template or load_default_prompt("generator_vision")
    sample = InputSample(
        row_idx=None,
        custom_id="single_image_question",
        question=question,
        ground_truth=None,
        question_type="single_image_question",
        image_path=str(image_path),
        raw_record={},
    )
    prompt = build_image_question_prompt(question=question, prompt_template=prompt_template)
    generation = generation_provider.generate(sample, prompt)
    result = {
        "image_path": str(image_path),
        "question": question,
        "final_answer": generation.final_answer,
        "reasoning": generation.reasoning,
        "raw_response": generation.raw_response,
        "provider_name": generation.provider_name,
    }
    if output_path is not None:
        _write_json(Path(output_path), result)
    return result


def run_evaluation(
    dataset_path: str | Path,
    output_dir: str | Path,
    generation_provider: GenerationProvider,
    judge_provider: JudgeProvider,
    dataset_adapter: str = "current",
    sample_per_type: int = 0,
    max_samples: int | None = None,
    seed: int = 42,
    generator_prompt_template: str | None = None,
    judge_prompt_template: str | None = None,
    images_dir: str | Path | None = None,
    score_justification: bool = True,
    judge_evidence: str = "img_graph",
    judge_images_dir: str | Path | None = None,
    judge_img_graph_file: str | Path | None = None,
    resume: bool = True,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    samples = _select_samples(
        dataset_path,
        dataset_adapter=dataset_adapter,
        sample_per_type=sample_per_type,
        max_samples=max_samples,
        seed=seed,
        images_dir=images_dir,
    )
    generator_prompt_template = generator_prompt_template or load_default_prompt("generator")
    judge_prompt_template = judge_prompt_template or load_default_prompt("judge")
    if judge_images_dir is not None and not Path(judge_images_dir).exists():
        raise ValueError(f"judge images-dir not found: {judge_images_dir}")
    scene_graph_lookup = _load_scene_graph_lookup(judge_img_graph_file)
    eval_records: list[EvalRecord] = []
    records_path = output_dir / "eval_records.jsonl"
    metrics_path = output_dir / "eval_metrics.json"
    report_md_path = output_dir / "eval_report.md"
    existing_records = [_eval_record_from_dict(row) for row in (_load_jsonl_if_exists(records_path) if resume else [])]
    existing_custom_ids = {record.custom_id for record in existing_records}
    if not resume or not records_path.exists():
        _ensure_parent_dir(records_path)
        records_path.write_text("", encoding="utf-8")
    eval_records.extend(existing_records)

    try:
        for sample in samples:
            if sample.custom_id in existing_custom_ids:
                continue
            generation_prompt = build_generator_prompt(sample, generator_prompt_template)
            generation = generation_provider.generate(sample, generation_prompt)
            sample = _enrich_sample_for_judge(
                sample=sample,
                judge_evidence=judge_evidence,
                judge_images_dir=judge_images_dir,
                scene_graph_lookup=scene_graph_lookup,
            )

            judge_prompt = build_judge_prompt(
                sample,
                generation,
                judge_prompt_template,
                score_justification=score_justification,
                judge_evidence=judge_evidence,
            )
            judge_output = judge_provider.judge(sample, generation, judge_prompt)
            record = _eval_record_from_sample(sample, generation, judge_output)
            eval_records.append(record)
            _append_jsonl_row(records_path, record.to_dict())
            existing_custom_ids.add(sample.custom_id)
    finally:
        metrics = _persist_eval_artifacts(eval_records, records_path, metrics_path, report_md_path)

    return {
        "metrics": metrics,
        "records_path": str(records_path),
        "metrics_path": str(metrics_path),
        "report_path": str(report_md_path),
    }


def run_image_question_generation_batch(
    dataset_path: str | Path,
    images_dir: str | Path,
    output_dir: str | Path,
    generation_provider: GenerationProvider,
    dataset_adapter: str = "testset-image-qa",
    sample_per_type: int = 0,
    max_samples: int | None = None,
    seed: int = 42,
    prompt_template: str | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    prompt_template = prompt_template or load_default_prompt("generator_vision")
    samples = _select_samples(
        dataset_path=dataset_path,
        dataset_adapter=dataset_adapter,
        sample_per_type=sample_per_type,
        max_samples=max_samples,
        seed=seed,
        images_dir=images_dir,
    )

    generation_rows: list[dict[str, Any]] = []
    total_samples = len(samples)
    print(f"[generate-iq-batch] Iniciando processamento de {total_samples} amostras.")
    batch_start = time.perf_counter()
    for idx, sample in enumerate(samples, start=1):
        item_start = time.perf_counter()
        print(f"[generate-iq-batch] [{idx}/{total_samples}] Processando custom_id={sample.custom_id}")
        prompt = build_image_question_prompt(question=sample.question, prompt_template=prompt_template)
        generation = generation_provider.generate(sample, prompt)
        item_elapsed = time.perf_counter() - item_start
        print(f"[generate-iq-batch] [{idx}/{total_samples}] Concluido em {item_elapsed:.2f}s")
        generation_rows.append(
            {
                "custom_id": sample.custom_id,
                "row_idx": sample.row_idx,
                "question_type": sample.question_type,
                "question": sample.question,
                "ground_truth": sample.ground_truth,
                "thinking": sample.raw_record.get("thinking"),
                "image_path": sample.image_path,
                "generation": generation.to_dict(),
            }
        )
    batch_elapsed = time.perf_counter() - batch_start
    print(f"[generate-iq-batch] Finalizado: {len(generation_rows)} amostras em {batch_elapsed:.2f}s")

    records_path = output_dir / "generation_records.jsonl"
    _write_jsonl(records_path, generation_rows)
    summary = {"total": len(generation_rows), "records_path": str(records_path)}
    _write_json(output_dir / "generation_summary.json", summary)
    return summary
