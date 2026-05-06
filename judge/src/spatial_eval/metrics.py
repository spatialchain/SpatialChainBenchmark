from __future__ import annotations

from collections import defaultdict
from typing import Any

from spatial_eval.judge import compute_verdict
from spatial_eval.schemas import EvalRecord


def compute_metrics(records: list[EvalRecord]) -> dict[str, Any]:
    total = len(records)
    if total == 0:
        return {
            "total": 0,
            "baseline_accuracy": None,
            "judge_pass_rate": None,
            "avg_answer_correctness": None,
            "avg_reasoning_faithfulness": None,
            "avg_reasoning_completeness": None,
            "by_type": {},
        }

    baseline_total = 0
    baseline_correct = 0
    judge_pass = 0
    sum_answer = 0.0
    sum_faithfulness = 0.0
    sum_completeness = 0.0
    per_type: dict[str, dict[str, float]] = defaultdict(lambda: {"total": 0, "baseline_total": 0, "baseline_correct": 0, "judge_pass": 0})

    for record in records:
        qtype = record.question_type or "unknown"
        per_type[qtype]["total"] += 1

        if record.baseline_match is not None:
            baseline_total += 1
            per_type[qtype]["baseline_total"] += 1
            if record.baseline_match:
                baseline_correct += 1
                per_type[qtype]["baseline_correct"] += 1

        computed_verdict = compute_verdict(
            answer_correctness=record.judge.answer_correctness,
            reasoning_faithfulness=record.judge.reasoning_faithfulness,
            reasoning_completeness=record.judge.reasoning_completeness,
        )
        if computed_verdict == "pass":
            judge_pass += 1
            per_type[qtype]["judge_pass"] += 1

        sum_answer += record.judge.answer_correctness
        sum_faithfulness += record.judge.reasoning_faithfulness
        sum_completeness += record.judge.reasoning_completeness

    by_type = {}
    for qtype, stats in per_type.items():
        b_total = int(stats["baseline_total"])
        by_type[qtype] = {
            "total": int(stats["total"]),
            "baseline_accuracy": (stats["baseline_correct"] / b_total) if b_total else None,
            "judge_pass_rate": stats["judge_pass"] / stats["total"] if stats["total"] else None,
        }

    return {
        "total": total,
        "baseline_accuracy": (baseline_correct / baseline_total) if baseline_total else None,
        "judge_pass_rate": judge_pass / total,
        "avg_answer_correctness": sum_answer / total,
        "avg_reasoning_faithfulness": sum_faithfulness / total,
        "avg_reasoning_completeness": sum_completeness / total,
        "by_type": by_type,
    }
