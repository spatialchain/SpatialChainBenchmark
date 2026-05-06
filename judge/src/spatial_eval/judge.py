from __future__ import annotations

import json
import re

from spatial_eval.schemas import GenerationOutput, InputSample, JudgeOutput

_VALID_JUDGE_EVIDENCE = {"image", "img_graph", "both"}


def _normalize_judge_evidence(judge_evidence: str) -> str:
    normalized = (judge_evidence or "").strip().lower()
    if normalized not in _VALID_JUDGE_EVIDENCE:
        raise ValueError("judge_evidence must be one of: image, img_graph, both")
    return normalized


def build_judge_prompt(
    sample: InputSample,
    generation: GenerationOutput,
    prompt_template: str | None = None,
    score_justification: bool = True,
    judge_evidence: str = "img_graph",
) -> str:
    if not prompt_template:
        raise ValueError("Judge prompt template is required.")
    evidence_mode = _normalize_judge_evidence(judge_evidence)
    template = prompt_template
    if score_justification:
        template = template.replace(
            "{{JUSTIFICATION_SCHEMA_SUFFIX}}",
            ',\n  "justification": "short explanation"',
        ).replace(
            "{{JUSTIFICATION_CONSTRAINT_LINE}}",
            "- `justification` must be concise and specific.",
        )
    else:
        template = template.replace("{{JUSTIFICATION_SCHEMA_SUFFIX}}", "").replace(
            "{{JUSTIFICATION_CONSTRAINT_LINE}}", ""
        )
    evidence_instruction_by_mode = {
        "image": "- The judge will receive the same image used in generation.\n- Scene graph will not be provided.",
        "img_graph": "- The judge will receive only the image scene graph.\n- No image will be provided.",
        "both": "- The judge will receive the same image used in generation.\n- The scene graph will also be provided.",
    }
    template = template.replace("{{EVIDENCE_AVAILABILITY}}", evidence_instruction_by_mode[evidence_mode])
    graph_text = sample.scene_graph or sample.raw_record.get("scene_graph") or "N/A"
    reference_reasoning = str(sample.raw_record.get("thinking", "")).strip() or "N/A"
    scene_graph_line = ""
    if evidence_mode in {"img_graph", "both"}:
        scene_graph_line = f"SceneGraph: {graph_text}\n"
    
    return (
        f"{template}\n\n"
        f"Question: {sample.question}\n"
        f"GroundTruth: {sample.ground_truth or 'N/A'}\n"
        f"{scene_graph_line}"
        f"CandidateAnswer: {generation.final_answer}\n"
        f"CandidateReasoning: {generation.reasoning}\n"
        f"ReferenceReasoning: {reference_reasoning}\n"
        "Return only JSON."
    )


def _to_float(value: object, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, parsed))


def compute_verdict(
    answer_correctness: float,
    reasoning_faithfulness: float,
    reasoning_completeness: float,
) -> str:
    avg_reasoning = (reasoning_faithfulness + reasoning_completeness) / 2.0
    return "pass" if answer_correctness >= 0.5 and avg_reasoning >= 0.5 else "fail"


def parse_judge_response(raw_text: str, provider_name: str = "unknown") -> JudgeOutput:
    text = (raw_text or "").strip()
    if not text:
        answer_correctness = 0.0
        reasoning_faithfulness = 0.0
        reasoning_completeness = 0.0
        return JudgeOutput(
            answer_correctness=answer_correctness,
            reasoning_faithfulness=reasoning_faithfulness,
            reasoning_completeness=reasoning_completeness,
            verdict=compute_verdict(answer_correctness, reasoning_faithfulness, reasoning_completeness),
            justification="Empty judge response.",
            raw_response=raw_text,
            provider_name=provider_name,
        )

    try:
        parsed = json.loads(text)
        answer_correctness = _to_float(parsed.get("answer_correctness"))
        reasoning_faithfulness = _to_float(parsed.get("reasoning_faithfulness"))
        reasoning_completeness = _to_float(parsed.get("reasoning_completeness"))
        return JudgeOutput(
            answer_correctness=answer_correctness,
            reasoning_faithfulness=reasoning_faithfulness,
            reasoning_completeness=reasoning_completeness,
            verdict=compute_verdict(answer_correctness, reasoning_faithfulness, reasoning_completeness),
            justification=str(parsed.get("justification", "")).strip() or "No justification.",
            raw_response=raw_text,
            provider_name=provider_name,
        )
    except json.JSONDecodeError:
        pass

    # Fallback flexível para resposta sem JSON.
    verdict_match = re.search(r"(?i)\b(pass|fail)\b", text)
    verdict = verdict_match.group(1).lower() if verdict_match else None
    score_matches = re.findall(r"(?:score|correctness|faithfulness|completeness)\s*[:=]\s*([0-1](?:\.\d+)?)", text, flags=re.I)
    score = _to_float(score_matches[0]) if score_matches else (1.0 if verdict == "pass" else 0.0)
    answer_correctness = score
    reasoning_faithfulness = score
    reasoning_completeness = score

    return JudgeOutput(
        answer_correctness=answer_correctness,
        reasoning_faithfulness=reasoning_faithfulness,
        reasoning_completeness=reasoning_completeness,
        verdict=compute_verdict(answer_correctness, reasoning_faithfulness, reasoning_completeness),
        justification=text[:500],
        raw_response=raw_text,
        provider_name=provider_name,
    )
