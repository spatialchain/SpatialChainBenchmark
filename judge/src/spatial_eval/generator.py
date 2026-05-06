from __future__ import annotations

import json
import re

from spatial_eval.schemas import GenerationOutput, InputSample


def build_generator_prompt(sample: InputSample, prompt_template: str | None = None) -> str:
    if not prompt_template:
        raise ValueError("Generator prompt template is required.")
    template = prompt_template
    graph_text = sample.scene_graph or sample.raw_record.get("scene_graph") or "N/A"
    return (
        f"{template}\n\n"
        f"Question: {sample.question}\n"
        f"SceneGraph: {graph_text}\n"
        "Return only JSON."
    )


def build_image_question_prompt(question: str, prompt_template: str | None = None) -> str:
    if not prompt_template:
        raise ValueError("Generator prompt template is required.")
    return (
        f"{prompt_template}\n\n"
        f"Question: {question}\n"
        "Return only JSON."
    )


def parse_generation_response(raw_text: str) -> tuple[str, str]:
    text = (raw_text or "").strip()
    if not text:
        return "", ""

    # 1) Caminho ideal: JSON estruturado.
    try:
        parsed = json.loads(text)
        final_answer = str(parsed.get("final_answer", "")).strip()
        reasoning = str(parsed.get("reasoning", "")).strip()
        if final_answer or reasoning:
            return final_answer, reasoning
    except json.JSONDecodeError:
        pass

    # 2) Fallback: extrair linha com ANSWER.
    answer_match = re.search(r"(?im)^\s*(?:final_answer|answer)\s*:\s*(.+?)\s*$", text)
    if answer_match:
        final_answer = answer_match.group(1).strip()
    else:
        # Último fallback: primeira linha não-vazia.
        final_answer = text.splitlines()[0].strip()

    # 3) Reasoning: todo texto exceto a linha de answer, se existir.
    reasoning = re.sub(r"(?im)^\s*(?:final_answer|answer)\s*:\s*.+?$", "", text).strip()
    if not reasoning:
        reasoning = text
    return final_answer, reasoning


def build_generation_output(
    sample: InputSample,
    provider_name: str,
    raw_text: str
) -> GenerationOutput:
    final_answer, reasoning = parse_generation_response(raw_text)
    return GenerationOutput(
        final_answer=final_answer,
        reasoning=reasoning,
        raw_response=raw_text,
        provider_name=provider_name
    )
