from __future__ import annotations

from spatial_eval.generator import build_generation_output
from spatial_eval.normalization import normalize_answer
from spatial_eval.providers.base import GenerationProvider, JudgeProvider
from spatial_eval.schemas import GenerationOutput, InputSample, JudgeOutput


class ReplayGenerationProvider(GenerationProvider):
    """
    Provider offline para MVP:
    reutiliza campos existentes do dataset (`model_answer`, `thinking`, `response`)
    como se fossem a saida de um modelo gerador.
    """

    name = "replay-generator"

    def generate(self, sample: InputSample, prompt: str) -> GenerationOutput:
        del prompt  # Mantem assinatura unificada para providers futuros.
        raw_answer = str(sample.raw_record.get("model_answer", "")).strip()
        raw_reasoning = str(sample.raw_record.get("thinking", "")).strip()
        raw_response = raw_answer
        if raw_reasoning:
            raw_response = f"final_answer: {raw_answer}\nreasoning: {raw_reasoning}"
        return build_generation_output(sample, self.name, raw_response)


class RuleBasedJudgeProvider(JudgeProvider):
    """
    Juiz deterministico para validacao offline.
    Serve como baseline enquanto o juiz LLM real nao estiver plugado.
    """

    name = "rule-based-judge"

    def judge(self, sample: InputSample, generation: GenerationOutput, prompt: str) -> JudgeOutput:
        del prompt
        pred = normalize_answer(generation.final_answer, sample.question_type, sample.question)
        gt = normalize_answer(sample.ground_truth or "", sample.question_type, sample.question)
        answer_correct = 1.0 if (pred is not None and gt is not None and pred == gt) else 0.0

        reasoning = (generation.reasoning or "").strip()
        reasoning_completeness = 1.0 if len(reasoning) >= 40 else 0.5 if reasoning else 0.0

        # Faithfulness simplificada: se menciona ao menos um conector espacial comum.
        lower_reasoning = reasoning.lower()
        has_spatial_signal = any(
            token in lower_reasoning
            for token in ("left", "right", "above", "below", "on", "under", "near", "inside")
        )
        reasoning_faithfulness = 1.0 if has_spatial_signal else 0.5 if reasoning else 0.0

        avg_reasoning = (reasoning_faithfulness + reasoning_completeness) / 2.0
        verdict = "pass" if answer_correct >= 0.5 and avg_reasoning >= 0.5 else "fail"
        justification = (
            f"pred={pred!r}, gt={gt!r}, answer_correctness={answer_correct:.1f}, "
            f"faithfulness={reasoning_faithfulness:.1f}, completeness={reasoning_completeness:.1f}"
        )
        return JudgeOutput(
            answer_correctness=answer_correct,
            reasoning_faithfulness=reasoning_faithfulness,
            reasoning_completeness=reasoning_completeness,
            verdict=verdict,
            justification=justification,
            raw_response=justification,
            provider_name=self.name,
        )
