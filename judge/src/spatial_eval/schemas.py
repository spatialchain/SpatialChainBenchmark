from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class InputSample:
    row_idx: int | None
    custom_id: str
    question: str
    ground_truth: str | None
    question_type: str | None
    scene_graph: str | None = None
    image_path: str | None = None
    raw_record: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class GenerationOutput:
    final_answer: str
    reasoning: str
    raw_response: str
    provider_name: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class JudgeOutput:
    answer_correctness: float
    reasoning_faithfulness: float
    reasoning_completeness: float
    verdict: str
    justification: str
    raw_response: str
    provider_name: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class EvalRecord:
    custom_id: str
    row_idx: int | None
    question_type: str | None
    question: str
    ground_truth: str | None
    reference_thinking: str | None
    generation: GenerationOutput
    judge: JudgeOutput
    normalized_prediction: str | None
    normalized_ground_truth: str | None
    baseline_match: bool | None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["generation"] = self.generation.to_dict()
        data["judge"] = self.judge.to_dict()
        return data
