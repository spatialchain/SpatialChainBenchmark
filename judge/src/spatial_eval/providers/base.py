from __future__ import annotations

from abc import ABC, abstractmethod

from spatial_eval.schemas import GenerationOutput, InputSample, JudgeOutput


class GenerationProvider(ABC):
    name: str = "base-generator"

    @abstractmethod
    def generate(self, sample: InputSample, prompt: str) -> GenerationOutput:
        raise NotImplementedError


class JudgeProvider(ABC):
    name: str = "base-judge"

    @abstractmethod
    def judge(
        self,
        sample: InputSample,
        generation: GenerationOutput,
        prompt: str,
    ) -> JudgeOutput:
        raise NotImplementedError
