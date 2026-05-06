from __future__ import annotations

import base64
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from spatial_eval.generator import build_generation_output
from spatial_eval.judge import parse_judge_response
from spatial_eval.providers.base import GenerationProvider, JudgeProvider
from spatial_eval.providers.utils import load_dotenv
from spatial_eval.schemas import GenerationOutput, InputSample, JudgeOutput


@dataclass(slots=True)
class OllamaClient:
    model: str
    base_url: str = "http://localhost:11434"
    env_path: str | Path = ".env"
    timeout_s: int = 60
    max_retries: int = 1
    retry_wait_s: float = 2.0
    temperature: float = 0.0
    max_tokens: int = 1024
    keep_alive: str | None = None
    _client: Any = None

    def __post_init__(self) -> None:
        load_dotenv(self.env_path)
        env_base_url = os.getenv("OLLAMA_BASE_URL")
        if env_base_url and self.base_url == "http://localhost:11434":
            self.base_url = env_base_url
        self.base_url = self.base_url.rstrip("/")

    def complete(self, user_prompt: str, system_prompt: str | None = None, image_path: str | None = None) -> str:
        client = self._get_client()
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        user_message: dict[str, Any] = {"role": "user", "content": user_prompt}
        if image_path:
            user_message["images"] = [self._image_base64(image_path)]
        messages.append(user_message)

        options: dict[str, Any] = {
            "temperature": self.temperature,
            "num_predict": self.max_tokens,
        }
        chat_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": options,
        }
        if self.keep_alive:
            chat_kwargs["keep_alive"] = self.keep_alive

        attempt = 0
        while True:
            try:
                response = client.chat(**chat_kwargs)
                break
            except Exception as exc:
                if attempt < self.max_retries:
                    attempt += 1
                    time.sleep(max(0.0, self.retry_wait_s))
                    continue
                raise RuntimeError(f"Ollama chat request failed: {exc}") from exc

        content = self._extract_content(response).strip()
        if not content:
            raise RuntimeError(f"Unexpected Ollama response: {response}")
        return content

    @staticmethod
    def _image_base64(image_path: str) -> str:
        path = Path(image_path)
        if not path.exists():
            raise RuntimeError(f"Image path not found: {image_path}")
        return base64.b64encode(path.read_bytes()).decode("utf-8")

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from ollama import Client  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "Package 'ollama' is required for Ollama provider. Install with: pip install ollama"
            ) from exc
        self._client = Client(host=self.base_url, timeout=self.timeout_s)
        return self._client

    @staticmethod
    def _extract_content(response: Any) -> str:
        # SDK resposta pode ser dict-like e/ou objeto com atributos.
        if isinstance(response, dict):
            message = response.get("message", {})
            return str(message.get("content", ""))

        message = getattr(response, "message", None)
        if message is None:
            return ""
        if isinstance(message, dict):
            return str(message.get("content", ""))
        return str(getattr(message, "content", ""))


class OllamaGenerationProvider(GenerationProvider):
    name = "ollama-generation-provider"

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        env_path: str | Path = ".env",
        timeout_s: int = 60,
        max_retries: int = 1,
        retry_wait_s: float = 2.0,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        keep_alive: str | None = None,
        system_prompt: str = "You are a precise spatial reasoning assistant.",
    ) -> None:
        self.client = OllamaClient(
            model=model,
            base_url=base_url,
            env_path=env_path,
            timeout_s=timeout_s,
            max_retries=max_retries,
            retry_wait_s=retry_wait_s,
            temperature=temperature,
            max_tokens=max_tokens,
            keep_alive=keep_alive,
        )
        self.name = f"ollama-generation-provider:{model}"
        self.system_prompt = system_prompt

    def generate(self, sample: InputSample, prompt: str) -> GenerationOutput:
        raw_text = self.client.complete(
            user_prompt=prompt,
            system_prompt=self.system_prompt,
            image_path=sample.image_path,
        )
        return build_generation_output(sample, provider_name=self.name, raw_text=raw_text)


class OllamaJudgeProvider(JudgeProvider):
    name = "ollama-judge-provider"

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        env_path: str | Path = ".env",
        timeout_s: int = 60,
        max_retries: int = 1,
        retry_wait_s: float = 2.0,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        keep_alive: str | None = None,
        system_prompt: str = "You are a strict evaluator of spatial reasoning responses.",
        judge_evidence: str = "img_graph",
    ) -> None:
        self.client = OllamaClient(
            model=model,
            base_url=base_url,
            env_path=env_path,
            timeout_s=timeout_s,
            max_retries=max_retries,
            retry_wait_s=retry_wait_s,
            temperature=temperature,
            max_tokens=max_tokens,
            keep_alive=keep_alive,
        )
        self.name = f"ollama-judge-provider:{model}"
        self.system_prompt = system_prompt
        self.judge_evidence = judge_evidence.strip().lower()

    def judge(self, sample: InputSample, generation: GenerationOutput, prompt: str) -> JudgeOutput:
        del generation
        include_image = self.judge_evidence in {"image", "both"}
        image_path = sample.image_path if include_image else None
        raw_text = self.client.complete(user_prompt=prompt, system_prompt=self.system_prompt, image_path=image_path)
        return parse_judge_response(raw_text=raw_text, provider_name=self.name)
