from __future__ import annotations

import base64
import mimetypes
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from spatial_eval.generator import build_generation_output
from spatial_eval.judge import parse_judge_response
from spatial_eval.providers.base import GenerationProvider, JudgeProvider
from spatial_eval.providers.utils import load_dotenv, post_json
from spatial_eval.schemas import GenerationOutput, InputSample, JudgeOutput


def _load_dotenv(env_path: str | Path = ".env") -> None:
    # Compatibilidade retroativa para testes/imports existentes.
    load_dotenv(env_path)


@dataclass(slots=True)
class LLMClient:
    backend: str
    model: str
    env_path: str | Path = ".env"
    timeout_s: int = 60
    max_retries: int = 1
    retry_wait_s: float = 2.0
    temperature: float = 0.0
    max_tokens: int = 1024

    def __post_init__(self) -> None:
        self.backend = self.backend.strip().lower()
        if self.backend not in {"openai", "gemini", "anthropic"}:
            raise ValueError("backend must be one of: openai, gemini, anthropic")
        load_dotenv(self.env_path)

    def complete(self, user_prompt: str, system_prompt: str | None = None, image_path: str | None = None) -> str:
        if self.backend == "openai":
            return self._complete_openai(user_prompt, system_prompt, image_path)
        if self.backend == "gemini":
            return self._complete_gemini(user_prompt, system_prompt, image_path)
        return self._complete_anthropic(user_prompt, system_prompt, image_path)

    def _complete_openai(self, user_prompt: str, system_prompt: str | None, image_path: str | None) -> str:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not found. Configure it in .env or environment.")

        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if image_path:
            image_part = self._openai_image_part(image_path)
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        image_part,
                    ],
                }
            )
        else:
            messages.append({"role": "user", "content": user_prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        data = post_json(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            payload=payload,
            timeout_s=self.timeout_s,
            max_retries=self.max_retries,
            retry_wait_s=self.retry_wait_s,
        )
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(f"Unexpected OpenAI response: {data}")
        message = choices[0].get("message", {})
        return str(message.get("content", "")).strip()

    def _complete_gemini(self, user_prompt: str, system_prompt: str | None, image_path: str | None) -> str:
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY/GOOGLE_API_KEY not found. Configure it in .env or environment.")

        parts: list[dict[str, Any]] = [{"text": user_prompt}]
        if image_path:
            inline_data = self._gemini_image_part(image_path)
            parts.append(inline_data)

        payload: dict[str, Any] = {
            "contents": [{"parts": parts}],
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": self.max_tokens,
            },
        }
        if system_prompt:
            payload["system_instruction"] = {"parts": [{"text": system_prompt}]}

        headers = {"Content-Type": "application/json"}
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={api_key}"
        data = post_json(
            url,
            headers=headers,
            payload=payload,
            timeout_s=self.timeout_s,
            max_retries=self.max_retries,
            retry_wait_s=self.retry_wait_s,
        )

        candidates = data.get("candidates") or []
        if not candidates:
            raise RuntimeError(f"Unexpected Gemini response: {data}")
        parts = candidates[0].get("content", {}).get("parts", [])
        text = "".join(str(part.get("text", "")) for part in parts)
        return text.strip()

    def _complete_anthropic(self, user_prompt: str, system_prompt: str | None, image_path: str | None) -> str:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not found. Configure it in .env or environment.")

        content: list[dict[str, Any]] = [{"type": "text", "text": user_prompt}]
        if image_path:
            content.insert(0, self._anthropic_image_part(image_path))

        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": [{"role": "user", "content": content}],
        }
        if system_prompt:
            payload["system"] = system_prompt

        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        data = post_json(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            payload=payload,
            timeout_s=self.timeout_s,
            max_retries=self.max_retries,
            retry_wait_s=self.retry_wait_s,
        )
        content = data.get("content") or []
        text_parts = [str(chunk.get("text", "")) for chunk in content if isinstance(chunk, dict)]
        text = "".join(text_parts).strip()
        if not text:
            raise RuntimeError(f"Unexpected Anthropic response: {data}")
        return text

    @staticmethod
    def _image_bytes_and_mime(image_path: str) -> tuple[bytes, str]:
        path = Path(image_path)
        if not path.exists():
            raise RuntimeError(f"Image path not found: {image_path}")
        mime, _ = mimetypes.guess_type(path.name)
        if not mime:
            mime = "image/png"
        return path.read_bytes(), mime

    def _openai_image_part(self, image_path: str) -> dict[str, Any]:
        image_bytes, mime = self._image_bytes_and_mime(image_path)
        encoded = base64.b64encode(image_bytes).decode("utf-8")
        return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}}

    def _gemini_image_part(self, image_path: str) -> dict[str, Any]:
        image_bytes, mime = self._image_bytes_and_mime(image_path)
        encoded = base64.b64encode(image_bytes).decode("utf-8")
        return {"inline_data": {"mime_type": mime, "data": encoded}}

    def _anthropic_image_part(self, image_path: str) -> dict[str, Any]:
        image_bytes, mime = self._image_bytes_and_mime(image_path)
        encoded = base64.b64encode(image_bytes).decode("utf-8")
        return {"type": "image", "source": {"type": "base64", "media_type": mime, "data": encoded}}


class LLMGenerationProvider(GenerationProvider):
    name = "llm-generation-provider"

    def __init__(
        self,
        backend: str,
        model: str,
        env_path: str | Path = ".env",
        timeout_s: int = 60,
        max_retries: int = 1,
        retry_wait_s: float = 2.0,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        system_prompt: str = "You are a precise spatial reasoning assistant.",
    ) -> None:
        self.client = LLMClient(
            backend=backend,
            model=model,
            env_path=env_path,
            timeout_s=timeout_s,
            max_retries=max_retries,
            retry_wait_s=retry_wait_s,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self.name = f"llm-generation-provider:{backend}:{model}"
        self.system_prompt = system_prompt

    def generate(self, sample: InputSample, prompt: str) -> GenerationOutput:
        raw_text = self.client.complete(
            user_prompt=prompt,
            system_prompt=self.system_prompt,
            image_path=sample.image_path,
        )
        return build_generation_output(sample, provider_name=self.name, raw_text=raw_text)


class LLMJudgeProvider(JudgeProvider):
    name = "llm-judge-provider"

    def __init__(
        self,
        backend: str,
        model: str,
        env_path: str | Path = ".env",
        timeout_s: int = 60,
        max_retries: int = 1,
        retry_wait_s: float = 2.0,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        system_prompt: str = "You are a strict evaluator of spatial reasoning responses.",
        judge_evidence: str = "img_graph",
    ) -> None:
        self.client = LLMClient(
            backend=backend,
            model=model,
            env_path=env_path,
            timeout_s=timeout_s,
            max_retries=max_retries,
            retry_wait_s=retry_wait_s,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self.name = f"llm-judge-provider:{backend}:{model}"
        self.system_prompt = system_prompt
        self.judge_evidence = judge_evidence.strip().lower()

    def judge(self, sample: InputSample, generation: GenerationOutput, prompt: str) -> JudgeOutput:
        del generation
        include_image = self.judge_evidence in {"image", "both"}
        image_path = sample.image_path if include_image else None
        raw_text = self.client.complete(user_prompt=prompt, system_prompt=self.system_prompt, image_path=image_path)
        return parse_judge_response(raw_text=raw_text, provider_name=self.name)
