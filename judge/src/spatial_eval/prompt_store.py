from __future__ import annotations

from pathlib import Path


PROMPT_DIR = Path(__file__).resolve().parent / "prompts"


def default_prompt_path(prompt_type: str) -> Path:
    normalized = prompt_type.strip().lower()
    if normalized == "generator":
        return PROMPT_DIR / "generator.md"
    if normalized == "generator_vision":
        return PROMPT_DIR / "generator_vision.md"
    if normalized == "judge":
        return PROMPT_DIR / "judge.md"
    raise ValueError(f"Unsupported prompt type: {prompt_type}")


def load_prompt_from_file(path: str | Path) -> str:
    content = Path(path).read_text(encoding="utf-8").strip()
    if not content:
        raise ValueError(f"Prompt file is empty: {path}")
    return content


def load_default_prompt(prompt_type: str) -> str:
    return load_prompt_from_file(default_prompt_path(prompt_type))
