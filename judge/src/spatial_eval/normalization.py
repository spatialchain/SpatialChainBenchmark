from __future__ import annotations

import re


YES_ALIASES = {"yes", "y", "true", "correct"}
NO_ALIASES = {"no", "n", "false", "incorrect"}


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def strip_markdown(text: str) -> str:
    cleaned = text.replace("**", "").replace("__", "").replace("`", "")
    return normalize_whitespace(cleaned)


def normalize_yes_no(text: str) -> str | None:
    cleaned = strip_markdown(text).lower()
    if not cleaned:
        return None

    if "answer:" in cleaned:
        cleaned = cleaned.split("answer:", maxsplit=1)[-1].strip()

    cleaned = re.sub(r"^[^a-z0-9]+", "", cleaned)
    token = cleaned.split(" ", maxsplit=1)[0].strip(".,;:!?")
    if token in YES_ALIASES:
        return "yes"
    if token in NO_ALIASES:
        return "no"
    return None


def extract_choose_attr_options(question: str) -> tuple[str, str] | None:
    if not question:
        return None

    normalized_question = normalize_whitespace(question).lower()
    match = re.search(r"\b([a-z][a-z0-9_-]*)\s+or\s+([a-z][a-z0-9_-]*)\b\??$", normalized_question)
    if not match:
        match = re.search(r"\b([a-z][a-z0-9_-]*)\s+or\s+([a-z][a-z0-9_-]*)\b", normalized_question)
        if not match:
            return None

    return match.group(1), match.group(2)


def normalize_choose_attr(answer: str, question: str) -> str | None:
    options = extract_choose_attr_options(question)
    if options is None:
        return strip_markdown(answer).lower() or None

    cleaned = strip_markdown(answer).lower()
    if "answer:" in cleaned:
        cleaned = cleaned.split("answer:", maxsplit=1)[-1].strip()

    for option in options:
        if re.search(rf"\b{re.escape(option)}\b", cleaned):
            return option
    return cleaned.split(" ", maxsplit=1)[0] if cleaned else None


def normalize_answer(answer: str, question_type: str | None, question: str) -> str | None:
    if answer is None:
        return None

    qtype = (question_type or "").strip()
    if qtype == "chooseAttr":
        return normalize_choose_attr(answer, question)

    yn = normalize_yes_no(answer)
    if yn is not None:
        return yn

    cleaned = strip_markdown(answer).lower()
    if "answer:" in cleaned:
        cleaned = cleaned.split("answer:", maxsplit=1)[-1].strip()
    return cleaned or None
