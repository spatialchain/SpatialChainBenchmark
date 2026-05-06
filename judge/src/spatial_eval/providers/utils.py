from __future__ import annotations

import json
import os
import socket
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def load_dotenv(env_path: str | Path = ".env") -> None:
    """
    Loader simples de .env para evitar dependencias externas.
    Apenas define variaveis que ainda nao existem no ambiente.
    """
    path = Path(env_path)
    if not path.exists():
        return

    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", maxsplit=1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def post_json(
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    timeout_s: int = 60,
    max_retries: int = 1,
    retry_wait_s: float = 2.0,
) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = Request(url=url, data=data, headers=headers, method="POST")
    attempt = 0
    while True:
        try:
            with urlopen(req, timeout=timeout_s) as resp:
                body = resp.read().decode("utf-8")
            break
        except HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="ignore")
            is_retryable = exc.code in {408, 409, 425, 429, 500, 502, 503, 504}
            if is_retryable and attempt < max_retries:
                attempt += 1
                time.sleep(max(0.0, retry_wait_s))
                continue
            raise RuntimeError(f"HTTP {exc.code} calling LLM API: {error_body}") from exc
        except (URLError, TimeoutError, socket.timeout) as exc:
            if attempt < max_retries:
                attempt += 1
                time.sleep(max(0.0, retry_wait_s))
                continue
            raise RuntimeError(f"Network/timeout error calling LLM API: {exc}") from exc

    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON response from LLM API: {body[:500]}") from exc
