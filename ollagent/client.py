"""Minimal Ollama HTTP client built on the standard library.

We deliberately avoid `requests` / `httpx` so Ollagent installs and runs with
zero third-party dependencies. Only `urllib` and `json` are used.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, List, Mapping, Optional

from .config import Config


class OllamaError(Exception):
    """Raised when the OpenAI-compatible endpoint is missing or malformed."""


class _Response:
    def __init__(self, body: bytes):
        self._body = body

    def json(self) -> Any:
        return json.loads(self._body.decode("utf-8"))


def _post(url: str, payload: Mapping[str, Any], timeout: int = 600) -> _Response:
    """POST JSON to Ollama. Raises OllamaError with a useful message on failure."""
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return _Response(resp.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise OllamaError(f"HTTP {exc.code} from {url}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise OllamaError(
            f"Could not reach Ollama at {url}. Is it running? "
            f"Start it with `ollama start` (or `ollama serve`). "
            f"Underlying error: {exc.reason}"
        ) from exc
    except TimeoutError as exc:
        raise OllamaError(f"Request to {url} timed out.") from exc


class Ollama:
    """Thin wrapper around the Ollama `/api/chat` endpoint."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.base = cfg.ollama_url.rstrip("/")

    # -- server status -------------------------------------------------
    def ping(self) -> bool:
        """Cheap connectivity check. Uses /api/tags (fast, no model load)."""
        try:
            with urllib.request.urlopen(f"{self.base}/api/tags", timeout=10) as resp:
                resp.read()
            return True
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
            return False

    def list_models(self) -> List[Mapping[str, Any]]:
        try:
            with urllib.request.urlopen(f"{self.base}/api/tags", timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data.get("models", [])
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
            return []

    # -- model interaction ---------------------------------------------
    def capabilities(self) -> List[str]:
        """Ask /api/show what the model supports (e.g. ['tools', 'thinking'])."""
        try:
            req = urllib.request.Request(
                f"{self.base}/api/show",
                data=json.dumps({"model": self.cfg.model}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                return list(json.loads(resp.read().decode("utf-8"))
                            .get("capabilities", []))
        except (urllib.error.URLError, urllib.error.HTTPError,
                TimeoutError, json.JSONDecodeError):
            return []

    def chat(
        self,
        messages: List[Mapping[str, Any]],
        *,
        temperature: float,
        num_ctx: int,
        format_json: bool = True,
        tools: Optional[List[Mapping[str, Any]]] = None,
    ) -> tuple[str, List[Mapping[str, Any]], str]:
        """Send a chat request.

        Returns:
            (content, tool_calls, done_reason) — the assistant's text reply,
            any native tool calls the model made, and why generation stopped
            ("stop", "length", ...). `content` may be empty when the stop
            reason is "length".
        """
        payload: dict[str, Any] = {
            "model": self.cfg.model,
            "messages": messages,
            "stream": False,
            "think": self.cfg.think,
            "options": {
                "temperature": temperature,
                "num_ctx": num_ctx,
            },
        }
        if tools:
            # Native tool calling: Ollama enforces the schema and returns
            # structured message.tool_calls. Mutually exclusive with the
            # JSON-format prompt hack.
            payload["tools"] = tools
        elif format_json:
            # Tells Ollama the response must be valid JSON.
            payload["format"] = "json"
        resp = _post(f"{self.base}/api/chat", payload)
        data = resp.json()
        msg = data.get("message") or {}
        tool_calls = msg.get("tool_calls") or []
        return (
            str(msg.get("content") or ""),
            [tc for tc in tool_calls if isinstance(tc, dict)],
            str(data.get("done_reason") or ""),
        )