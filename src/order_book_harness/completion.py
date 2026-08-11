from __future__ import annotations

import json
import time
import urllib.error
from dataclasses import dataclass
from typing import Any, Callable, Protocol, TypeVar

T = TypeVar("T")


class CompletionClient(Protocol):
    def complete(self, payload: dict, timeout: int) -> dict: ...


@dataclass(frozen=True)
class CompletionRun:
    parsed: T | None
    raw: str | None
    response: dict | None
    attempts: tuple[Any, ...]
    error: str | None
    latency_ms: int
    retry_count: int
    usage: dict[str, Any]
    choice: dict[str, Any]


class OpenRouterResponseError(Exception):
    def __init__(self, response: dict, status: int | None = None, missing_choices: bool = False):
        self.api_error = _safe_api_error(response, status, missing_choices)
        self.retryable = _retryable_api_error(self.api_error)
        super().__init__(self.api_error["message"])


def complete_with_retries(
    client: CompletionClient,
    payload: dict,
    timeout_seconds: int,
    max_retries: int,
    parse_response: Callable[[dict], T],
    build_attempt: Callable[[int, dict | None, float, str | None, dict | None], Any],
) -> CompletionRun[T]:
    started = time.monotonic()
    attempts: list[Any] = []
    response: dict | None = None
    parsed: T | None = None
    raw: str | None = None
    error: str | None = None
    for index in range(max_retries + 1):
        attempt_start = time.monotonic()
        try:
            response = client.complete(payload, timeout_seconds)
            if "error" in response or not isinstance(response.get("choices"), list) or not response["choices"]:
                raise OpenRouterResponseError(response, missing_choices=True)
            raw = response["choices"][0]["message"].get("content")
            parsed = parse_response(response)
            attempts.append(build_attempt(index, response, attempt_start, None, None))
            error = None
            break
        except OpenRouterResponseError as exc:
            error = f"OpenRouterResponseError: {exc}"
            attempts.append(build_attempt(index, response, attempt_start, error, exc.api_error))
            if not exc.retryable or index == max_retries:
                break
            response = None
        except (TimeoutError, urllib.error.URLError) as exc:
            error = f"{type(exc).__name__}: {exc}"
            attempts.append(build_attempt(index, None, attempt_start, error, None))
            if not _is_transient(exc) or index == max_retries:
                break
        except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
            error = f"{type(exc).__name__}: {exc}"
            attempts.append(build_attempt(index, response, attempt_start, error, None))
            parsed = None
            break
    usage = response.get("usage", {}) if response else {}
    choices = response.get("choices") if response else None
    choice = choices[0] if isinstance(choices, list) and choices else {}
    return CompletionRun(
        parsed,
        raw,
        response,
        tuple(attempts),
        error,
        round((time.monotonic() - started) * 1000),
        max(0, len(attempts) - 1),
        usage,
        choice,
    )


def response_usage(response: dict | None) -> dict[str, Any]:
    return response.get("usage", {}) if response else {}


def _is_transient(error: BaseException) -> bool:
    if isinstance(error, urllib.error.HTTPError):
        return error.code in (408, 409, 425, 429) or error.code >= 500
    return isinstance(error, (TimeoutError, urllib.error.URLError))


def _safe_api_error(response: dict, status: int | None, missing_choices: bool) -> dict:
    error = response.get("error") if isinstance(response, dict) else None
    error = error if isinstance(error, dict) else {}
    message = error.get("message")
    if not isinstance(message, str):
        message = "OpenRouter response did not contain choices" if missing_choices else "OpenRouter API error"
    return {
        "status": status,
        "code": error.get("code"),
        "type": error.get("type"),
        "message": message[:2000],
        "missing_choices": missing_choices,
        "response_keys": sorted(str(key) for key in response.keys()) if isinstance(response, dict) else [],
    }


def _retryable_api_error(error: dict) -> bool:
    values = (error.get("status"), error.get("code"))
    for value in values:
        try:
            code = int(value)
        except (TypeError, ValueError):
            continue
        if 400 <= code < 500:
            return code in (408, 409, 425, 429)
        if code >= 500:
            return True
    error_type = str(error.get("type") or "").lower()
    if any(marker in error_type for marker in ("invalid", "auth", "permission", "billing")):
        return False
    if error.get("missing_choices"):
        return True
    return any(marker in error_type for marker in ("rate_limit", "timeout", "provider", "server"))
