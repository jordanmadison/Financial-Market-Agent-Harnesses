from __future__ import annotations

import hashlib
import http.client
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from typing import Literal

from .completion import CompletionClient, OpenRouterResponseError, complete_with_retries, response_usage
from .domain import InferenceAttempt, InferenceRecord, PlacementAction

MODEL_ID = "deepseek/deepseek-v4-flash-0731"


@dataclass(frozen=True)
class AgentConfig:
    provider: Literal["openrouter"] = "openrouter"
    model_id: Literal["deepseek/deepseek-v4-flash-0731"] = MODEL_ID
    upstream_provider: str = ""
    allow_fallbacks: Literal[False] = False
    temperature: float = 0.0
    top_p: float = 1.0
    max_output_tokens: int | None = 16_384
    reasoning_effort: Literal["medium"] = "medium"
    exclude_reasoning: Literal[True] = True
    seed: int | None = None
    timeout_seconds: int = 180
    max_retries: int = 2

    def __post_init__(self) -> None:
        if self.model_id != MODEL_ID or self.allow_fallbacks is not False:
            raise ValueError("model and fallback policy are fixed")

    @property
    def config_hash(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()


class OpenRouterClient:
    endpoint = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self):
        self._api_key = os.environ.get("OPENROUTER_API_KEY")
        if not self._api_key:
            raise ValueError("OPENROUTER_API_KEY is required")

    def complete(self, payload: dict, timeout: int) -> dict:
        started = time.monotonic()
        request = urllib.request.Request(
            self.endpoint,
            json.dumps(payload).encode(),
            {
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Connection": "close",
                "User-Agent": "order-book-harness/0.1",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return _read_json_with_deadline(response, timeout, started)
        except urllib.error.HTTPError as exc:
            try:
                body = _read_json_with_deadline(exc, timeout, started)
            except (TimeoutError, ValueError, json.JSONDecodeError, UnicodeDecodeError, OSError):
                body = {"error": {"message": str(exc)}}
            raise OpenRouterResponseError(body, status=exc.code) from exc
        except TimeoutError:
            raise
        except (http.client.IncompleteRead, http.client.RemoteDisconnected, ConnectionError, OSError) as exc:
            raise urllib.error.URLError(f"incomplete OpenRouter response: {type(exc).__name__}") from exc


def _read_json_with_deadline(response, timeout: int, started: float) -> dict:
    """Read a chunked response under one wall-clock deadline.

    ``urlopen(timeout=...)`` applies the timeout to individual socket operations;
    provider keepalive chunks can otherwise keep a response alive indefinitely.
    """
    deadline = started + timeout
    body = bytearray()
    raw = getattr(getattr(response, "fp", None), "raw", None)
    sock = getattr(raw, "_sock", None)
    read = response.read1 if hasattr(response, "read1") else response.read
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"OpenRouter response exceeded {timeout}s total deadline")
        if sock is not None:
            sock.settimeout(max(0.001, remaining))
        chunk = read(65_536)
        if not chunk:
            break
        body.extend(chunk)
        if len(body) > 10_000_000:
            raise ValueError("OpenRouter response exceeded 10 MB limit")
    try:
        value = json.loads(body)
    except json.JSONDecodeError as exc:
        raise urllib.error.URLError("OpenRouter returned an invalid JSON envelope") from exc
    if not isinstance(value, dict):
        raise urllib.error.URLError("OpenRouter returned a non-object JSON envelope")
    return value


def parse_placement_action(text: str | None) -> PlacementAction:
    if not text:
        raise ValueError("empty response")
    value = json.loads(text)
    if not isinstance(value, dict) or type(value.get("offset_ticks")) is not int:
        raise ValueError("response must contain integer offset_ticks")
    return PlacementAction(value["offset_ticks"])


class FixedAgent:
    def __init__(self, config: AgentConfig, client: CompletionClient):
        self.config = config
        self.client = client

    def act(self, messages: list[dict], harness_id: str) -> tuple[PlacementAction | None, InferenceRecord]:
        payload = {
            "model": self.config.model_id,
            "messages": messages,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "stream": False,
            "reasoning": {"effort": self.config.reasoning_effort, "exclude": self.config.exclude_reasoning},
            "provider": {"only": [self.config.upstream_provider], "allow_fallbacks": False},
        }
        if self.config.max_output_tokens is not None:
            payload["max_tokens"] = self.config.max_output_tokens
        if self.config.seed is not None:
            payload["seed"] = self.config.seed
        prompt_hash = hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest()

        def parse_response(response: dict) -> PlacementAction:
            raw = response["choices"][0]["message"].get("content")
            return parse_placement_action(raw)

        run = complete_with_retries(
            self.client,
            payload,
            self.config.timeout_seconds,
            self.config.max_retries,
            parse_response,
            _attempt,
        )
        usage = response_usage(run.response)
        reasoning_tokens = _reasoning_tokens(usage)
        api_error = next((attempt.api_error for attempt in reversed(run.attempts) if attempt.api_error is not None), None) if run.parsed is None else None
        record = InferenceRecord(
            run.response.get("id") if run.response else None,
            self.config.model_id,
            run.response.get("model") if run.response else None,
            run.response.get("provider") if run.response else None,
            self.config.config_hash,
            harness_id,
            prompt_hash,
            run.raw,
            run.parsed,
            usage.get("prompt_tokens"),
            usage.get("completion_tokens"),
            run.latency_ms,
            run.retry_count,
            run.error,
            reasoning_tokens,
            run.choice.get("finish_reason"),
            len(run.raw or ""),
            tuple(run.attempts),
            api_error,
        )
        return run.parsed, record


def _reasoning_tokens(usage: dict) -> int | None:
    details = usage.get("completion_tokens_details") or usage.get("completionTokensDetails") or {}
    return details.get("reasoning_tokens", details.get("reasoningTokens"))


def _attempt(index: int, response: dict | None, started: float, error: str | None, api_error: dict | None = None) -> InferenceAttempt:
    usage = response_usage(response)
    raw = None
    finish_reason = None
    if response:
        choice = response.get("choices", [{}])[0]
        raw = choice.get("message", {}).get("content")
        finish_reason = choice.get("finish_reason")
    return InferenceAttempt(
        index,
        response.get("id") if response else None,
        response.get("model") if response else None,
        response.get("provider") if response else None,
        usage.get("prompt_tokens"),
        usage.get("completion_tokens"),
        _reasoning_tokens(usage),
        finish_reason,
        len(raw or ""),
        round((time.monotonic() - started) * 1000),
        error,
        api_error,
    )


class HeuristicAgent:
    """Deterministic offline agent for tests, baselines, and API-free experiments."""

    config_hash = hashlib.sha256(b"offline-heuristic-v1").hexdigest()

    def act(self, messages: list[dict], harness_id: str) -> tuple[PlacementAction, InferenceRecord]:
        start = time.monotonic()
        context = json.loads(messages[-1]["content"])
        features = context.get("features", {})
        imbalance = features.get("multilevel_imbalance", features.get("level1_imbalance", 0))
        side = context["parent_order"]["side"]
        pressure = imbalance if side == "BUY" else -imbalance
        offset = 0 if pressure > 0.2 else (2 if pressure < -0.2 else 1)
        allowed = context["allowed_offsets"]
        if offset not in allowed:
            offset = min(allowed, key=lambda x: abs(x - offset))
        action = PlacementAction(offset)
        raw = json.dumps({"offset_ticks": offset})
        record = InferenceRecord(
            None,
            "offline-heuristic-v1",
            "offline-heuristic-v1",
            "local",
            self.config_hash,
            harness_id,
            hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest(),
            raw,
            action,
            None,
            None,
            round((time.monotonic() - start) * 1000),
            0,
            None,
        )
        return action, record
