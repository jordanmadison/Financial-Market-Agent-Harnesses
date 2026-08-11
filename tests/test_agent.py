import unittest
import os
import time
from unittest.mock import patch
import urllib.error

from order_book_harness.agent import AgentConfig, FixedAgent, OpenRouterClient, _read_json_with_deadline
from order_book_harness.domain import PlacementAction


class FakeClient:
    def __init__(self, results):
        self.results = list(results)
        self.payloads = []

    def complete(self, payload, timeout):
        self.payloads.append(payload)
        result = self.results.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result


def response(content='{"offset_ticks": 2}', reasoning_tokens=700):
    return {
        "id": "request-1",
        "model": "deepseek/deepseek-v4-flash-0731",
        "provider": "DeepInfra",
        "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 20, "completion_tokens": 712, "completion_tokens_details": {"reasoning_tokens": reasoning_tokens}},
    }


class FixedAgentTests(unittest.TestCase):
    def test_chunked_response_obeys_total_deadline(self):
        class NeverRead:
            fp = None

            def read1(self, size):
                raise AssertionError("expired response must not be read")

        with self.assertRaises(TimeoutError):
            _read_json_with_deadline(NeverRead(), 1, time.monotonic() - 2)

    def test_invalid_http_json_envelope_is_retryable_transport_error(self):
        class InvalidEnvelope:
            fp = None

            def __init__(self):
                self.done = False

            def read1(self, size):
                if self.done:
                    return b""
                self.done = True
                return b"not-json"

        with self.assertRaises(urllib.error.URLError):
            _read_json_with_deadline(InvalidEnvelope(), 1, time.monotonic())

    def test_openrouter_client_requires_environment_credential(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "OPENROUTER_API_KEY"):
                OpenRouterClient()
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-placeholder"}, clear=True):
            client = OpenRouterClient()
            self.assertFalse(hasattr(client, "api_key"))

    messages = [{"role": "user", "content": "choose"}]

    def test_capped_explicit_reasoning_and_telemetry(self):
        client = FakeClient([response()])
        action, record = FixedAgent(AgentConfig(upstream_provider="DeepInfra"), client).act(self.messages, "h")
        self.assertEqual(action, PlacementAction(2))
        self.assertEqual(client.payloads[0]["max_tokens"], 16_384)
        self.assertEqual(client.payloads[0]["reasoning"], {"effort": "medium", "exclude": True})
        self.assertEqual(AgentConfig().timeout_seconds, 180)
        self.assertEqual(record.reasoning_tokens, 700)
        self.assertEqual(record.finish_reason, "stop")
        self.assertEqual(record.response_length_chars, len('{"offset_ticks": 2}'))
        self.assertEqual(len(record.attempts), 1)

    def test_parse_failure_is_not_retried(self):
        client = FakeClient([response(None), response()])
        action, record = FixedAgent(AgentConfig(upstream_provider="DeepInfra"), client).act(self.messages, "h")
        self.assertIsNone(action)
        self.assertEqual(len(client.payloads), 1)
        self.assertEqual(record.retry_count, 0)
        self.assertIn("empty response", record.error)
        self.assertEqual(len(record.attempts), 1)

    def test_transient_failure_is_retried_and_all_attempts_recorded(self):
        client = FakeClient([urllib.error.URLError("temporary"), response()])
        action, record = FixedAgent(AgentConfig(upstream_provider="DeepInfra"), client).act(self.messages, "h")
        self.assertEqual(action, PlacementAction(2))
        self.assertEqual(len(client.payloads), 2)
        self.assertEqual(record.retry_count, 1)
        self.assertEqual(len(record.attempts), 2)
        self.assertIn("temporary", record.attempts[0].error)
        self.assertIsNone(record.attempts[1].error)

    def test_missing_choices_envelope_is_preserved_and_retried(self):
        envelope = {"error": {"code": 502, "type": "provider_error", "message": "upstream unavailable"}}
        client = FakeClient([envelope, response()])
        action, record = FixedAgent(AgentConfig(upstream_provider="DeepInfra"), client).act(self.messages, "h")
        self.assertEqual(action, PlacementAction(2))
        self.assertEqual(len(client.payloads), 2)
        self.assertEqual(record.retry_count, 1)
        self.assertEqual(record.attempts[0].api_error["code"], 502)
        self.assertEqual(record.attempts[0].api_error["message"], "upstream unavailable")
        self.assertEqual(record.attempts[0].api_error["response_keys"], ["error"])
        self.assertIsNone(record.api_error)

    def test_nonretryable_api_error_is_preserved_without_retry(self):
        envelope = {"error": {"code": 400, "type": "invalid_request", "message": "bad parameter"}}
        client = FakeClient([envelope, response()])
        action, record = FixedAgent(AgentConfig(upstream_provider="DeepInfra"), client).act(self.messages, "h")
        self.assertIsNone(action)
        self.assertEqual(len(client.payloads), 1)
        self.assertEqual(record.retry_count, 0)
        self.assertEqual(record.api_error["code"], 400)
        self.assertEqual(record.api_error["message"], "bad parameter")


if __name__ == "__main__":
    unittest.main()
