"""
End-to-end tests for the X-Use-ServerSide-System-Prompt header.

Exercises the full pipeline:
  router parses header -> dispatcher.chat() -> call_provider_api() -> mock provider

Validates that agent frameworks (LangChain, AutoGen, CrewAI) get verbatim
message forwarding when they opt out of server-side system prompt injection.
"""
import json
import logging
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

# ---------------------------------------------------------------------------
# Compatibility shim: FastAPI 0.115.2 passes on_startup/on_shutdown to
# Starlette Router.__init__(), which was removed in Starlette >= 1.0.
# This is a pre-existing environment issue, not related to our change.
# ---------------------------------------------------------------------------
import starlette.routing
_orig_router_init = starlette.routing.Router.__init__
def _compat_router_init(self, *args, **kwargs):
    kwargs.pop("on_startup", None)
    kwargs.pop("on_shutdown", None)
    _orig_router_init(self, *args, **kwargs)
starlette.routing.Router.__init__ = _compat_router_init
# ---------------------------------------------------------------------------

logging.disable(logging.CRITICAL)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.config import settings
from src.model_dispatcher import ModelDispatcher
from src.model_selector import ModelSelector
from src.provider_registry import ProviderRegistry
from src.router import api_router

# FastAPI's include_router expects on_startup/on_shutdown on the APIRouter
# (Starlette 1.x dropped these attrs; FastAPI 0.115.x hasn't caught up).
if not hasattr(api_router, "on_startup"):
    api_router.on_startup = []
if not hasattr(api_router, "on_shutdown"):
    api_router.on_shutdown = []

# Build a minimal test app (no lifespan, no real providers).
app = FastAPI()
app.include_router(api_router)

# ---------------------------------------------------------------------------
# Realistic agent-style payloads
# ---------------------------------------------------------------------------

AGENT_MULTI_TURN = {
    "model": "meta-model",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant powered by tools."},
        {"role": "user", "content": "What is the weather in Paris?"},
        {"role": "assistant", "content": "Let me check the weather tool."},
        {
            "role": "assistant",
            "content": 'Tool result: {"temperature": 22, "condition": "sunny"}',
        },
        {"role": "assistant", "content": "The weather in Paris is currently 22C and sunny."},
        {"role": "user", "content": "Great, thanks for checking!"},
    ],
}

AGENT_NO_SYSTEM = {
    "model": "meta-model",
    "messages": [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "reply"},
        {"role": "user", "content": "second"},
    ],
}

AGENT_SINGLE_USER = {
    "model": "meta-model",
    "messages": [
        {"role": "user", "content": "hello"},
    ],
}

AGENT_STRUCTURED = {
    "model": "meta-model",
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "describe "},
                {"type": "image_url", "url": "https://example.com/img.png"},
                {"type": "text", "text": " this"},
            ],
        },
    ],
}

HEADER_NAME = "X-Use-ServerSide-System-Prompt"


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def dispatcher_and_mocks():
    """Build a real ModelDispatcher with mocked deps and patch get_dispatcher."""
    mock_registry = MagicMock(spec=ProviderRegistry)
    mock_selector = MagicMock(spec=ModelSelector)
    mock_selector.providers = {}
    mock_selector.select.return_value = ("Gemini", "gemini-2.0-flash", 0.0)
    mock_selector.estimate_tokens.return_value = 10
    mock_usage_tracker = MagicMock()

    mock_gemini = AsyncMock()
    mock_gemini.call_model_api.return_value = "Mock response"
    mock_registry.get_client.return_value = mock_gemini
    mock_registry.list_providers.return_value = ["Gemini"]

    dispatcher = ModelDispatcher(mock_registry, mock_selector, mock_usage_tracker)
    dispatcher._calculate_target_context_tokens = lambda *a, **kw: 100_000

    with patch("src.router.get_dispatcher", return_value=dispatcher):
        yield dispatcher, mock_gemini, mock_selector


# ---------------------------------------------------------------------------
# Tests: header absent / default-true (legacy path)
# ---------------------------------------------------------------------------


class TestDefaultTrueLegacyPath:
    """No header (or header=true) -> legacy reconstruction with system injection."""

    def test_no_header_legacy_reconstruction(self, dispatcher_and_mocks):
        """Absent header falls back to server default (True) -> legacy path."""
        _, mock_gemini, _ = dispatcher_and_mocks
        with TestClient(app) as client:
            resp = client.post("/v1/chat/completions", json=AGENT_MULTI_TURN)
        assert resp.status_code == 200
        msgs = mock_gemini.call_model_api.call_args.kwargs["messages"]
        # Legacy: only the last user prompt is forwarded, system is injected.
        assert msgs[0]["role"] == "system"
        assert settings.STANDARD_SYSTEM_PROMPT in msgs[0]["content"]
        assert len(msgs) >= 1
        assert msgs[-1]["role"] == "user"
        assert msgs[-1]["content"] == "Great, thanks for checking!"

    def test_header_true_legacy_path(self, dispatcher_and_mocks):
        """Explicit X-Use-ServerSide-System-Prompt: true -> legacy path."""
        _, mock_gemini, _ = dispatcher_and_mocks
        with TestClient(app) as client:
            resp = client.post(
                "/v1/chat/completions",
                json=AGENT_MULTI_TURN,
                headers={HEADER_NAME: "true"},
            )
        assert resp.status_code == 200
        msgs = mock_gemini.call_model_api.call_args.kwargs["messages"]
        assert msgs[0]["role"] == "system"
        assert settings.STANDARD_SYSTEM_PROMPT in msgs[0]["content"]

    def test_no_header_single_user_legacy(self, dispatcher_and_mocks):
        """Single user message, no header -> works (no crash)."""
        _, mock_gemini, _ = dispatcher_and_mocks
        with TestClient(app) as client:
            resp = client.post("/v1/chat/completions", json=AGENT_SINGLE_USER)
        assert resp.status_code == 200
        msgs = mock_gemini.call_model_api.call_args.kwargs["messages"]
        assert msgs[-1]["content"] == "hello"


# ---------------------------------------------------------------------------
# Tests: header=false (verbatim agent path)
# ---------------------------------------------------------------------------


class TestFalseVerbatimPath:
    """X-Use-ServerSide-System-Prompt: false -> forward client array as-is."""

    def test_agent_multi_turn_preserved(self, dispatcher_and_mocks):
        """Full agent conversation with tool messages forwarded verbatim."""
        _, mock_gemini, _ = dispatcher_and_mocks
        with TestClient(app) as client:
            resp = client.post(
                "/v1/chat/completions",
                json=AGENT_MULTI_TURN,
                headers={HEADER_NAME: "false"},
            )
        assert resp.status_code == 200
        msgs = mock_gemini.call_model_api.call_args.kwargs["messages"]
        # All 6 messages forwarded, preserving roles and order.
        assert len(msgs) == 6
        assert [m["role"] for m in msgs] == [
            "system",
            "user",
            "assistant",
            "assistant",
            "assistant",
            "user",
        ]
        assert msgs[0]["content"] == "You are a helpful assistant powered by tools."
        # Tool-style content preserved.
        assert "Tool result:" in msgs[3]["content"]
        # No standard prompt injected.
        all_text = " ".join(m["content"] for m in msgs)
        assert settings.STANDARD_SYSTEM_PROMPT not in all_text

    def test_no_client_system_no_injection(self, dispatcher_and_mocks):
        """Client sends no system message -> no system sent to provider."""
        _, mock_gemini, _ = dispatcher_and_mocks
        with TestClient(app) as client:
            resp = client.post(
                "/v1/chat/completions",
                json=AGENT_NO_SYSTEM,
                headers={HEADER_NAME: "false"},
            )
        assert resp.status_code == 200
        msgs = mock_gemini.call_model_api.call_args.kwargs["messages"]
        assert len(msgs) == 3
        assert [m["role"] for m in msgs] == ["user", "assistant", "user"]

    def test_single_user_verbatim(self, dispatcher_and_mocks):
        """Single user message, header=false -> forwarded."""
        _, mock_gemini, _ = dispatcher_and_mocks
        with TestClient(app) as client:
            resp = client.post(
                "/v1/chat/completions",
                json=AGENT_SINGLE_USER,
                headers={HEADER_NAME: "false"},
            )
        assert resp.status_code == 200
        msgs = mock_gemini.call_model_api.call_args.kwargs["messages"]
        assert len(msgs) == 1
        assert msgs[0]["content"] == "hello"

    def test_structured_content_flattened_e2e(self, dispatcher_and_mocks):
        """Content parts are flattened via get_text() before provider."""
        _, mock_gemini, _ = dispatcher_and_mocks
        with TestClient(app) as client:
            resp = client.post(
                "/v1/chat/completions",
                json=AGENT_STRUCTURED,
                headers={HEADER_NAME: "false"},
            )
        assert resp.status_code == 200
        msgs = mock_gemini.call_model_api.call_args.kwargs["messages"]
        assert len(msgs) == 1
        assert msgs[0]["content"] == "describe   this"

    def test_invalid_header_falls_back_to_default(self, dispatcher_and_mocks):
        """Invalid header value -> None -> falls back to server default (True)."""
        _, mock_gemini, _ = dispatcher_and_mocks
        with TestClient(app) as client:
            resp = client.post(
                "/v1/chat/completions",
                json=AGENT_MULTI_TURN,
                headers={HEADER_NAME: "maybe"},
            )
        assert resp.status_code == 200
        msgs = mock_gemini.call_model_api.call_args.kwargs["messages"]
        # Falls back to True -> legacy path with system injection.
        assert msgs[0]["role"] == "system"
        assert settings.STANDARD_SYSTEM_PROMPT in msgs[0]["content"]


# ---------------------------------------------------------------------------
# Tests: streaming path
# ---------------------------------------------------------------------------


class TestStreamingPath:
    """Streaming path also respects the header."""

    @pytest.mark.parametrize("header_value,expect_verbatim", [
        ("false", True),
        ("true", False),
        (None, False),  # absent -> default true
    ])
    def test_streaming_respects_header(
        self, dispatcher_and_mocks, header_value, expect_verbatim
    ):
        """Streaming requests respect X-Use-ServerSide-System-Prompt."""
        _, mock_gemini, _ = dispatcher_and_mocks

        async def mock_stream():
            yield "chunk1"
            yield "chunk2"

        mock_gemini.call_model_api.return_value = mock_stream()

        headers = {}
        if header_value is not None:
            headers[HEADER_NAME] = header_value

        with TestClient(app) as client:
            with client.stream(
                "POST", "/v1/chat/completions",
                json={
                    "model": "meta-model",
                    "messages": AGENT_MULTI_TURN["messages"],
                    "stream": True,
                },
                headers=headers,
            ) as response:
                assert response.status_code == 200
                chunks = []
                for line in response.iter_lines():
                    if line.startswith("data: "):
                        data_str = line[len("data: "):]
                        if data_str == "[DONE]":
                            break
                        chunks.append(json.loads(data_str))
                assert len(chunks) >= 2  # meta + content chunks

        msgs = mock_gemini.call_model_api.call_args.kwargs["messages"]
        if expect_verbatim:
            assert len(msgs) == 6
            assert msgs[0]["role"] == "system"
            assert msgs[0]["content"] == "You are a helpful assistant powered by tools."
        else:
            assert msgs[0]["role"] == "system"
            assert settings.STANDARD_SYSTEM_PROMPT in msgs[0]["content"]


# ---------------------------------------------------------------------------
# Tests: response content
# ---------------------------------------------------------------------------


class TestResponseContent:
    """Verify the HTTP response body is well-formed."""

    def test_response_has_choices(self, dispatcher_and_mocks):
        """Non-streaming response returns standard OpenAI structure."""
        _, mock_gemini, _ = dispatcher_and_mocks
        with TestClient(app) as client:
            resp = client.post(
                "/v1/chat/completions",
                json=AGENT_MULTI_TURN,
                headers={HEADER_NAME: "false"},
            )
        assert resp.status_code == 200
        body = resp.json()
        assert "choices" in body
        assert len(body["choices"]) > 0
        assert body["choices"][0]["message"]["content"] == "Mock response"

    def test_response_meta_provider(self, dispatcher_and_mocks):
        """Response includes provider/model metadata."""
        _, mock_gemini, _ = dispatcher_and_mocks
        with TestClient(app) as client:
            resp = client.post(
                "/v1/chat/completions",
                json=AGENT_MULTI_TURN,
                headers={HEADER_NAME: "false"},
            )
        body = resp.json()
        assert body.get("model") == "gemini-2.0-flash"
