"""
Tests for the USE_SERVER_SIDE_SYSTEM_PROMPT feature (single-header opt-in).

Covers the per-request header X-Use-ServerSide-System-Prompt that lets
agent frameworks (LangChain, AutoGen, CrewAI) opt out of server-side
system prompt injection and array reconstruction, receiving their
messages array verbatim.

Design:
  true  (default) → legacy path:  [system+STANDARD+style, ...context, user]
  false            → verbatim path: client array as-is, no injection
"""
import logging
import os
import sys
import unittest
from unittest.mock import AsyncMock, MagicMock

logging.disable(logging.CRITICAL)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import settings
from src.models import ChatMessage
from src.model_dispatcher import ModelDispatcher
from src.model_selector import ModelSelector
from src.provider_registry import ProviderRegistry

# Reusable client arrays for behaviour-matrix tests
CLIENT_WITH_SYSTEM = [
    ChatMessage(role="system", content="You are a strict JSON API."),
    ChatMessage(role="user", content="hi"),
]

CLIENT_WITHOUT_SYSTEM = [
    ChatMessage(role="user", content="first"),
    ChatMessage(role="assistant", content="first-reply"),
    ChatMessage(role="user", content="second"),
]


# ---------------------------------------------------------------------------
# Header parsing unit tests (pure function, copy to avoid FastAPI import)
# ---------------------------------------------------------------------------

def _parse_bool_header(value: str | None) -> bool | None:
    """Parse a boolean HTTP header value. Returns None if absent or invalid."""
    if value is None:
        return None
    stripped = value.strip().lower()
    if stripped == "true":
        return True
    if stripped == "false":
        return False
    return None


class TestHeaderParsing(unittest.TestCase):
    """_parse_bool_header must handle the full range of inputs."""

    def test_absent(self):
        self.assertIsNone(_parse_bool_header(None))

    def test_true_lower(self):
        self.assertTrue(_parse_bool_header("true"))

    def test_true_capitalised(self):
        self.assertTrue(_parse_bool_header("True"))

    def test_true_upper(self):
        self.assertTrue(_parse_bool_header("TRUE"))

    def test_true_leading_trailing_whitespace(self):
        self.assertTrue(_parse_bool_header("  true  "))

    def test_false_lower(self):
        self.assertFalse(_parse_bool_header("false"))

    def test_false_capitalised(self):
        self.assertFalse(_parse_bool_header("False"))

    def test_false_upper(self):
        self.assertFalse(_parse_bool_header("FALSE"))

    def test_invalid_returns_none(self):
        self.assertIsNone(_parse_bool_header("maybe"))

    def test_empty_string_returns_none(self):
        self.assertIsNone(_parse_bool_header(""))

    def test_yes_returns_none(self):
        self.assertIsNone(_parse_bool_header("yes"))

    def test_one_returns_none(self):
        self.assertIsNone(_parse_bool_header("1"))


# ---------------------------------------------------------------------------
# Direct call_provider_api tests — the behaviour matrix
# ---------------------------------------------------------------------------

class _DispatcherTestBase(unittest.IsolatedAsyncioTestCase):
    """Common setUp for tests that exercise call_provider_api directly."""

    def setUp(self):
        self.mock_registry = MagicMock(spec=ProviderRegistry)
        self.mock_selector = MagicMock(spec=ModelSelector)
        self.mock_selector.providers = {}
        self.mock_usage_tracker = MagicMock()

        self.mock_gemini = AsyncMock()
        self.mock_registry.get_client.return_value = self.mock_gemini
        self.mock_gemini.call_model_api.return_value = "Response"

        self.dispatcher = ModelDispatcher(
            self.mock_registry, self.mock_selector, self.mock_usage_tracker
        )
        self.dispatcher._calculate_target_context_tokens = lambda *a, **kw: 100_000

    def _last_messages(self):
        self.mock_gemini.call_model_api.assert_called()
        return self.mock_gemini.call_model_api.call_args.kwargs["messages"]


class TestDefaultTrueBehaviour(_DispatcherTestBase):
    """USE_SERVER_SIDE_SYSTEM_PROMPT = True (default)."""

    def setUp(self):
        super().setUp()
        # This class is for tests that check the "true" path explicitly.
        self._saved_default = settings.USE_SERVER_SIDE_SYSTEM_PROMPT
        settings.USE_SERVER_SIDE_SYSTEM_PROMPT = True

    def tearDown(self):
        settings.USE_SERVER_SIDE_SYSTEM_PROMPT = self._saved_default

    async def test_client_system_ignored_standard_injected(self):
        """Header true → client system ignored, STANDARD + style injected."""
        await self.dispatcher.call_provider_api(
            "Gemini", "gemini-2.0-flash",
            user_prompt="hi",
            system_prompt="client-sys",
            client_messages=None,
        )
        msgs = self._last_messages()
        self.assertEqual(msgs[0]["role"], "system")
        self.assertIn(settings.STANDARD_SYSTEM_PROMPT, msgs[0]["content"])
        # Last message is the new user prompt (legacy reconstruction).
        self.assertEqual(msgs[-1]["role"], "user")
        self.assertEqual(msgs[-1]["content"], "hi")

    async def test_client_messages_ignored_when_true(self):
        """When server default is True and no client_messages passed,
        legacy path runs (system prompt injected, user prompt appended)."""
        await self.dispatcher.call_provider_api(
            "Gemini", "gemini-2.0-flash",
            user_prompt="new-prompt",
            system_prompt="sys",
        )
        msgs = self._last_messages()
        self.assertEqual(msgs[0]["role"], "system")
        self.assertIn("sys", msgs[0]["content"])
        self.assertIn(settings.STANDARD_SYSTEM_PROMPT, msgs[0]["content"])
        self.assertEqual(msgs[-1]["content"], "new-prompt")


class TestFalseVerbatimPath(_DispatcherTestBase):
    """USE_SERVER_SIDE_SYSTEM_PROMPT = False — verbatim client array."""

    def setUp(self):
        super().setUp()
        self._saved_default = settings.USE_SERVER_SIDE_SYSTEM_PROMPT
        settings.USE_SERVER_SIDE_SYSTEM_PROMPT = False

    def tearDown(self):
        settings.USE_SERVER_SIDE_SYSTEM_PROMPT = self._saved_default

    async def test_client_array_forwarded_verbatim(self):
        """Client array forwarded, no system injection."""
        await self.dispatcher.call_provider_api(
            "Gemini", "gemini-2.0-flash",
            user_prompt="ignored",
            system_prompt="ignored",
            client_messages=CLIENT_WITHOUT_SYSTEM,
        )
        msgs = self._last_messages()
        self.assertEqual(
            [m["role"] for m in msgs],
            ["user", "assistant", "user"],
        )
        self.assertEqual(
            [m["content"] for m in msgs],
            ["first", "first-reply", "second"],
        )

    async def test_client_system_respected(self):
        """Client's system message is respected, no server injection."""
        await self.dispatcher.call_provider_api(
            "Gemini", "gemini-2.0-flash",
            user_prompt="ignored",
            system_prompt="ignored",
            client_messages=CLIENT_WITH_SYSTEM,
        )
        msgs = self._last_messages()
        self.assertEqual(msgs[0]["role"], "system")
        self.assertEqual(msgs[0]["content"], "You are a strict JSON API.")
        self.assertNotIn(settings.STANDARD_SYSTEM_PROMPT, msgs[0]["content"])

    async def test_no_client_system_no_injection(self):
        """When client sends no system, nothing is injected."""
        await self.dispatcher.call_provider_api(
            "Gemini", "gemini-2.0-flash",
            user_prompt="ignored",
            system_prompt="ignored",
            client_messages=[
                ChatMessage(role="user", content="just this"),
            ],
        )
        msgs = self._last_messages()
        self.assertEqual(len(msgs), 1)
        self.assertEqual(msgs[0]["role"], "user")
        self.assertEqual(msgs[0]["content"], "just this")

    async def test_structured_content_flattened(self):
        """List content flattened via get_text()."""
        await self.dispatcher.call_provider_api(
            "Gemini", "gemini-2.0-flash",
            user_prompt="ignored",
            system_prompt="ignored",
            client_messages=[
                ChatMessage(role="user", content=[
                    {"type": "text", "text": "describe this"},
                    {"type": "image_url", "url": "https://example.com/x.png"},
                    {"type": "text", "text": "image"},
                ]),
            ],
        )
        msgs = self._last_messages()
        self.assertEqual(len(msgs), 1)
        self.assertEqual(msgs[0]["content"], "describe this image")

    async def test_empty_messages_dropped(self):
        """Empty/whitespace-only messages filtered out."""
        await self.dispatcher.call_provider_api(
            "Gemini", "gemini-2.0-flash",
            user_prompt="ignored",
            system_prompt="ignored",
            client_messages=[
                ChatMessage(role="user", content="keep"),
                ChatMessage(role="user", content=""),
                ChatMessage(role="user", content="   "),
                ChatMessage(role="user", content="last"),
            ],
        )
        msgs = self._last_messages()
        self.assertEqual([m["content"] for m in msgs], ["keep", "last"])

    async def test_context_manager_trims_on_overflow(self):
        """Hard token cap still enforced via context manager."""
        self.dispatcher.context_manager.context_management_mode = "static"
        self.dispatcher.context_manager.static_recent_keep = 3
        client = [ChatMessage(role="user", content=str(i)) for i in range(10)]
        await self.dispatcher.call_provider_api(
            "Gemini", "gemini-2.0-flash",
            user_prompt="ignored",
            system_prompt="ignored",
            client_messages=client,
        )
        msgs = self._last_messages()
        # 3 client messages (oldest 7 dropped).
        self.assertEqual(len(msgs), 3)
        self.assertEqual(msgs[0]["content"], "7")
        self.assertEqual(msgs[-1]["content"], "9")


# ---------------------------------------------------------------------------
# Chat method wiring tests — the flag resolution lives in chat()
# ---------------------------------------------------------------------------

class TestChatWiring(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        self.mock_registry = MagicMock(spec=ProviderRegistry)
        self.mock_selector = MagicMock(spec=ModelSelector)
        self.mock_selector.providers = {}
        self.mock_gemini = AsyncMock()
        self.mock_registry.get_client.return_value = self.mock_gemini
        self.mock_gemini.call_model_api.return_value = "Response"
        self.mock_selector.select.return_value = ("Gemini", "gemini-2.0-flash", 0.0)
        self.mock_selector.estimate_tokens.return_value = 1
        self.dispatcher = ModelDispatcher(self.mock_registry, self.mock_selector)
        self.dispatcher._calculate_target_context_tokens = lambda *a, **kw: 100_000
        self._saved = settings.USE_SERVER_SIDE_SYSTEM_PROMPT

    def tearDown(self):
        settings.USE_SERVER_SIDE_SYSTEM_PROMPT = self._saved

    async def test_chat_verbatim_when_default_false(self):
        """Server default False → chat() forwards client messages."""
        from src.models import ChatCompletionRequest
        settings.USE_SERVER_SIDE_SYSTEM_PROMPT = False
        request = ChatCompletionRequest(
            model="meta-model",
            messages=[
                ChatMessage(role="user", content="a"),
                ChatMessage(role="assistant", content="b"),
                ChatMessage(role="user", content="c"),
            ],
        )
        await self.dispatcher.chat(request)
        msgs = self.mock_gemini.call_model_api.call_args.kwargs["messages"]
        # All three client messages forwarded (no injection).
        self.assertEqual([m["content"] for m in msgs], ["a", "b", "c"])

    async def test_chat_legacy_when_default_true(self):
        """Server default True → legacy reconstruction."""
        from src.models import ChatCompletionRequest
        settings.USE_SERVER_SIDE_SYSTEM_PROMPT = True
        request = ChatCompletionRequest(
            model="meta-model",
            messages=[
                ChatMessage(role="user", content="a"),
                ChatMessage(role="assistant", content="b"),
                ChatMessage(role="user", content="c"),
            ],
        )
        await self.dispatcher.chat(request)
        msgs = self.mock_gemini.call_model_api.call_args.kwargs["messages"]
        # Only the last user message is sent (legacy, no conversation_history).
        self.assertEqual(msgs[-1]["content"], "c")
        self.assertNotIn("a", [m["content"] for m in msgs])

    async def test_chat_per_request_header_override(self):
        """Server default True + per-request override False → verbatim."""
        from src.models import ChatCompletionRequest
        settings.USE_SERVER_SIDE_SYSTEM_PROMPT = True
        request = ChatCompletionRequest(
            model="meta-model",
            messages=[ChatMessage(role="user", content="only-msg")],
        )
        await self.dispatcher.chat(
            request,
            use_server_side_system_prompt=False,
        )
        msgs = self.mock_gemini.call_model_api.call_args.kwargs["messages"]
        # Verbatim: no system injection.
        self.assertEqual(len(msgs), 1)
        self.assertEqual(msgs[0]["content"], "only-msg")

    async def test_chat_per_request_header_no_override(self):
        """Per-request not set, falls back to server default (True)."""
        from src.models import ChatCompletionRequest
        settings.USE_SERVER_SIDE_SYSTEM_PROMPT = True
        request = ChatCompletionRequest(
            model="meta-model",
            messages=[ChatMessage(role="user", content="only-msg")],
        )
        await self.dispatcher.chat(request)
        msgs = self.mock_gemini.call_model_api.call_args.kwargs["messages"]
        # Legacy: conversation_history=None, so just the user prompt.
        self.assertEqual(len(msgs), 1)
        self.assertEqual(msgs[0]["content"], "only-msg")


if __name__ == "__main__":
    unittest.main()
