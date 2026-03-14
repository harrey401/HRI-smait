"""LLM dialogue: Phi-4 Mini (local, Ollama) + GPT-4o-mini (API fallback)."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass
from typing import AsyncGenerator, Callable, Optional

import requests

from smait.core.config import Config
from smait.core.events import EventBus, EventType

logger = logging.getLogger(__name__)

# Phrases indicating the user wants to end the conversation
GOODBYE_PATTERNS = re.compile(
    r"\b(goodbye|bye|see you|take care|nice meeting you|"
    r"gotta go|have to go|i('m| am) leaving)\b",
    re.IGNORECASE,
)


@dataclass
class DialogueResponse:
    """Result from LLM dialogue."""
    text: str
    latency_ms: float
    model_used: str
    tokens_used: int = 0
    is_farewell: bool = False


class DialogueManager:
    """Hybrid LLM: Phi-4 Mini (local, Ollama) + GPT-4o-mini (API fallback).

    - try_local_first: attempts Ollama first, falls back to API on timeout/error
    - Conversation memory: sliding window of last 10 turns
    - System prompt: Jackie persona
    - Goodbye detection: analyze response for farewell intent
    - ask_streaming: yields partial responses for sentence-level TTS
    """

    def __init__(self, config: Config, event_bus: EventBus) -> None:
        self._config = config.dialogue
        self._event_bus = event_bus
        self._history: list[dict[str, str]] = []
        self._openai_client = None
        self._ollama_url = "http://localhost:11434"
        self._tools: list[dict] = []
        self._tool_handlers: dict[str, Callable] = {}

    async def init(self) -> None:
        """Initialize LLM backends."""
        # Test Ollama connectivity
        if self._config.try_local_first:
            try:
                resp = requests.get(f"{self._ollama_url}/api/tags", timeout=3)
                if resp.status_code == 200:
                    models = [m["name"] for m in resp.json().get("models", [])]
                    if any(self._config.local_model in m for m in models):
                        logger.info("Ollama available with model: %s", self._config.local_model)
                    else:
                        logger.warning("Ollama running but model '%s' not found. Available: %s",
                                       self._config.local_model, models)
                else:
                    logger.warning("Ollama not reachable (status %d)", resp.status_code)
            except Exception:
                logger.warning("Ollama not available at %s", self._ollama_url)

        # Initialize OpenAI client for API fallback
        try:
            import openai
            self._openai_client = openai.AsyncOpenAI()
            logger.info("OpenAI API client initialized (model: %s)", self._config.api_model)
        except ImportError:
            logger.warning("OpenAI package not installed. API fallback unavailable.")
        except Exception:
            logger.warning("OpenAI client init failed. API fallback unavailable.")

    async def ask(self, user_text: str) -> DialogueResponse:
        """Generate a response to user input.

        Tries local Ollama first, then falls back to API.
        """
        self._history.append({"role": "user", "content": user_text})
        self._trim_history()

        start = time.monotonic()

        # Try local first
        if self._config.try_local_first:
            response = await self._ask_ollama(user_text)
            if response is not None:
                return self._finalize_response(response, start)

        # Fallback to API
        response = await self._ask_api(user_text)
        if response is not None:
            return self._finalize_response(response, start)

        # Both failed — return a graceful error response
        fallback = DialogueResponse(
            text="Sorry, I'm having trouble thinking right now. Could you try again?",
            latency_ms=(time.monotonic() - start) * 1000,
            model_used="fallback",
        )
        return self._finalize_response(fallback, start)

    async def ask_streaming(self, user_text: str) -> AsyncGenerator[str, None]:
        """Generate a streaming response, yielding partial text.

        Yields sentence fragments for real-time TTS synthesis.
        Emits DIALOGUE_STREAM events for each partial.
        """
        self._history.append({"role": "user", "content": user_text})
        self._trim_history()

        full_text = ""
        model_used = "unknown"

        # Try local streaming first
        if self._config.try_local_first:
            try:
                async for chunk in self._stream_ollama(user_text):
                    full_text += chunk
                    self._event_bus.emit(EventType.DIALOGUE_STREAM, {"text": chunk, "full_text": full_text})
                    yield chunk
                model_used = self._config.local_model
            except Exception:
                logger.debug("Ollama streaming failed, trying API")
                full_text = ""

        # Fallback to API streaming
        if not full_text:
            try:
                async for chunk in self._stream_api(user_text):
                    full_text += chunk
                    self._event_bus.emit(EventType.DIALOGUE_STREAM, {"text": chunk, "full_text": full_text})
                    yield chunk
                model_used = self._config.api_model
            except Exception:
                logger.exception("API streaming also failed")
                fallback = "Sorry, I'm having trouble right now."
                full_text = fallback
                yield fallback

        # Record assistant response in history
        if full_text:
            self._history.append({"role": "assistant", "content": full_text})

        # Check for farewell
        is_farewell = self._detect_goodbye(user_text, full_text)
        if is_farewell:
            self._event_bus.emit(EventType.SESSION_END, {"reason": "goodbye_detected"})

    def register_tools(self, tools: list[dict], handlers: dict[str, Callable]) -> None:
        """Register LLM tool definitions and async handler callables.

        Call this once at startup with the tools and handlers from WayfindingManager.
        Registered tools are passed to OpenAI API calls via tools= and tool_choice="auto".
        NOTE: Ollama does not receive tools (unreliable tool-calling in local models).
        """
        self._tools = tools
        self._tool_handlers = handlers

    async def _ask_ollama(self, user_text: str) -> Optional[DialogueResponse]:
        """Query Ollama local LLM."""
        messages = self._build_messages()
        try:
            resp = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: requests.post(
                    f"{self._ollama_url}/api/chat",
                    json={
                        "model": self._config.local_model,
                        "messages": messages,
                        "stream": False,
                        "options": {
                            "num_predict": self._config.max_tokens,
                            "temperature": self._config.temperature,
                        },
                    },
                    timeout=10,
                ),
            )
            if resp.status_code == 200:
                data = resp.json()
                text = data.get("message", {}).get("content", "")
                tokens = data.get("eval_count", 0)
                return DialogueResponse(
                    text=text,
                    latency_ms=0,  # Will be set by _finalize_response
                    model_used=self._config.local_model,
                    tokens_used=tokens,
                )
        except Exception:
            logger.debug("Ollama request failed")
        return None

    async def _ask_api(self, user_text: str) -> Optional[DialogueResponse]:
        """Query OpenAI API, with optional tool-call two-round-trip support.

        If tools are registered via register_tools(), passes them to the first
        API call. If the response contains tool_calls, executes the handlers and
        makes a second call to get the verbal response. Ollama does NOT receive
        tools (graceful degradation for local models).
        """
        if self._openai_client is None:
            return None

        messages = self._build_messages()
        try:
            kwargs: dict = {
                "model": self._config.api_model,
                "messages": messages,
                "max_tokens": self._config.max_tokens,
                "temperature": self._config.temperature,
            }
            if self._tools:
                kwargs["tools"] = self._tools
                kwargs["tool_choice"] = "auto"

            response = await self._openai_client.chat.completions.create(**kwargs)
            message = response.choices[0].message

            if message.tool_calls:
                return await self._handle_tool_calls(message, messages)

            text = message.content or ""
            tokens = response.usage.total_tokens if response.usage else 0
            return DialogueResponse(
                text=text,
                latency_ms=0,
                model_used=self._config.api_model,
                tokens_used=tokens,
            )
        except Exception:
            logger.debug("OpenAI API request failed")
        return None

    async def _handle_tool_calls(self, message, messages: list) -> Optional[DialogueResponse]:
        """Execute tool calls and make follow-up LLM call with results.

        Appends the assistant tool_calls message and each tool result to the
        message list, then makes a second (no-tools) LLM call to produce the
        verbal response.
        """
        # Append assistant message with tool_calls for the second call context
        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [tc.model_dump() for tc in message.tool_calls],
        })

        # Execute each tool call and inject result as "tool" role message
        for tool_call in message.tool_calls:
            handler = self._tool_handlers.get(tool_call.function.name)
            if handler is None:
                logger.warning("No handler registered for tool: %s", tool_call.function.name)
                continue
            args = json.loads(tool_call.function.arguments)
            result = await handler(args)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result),
            })

        # Second LLM call — no tools, get verbal response
        try:
            second_resp = await self._openai_client.chat.completions.create(
                model=self._config.api_model,
                messages=messages,
                max_tokens=self._config.max_tokens,
                temperature=self._config.temperature,
            )
            text = second_resp.choices[0].message.content or ""
            tokens = second_resp.usage.total_tokens if second_resp.usage else 0
            return DialogueResponse(
                text=text,
                latency_ms=0,
                model_used=self._config.api_model,
                tokens_used=tokens,
            )
        except Exception:
            logger.exception("Tool follow-up call failed")
            return None

    async def _stream_ollama(self, user_text: str) -> AsyncGenerator[str, None]:
        """Stream from Ollama local LLM."""
        messages = self._build_messages()
        resp = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: requests.post(
                f"{self._ollama_url}/api/chat",
                json={
                    "model": self._config.local_model,
                    "messages": messages,
                    "stream": True,
                    "options": {
                        "num_predict": self._config.max_tokens,
                        "temperature": self._config.temperature,
                    },
                },
                timeout=15,
                stream=True,
            ),
        )

        if resp.status_code != 200:
            raise RuntimeError(f"Ollama returned status {resp.status_code}")

        import json
        for line in resp.iter_lines():
            if line:
                data = json.loads(line)
                content = data.get("message", {}).get("content", "")
                if content:
                    yield content
                if data.get("done"):
                    break

    async def _stream_api(self, user_text: str) -> AsyncGenerator[str, None]:
        """Stream from OpenAI API."""
        if self._openai_client is None:
            raise RuntimeError("OpenAI client not available")

        messages = self._build_messages()
        stream = await self._openai_client.chat.completions.create(
            model=self._config.api_model,
            messages=messages,
            max_tokens=self._config.max_tokens,
            temperature=self._config.temperature,
            stream=True,
        )

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def _build_messages(self) -> list[dict[str, str]]:
        """Build the message list with system prompt and history."""
        return [
            {"role": "system", "content": self._config.system_prompt},
            *self._history,
        ]

    def _trim_history(self) -> None:
        """Keep only the last N turns (each turn = user + assistant)."""
        max_messages = self._config.max_history_turns * 2
        if len(self._history) > max_messages:
            self._history = self._history[-max_messages:]

    def _finalize_response(self, response: DialogueResponse, start: float) -> DialogueResponse:
        """Set latency, record in history, check for goodbye."""
        response.latency_ms = (time.monotonic() - start) * 1000

        # Record in history
        self._history.append({"role": "assistant", "content": response.text})

        # Check farewell
        user_text = self._history[-2]["content"] if len(self._history) >= 2 else ""
        response.is_farewell = self._detect_goodbye(user_text, response.text)

        # Emit response event
        self._event_bus.emit(EventType.DIALOGUE_RESPONSE, response)

        if response.is_farewell:
            self._event_bus.emit(EventType.SESSION_END, {"reason": "goodbye_detected"})

        logger.info("Dialogue: '%s' (model=%s, %.0fms)",
                     response.text[:80], response.model_used, response.latency_ms)

        return response

    @staticmethod
    def _detect_goodbye(user_text: str, robot_text: str) -> bool:
        """Detect if the conversation is ending."""
        return bool(GOODBYE_PATTERNS.search(user_text) or GOODBYE_PATTERNS.search(robot_text))

    def clear_history(self) -> None:
        """Reset conversation history (e.g., on session end)."""
        self._history.clear()
