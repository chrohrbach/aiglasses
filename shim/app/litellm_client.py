"""Streaming client for LiteLLM (or any OpenAI-compat endpoint).

Two layers:
  * stream_events(): low-level — yields typed deltas (text, tool_call, finish).
                     Used by the agent_loop when tools may be invoked.
  * stream_chat():   text-only convenience — yields raw content chunks. Used
                     by the fast and vision paths that never call tools.
"""

import json
import os
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Union

import httpx

LITELLM_URL = os.environ.get("LITELLM_URL", "http://litellm:4000/v1").rstrip("/")
LITELLM_API_KEY = os.environ.get("LITELLM_API_KEY", "sk-anything")
FAST_MODEL = os.environ.get("ROKID_FAST_MODEL", "infomaniak-ministral")
VISION_MODEL = os.environ.get("ROKID_VISION_MODEL", "purpose-vision")
FULL_MODEL = os.environ.get("ROKID_FULL_MODEL", "purpose-tool-calling")
# Vision-capable model used on the FULL (tools) path when the conversation
# carries an image — lets the model both see the photo and call MCP tools
# (e.g. remember + attach_asset to archive a memory). Defaults to FULL_MODEL,
# so this is safe even without a dedicated vision-tool model configured.
VISION_TOOL_MODEL = os.environ.get("ROKID_VISION_TOOL_MODEL", FULL_MODEL)
LITELLM_TEMPERATURE = float(os.environ.get("LITELLM_TEMPERATURE", "0.4"))


@dataclass
class TextDelta:
    text: str


@dataclass
class ToolCallDelta:
    """Streaming chunk of a single tool_call. Multiple deltas accumulate."""

    index: int
    id: str | None
    name: str | None
    arguments_delta: str  # partial JSON string fragment


@dataclass
class Finish:
    reason: str  # "stop" | "tool_calls" | "length" | ...


Event = Union[TextDelta, ToolCallDelta, Finish]


async def stream_events(
    messages: list[dict],
    *,
    model: str,
    tools: list[dict] | None = None,
    temperature: float | None = None,
) -> AsyncIterator[Event]:
    """Stream typed events from an OpenAI-compat chat-completions endpoint."""
    payload: dict = {
        "model": model,
        "messages": messages,
        "stream": True,
        "temperature": temperature if temperature is not None else LITELLM_TEMPERATURE,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"

    headers = {
        "Authorization": f"Bearer {LITELLM_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }
    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0, read=None)) as client:
        async with client.stream(
            "POST",
            f"{LITELLM_URL}/chat/completions",
            json=payload,
            headers=headers,
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    return
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                ch = choices[0]
                delta = ch.get("delta") or {}
                content = delta.get("content")
                if content:
                    yield TextDelta(content)
                for tc in delta.get("tool_calls") or []:
                    fn = tc.get("function") or {}
                    yield ToolCallDelta(
                        index=tc.get("index", 0),
                        id=tc.get("id"),
                        name=fn.get("name"),
                        arguments_delta=fn.get("arguments") or "",
                    )
                if ch.get("finish_reason"):
                    yield Finish(ch["finish_reason"])


async def stream_chat(messages: list[dict], *, model: str) -> AsyncIterator[str]:
    """Text-only convenience wrapper for the no-tools paths."""
    async for ev in stream_events(messages, model=model):
        if isinstance(ev, TextDelta):
            yield ev.text
