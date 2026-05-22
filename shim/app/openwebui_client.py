"""Async streaming client for OpenWebUI's OpenAI-compatible chat-completions endpoint."""

import json
import os
from collections.abc import AsyncIterator

import httpx

OPENWEBUI_URL = os.environ["OPENWEBUI_URL"].rstrip("/")
OPENWEBUI_API_KEY = os.environ["OPENWEBUI_API_KEY"]
OPENWEBUI_MODEL = os.environ["OPENWEBUI_MODEL"]
OPENWEBUI_TEMPERATURE = float(os.environ.get("OPENWEBUI_TEMPERATURE", "0.4"))


async def stream_chat(messages: list[dict]) -> AsyncIterator[str]:
    """Stream the assistant's text deltas from OpenWebUI.

    Yields raw text chunks as they arrive. Tool-call detection and the
    Rokid SSE wrapping happen one layer up.
    """
    payload = {
        "model": OPENWEBUI_MODEL,
        "messages": messages,
        "stream": True,
        "temperature": OPENWEBUI_TEMPERATURE,
    }
    headers = {
        "Authorization": f"Bearer {OPENWEBUI_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }

    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0, read=None)) as client:
        async with client.stream(
            "POST",
            f"{OPENWEBUI_URL}/api/chat/completions",
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
                delta = choices[0].get("delta") or {}
                text = delta.get("content")
                if text:
                    yield text
