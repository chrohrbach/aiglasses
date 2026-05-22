"""Direct streaming client for LiteLLM (bypasses OpenWebUI).

Used for the fast and vision paths — no MCP tool fanout, lower latency.
Speaks the standard OpenAI Chat Completions shape over /v1/chat/completions.
"""

import json
import os
from collections.abc import AsyncIterator

import httpx

LITELLM_URL = os.environ.get("LITELLM_URL", "http://litellm:4000/v1").rstrip("/")
LITELLM_API_KEY = os.environ.get("LITELLM_API_KEY", "sk-anything")  # LiteLLM accepts any key by default
FAST_MODEL = os.environ.get("ROKID_FAST_MODEL", "infomaniak-ministral")
VISION_MODEL = os.environ.get("ROKID_VISION_MODEL", "purpose-vision")
LITELLM_TEMPERATURE = float(os.environ.get("LITELLM_TEMPERATURE", "0.4"))


async def stream_chat(
    messages: list[dict],
    *,
    model: str,
) -> AsyncIterator[str]:
    """Stream text deltas from LiteLLM. Yields raw content chunks."""
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "temperature": LITELLM_TEMPERATURE,
    }
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
                delta = choices[0].get("delta") or {}
                text = delta.get("content")
                if text:
                    yield text
