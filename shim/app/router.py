"""Decide which backend handles a given Rokid request.

Three paths:
    - "vision": request contains at least one image item → direct to LiteLLM
                with a vision-capable model. No MCP tools.
    - "fast":   short text-only request that doesn't hint at MCP tool usage →
                direct to LiteLLM with a small/fast EU model. No OpenWebUI hop.
    - "full":   anything else → through agent_loop with MCP tools wired in.
"""

import os
from collections.abc import AsyncIterator
from typing import Literal

from . import agent_loop, litellm_client
from .mcp_tools import McpToolCatalog
from .rokid_types import RokidRequest

Path = Literal["vision", "fast", "full"]

FAST_MAX_CHARS = int(os.environ.get("ROKID_FAST_MAX_CHARS", "120"))

# Keywords that suggest the user wants something from the MCP fleet.
# French + English mix; the LLM still re-checks but this filter biases routing.
_TOOL_HINT_KEYWORDS = {
    # Mail
    "mail", "email", "courrier", "boite", "boîte", "inbox",
    # Calendar
    "rdv", "rendez", "rendez-vous", "agenda", "calendar", "calendrier",
    "réunion", "meeting",
    # Comms
    "envoie", "envoyer", "réponds", "repondre", "répond", "sms", "whatsapp",
    "message",
    # Contacts
    "contact", "numéro", "tél", "telephone", "téléphone",
    # Knowledge / notes
    "note", "rappel", "rappelle", "souvient", "remember", "mémorise", "knowledge",
    # Smart home
    "lumière", "lumiere", "stores", "store", "chauffage", "thermostat",
    "casasmooth", "allume", "éteins", "eteins",
    # GitHub / dev
    "pr ", " pr,", "github", "commit", "branche", "branch",
    # Tasks
    "tâche", "tache", "todo", "task",
    # Web / news
    "cherche", "search", "actualité", "actualite", "news", "google",
}

# Keywords signalling the user wants to ARCHIVE/REMEMBER the current photo.
# When an image is present together with one of these, we route to the FULL
# (tools) path so the agent can call remember + attach_asset, instead of the
# tool-less vision path used for plain "what is this?" questions.
_MEMORY_INTENT_KEYWORDS = {
    # French
    "souviens", "souvenir", "mémorise", "memorise", "retiens", "garde ça",
    "garde cette", "archive", "immortalise", "capture ce moment",
    "note cette photo",
    # English
    "remember", "save this", "keep this photo", "memorize",
}


def _messages_have_image(messages: list[dict]) -> bool:
    """True when any message's content is a list containing an image_url part."""
    for m in messages:
        content = m.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    return True
    return False


def pick_path(req: RokidRequest) -> Path:
    has_image = any(item.type == "image" and item.image_url for item in req.message)

    last_user_text = ""
    for item in reversed(req.message):
        if item.role == "user" and item.type == "text" and item.text:
            last_user_text = item.text
            break

    lower = last_user_text.lower()

    if has_image:
        # Memory-intent + photo → tool path (remember + attach_asset).
        if any(kw in lower for kw in _MEMORY_INTENT_KEYWORDS):
            return "full"
        return "vision"

    if not last_user_text:
        return "full"

    if any(kw in lower for kw in _TOOL_HINT_KEYWORDS):
        return "full"

    if len(last_user_text) <= FAST_MAX_CHARS:
        return "fast"

    return "full"


async def stream_for_path(
    path: Path,
    messages: list[dict],
    *,
    catalog: McpToolCatalog,
) -> AsyncIterator[str]:
    if path == "vision":
        async for chunk in litellm_client.stream_chat(messages, model=litellm_client.VISION_MODEL):
            yield chunk
    elif path == "fast":
        async for chunk in litellm_client.stream_chat(messages, model=litellm_client.FAST_MODEL):
            yield chunk
    else:
        has_image = _messages_have_image(messages)
        model = litellm_client.VISION_TOOL_MODEL if has_image else litellm_client.FULL_MODEL
        async for chunk in agent_loop.stream_with_tools(
            messages,
            model=model,
            catalog=catalog,
        ):
            yield chunk
