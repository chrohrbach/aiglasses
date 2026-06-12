"""Talk to mcp-hub's per-profile OpenAPI + REST surface.

The hub exposes each named profile (mail / personal / knowledge / dev / agents)
at two parallel paths:

    GET  /profiles/<name>/openapi.json   — full OpenAPI 3.1 spec, one POST per tool
    POST /profiles/<name>/tools/<tool>   — execute the tool with a JSON body

This module:
  1. Fetches the OpenAPI spec for the configured profile (with TTL cache).
  2. Translates each operation into an OpenAI-style function tool definition.
  3. Provides a `dispatch(name, args)` that POSTs to the right endpoint and
     returns the JSON result as a string the LLM can consume.

Auth: when MCP_HUB_AUTH_TOKEN is set, every request sends
`Authorization: Bearer <token>`. Inside mcp-network on the LXC the hub
trusts cluster-local traffic so the token is optional in practice.
"""

import asyncio
import json
import logging
import os
import time
from typing import Any

import httpx

logger = logging.getLogger(__name__)

MCP_HUB_URL = os.environ.get("MCP_HUB_URL", "http://mcp-hub:8012").rstrip("/")
MCP_HUB_AUTH_TOKEN = os.environ.get("MCP_HUB_AUTH_TOKEN", "")
# Comma-separated list of hub profiles to merge. Default covers everything
# voice-relevant for daily glasses use: mail/calendar/contacts (mail) +
# sms/whatsapp/memory/tasks (personal) + notes/news (knowledge). Excludes
# dev (github/infra) and agents (browser/social) which are rarely voice asks.
# Legacy single-profile env ROKID_MCP_PROFILE is honored if ROKID_MCP_PROFILES
# isn't set.
_default_profiles = os.environ.get("ROKID_MCP_PROFILE") or "mail,personal,knowledge"
ROKID_MCP_PROFILES = [
    p.strip() for p in os.environ.get("ROKID_MCP_PROFILES", _default_profiles).split(",") if p.strip()
]
MCP_TOOLS_CACHE_TTL = int(os.environ.get("MCP_TOOLS_CACHE_TTL", "300"))
MCP_TOOL_TIMEOUT = float(os.environ.get("MCP_TOOL_TIMEOUT", "30"))


def _auth_headers() -> dict[str, str]:
    if MCP_HUB_AUTH_TOKEN:
        return {"Authorization": f"Bearer {MCP_HUB_AUTH_TOKEN}"}
    return {}


def _operation_to_tool(operation_id: str, op: dict) -> dict | None:
    """Translate one OpenAPI operation into an OpenAI function-tool definition."""
    description = (op.get("description") or op.get("summary") or operation_id).strip()
    # Truncate over-long descriptions — Anthropic and others cap at ~1024 chars
    if len(description) > 1024:
        description = description[:1020] + "..."

    schema: dict[str, Any] = {"type": "object", "properties": {}, "required": []}
    body = op.get("requestBody") or {}
    body_schema = (
        body.get("content", {})
        .get("application/json", {})
        .get("schema")
    )
    if isinstance(body_schema, dict) and body_schema.get("type") == "object":
        schema["properties"] = body_schema.get("properties", {}) or {}
        if body_schema.get("required"):
            schema["required"] = body_schema["required"]

    return {
        "type": "function",
        "function": {
            "name": operation_id,
            "description": description,
            "parameters": schema,
        },
    }


class McpToolCatalog:
    """Caches the OpenAPI specs for one or more hub profiles and dispatches calls.

    Tools across profiles are merged into one flat OpenAI tool list. The first
    profile to define a given operationId wins (later duplicates skipped). The
    catalog remembers which profile each tool came from so `dispatch()` POSTs
    to the right URL.
    """

    def __init__(
        self,
        *,
        hub_url: str = MCP_HUB_URL,
        profiles: list[str] | None = None,
        auth_token: str = MCP_HUB_AUTH_TOKEN,
        cache_ttl: int = MCP_TOOLS_CACHE_TTL,
        timeout: float = MCP_TOOL_TIMEOUT,
        user_id: str | None = None,
    ):
        self.hub_url = hub_url.rstrip("/")
        self.user_id = user_id
        base_profiles = profiles or list(ROKID_MCP_PROFILES)
        if user_id:
            self.profiles = [f"{user_id}_{p}" for p in base_profiles]
        else:
            self.profiles = base_profiles
        self.auth_token = auth_token
        self.cache_ttl = cache_ttl
        self.timeout = timeout
        self._tools_cache: list[dict] | None = None
        self._tool_to_profile: dict[str, str] = {}
        self._cached_at: float = 0.0
        self._lock = asyncio.Lock()

    # Back-compat: some callers (health route) read `.profile`.
    @property
    def profile(self) -> str:
        return ",".join(self.profiles)

    @property
    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.auth_token}"} if self.auth_token else {}

    async def _fetch_one_profile(self, client: httpx.AsyncClient, profile: str) -> dict:
        url = f"{self.hub_url}/profiles/{profile}/openapi.json"
        r = await client.get(url, headers=self._headers)
        r.raise_for_status()
        return r.json()

    async def get_tools(self) -> list[dict]:
        """Return OpenAI-format tool defs merged across all configured profiles."""
        async with self._lock:
            if (
                self._tools_cache is not None
                and time.time() - self._cached_at < self.cache_ttl
            ):
                return self._tools_cache

            merged: list[dict] = []
            self._tool_to_profile = {}
            async with httpx.AsyncClient(timeout=10.0) as c:
                specs = await asyncio.gather(
                    *(self._fetch_one_profile(c, p) for p in self.profiles),
                    return_exceptions=True,
                )
            for profile, spec in zip(self.profiles, specs, strict=True):
                if isinstance(spec, Exception):
                    logger.warning("failed to fetch profile %s: %s", profile, spec)
                    continue
                paths = spec.get("paths", {})
                added = 0
                for path, methods in paths.items():
                    if not path.startswith("/tools/"):
                        continue
                    for method, op in methods.items():
                        if method.lower() != "post":
                            continue
                        op_id = op.get("operationId") or path.removeprefix("/tools/")
                        if op_id in self._tool_to_profile:
                            continue  # earlier profile already provides this tool
                        tool = _operation_to_tool(op_id, op)
                        if not tool:
                            continue
                        merged.append(tool)
                        self._tool_to_profile[op_id] = profile
                        added += 1
                logger.info("profile %s contributed %d tools", profile, added)

            self._tools_cache = merged
            self._cached_at = time.time()
            logger.info("catalog loaded %d total tools across %d profiles", len(merged), len(self.profiles))
            return merged

    async def dispatch(self, name: str, args: dict | None) -> str:
        """Call one tool. Returns a string suitable for an OpenAI tool message."""
        # Ensure the routing table is populated (lazy init)
        if not self._tool_to_profile:
            await self.get_tools()

        profile = self._tool_to_profile.get(name)
        if not profile:
            return json.dumps({"error": f"unknown tool {name!r}"})

        url = f"{self.hub_url}/profiles/{profile}/tools/{name}"
        payload = args if isinstance(args, dict) else {}
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as c:
                r = await c.post(url, headers=self._headers, json=payload)
        except httpx.TimeoutException:
            return json.dumps({"error": "tool timed out", "tool": name})
        except httpx.RequestError as e:
            return json.dumps({"error": f"network error: {e}", "tool": name})

        if r.status_code >= 400:
            return json.dumps({
                "error": f"tool returned HTTP {r.status_code}",
                "body": r.text[:500],
                "tool": name,
            })
        try:
            data = r.json()
            return json.dumps(data, ensure_ascii=False)[:8000]
        except ValueError:
            return r.text[:8000]
