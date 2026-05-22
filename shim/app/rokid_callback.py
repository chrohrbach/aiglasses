"""Push an async message FROM us back TO the glasses, via Rokid's callback endpoint.

Per spec: POST {ROKID_CALLBACK_URL} with Authorization: Bearer <sk-...>
Body: {"message_id": "...", "account_id": "...", "message": {"agent_id": "...", "content": "..."}}

The sk token is generated once at https://account-web.rokid.com/token and put
into ROKID_SK_TOKEN. The callback host is provided via ROKID_CALLBACK_URL —
the spec gives the path /metis/callback/message but not the host explicitly;
the operator must confirm and set the full URL.
"""

import logging
import os
import uuid

import httpx

logger = logging.getLogger(__name__)

ROKID_CALLBACK_URL = os.environ.get("ROKID_CALLBACK_URL", "").rstrip("/")
ROKID_SK_TOKEN = os.environ.get("ROKID_SK_TOKEN", "")


class CallbackNotConfigured(Exception):
    pass


class CallbackFailed(Exception):
    pass


async def push(*, account_id: str, agent_id: str, content: str, message_id: str | None = None) -> dict:
    """Forward an outbound message to Rokid. Raises on misconfiguration or failure."""
    if not ROKID_CALLBACK_URL or not ROKID_SK_TOKEN:
        raise CallbackNotConfigured("ROKID_CALLBACK_URL and ROKID_SK_TOKEN must be set")

    body = {
        "message_id": message_id or f"shim-{uuid.uuid4().hex}",
        "account_id": account_id,
        "message": {"agent_id": agent_id, "content": content},
    }
    headers = {
        "Authorization": f"Bearer {ROKID_SK_TOKEN}",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=10.0) as c:
        resp = await c.post(ROKID_CALLBACK_URL, json=body, headers=headers)
    try:
        data = resp.json()
    except ValueError:
        raise CallbackFailed(f"non-json response: HTTP {resp.status_code} body={resp.text[:200]!r}")
    if resp.status_code >= 400 or data.get("code") != 1:
        raise CallbackFailed(f"HTTP {resp.status_code} code={data.get('code')} msg={data.get('msg')!r}")
    logger.info("pushed to account=%s agent=%s msg_id=%s", account_id, agent_id, body["message_id"])
    return data
