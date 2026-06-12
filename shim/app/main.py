"""FastAPI shim that bridges Rokid Lingzhu custom-agent SSE to Plexus.

Endpoints:
    POST /rokid/agent       — Rokid calls this. Routed (fast / vision / full+tools).
    POST /push              — Internal. Push an async message back to Rokid.
    GET  /photos/{name}     — Serves cached camera frames.
    GET  /health            — Liveness + tool count.

Rokid contract:  https://rokid.yuque.com/ub8h5n/hth52o/qq4gs616xz4ellh1
"""

import asyncio
import logging
import os

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from . import photo_cache, rokid_callback, router
from .mcp_tools import McpToolCatalog
from .rokid_types import RokidEventPayload, RokidRequest, RokidToolCall
from .tool_extractor import split_stream
from .translator import rokid_to_openai_messages

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger("rokid-shim")

ROKID_AK = os.environ["ROKID_AK"]
PUSH_SHARED_SECRET = os.environ.get("PUSH_SHARED_SECRET", "")

# Per-user authorization. Comma-separated Rokid user_ids that may invoke this
# agent. CRITICAL when the agent is published to the Rokid store (智能体商店),
# otherwise ANY Rokid user who installs it would invoke our shim with the
# shared AK and reach our MCP tools (Gmail, Office, casasmooth…).
# Empty / unset (default) = allowlist DISABLED — every authenticated caller
# is allowed. Safe ONLY while the agent stays in 草稿/private. Find your own
# user_id by checking `docker logs rokid-shim` after a real glasses call.
ROKID_ALLOWED_USER_IDS = {
    u.strip() for u in os.environ.get("ROKID_ALLOWED_USER_IDS", "").split(",") if u.strip()
}
if ROKID_ALLOWED_USER_IDS:
    logger.warning(
        "user_id allowlist ENFORCED (%d entries) — rejecting any other Rokid user",
        len(ROKID_ALLOWED_USER_IDS),
    )
else:
    logger.warning(
        "ROKID_ALLOWED_USER_IDS is empty — allowlist DISABLED. Safe only while "
        "the agent stays in 草稿/private. Set this env var BEFORE 提审/发布."
    )

app = FastAPI(title="Rokid → Plexus shim")
catalog = McpToolCatalog()

_catalogs_cache: dict[str | None, McpToolCatalog] = {}
_catalogs_lock = asyncio.Lock()


async def get_catalog_for_user(user_id: str | None) -> McpToolCatalog:
    async with _catalogs_lock:
        if user_id not in _catalogs_cache:
            _catalogs_cache[user_id] = McpToolCatalog(user_id=user_id)
        return _catalogs_cache[user_id]


@app.get("/health")
async def health():
    try:
        tools = await catalog.get_tools()
        tool_count = len(tools)
        tool_err = None
    except Exception as e:
        tool_count = 0
        tool_err = f"{type(e).__name__}: {e}"
    return {
        "status": "ok",
        "mcp_profile": catalog.profile,
        "mcp_tool_count": tool_count,
        "mcp_tool_error": tool_err,
    }


def _check_rokid_auth(authorization: str | None) -> None:
    expected = f"Bearer {ROKID_AK}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="invalid auth")


def _check_user_id(user_id: str | None) -> None:
    """Enforce ROKID_ALLOWED_USER_IDS allowlist if configured."""
    if not ROKID_ALLOWED_USER_IDS:
        return  # disabled — allow all
    if not user_id:
        logger.warning("REJECT: request has no user_id but allowlist is enforced")
        raise HTTPException(status_code=403, detail="user_id required")
    if user_id not in ROKID_ALLOWED_USER_IDS:
        logger.warning("REJECT: user_id=%r not in allowlist", user_id)
        raise HTTPException(status_code=403, detail="user not authorized")


@app.post("/rokid/agent")
async def rokid_agent(
    request: Request,
    authorization: str | None = Header(default=None),
):
    _check_rokid_auth(authorization)

    try:
        body = await request.json()
        req = RokidRequest.model_validate(body)
    except Exception as e:
        logger.warning("bad request: %s", e)
        return JSONResponse({"error": str(e)}, status_code=400)

    # Log the user_id before any auth decision so the operator can grab it from
    # the logs to populate ROKID_ALLOWED_USER_IDS later.
    logger.info(
        "INCOMING agent_id=%s message_id=%s user_id=%r items=%d",
        req.agent_id, req.message_id, req.user_id, len(req.message),
    )
    _check_user_id(req.user_id)

    # Replace incoming Rokid CDN URLs with cached local URLs / inlined base64
    # before the model sees them.
    await photo_cache.cache_request_images(req)

    path = router.pick_path(req)
    messages = rokid_to_openai_messages(req)
    logger.info(
        "path=%s agent_id=%s message_id=%s items=%d turns=%d",
        path,
        req.agent_id,
        req.message_id,
        len(req.message),
        len(messages),
    )

    req_catalog = await get_catalog_for_user(req.user_id)

    async def event_stream():
        pending_tool: dict | None = None
        try:
            async for part in split_stream(
                router.stream_for_path(path, messages, catalog=req_catalog)
            ):
                if part.kind == "text" and part.text:
                    yield RokidEventPayload(
                        message_id=req.message_id,
                        agent_id=req.agent_id,
                        is_finish=False,
                        type="answer",
                        answer_stream=part.text,
                    ).to_sse("message")
                elif part.kind == "tool_call" and part.tool_call:
                    pending_tool = part.tool_call
        except Exception as e:
            logger.exception("upstream error")
            yield RokidEventPayload(
                message_id=req.message_id,
                agent_id=req.agent_id,
                is_finish=False,
                type="answer",
                answer_stream=f"[shim error: {type(e).__name__}]",
            ).to_sse("message")

        if pending_tool is not None:
            try:
                tool = RokidToolCall.model_validate(pending_tool)
                yield RokidEventPayload(
                    message_id=req.message_id,
                    agent_id=req.agent_id,
                    is_finish=False,
                    type="tool_call",
                    tool_call=tool,
                ).to_sse("message")
            except Exception as e:
                logger.warning("tool_call validation failed: %s — payload=%r", e, pending_tool)

        yield RokidEventPayload(
            message_id=req.message_id,
            agent_id=req.agent_id,
            is_finish=True,
            type="answer",
        ).to_sse("done")

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/photos/{name}")
async def get_photo(name: str):
    path = photo_cache.get_path(name)
    if path is None:
        raise HTTPException(status_code=404, detail="not found")
    return FileResponse(path)


class PushRequest(BaseModel):
    account_id: str
    agent_id: str
    content: str
    message_id: str | None = None


@app.post("/push")
async def push(
    body: PushRequest,
    x_push_secret: str | None = Header(default=None, alias="X-Push-Secret"),
):
    """Forward an outbound message to a glasses user via Rokid callback."""
    if PUSH_SHARED_SECRET and x_push_secret != PUSH_SHARED_SECRET:
        raise HTTPException(status_code=401, detail="invalid push secret")
    try:
        result = await rokid_callback.push(
            account_id=body.account_id,
            agent_id=body.agent_id,
            content=body.content,
            message_id=body.message_id,
        )
    except rokid_callback.CallbackNotConfigured as e:
        raise HTTPException(status_code=503, detail=str(e))
    except rokid_callback.CallbackFailed as e:
        raise HTTPException(status_code=502, detail=str(e))
    return {"ok": True, "rokid": result}
