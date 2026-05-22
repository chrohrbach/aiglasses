"""FastAPI shim that bridges Rokid Lingzhu custom-agent SSE to Plexus.

Endpoints:
    POST /rokid/agent       — Rokid calls this. Routed (fast / vision / full).
    POST /push              — Internal. Push an async message back to Rokid.
    GET  /photos/{name}     — Serves cached camera frames.
    GET  /health            — Liveness.

Rokid contract:  https://rokid.yuque.com/ub8h5n/hth52o/qq4gs616xz4ellh1
"""

import logging
import os

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from . import photo_cache, rokid_callback, router
from .rokid_types import RokidEventPayload, RokidRequest, RokidToolCall
from .tool_extractor import split_stream
from .translator import rokid_to_openai_messages

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger("rokid-shim")

ROKID_AK = os.environ["ROKID_AK"]
PUSH_SHARED_SECRET = os.environ.get("PUSH_SHARED_SECRET", "")

app = FastAPI(title="Rokid → Plexus shim")


@app.get("/health")
async def health():
    return {"status": "ok"}


def _check_rokid_auth(authorization: str | None) -> None:
    expected = f"Bearer {ROKID_AK}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="invalid auth")


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

    # Replace incoming Rokid CDN URLs with cached local URLs before the model
    # sees them — guarantees later turns / tool calls can still reach the photo.
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

    async def event_stream():
        pending_tool: dict | None = None
        try:
            async for part in split_stream(router.stream_for_path(path, messages)):
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


# --- /photos/{name} ---------------------------------------------------------

@app.get("/photos/{name}")
async def get_photo(name: str):
    path = photo_cache.get_path(name)
    if path is None:
        raise HTTPException(status_code=404, detail="not found")
    return FileResponse(path)


# --- /push ------------------------------------------------------------------

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
    """Forward an outbound message to a glasses user via Rokid callback.

    Auth: if PUSH_SHARED_SECRET is set in env, callers must send the matching
    X-Push-Secret header. Empty env disables the check (intra-cluster only).
    """
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
