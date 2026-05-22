"""FastAPI shim that bridges Rokid Lingzhu custom-agent SSE to OpenWebUI.

Endpoint contract (Rokid side): https://rokid.yuque.com/ub8h5n/hth52o/qq4gs616xz4ellh1
"""

import logging
import os

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .openwebui_client import stream_chat
from .rokid_types import RokidEventPayload, RokidRequest, RokidToolCall
from .tool_extractor import split_stream
from .translator import rokid_to_openai_messages

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger("rokid-shim")

ROKID_AK = os.environ["ROKID_AK"]

app = FastAPI(title="Rokid → Plexus shim")


@app.get("/health")
async def health():
    return {"status": "ok"}


def _check_auth(authorization: str | None) -> None:
    expected = f"Bearer {ROKID_AK}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="invalid auth")


@app.post("/rokid/agent")
async def rokid_agent(
    request: Request,
    authorization: str | None = Header(default=None),
):
    _check_auth(authorization)

    try:
        body = await request.json()
        req = RokidRequest.model_validate(body)
    except Exception as e:
        logger.warning("bad request: %s", e)
        return JSONResponse({"error": str(e)}, status_code=400)

    messages = rokid_to_openai_messages(req)
    logger.info(
        "agent_id=%s message_id=%s items=%d turns=%d",
        req.agent_id,
        req.message_id,
        len(req.message),
        len(messages),
    )

    async def event_stream():
        emitted_any = False
        pending_tool: dict | None = None
        try:
            async for part in split_stream(stream_chat(messages)):
                if part.kind == "text" and part.text:
                    emitted_any = True
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
            # Best-effort error surface in Rokid's expected shape
            yield RokidEventPayload(
                message_id=req.message_id,
                agent_id=req.agent_id,
                is_finish=False,
                type="answer",
                answer_stream=f"[shim error: {type(e).__name__}]",
            ).to_sse("message")

        # If a tool_call was queued, emit it just before the done event.
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

        # Final done event — always.
        yield RokidEventPayload(
            message_id=req.message_id,
            agent_id=req.agent_id,
            is_finish=True,
            type="answer",
        ).to_sse("done")

    return StreamingResponse(event_stream(), media_type="text/event-stream")
