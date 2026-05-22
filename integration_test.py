"""End-to-end test of the v3 rokid shim against mocks.

Mocks: LiteLLM (multi-round agent), mcp-hub (OpenAPI + tool dispatch),
Rokid callback, fake camera CDN.

Run from the repo root:  python integration_test.py
"""
import asyncio
import json
import os
import sys
import tempfile
import threading
import time

# Mock ports
MOCK_LITELLM_PORT = 9991
MOCK_MCP_HUB_PORT = 9992
MOCK_ROKID_PORT = 9993
PHOTO_HOST_PORT = 9994
SHIM_PORT = 9995

# Env must be set BEFORE importing the shim app (modules read env at import).
os.environ["ROKID_AK"] = "test-ak-secret"
os.environ["LITELLM_URL"] = f"http://127.0.0.1:{MOCK_LITELLM_PORT}/v1"
os.environ["LITELLM_API_KEY"] = "sk-anything"
os.environ["ROKID_FAST_MODEL"] = "fast-model"
os.environ["ROKID_VISION_MODEL"] = "vision-model"
os.environ["ROKID_FULL_MODEL"] = "full-model"
os.environ["ROKID_FAST_MAX_CHARS"] = "120"
os.environ["MCP_HUB_URL"] = f"http://127.0.0.1:{MOCK_MCP_HUB_PORT}"
os.environ["MCP_HUB_AUTH_TOKEN"] = ""
os.environ["ROKID_MCP_PROFILE"] = "personal"
os.environ["ROKID_CALLBACK_URL"] = f"http://127.0.0.1:{MOCK_ROKID_PORT}/metis/callback/message"
os.environ["ROKID_SK_TOKEN"] = "sk-rokid-test"
os.environ["PUSH_SHARED_SECRET"] = "push-secret"
os.environ["PHOTO_CACHE_DIR"] = tempfile.mkdtemp(prefix="rokid-photos-")
os.environ["PHOTOS_PUBLIC_URL"] = f"http://127.0.0.1:{SHIM_PORT}"
os.environ["LOG_LEVEL"] = "WARNING"

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse


# --- Mock LiteLLM (used by all 3 paths) -------------------------------------
mock_litellm = FastAPI()
litellm_calls: list[dict] = []


@mock_litellm.post("/v1/chat/completions")
async def litellm_completions(request: Request):
    body = await request.json()
    litellm_calls.append(body)
    model = body.get("model", "")
    messages = body["messages"]
    last_role = messages[-1]["role"]
    has_tools = bool(body.get("tools"))

    # Vision path: model=vision-model
    if model == "vision-model":
        return StreamingResponse(_text_sse(["Je ", "vois ", "l'image."]), media_type="text/event-stream")

    # Fast path: model=fast-model. The fence-extraction prompt ("photo") gets
    # back a reply with a rokid_action block at the end.
    if model == "fast-model":
        user_text = _extract_user_text(messages)
        if "photo" in user_text.lower():
            return StreamingResponse(
                _text_sse([
                    "Voici votre photo.\n\n",
                    "```rokid_action\n",
                    '{"command": "take_photo"}',
                    "\n```",
                ]),
                media_type="text/event-stream",
            )
        return StreamingResponse(_text_sse(["Réponse ", "rapide."]), media_type="text/event-stream")

    # Full path: model=full-model, tools attached.
    if model == "full-model":
        if has_tools and last_role == "user":
            # Round 1: emit a tool_call asking to list mails
            return StreamingResponse(_tool_call_sse(
                index=0, id="call_test_1", name="gmail_list_emails",
                arguments_json='{"count":3}',
            ), media_type="text/event-stream")
        if last_role == "tool":
            # Round 2: use the tool result to build a final answer
            tool_msg = messages[-1]["content"]
            return StreamingResponse(_text_sse([
                "Tu as 3 mails: ",
                "voici le résultat de l'outil: ",
                tool_msg[:200],
            ]), media_type="text/event-stream")

    return StreamingResponse(_text_sse(["???"]), media_type="text/event-stream")


# --- Mock mcp-hub -----------------------------------------------------------
mock_hub = FastAPI()
hub_tool_calls: list[dict] = []

MOCK_OPENAPI = {
    "openapi": "3.1.0",
    "info": {"title": "mock-personal", "version": "1.0"},
    "paths": {
        "/tools/gmail_list_emails": {
            "post": {
                "summary": "List recent emails",
                "description": "List recent emails from Gmail. Returns subject + sender of each.",
                "operationId": "gmail_list_emails",
                "requestBody": {
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "count": {"type": "integer", "default": 20},
                                    "label": {"type": "string", "default": "INBOX"},
                                },
                                "required": [],
                            }
                        }
                    }
                },
            }
        },
        "/tools/office_get_due_today": {
            "post": {
                "summary": "Get today's calendar events",
                "description": "Return the user's calendar events scheduled for today.",
                "operationId": "office_get_due_today",
                "requestBody": {
                    "content": {
                        "application/json": {
                            "schema": {"type": "object", "properties": {}, "required": []}
                        }
                    }
                },
            }
        },
    },
}


@mock_hub.get("/profiles/{name}/openapi.json")
async def hub_openapi(name: str):
    return MOCK_OPENAPI


@mock_hub.post("/profiles/{profile}/tools/{tool}")
async def hub_tool(profile: str, tool: str, request: Request):
    args = await request.json()
    hub_tool_calls.append({"profile": profile, "tool": tool, "args": args})
    # Return a canned response per tool
    if tool == "gmail_list_emails":
        return [
            {"subject": "Confirmation commande", "from": "Amazon", "date": "2026-05-22"},
            {"subject": "Réunion équipe", "from": "Julie Martin", "date": "2026-05-22"},
            {"subject": "Newsletter Hebdo", "from": "TechNews", "date": "2026-05-21"},
        ]
    return {"error": "unknown tool"}


# --- Mock Rokid callback -----------------------------------------------------
mock_rokid = FastAPI()
rokid_callback_log: list = []


@mock_rokid.post("/metis/callback/message")
async def rokid_callback(request: Request):
    headers = dict(request.headers)
    body = await request.json()
    rokid_callback_log.append({"headers": headers, "body": body})
    if headers.get("authorization") != "Bearer sk-rokid-test":
        return {"code": -1, "msg": "sk invalid"}
    return {"code": 1, "msg": "success", "timestamp": int(time.time() * 1000),
            "uuid": "test-uuid", "data": {"success": True, "messageId": body.get("message_id")}}


# --- Mock photo CDN ----------------------------------------------------------
mock_photo_host = FastAPI()


@mock_photo_host.get("/cdn/{name}")
async def serve_photo(name: str):
    return Response(content=b"\xff\xd8\xff\xe0FAKE_JPEG_BYTES", media_type="image/jpeg")


# --- Helpers -----------------------------------------------------------------
def _extract_user_text(messages):
    parts = []
    for m in messages:
        if m["role"] == "user":
            c = m["content"]
            if isinstance(c, str):
                parts.append(c)
            else:
                for p in c:
                    if p.get("type") == "text":
                        parts.append(p["text"])
    return " ".join(parts)


async def _text_sse(chunks):
    for ch in chunks:
        payload = {"choices": [{"delta": {"content": ch}}]}
        yield f"data: {json.dumps(payload)}\n\n"
        await asyncio.sleep(0.005)
    yield "data: [DONE]\n\n"


async def _tool_call_sse(*, index, id, name, arguments_json):
    """Emit a streaming tool_call response — split args across chunks for realism."""
    yield f'data: {json.dumps({"choices":[{"delta":{"tool_calls":[{"index":index,"id":id,"type":"function","function":{"name":name,"arguments":""}}]}}]})}\n\n'
    await asyncio.sleep(0.005)
    yield f'data: {json.dumps({"choices":[{"delta":{"tool_calls":[{"index":index,"function":{"arguments":arguments_json[:5]}}]}}]})}\n\n'
    await asyncio.sleep(0.005)
    yield f'data: {json.dumps({"choices":[{"delta":{"tool_calls":[{"index":index,"function":{"arguments":arguments_json[5:]}}]}}]})}\n\n'
    await asyncio.sleep(0.005)
    yield f'data: {json.dumps({"choices":[{"finish_reason":"tool_calls"}]})}\n\n'
    yield "data: [DONE]\n\n"


def run_server(app, port):
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    asyncio.run(uvicorn.Server(config).serve())


def start_in_thread(app, port):
    t = threading.Thread(target=run_server, args=(app, port), daemon=True)
    t.start()
    return t


async def wait_for(url, timeout=5):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            async with httpx.AsyncClient(timeout=1.0) as c:
                r = await c.get(url)
            if r.status_code < 500:
                return
        except Exception:
            pass
        await asyncio.sleep(0.1)
    raise RuntimeError(f"server at {url} did not come up")


async def call_shim(payload, ak="test-ak-secret"):
    events = []
    async with httpx.AsyncClient(timeout=15.0) as c:
        async with c.stream(
            "POST",
            f"http://127.0.0.1:{SHIM_PORT}/rokid/agent",
            json=payload,
            headers={"Authorization": f"Bearer {ak}", "Content-Type": "application/json"},
        ) as r:
            cur_event = None
            async for line in r.aiter_lines():
                if line.startswith("event:"):
                    cur_event = line.split(":", 1)[1].strip()
                elif line.startswith("data:"):
                    data = json.loads(line.split(":", 1)[1].strip())
                    events.append((cur_event, data))
            assert r.status_code == 200, f"status={r.status_code}"
    return events


async def main():
    start_in_thread(mock_litellm, MOCK_LITELLM_PORT)
    start_in_thread(mock_hub, MOCK_MCP_HUB_PORT)
    start_in_thread(mock_rokid, MOCK_ROKID_PORT)
    start_in_thread(mock_photo_host, PHOTO_HOST_PORT)
    for port in (MOCK_LITELLM_PORT, MOCK_MCP_HUB_PORT, MOCK_ROKID_PORT, PHOTO_HOST_PORT):
        await wait_for(f"http://127.0.0.1:{port}/openapi.json")

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "shim"))
    from app.main import app as shim_app
    start_in_thread(shim_app, SHIM_PORT)
    await wait_for(f"http://127.0.0.1:{SHIM_PORT}/health")

    failures = []

    # --- T1: short text -> fast path (LiteLLM, fast-model) ---
    litellm_calls.clear()
    ev = await call_shim({
        "message_id": "m1", "agent_id": "test",
        "message": [{"role": "user", "type": "text", "text": "Bonjour"}],
    })
    text = "".join(d.get("answer_stream", "") for _, d in ev if d.get("type") == "answer")
    fast_call = [c for c in litellm_calls if c.get("model") == "fast-model"]
    print(f"[T1 fast] calls={len(fast_call)} text={text!r}")
    if not fast_call: failures.append("T1 fast-model not called")
    if text != "Réponse rapide.": failures.append(f"T1 text: {text!r}")

    # --- T2: image -> vision path, base64 inlined, disk cache populated ---
    litellm_calls.clear()
    ev = await call_shim({
        "message_id": "m2", "agent_id": "test",
        "message": [
            {"role": "user", "type": "text", "text": "Que vois-tu ?"},
            {"role": "user", "type": "image", "image_url": f"http://127.0.0.1:{PHOTO_HOST_PORT}/cdn/test1.jpg"},
        ],
    })
    text = "".join(d.get("answer_stream", "") for _, d in ev if d.get("type") == "answer")
    vision_call = [c for c in litellm_calls if c.get("model") == "vision-model"]
    print(f"[T2 vision] calls={len(vision_call)} text={text!r}")
    if not vision_call: failures.append("T2 vision-model not called")
    if not vision_call: pass
    else:
        user_msg = [m for m in vision_call[0]["messages"] if m["role"] == "user"][0]
        img_part = next((p for p in user_msg["content"] if p.get("type") == "image_url"), None)
        img_url_sent = img_part["image_url"]["url"] if img_part else ""
        if not img_url_sent.startswith("data:image/"):
            failures.append(f"T2 image url not inlined: {img_url_sent[:60]}")
        else:
            import base64 as _b64
            decoded = _b64.b64decode(img_url_sent.split(",", 1)[1])
            if b"FAKE_JPEG_BYTES" not in decoded:
                failures.append(f"T2 inlined bytes wrong: {decoded[:30]!r}")
            print(f"[T2 inline] decoded {len(decoded)} bytes OK")
    photos = list(os.scandir(os.environ["PHOTO_CACHE_DIR"]))
    if not photos: failures.append("T2 disk cache empty")

    # --- T3: tool-keyword text -> full path, agent loop, MCP tool dispatched ---
    litellm_calls.clear()
    hub_tool_calls.clear()
    ev = await call_shim({
        "message_id": "m3", "agent_id": "test",
        "message": [{"role": "user", "type": "text", "text": "Liste mes 3 derniers mails"}],
    })
    full_calls = [c for c in litellm_calls if c.get("model") == "full-model"]
    text = "".join(d.get("answer_stream", "") for _, d in ev
                   if d.get("type") == "answer" and not d.get("is_finish"))
    print(f"[T3 full] litellm_rounds={len(full_calls)} hub_calls={len(hub_tool_calls)} text={text[:120]!r}")
    if len(full_calls) != 2:
        failures.append(f"T3 expected 2 litellm rounds, got {len(full_calls)}")
    else:
        # Round 1 must include tools; round 2 must include the tool message
        if not full_calls[0].get("tools"):
            failures.append("T3 round 1 missing tools field")
        if full_calls[1]["messages"][-1]["role"] != "tool":
            failures.append("T3 round 2 last message should be a tool result")
    if len(hub_tool_calls) != 1 or hub_tool_calls[0]["tool"] != "gmail_list_emails":
        failures.append(f"T3 hub dispatch: {hub_tool_calls}")
    if hub_tool_calls and hub_tool_calls[0]["args"].get("count") != 3:
        failures.append(f"T3 wrong args: {hub_tool_calls[0]['args']}")
    if "Tu as 3 mails" not in text:
        failures.append(f"T3 final answer didn't incorporate tool result: {text!r}")

    # --- T4: fence extraction still works on the FAST path ---
    litellm_calls.clear()
    ev = await call_shim({
        "message_id": "m4", "agent_id": "test",
        "message": [{"role": "user", "type": "text", "text": "Prends une photo."}],
    })
    tools_emitted = [d for _, d in ev if d.get("type") == "tool_call"]
    answer_text = "".join(d.get("answer_stream", "") for _, d in ev
                          if d.get("type") == "answer" and not d.get("is_finish"))
    print(f"[T4 fence] tools={[t['tool_call'] for t in tools_emitted]} answer={answer_text!r}")
    if not tools_emitted or tools_emitted[0]["tool_call"]["command"] != "take_photo":
        failures.append(f"T4 expected take_photo: {tools_emitted}")
    if "```" in answer_text:
        failures.append(f"T4 fence leaked: {answer_text!r}")

    # --- T5: bad AK -> 401 ---
    async with httpx.AsyncClient() as c:
        r = await c.post(
            f"http://127.0.0.1:{SHIM_PORT}/rokid/agent",
            json={"message_id": "x", "agent_id": "x", "message": []},
            headers={"Authorization": "Bearer WRONG"},
        )
    print(f"[T5 bad AK] status={r.status_code}")
    if r.status_code != 401: failures.append(f"T5 should be 401: {r.status_code}")

    # --- T6: /push proxies to Rokid callback ---
    rokid_callback_log.clear()
    async with httpx.AsyncClient() as c:
        r = await c.post(
            f"http://127.0.0.1:{SHIM_PORT}/push",
            json={"account_id": "user-42", "agent_id": "plexus-glasses",
                  "content": "Test push."},
            headers={"X-Push-Secret": "push-secret"},
        )
    print(f"[T6 push] status={r.status_code} ok={r.json().get('ok') if r.status_code==200 else 'n/a'}")
    if r.status_code != 200 or not r.json().get("ok"):
        failures.append(f"T6 push failed: {r.status_code}")
    if not rokid_callback_log: failures.append("T6 rokid callback not hit")

    # --- T7: /push without secret -> 401 ---
    async with httpx.AsyncClient() as c:
        r = await c.post(
            f"http://127.0.0.1:{SHIM_PORT}/push",
            json={"account_id": "u", "agent_id": "a", "content": "x"},
        )
    print(f"[T7 no-secret] status={r.status_code}")
    if r.status_code != 401: failures.append(f"T7 should be 401: {r.status_code}")

    # --- T8: /health reports tool count from mock hub ---
    async with httpx.AsyncClient() as c:
        r = await c.get(f"http://127.0.0.1:{SHIM_PORT}/health")
    health = r.json()
    print(f"[T8 health] {health}")
    if health.get("mcp_tool_count") != 2:
        failures.append(f"T8 expected 2 tools from mock hub, got {health.get('mcp_tool_count')}")

    # --- T9-T12: ROKID_ALLOWED_USER_IDS allowlist ---
    # The env was empty at import time so the allowlist is currently OFF
    # (every authenticated caller allowed). Mutate the module attribute to
    # exercise the enforcing branch.
    from app import main as shim_main

    async def post_raw(user_id):
        body = {"message_id": "alw", "agent_id": "t",
                "message": [{"role": "user", "type": "text", "text": "Bonjour"}]}
        if user_id is not None:
            body["user_id"] = user_id
        async with httpx.AsyncClient(timeout=10.0) as c:
            return await c.post(
                f"http://127.0.0.1:{SHIM_PORT}/rokid/agent",
                json=body,
                headers={"Authorization": "Bearer test-ak-secret",
                         "Content-Type": "application/json"},
            )

    # T9: empty allowlist (current state) — any user_id passes
    shim_main.ROKID_ALLOWED_USER_IDS = set()
    r = await post_raw("rando-user-42")
    print(f"[T9 empty-allowlist any user] status={r.status_code}")
    if r.status_code != 200:
        failures.append(f"T9 empty allowlist should accept everyone, got {r.status_code}")

    # T10: allowlist = {"alice"}, caller with user_id=alice -> 200
    shim_main.ROKID_ALLOWED_USER_IDS = {"alice-id"}
    r = await post_raw("alice-id")
    print(f"[T10 listed user] status={r.status_code}")
    if r.status_code != 200:
        failures.append(f"T10 listed user should pass, got {r.status_code}")

    # T11: allowlist = {"alice"}, caller with user_id=bob -> 403
    r = await post_raw("bob-id")
    print(f"[T11 unlisted user] status={r.status_code} body={r.text[:120]!r}")
    if r.status_code != 403:
        failures.append(f"T11 unlisted user must be 403, got {r.status_code}")

    # T12: allowlist enforced, no user_id in request -> 403 (refuse anon)
    r = await post_raw(None)
    print(f"[T12 no user_id when enforced] status={r.status_code}")
    if r.status_code != 403:
        failures.append(f"T12 missing user_id must be 403 when enforced, got {r.status_code}")

    # Reset to off so any later mock-test reuse stays open
    shim_main.ROKID_ALLOWED_USER_IDS = set()

    print()
    if failures:
        print("FAILURES:")
        for f in failures: print("  -", f)
        sys.exit(1)
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    asyncio.run(main())
