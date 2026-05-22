"""End-to-end test of the rokid shim against mocks (OpenWebUI + LiteLLM + Rokid callback).

Run from the repo root:  python integration_test.py
"""
import asyncio
import json
import os
import sys
import tempfile
import threading
import time

# Mock ports (chosen to avoid the real shim port 8024)
MOCK_OPENWEBUI_PORT = 9990
MOCK_LITELLM_PORT = 9991
MOCK_ROKID_PORT = 9992
SHIM_PORT = 9993
PHOTO_HOST_PORT = 9994

# Env must be set BEFORE importing the shim app (modules read env at import).
os.environ["ROKID_AK"] = "test-ak-secret"
os.environ["OPENWEBUI_URL"] = f"http://127.0.0.1:{MOCK_OPENWEBUI_PORT}"
os.environ["OPENWEBUI_API_KEY"] = "sk-mock"
os.environ["OPENWEBUI_MODEL"] = "openwebui-model"
os.environ["LITELLM_URL"] = f"http://127.0.0.1:{MOCK_LITELLM_PORT}/v1"
os.environ["LITELLM_API_KEY"] = "sk-anything"
os.environ["ROKID_FAST_MODEL"] = "fast-model"
os.environ["ROKID_VISION_MODEL"] = "vision-model"
os.environ["ROKID_FAST_MAX_CHARS"] = "120"
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


# --- Mock OpenWebUI (used by "full" path) -----------------------------------
mock_owui = FastAPI()
owui_seen: dict = {}


@mock_owui.post("/openai/chat/completions")
async def owui_completions(request: Request):
    body = await request.json()
    owui_seen.clear()
    owui_seen.update(body)
    user_text = _extract_user_text(body["messages"])

    if "photo" in user_text.lower():
        chunks = ["Voici votre photo.\n\n", "```rokid_action\n",
                  '{"command": "take_photo"}', "\n```"]
    else:
        chunks = ["Réponse ", "complète ", "via OpenWebUI."]
    return StreamingResponse(_sse_stream(chunks), media_type="text/event-stream")


# --- Mock LiteLLM (used by "fast" and "vision" paths) -----------------------
mock_litellm = FastAPI()
litellm_seen: dict = {}


@mock_litellm.post("/v1/chat/completions")
async def litellm_completions(request: Request):
    body = await request.json()
    litellm_seen.clear()
    litellm_seen.update(body)
    user_text = _extract_user_text(body["messages"])
    model = body.get("model", "")

    if model == "vision-model":
        chunks = ["Je ", "vois ", "l'image."]
    elif model == "fast-model":
        chunks = ["Réponse ", "rapide."]
    else:
        chunks = ["???"]
    return StreamingResponse(_sse_stream(chunks), media_type="text/event-stream")


# --- Mock Rokid callback endpoint -------------------------------------------
mock_rokid = FastAPI()
rokid_callback_log: list = []


@mock_rokid.post("/metis/callback/message")
async def rokid_callback(request: Request):
    headers = dict(request.headers)
    body = await request.json()
    rokid_callback_log.append({"headers": headers, "body": body})
    auth = headers.get("authorization", "")
    if auth != "Bearer sk-rokid-test":
        return {"code": -1, "msg": "sk invalid"}  # spec says "sk不存在", ASCII-only here for Windows console
    return {
        "code": 1, "msg": "success",
        "timestamp": int(time.time() * 1000),
        "uuid": "test-uuid",
        "data": {"success": True, "messageId": body.get("message_id")},
    }


# --- Mock photo host (serves a fake image as if it were Rokid's CDN) --------
mock_photo_host = FastAPI()


@mock_photo_host.get("/cdn/{name}")
async def serve_photo(name: str):
    return Response(content=b"\xff\xd8\xff\xe0FAKE_JPEG_BYTES", media_type="image/jpeg")


# --- Helpers ---------------------------------------------------------------
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


async def _sse_stream(chunks):
    for ch in chunks:
        payload = {"choices": [{"delta": {"content": ch}}]}
        yield f"data: {json.dumps(payload)}\n\n"
        await asyncio.sleep(0.005)
    yield "data: [DONE]\n\n"


def run_server(app, port):
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    asyncio.run(server.serve())


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
    async with httpx.AsyncClient(timeout=10.0) as c:
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
    start_in_thread(mock_owui, MOCK_OPENWEBUI_PORT)
    start_in_thread(mock_litellm, MOCK_LITELLM_PORT)
    start_in_thread(mock_rokid, MOCK_ROKID_PORT)
    start_in_thread(mock_photo_host, PHOTO_HOST_PORT)
    await wait_for(f"http://127.0.0.1:{MOCK_OPENWEBUI_PORT}/openapi.json")
    await wait_for(f"http://127.0.0.1:{MOCK_LITELLM_PORT}/openapi.json")
    await wait_for(f"http://127.0.0.1:{MOCK_ROKID_PORT}/openapi.json")
    await wait_for(f"http://127.0.0.1:{PHOTO_HOST_PORT}/openapi.json")

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "shim"))
    from app.main import app as shim_app
    start_in_thread(shim_app, SHIM_PORT)
    await wait_for(f"http://127.0.0.1:{SHIM_PORT}/health")

    failures = []

    # --- T1: short text -> fast path (LiteLLM, fast-model) ---
    ev = await call_shim({
        "message_id": "m1", "agent_id": "test",
        "message": [{"role": "user", "type": "text", "text": "Bonjour"}],
    })
    text = "".join(d.get("answer_stream", "") for _, d in ev if d.get("type") == "answer")
    print(f"[T1 fast] model_called={litellm_seen.get('model')!r} text={text!r}")
    if litellm_seen.get("model") != "fast-model":
        failures.append(f"T1 expected fast-model, got {litellm_seen.get('model')}")
    if text != "Réponse rapide.":
        failures.append(f"T1 text mismatch: {text!r}")

    # --- T2: image present -> vision path (LiteLLM, vision-model) ---
    litellm_seen.clear()
    ev = await call_shim({
        "message_id": "m2", "agent_id": "test",
        "message": [
            {"role": "user", "type": "text", "text": "Que vois-tu ?"},
            {"role": "user", "type": "image",
             "image_url": f"http://127.0.0.1:{PHOTO_HOST_PORT}/cdn/test1.jpg"},
        ],
    })
    text = "".join(d.get("answer_stream", "") for _, d in ev if d.get("type") == "answer")
    print(f"[T2 vision] model_called={litellm_seen.get('model')!r} text={text!r}")
    if litellm_seen.get("model") != "vision-model":
        failures.append(f"T2 expected vision-model, got {litellm_seen.get('model')}")
    # Image URL should now be a data: URL with the fake bytes inlined
    user_msg = [m for m in litellm_seen["messages"] if m["role"] == "user"][0]
    img_part = next((p for p in user_msg["content"] if p.get("type") == "image_url"), None)
    img_url_sent = img_part["image_url"]["url"] if img_part else ""
    if not img_url_sent.startswith("data:image/"):
        failures.append(f"T2 image url not inlined as data: {img_url_sent[:80]}")
    else:
        import base64 as _b64
        b64_part = img_url_sent.split(",", 1)[1]
        decoded = _b64.b64decode(b64_part)
        print(f"[T2 inline] data URL len={len(img_url_sent)} decoded={len(decoded)} bytes")
        if b"FAKE_JPEG_BYTES" not in decoded:
            failures.append(f"T2 inlined bytes wrong: {decoded[:40]!r}")
    # AND the disk cache should still hold a copy for downstream tools
    photos_on_disk = list(os.scandir(os.environ["PHOTO_CACHE_DIR"]))
    print(f"[T2 disk] photos cached: {[p.name for p in photos_on_disk]}")
    if not photos_on_disk:
        failures.append("T2 disk cache empty after vision turn")
    else:
        # Verify the local /photos/{name} route still serves them
        first = photos_on_disk[0].name
        async with httpx.AsyncClient() as c:
            r = await c.get(f"http://127.0.0.1:{SHIM_PORT}/photos/{first}")
        print(f"[T2 served] GET /photos/{first} -> {r.status_code} ({len(r.content)} bytes)")
        if r.status_code != 200 or b"FAKE_JPEG_BYTES" not in r.content:
            failures.append(f"T2 disk-served photo wrong: status={r.status_code}")

    # --- T3: tool-keyword text -> full path (OpenWebUI) ---
    owui_seen.clear()
    ev = await call_shim({
        "message_id": "m3", "agent_id": "test",
        "message": [{"role": "user", "type": "text",
                     "text": "Envoie un mail à Marie pour confirmer le rdv"}],
    })
    text = "".join(d.get("answer_stream", "") for _, d in ev if d.get("type") == "answer")
    print(f"[T3 full] owui_called_with_model={owui_seen.get('model')!r} text={text!r}")
    if owui_seen.get("model") != "openwebui-model":
        failures.append(f"T3 OpenWebUI was not called: owui_seen={owui_seen}")
    if text != "Réponse complète via OpenWebUI.":
        failures.append(f"T3 text mismatch: {text!r}")

    # --- T4: long text without keywords -> full path (length triggers) ---
    owui_seen.clear()
    long_text = "Peux-tu m'expliquer en détail comment fonctionne la fission nucléaire " * 3
    ev = await call_shim({
        "message_id": "m4", "agent_id": "test",
        "message": [{"role": "user", "type": "text", "text": long_text}],
    })
    print(f"[T4 long] owui_called_with_model={owui_seen.get('model')!r}")
    if owui_seen.get("model") != "openwebui-model":
        failures.append(f"T4 long text should route to full path: owui_seen={owui_seen}")

    # --- T5: tool_call fence extraction still works on the full path ---
    ev = await call_shim({
        "message_id": "m5", "agent_id": "test",
        "message": [{"role": "user", "type": "text", "text": "Prends une photo et envoie un mail"}],
    })
    tools = [d for _, d in ev if d.get("type") == "tool_call"]
    answer_text = "".join(d.get("answer_stream", "") for _, d in ev
                          if d.get("type") == "answer" and not d.get("is_finish"))
    print(f"[T5 tool] tools={[t['tool_call'] for t in tools]} answer={answer_text!r}")
    if len(tools) != 1 or tools[0]["tool_call"]["command"] != "take_photo":
        failures.append(f"T5 expected one take_photo tool: {tools}")
    if "```" in answer_text:
        failures.append(f"T5 fence leaked: {answer_text!r}")

    # --- T6: bad AK rejected ---
    async with httpx.AsyncClient() as c:
        r = await c.post(
            f"http://127.0.0.1:{SHIM_PORT}/rokid/agent",
            json={"message_id": "x", "agent_id": "x", "message": []},
            headers={"Authorization": "Bearer WRONG"},
        )
    print(f"[T6 bad AK] status={r.status_code}")
    if r.status_code != 401:
        failures.append(f"T6 bad AK should be 401, got {r.status_code}")

    # --- T7: /push proxies to Rokid callback with correct sk ---
    rokid_callback_log.clear()
    async with httpx.AsyncClient() as c:
        r = await c.post(
            f"http://127.0.0.1:{SHIM_PORT}/push",
            json={"account_id": "user-42", "agent_id": "plexus-glasses",
                  "content": "Ton train part dans 5 minutes."},
            headers={"X-Push-Secret": "push-secret"},
        )
    print(f"[T7 push] status={r.status_code} body={r.json()}")
    if r.status_code != 200 or not r.json().get("ok"):
        failures.append(f"T7 push failed: {r.status_code} {r.text}")
    if not rokid_callback_log:
        failures.append("T7 rokid callback was not hit")
    else:
        cb = rokid_callback_log[0]
        if cb["headers"].get("authorization") != "Bearer sk-rokid-test":
            failures.append(f"T7 wrong sk forwarded: {cb['headers'].get('authorization')!r}")
        if cb["body"]["message"]["agent_id"] != "plexus-glasses":
            failures.append(f"T7 wrong agent_id: {cb['body']}")

    # --- T8: /push without secret rejected ---
    async with httpx.AsyncClient() as c:
        r = await c.post(
            f"http://127.0.0.1:{SHIM_PORT}/push",
            json={"account_id": "user-42", "agent_id": "a", "content": "x"},
        )
    print(f"[T8 push no-secret] status={r.status_code}")
    if r.status_code != 401:
        failures.append(f"T8 should be 401, got {r.status_code}")

    print()
    if failures:
        print("FAILURES:")
        for f in failures: print("  -", f)
        sys.exit(1)
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    asyncio.run(main())
