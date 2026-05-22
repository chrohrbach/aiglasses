"""End-to-end test of the rokid shim against a mock OpenWebUI.

Run from the repo root:  python integration_test.py
"""
import asyncio
import json
import os
import sys
import threading
import time
from contextlib import asynccontextmanager

# Mock OpenWebUI runs on this port, shim runs on the next.
MOCK_PORT = 9998
SHIM_PORT = 9999

os.environ["ROKID_AK"] = "test-ak-secret"
os.environ["OPENWEBUI_URL"] = f"http://127.0.0.1:{MOCK_PORT}"
os.environ["OPENWEBUI_API_KEY"] = "sk-mock"
os.environ["OPENWEBUI_MODEL"] = "mock-model"
os.environ["LOG_LEVEL"] = "WARNING"

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

# --- Mock OpenWebUI ---------------------------------------------------------
mock = FastAPI()
last_seen_request: dict = {}


@mock.post("/api/chat/completions")
async def mock_completions(request: Request):
    body = await request.json()
    last_seen_request.clear()
    last_seen_request.update(body)
    user_text = ""
    for m in body.get("messages", []):
        if m["role"] == "user":
            c = m["content"]
            if isinstance(c, str):
                user_text += c
            else:
                for part in c:
                    if part.get("type") == "text":
                        user_text += part["text"]

    if "photo" in user_text.lower():
        reply_chunks = [
            "Voici votre photo.\n\n",
            "```rokid_action\n",
            '{"command": "take_photo"}',
            "\n```",
        ]
    elif "image" in user_text.lower():
        reply_chunks = ["Je vois l'image."]
    else:
        reply_chunks = ["Bon", "jour", " le", " monde."]

    async def gen():
        for ch in reply_chunks:
            payload = {"choices": [{"delta": {"content": ch}}]}
            yield f"data: {json.dumps(payload)}\n\n"
            await asyncio.sleep(0.01)
        yield "data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


# --- Spin up both servers ---------------------------------------------------
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
            if r.status_code == 200:
                return
        except Exception:
            pass
        await asyncio.sleep(0.1)
    raise RuntimeError(f"server at {url} did not come up")


# --- Test driver ------------------------------------------------------------
async def call_shim(payload: dict, ak: str = "test-ak-secret"):
    events = []
    async with httpx.AsyncClient(timeout=10.0) as c:
        async with c.stream(
            "POST",
            f"http://127.0.0.1:{SHIM_PORT}/rokid/agent",
            json=payload,
            headers={
                "Authorization": f"Bearer {ak}",
                "Content-Type": "application/json",
            },
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
    # Start mock first
    start_in_thread(mock, MOCK_PORT)
    await wait_for(f"http://127.0.0.1:{MOCK_PORT}/openapi.json")

    # Then the shim
    sys.path.insert(0, str(os.path.join(os.path.dirname(__file__), "shim")))
    from app.main import app as shim_app
    start_in_thread(shim_app, SHIM_PORT)
    await wait_for(f"http://127.0.0.1:{SHIM_PORT}/health")

    failures = []

    # --- Test 1: plain text query ---
    ev = await call_shim({
        "message_id": "m1", "agent_id": "test",
        "message": [{"role": "user", "type": "text", "text": "Bonjour"}],
    })
    text = "".join(d.get("answer_stream", "") for kind, d in ev if d.get("type") == "answer")
    done = [d for kind, d in ev if kind == "done"]
    print(f"[T1 plain] events={len(ev)} text={text!r} done_count={len(done)}")
    if text != "Bonjour le monde.": failures.append(f"T1 text mismatch: {text!r}")
    if len(done) != 1: failures.append(f"T1 done count: {len(done)}")
    if not done[0]["is_finish"]: failures.append("T1 done is_finish should be True")

    # --- Test 2: photo action — should emit tool_call ---
    ev = await call_shim({
        "message_id": "m2", "agent_id": "test",
        "message": [{"role": "user", "type": "text", "text": "Prends une photo"}],
    })
    answer_text = "".join(d.get("answer_stream", "") for kind, d in ev if d.get("type") == "answer" and not d["is_finish"])
    tools = [d for kind, d in ev if d.get("type") == "tool_call"]
    done = [d for kind, d in ev if kind == "done"]
    print(f"[T2 photo] events={len(ev)} answer={answer_text!r} tools={tools} done_count={len(done)}")
    if "```" in answer_text: failures.append(f"T2 fence leaked into text: {answer_text!r}")
    if answer_text.strip() != "Voici votre photo.": failures.append(f"T2 text mismatch: {answer_text!r}")
    if len(tools) != 1 or tools[0]["tool_call"]["command"] != "take_photo":
        failures.append(f"T2 tool_call: {tools}")
    if len(done) != 1: failures.append(f"T2 done count: {len(done)}")

    # --- Test 3: multimodal image input reaches the mock ---
    ev = await call_shim({
        "message_id": "m3", "agent_id": "test",
        "message": [
            {"role": "user", "type": "text", "text": "Que vois-tu sur cette image?"},
            {"role": "user", "type": "image", "image_url": "https://example.com/x.jpg"},
        ],
    })
    user_msg = [m for m in last_seen_request.get("messages", []) if m["role"] == "user"][0]
    print(f"[T3 image] user_content={user_msg['content']}")
    if not isinstance(user_msg["content"], list):
        failures.append("T3 user content should be list (multimodal)")
    else:
        parts = user_msg["content"]
        has_img = any(p.get("type") == "image_url" and p["image_url"]["url"] == "https://example.com/x.jpg" for p in parts)
        if not has_img: failures.append(f"T3 image part missing: {parts}")

    # --- Test 4: bad AK is rejected ---
    try:
        async with httpx.AsyncClient(timeout=5.0) as c:
            r = await c.post(
                f"http://127.0.0.1:{SHIM_PORT}/rokid/agent",
                json={"message_id": "x", "agent_id": "x", "message": []},
                headers={"Authorization": "Bearer WRONG", "Content-Type": "application/json"},
            )
        print(f"[T4 bad AK] status={r.status_code}")
        if r.status_code != 401: failures.append(f"T4 bad AK should be 401, got {r.status_code}")
    except Exception as e:
        failures.append(f"T4 unexpected: {e}")

    # --- Test 5: device context injected into system prompt ---
    ev = await call_shim({
        "message_id": "m5", "agent_id": "test",
        "message": [{"role": "user", "type": "text", "text": "Hello"}],
        "metadata": {"context": {"location": "Geneva", "weather": "sunny", "battery": "73"}},
    })
    sys_msg = [m for m in last_seen_request["messages"] if m["role"] == "system"][0]
    print(f"[T5 context] system tail: ...{sys_msg['content'][-200:]!r}")
    if "Geneva" not in sys_msg["content"]: failures.append("T5 location missing")
    if "sunny" not in sys_msg["content"]: failures.append("T5 weather missing")
    if "73" not in sys_msg["content"]: failures.append("T5 battery missing")

    print()
    if failures:
        print("FAILURES:")
        for f in failures: print("  -", f)
        sys.exit(1)
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    asyncio.run(main())
