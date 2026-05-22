"""v3 real-stack smoke test.

Drives the shim against the live Plexus stack on the LAN:
  - LiteLLM proxy via OpenWebUI's /openai endpoint (LiteLLM :4000 isn't on LAN)
  - mcp-hub at 192.168.68.86:8012 (LAN-exposed)
  - Auth via mcp-servers/.env OPENWEBUI_EMAIL/PASSWORD pair

The FULL path now drives the agent loop itself — "Liste mes mails non lus"
should actually invoke gmail_list_emails on the real Gmail MCP server.

Run from the repo root:  python test_real.py
"""
import asyncio
import os
import re
import sys
import tempfile
import threading
import time
from pathlib import Path

# Reconfigure stdout/stderr to UTF-8 — LLM replies contain emoji/CJK.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

OPENWEBUI_LAN_URL = "http://192.168.68.86:3000"
MCP_HUB_LAN_URL = "http://192.168.68.86:8012"
MCP_SERVERS_ENV = Path("C:/Users/crohr/MyProjects/mcp-servers/.env")
SHIM_PORT = 9995

import httpx

env_text = MCP_SERVERS_ENV.read_text(encoding="utf-8")


def _grab(var):
    m = re.search(rf'^{var}=(.*)$', env_text, re.MULTILINE)
    if not m:
        raise RuntimeError(f"{var} not in {MCP_SERVERS_ENV}")
    return m.group(1).strip().strip('"').strip("'")


email = _grab("OPENWEBUI_EMAIL")
password = _grab("OPENWEBUI_PASSWORD")

print(f"Signing in to {OPENWEBUI_LAN_URL} as {email[:3]}***...")
r = httpx.post(
    f"{OPENWEBUI_LAN_URL}/api/v1/auths/signin",
    json={"email": email, "password": password},
    timeout=10.0,
)
r.raise_for_status()
jwt = r.json()["token"]
print(f"  -> JWT length {len(jwt)}")

os.environ["ROKID_AK"] = "test-real-ak"
os.environ["LITELLM_URL"] = f"{OPENWEBUI_LAN_URL}/openai"
os.environ["LITELLM_API_KEY"] = jwt
os.environ["ROKID_FAST_MODEL"] = "infomaniak-ministral"
os.environ["ROKID_VISION_MODEL"] = "purpose-vision"
os.environ["ROKID_FULL_MODEL"] = "purpose-tool-calling"
os.environ["ROKID_FAST_MAX_CHARS"] = "120"
os.environ["MCP_HUB_URL"] = MCP_HUB_LAN_URL
os.environ["MCP_HUB_AUTH_TOKEN"] = ""
os.environ["ROKID_MCP_PROFILES"] = os.environ.get("ROKID_MCP_PROFILES", "mail,personal,knowledge")
os.environ["PHOTO_CACHE_DIR"] = tempfile.mkdtemp(prefix="rokid-photos-real-")
os.environ["PHOTOS_PUBLIC_URL"] = f"http://127.0.0.1:{SHIM_PORT}"
os.environ["LOG_LEVEL"] = "WARNING"
os.environ["PUSH_SHARED_SECRET"] = ""

import uvicorn  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "shim"))
from app import router as router_mod  # noqa: E402
from app.main import app as shim_app  # noqa: E402


def run_server(app, port):
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    asyncio.run(uvicorn.Server(config).serve())


def start_in_thread(app, port):
    t = threading.Thread(target=run_server, args=(app, port), daemon=True)
    t.start()
    return t


async def wait_for(url, timeout=10):
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


async def call_shim(prompt: str, image_url: str | None = None, message_id: str = "t"):
    items = [{"role": "user", "type": "text", "text": prompt}]
    if image_url:
        items.append({"role": "user", "type": "image", "image_url": image_url})
    payload = {"message_id": message_id, "agent_id": "real-test", "message": items}

    from app.rokid_types import RokidRequest
    actual_path = router_mod.pick_path(RokidRequest.model_validate(payload))

    t0 = time.perf_counter()
    ttft_ms: float | None = None
    answer = ""
    err: str | None = None

    try:
        async with httpx.AsyncClient(timeout=120.0) as c:
            async with c.stream(
                "POST",
                f"http://127.0.0.1:{SHIM_PORT}/rokid/agent",
                json=payload,
                headers={"Authorization": "Bearer test-real-ak", "Content-Type": "application/json"},
            ) as r:
                async for line in r.aiter_lines():
                    if line.startswith("data:"):
                        import json as _json
                        try:
                            data = _json.loads(line.split(":", 1)[1].strip())
                        except Exception:
                            continue
                        if data.get("type") == "answer" and data.get("answer_stream"):
                            if ttft_ms is None:
                                ttft_ms = (time.perf_counter() - t0) * 1000
                            answer += data["answer_stream"]
                if r.status_code != 200:
                    err = f"HTTP {r.status_code}"
    except Exception as e:
        err = f"{type(e).__name__}: {e}"

    total_ms = (time.perf_counter() - t0) * 1000
    return {"path": actual_path, "ttft_ms": ttft_ms, "total_ms": total_ms,
            "answer": answer.strip(), "error": err}


# Non-mutating prompts only — no email send / calendar create / lights on / note write.
PROMPTS = [
    # FAST path
    ("fast.greet",   "Bonjour, comment vas-tu ?", None),
    ("fast.trad-it", "Traduis 'where is the toilet' en italien", None),
    ("fast.math",    "Combien font 17 fois 23 ?", None),
    ("fast.geo",     "Quelle est la capitale du Burkina Faso ?", None),
    ("fast.proverb", "Donne-moi un proverbe chinois en une phrase", None),
    ("fast.trad-sw", "Comment on dit bonjour en swahili ?", None),
    ("fast.def",     "Definition rapide d'entropie en une phrase", None),
    ("fast.timez",   "Quelle heure est-il a Tokyo si il est 14h a Paris ?", None),

    # VISION path
    ("vision.cat",      "Que vois-tu sur cette image, en une phrase ?",
        "https://picsum.photos/seed/cat/400/300.jpg"),
    ("vision.landscape", "Decris cette photo en une phrase",
        "https://picsum.photos/seed/landscape/400/300.jpg"),

    # FULL path — read-only MCP tool fanout via v3 agent loop
    ("full.mail",    "Liste mes 3 derniers mails, juste sujet et expediteur", None),
    ("full.unread",  "Combien j'ai de mails non lus dans Gmail ?", None),
    ("full.cal",     "Quel est mon prochain rendez-vous ?", None),
    ("full.know",    "Cherche dans mes notes Plexus", None),
]


async def main():
    start_in_thread(shim_app, SHIM_PORT)
    await wait_for(f"http://127.0.0.1:{SHIM_PORT}/health")

    # Print health (shows tool count from mcp-hub)
    async with httpx.AsyncClient() as c:
        r = await c.get(f"http://127.0.0.1:{SHIM_PORT}/health")
    print(f"shim health: {r.json()}\n")

    rows = []
    for label, prompt, image in PROMPTS:
        print(f"  testing {label!r}... ", end="", flush=True)
        res = await call_shim(prompt, image, message_id=label)
        rows.append((label, prompt, image, res))
        status = "OK" if not res["error"] else f"ERR({res['error']})"
        ttft = f"{res['ttft_ms']:.0f}ms" if res['ttft_ms'] else "—"
        print(f"{status} ttft={ttft} total={res['total_ms']:.0f}ms chars={len(res['answer'])}")

    print()
    print("=" * 120)
    print(f"{'LABEL':<18}{'PATH':<8}{'TTFT':>10}{'TOTAL':>10}  ANSWER (first 240 chars)")
    print("=" * 120)
    for label, prompt, image, res in rows:
        ttft = f"{res['ttft_ms']:.0f}ms" if res['ttft_ms'] else "—"
        total = f"{res['total_ms']:.0f}ms"
        ans = (res['answer'].replace('\n', ' ')[:240] or res['error'] or "(empty)")
        print(f"{label:<18}{res['path']:<8}{ttft:>10}{total:>10}  {ans}")

    print()
    by_path: dict[str, list[float]] = {}
    for _, _, _, res in rows:
        if res['ttft_ms'] is not None:
            by_path.setdefault(res['path'], []).append(res['ttft_ms'])
    print("Per-path TTFT median (ms):")
    for path, ttfts in sorted(by_path.items()):
        ts = sorted(ttfts)
        print(f"  {path:<8} n={len(ts)}  median={ts[len(ts)//2]:.0f}  min={min(ts):.0f}  max={max(ts):.0f}")


if __name__ == "__main__":
    asyncio.run(main())
