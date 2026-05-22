"""Real-stack smoke test for the rokid shim.

Drives the shim against the live Plexus OpenWebUI on the LAN (192.168.68.86:3000),
authenticating via the mcp-servers/.env OPENWEBUI_EMAIL/PASSWORD pair. Skips
mutating prompts (no email send / calendar create / smart home action / note write).

Run from the repo root: python test_real.py
"""
import asyncio
import os
import re
import sys
import tempfile
import threading
import time
from pathlib import Path

# Windows console defaults to cp1252; LLM replies happily contain emoji/CJK.
# Reconfigure stdout/stderr to UTF-8 before any print.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

OPENWEBUI_LAN_URL = "http://192.168.68.86:3000"
MCP_SERVERS_ENV = Path("C:/Users/crohr/MyProjects/mcp-servers/.env")
SHIM_PORT = 9995

# Sign in to OpenWebUI to grab a JWT, then set env vars BEFORE importing the shim
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

# Shim env. The shim's LiteLLM client also speaks /chat/completions, so we
# point LITELLM_URL at OpenWebUI's /api endpoint — the fast and vision paths
# will hop OpenWebUI (no tool fanout since we don't send a tools field) and
# reach the same models via the underlying litellm.yaml routing.
os.environ["ROKID_AK"] = "test-real-ak"
os.environ["OPENWEBUI_URL"] = OPENWEBUI_LAN_URL
os.environ["OPENWEBUI_API_KEY"] = jwt
os.environ["OPENWEBUI_MODEL"] = "purpose-tool-calling"
os.environ["LITELLM_URL"] = f"{OPENWEBUI_LAN_URL}/openai"
os.environ["LITELLM_API_KEY"] = jwt
os.environ["ROKID_FAST_MODEL"] = "infomaniak-ministral"
os.environ["ROKID_VISION_MODEL"] = "purpose-vision"
os.environ["ROKID_FAST_MAX_CHARS"] = "120"
os.environ["PHOTO_CACHE_DIR"] = tempfile.mkdtemp(prefix="rokid-photos-real-")
os.environ["PHOTOS_PUBLIC_URL"] = f"http://127.0.0.1:{SHIM_PORT}"
os.environ["LOG_LEVEL"] = "WARNING"
os.environ["PUSH_SHARED_SECRET"] = ""  # not testing push here

import uvicorn  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "shim"))
from app import router as router_mod  # noqa: E402
from app.main import app as shim_app  # noqa: E402


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


async def call_shim(prompt: str, image_url: str | None = None, message_id: str = "t"):
    """Returns (path_actual, ttft_ms, total_ms, answer_text, tool_calls)."""
    items = [{"role": "user", "type": "text", "text": prompt}]
    if image_url:
        items.append({"role": "user", "type": "image", "image_url": image_url})
    payload = {"message_id": message_id, "agent_id": "real-test", "message": items}

    from app.rokid_types import RokidRequest
    actual_path = router_mod.pick_path(RokidRequest.model_validate(payload))

    t0 = time.perf_counter()
    ttft_ms: float | None = None
    answer = ""
    tools: list = []
    err: str | None = None

    try:
        async with httpx.AsyncClient(timeout=60.0) as c:
            async with c.stream(
                "POST",
                f"http://127.0.0.1:{SHIM_PORT}/rokid/agent",
                json=payload,
                headers={"Authorization": "Bearer test-real-ak", "Content-Type": "application/json"},
            ) as r:
                cur_event = None
                async for line in r.aiter_lines():
                    if line.startswith("event:"):
                        cur_event = line.split(":", 1)[1].strip()
                    elif line.startswith("data:"):
                        import json
                        try:
                            data = json.loads(line.split(":", 1)[1].strip())
                        except Exception:
                            continue
                        if data.get("type") == "answer" and data.get("answer_stream"):
                            if ttft_ms is None:
                                ttft_ms = (time.perf_counter() - t0) * 1000
                            answer += data["answer_stream"]
                        elif data.get("type") == "tool_call":
                            tools.append(data.get("tool_call"))
                if r.status_code != 200:
                    err = f"HTTP {r.status_code}"
    except Exception as e:
        err = f"{type(e).__name__}: {e}"

    total_ms = (time.perf_counter() - t0) * 1000
    return {
        "path_expected": actual_path,
        "ttft_ms": ttft_ms,
        "total_ms": total_ms,
        "answer": answer.strip(),
        "tools": tools,
        "error": err,
    }


# Non-mutating prompts only — no email send, calendar create, smart home, note write.
PROMPTS = [
    # Path FAST: short text, no tool keyword
    ("fast.greet",   "Bonjour, comment vas-tu ?", None),
    ("fast.trad-it", "Traduis 'where is the toilet' en italien", None),
    ("fast.math",    "Combien font 17 fois 23 ?", None),
    ("fast.geo",     "Quelle est la capitale du Burkina Faso ?", None),
    ("fast.proverb", "Donne-moi un proverbe chinois en une phrase", None),
    ("fast.trad-sw", "Comment on dit bonjour en swahili ?", None),
    ("fast.def",     "Definition rapide d'entropie en une phrase", None),
    ("fast.timez",   "Quelle heure est-il a Tokyo si il est 14h a Paris ?", None),

    # Path VISION: image_url present. picsum.photos serves random JPEGs without
    # blocking client UAs (Wikimedia 403's the default httpx UA). The seeded
    # variants return the SAME image across calls — deterministic for the test.
    ("vision.cat",   "Que vois-tu sur cette image, en une phrase ?",
        "https://picsum.photos/seed/cat/400/300.jpg"),
    ("vision.landscape", "Decris cette photo en une phrase",
        "https://picsum.photos/seed/landscape/400/300.jpg"),

    # Path FULL but read-only (mail/calendar/search/web — no mutation)
    ("full.long",    "Peux-tu expliquer en deux phrases comment fonctionne la fission nucleaire et a quoi elle sert dans une centrale ?", None),
    ("full.mail",    "Liste mes 3 derniers mails non lus, juste le sujet et l'expediteur", None),
    ("full.cal",     "Quel est mon prochain rdv ?", None),
    ("full.search",  "Cherche les news IA aujourd'hui en une phrase", None),
    ("full.know",    "Cherche dans mes notes ce qui parle de Plexus", None),
]


async def main():
    start_in_thread(shim_app, SHIM_PORT)
    await wait_for(f"http://127.0.0.1:{SHIM_PORT}/health")
    print(f"shim up on :{SHIM_PORT} -> OpenWebUI {OPENWEBUI_LAN_URL}\n")

    rows = []
    for label, prompt, image in PROMPTS:
        print(f"  testing {label!r}... ", end="", flush=True)
        res = await call_shim(prompt, image, message_id=label)
        rows.append((label, prompt, image, res))
        status = "OK" if not res["error"] else f"ERR({res['error']})"
        ttft = f"{res['ttft_ms']:.0f}ms" if res['ttft_ms'] else "—"
        print(f"{status} ttft={ttft} total={res['total_ms']:.0f}ms answer_chars={len(res['answer'])}")

    print()
    print("=" * 110)
    print(f"{'LABEL':<18}{'PATH':<8}{'TTFT':>10}{'TOTAL':>10}  ANSWER (first 200 chars)")
    print("=" * 110)
    for label, prompt, image, res in rows:
        ttft = f"{res['ttft_ms']:.0f}ms" if res['ttft_ms'] else "—"
        total = f"{res['total_ms']:.0f}ms"
        ans = (res['answer'].replace('\n', ' ')[:200] or res['error'] or "(empty)")
        print(f"{label:<18}{res['path_expected']:<8}{ttft:>10}{total:>10}  {ans}")

    # Quick per-path latency summary
    print()
    by_path: dict[str, list[float]] = {}
    for _, _, _, res in rows:
        if res['ttft_ms'] is not None:
            by_path.setdefault(res['path_expected'], []).append(res['ttft_ms'])
    print("Per-path TTFT median (ms):")
    for path, ttfts in sorted(by_path.items()):
        ttfts_sorted = sorted(ttfts)
        med = ttfts_sorted[len(ttfts_sorted) // 2]
        print(f"  {path:<8} n={len(ttfts)}  median={med:.0f}ms  min={min(ttfts):.0f}ms  max={max(ttfts):.0f}ms")


if __name__ == "__main__":
    asyncio.run(main())
