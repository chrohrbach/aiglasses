# aiglasses

Bridges the **Rokid AI Glasses** (Lingzhu / 灵珠 custom-agent slot) to the local
**Plexus stack** (OpenWebUI + 12 MCP tool servers, deployed via
`mcp-servers/`). Rokid POSTs an SSE request; the shim translates to
OpenWebUI's OpenAI-compatible chat-completions, streams the reply back in
Rokid's expected event format, and can emit AR device commands
(take_photo / take_navigation / control_calendar / notify_agent_off).

## Why a shim is needed

Rokid's Lingzhu platform is a fork of ByteDance **Coze Studio**, not
OpenAI-compatible. Its custom-agent contract uses:

- `POST` to your URL, `Authorization: Bearer <AK>`
- Body: `{ message_id, agent_id, message: [{role,type,text|image_url}], metadata }`
- Response: SSE with `event: message` (chunks) and `event: done` (end),
  each carrying `{ role:"agent", message_id, agent_id, answer_stream,
  is_finish, type:"answer"|"tool_call"|"follow_up", ... }`

OpenWebUI speaks plain OpenAI chat-completions. The shim is the adapter.

Full spec extracted from the official Rokid Yuque doc and saved at
`docs/rokid-spec.md` (and in Claude memory for cross-session reference).

## Architecture

```
Rokid Glasses ──POST /rokid/agent──► rokid-shim (FastAPI, :8024)
                                         │
                                         │  router.pick_path(req)
                                         │     ├─ image in payload   ──► VISION  ──► LiteLLM (purpose-vision)
                                         │     ├─ short + no keyword ──► FAST    ──► LiteLLM (infomaniak-ministral)
                                         │     └─ otherwise           ──► FULL    ──► OpenWebUI
                                         │                                            │
                                         │                                            └─► 12 MCP tool servers
                                         │
                                         ├──► POST /push   (async push back to glasses via Rokid /metis/callback/message)
                                         └──► GET  /photos/{hash}  (serves cached camera frames)
```

The shim is deployed on the same Proxmox LXC as `mcp-servers/` and joins the
existing `mcp-network` Docker network. Caddy on the LXC routes
`glasses.rohrbach.app` to the shim; Cloudflare Tunnel exposes that hostname
publicly with valid TLS so Rokid accepts it.

### Routing rules

| Trigger | Path | Model | Why |
|---|---|---|---|
| Any `image_url` item in `message[]` | vision | `ROKID_VISION_MODEL` (default `purpose-vision`) | OpenWebUI multimodal tool fanout is overkill for "what is this?" — direct LiteLLM is faster |
| Last user text ≤ `ROKID_FAST_MAX_CHARS` chars AND no tool-hint keyword | fast | `ROKID_FAST_MODEL` (default `infomaniak-ministral`) | Sub-second TTFT for voice via Swiss-AI |
| Anything else | full | `OPENWEBUI_MODEL` (default `purpose-tool-calling` → `infomaniak-gemma4`) | OpenWebUI dispatches MCP tools as needed |

Tool-hint keywords are in [`shim/app/router.py`](shim/app/router.py)
(`_TOOL_HINT_KEYWORDS`) — French + English mix covering mail, calendar,
contacts, smart home, knowledge, github, tasks, web search.

### Per-user identity (Plexus principal propagation)

Per-user MCP tools (Gmail, Office, …) don't take a username argument: each
Plexus backend resolves the **caller** from gateway-injected `X-Plexus-*`
headers, then looks up that principal's stored OAuth token in the credential
broker. On the public OpenWebUI path the gateway injects those headers from the
SSO session; the shim talks to `mcp-hub` directly, so it must inject them
itself — otherwise the backend sees `caller email = None` and replies
"connect your account".

[`shim/app/identity.py`](shim/app/identity.py) maps the Rokid `user_id` to a
Plexus principal and emits three headers on every `mcp-hub` request (both the
OpenAPI fetch and the tool dispatch):

| Header | Source | Notes |
|---|---|---|
| `X-Plexus-Principal` | `ROKID_PRINCIPAL_SUB`, else the email | **Required** — Plexus returns no principal without it |
| `X-Plexus-Principal-Type` | `ROKID_PRINCIPAL_TYPE` (default `user`) | |
| `X-Plexus-Email` | `ROKID_PRINCIPAL_EMAIL` | Must equal the email the account was connected with in Plexus |

Configuration (see [`.env.example`](.env.example)):

- `ROKID_PRINCIPAL_EMAIL` — default principal for all glasses (single-owner setup).
- `ROKID_USER_PRINCIPAL_MAP` — optional JSON `{"<rokid_user_id>": "<email>"}`
  to map individual glasses to different Plexus accounts.

If neither is set the shim logs a warning and per-user tools run without a
caller identity (unchanged legacy behaviour).

### Photo cache

Rokid serves camera frames from its own CDN with short-lived URLs. The shim
downloads each `image_url` once into a persistent volume and rewrites the
in-memory request so the model sees `${PHOTOS_PUBLIC_URL}/photos/<sha>.<ext>`
instead. Benefits:

- Photos remain reachable for later turns and downstream tools
  (e.g. attach to email via mcp-gmail).
- The model fetches from inside `mcp-network` rather than out to Rokid,
  saving a hop.
- Configurable retention (`PHOTO_RETENTION_HOURS`, default 48h).

### Async push

`POST /push` on the shim forwards to Rokid's `/metis/callback/message`
endpoint, letting any in-cluster service (n8n, mcp-tasks, custom cron)
push a proactive message to the glasses. Auth via `X-Push-Secret`
header (configurable in `.env`). The `sk` token used against Rokid is
the `ROKID_SK_TOKEN` generated once at
<https://account-web.rokid.com/token>.

Example trigger from inside the LXC:
```bash
curl -X POST http://rokid-shim:8000/push \
  -H "X-Push-Secret: $PUSH_SHARED_SECRET" \
  -H "Content-Type: application/json" \
  -d '{"account_id":"<rokid-user-id>","agent_id":"plexus-glasses","content":"Ton train part dans 5 min."}'
```

## Files

```
shim/
├── Dockerfile
├── requirements.txt
└── app/
    ├── main.py              FastAPI app — /rokid/agent + /push + /photos/{name} + /health
    ├── rokid_types.py       Pydantic models for the Rokid contract
    ├── translator.py        Rokid message[] -> OpenAI messages[] + system prompt
    ├── router.py            Decide fast / vision / full per request
    ├── openwebui_client.py  Async streaming client for the "full" path
    ├── litellm_client.py    Async streaming client for the "fast" + "vision" paths
    ├── photo_cache.py       Download Rokid camera frames, expose locally
    ├── rokid_callback.py    POST to Rokid /metis/callback/message
    └── tool_extractor.py    Parse fenced ```rokid_action JSON from LLM output

deploy/
└── glasses_hostname.caddyfile   Caddy site block, drop into mcp-servers/deploy-local/local-caddy/sites/

docker-compose.yml          Joins mcp-network external, exposes :8024, photo_cache volume
.env.example                ROKID_*, OPENWEBUI_*, LITELLM_*, PHOTOS_*, PUSH_*
```

## Deploy (LXC 500)

Assumes `mcp-servers/` is already running on LXC 500 with the
`mcp-network` Docker network up and Caddy fronting it.

1. **On your dev machine** — generate the AK and fill `.env`:

   ```bash
   cp .env.example .env
   # ROKID_AK: a long random string (paste this back into Lingzhu later)
   openssl rand -hex 24
   # OPENWEBUI_API_KEY: generated in OpenWebUI → Settings → Account → API Key
   ```

2. **Ship the repo to the LXC** — same pattern as the rest of `mcp-servers/`
   (tar → sftp to Proxmox host → `pct push 500` → `pct exec 500 -- tar xzf`).
   Target path: `/opt/aiglasses/`.

3. **Drop the Caddy snippet** into the existing mount:

   ```bash
   cp /opt/aiglasses/deploy/glasses_hostname.caddyfile \
      /opt/mcp-servers/deploy-local/local-caddy/sites/
   docker restart caddy
   ```

4. **Add a Cloudflare Tunnel entry** in the CF dashboard:
   `glasses.rohrbach.app` → `http://<LXC-500-IP>:80`.

5. **Bring up the shim**:

   ```bash
   cd /opt/aiglasses
   docker compose --env-file .env up -d --build
   ```

6. **Smoke test from the LXC**:

   ```bash
   curl https://glasses.rohrbach.app/health
   # {"status":"ok"}

   curl -N -X POST https://glasses.rohrbach.app/rokid/agent \
     -H "Authorization: Bearer $ROKID_AK" \
     -H "Content-Type: application/json" \
     -d '{"message_id":"t1","agent_id":"dev","message":[{"role":"user","type":"text","text":"Bonjour, qui es-tu ?"}]}'
   # Should stream: event: message ... event: done
   ```

## Register on Lingzhu

1. Sign in at <https://rizon.rokid.com/> with the Rokid account that owns
   the glasses.
2. Project Development → Third-party Agents → Create.
3. Fill in:
   - **三方厂商** (vendor): 自定义 (custom)
   - **智能体ID** (agent ID): any string you want, e.g. `plexus-glasses`
   - **智能体 SSE 接口地址** (SSE endpoint): `https://glasses.rohrbach.app/rokid/agent`
   - **智能体鉴权 AK**: the `ROKID_AK` value from your `.env`
   - **入参类型** (input types): text + image (enable image if you want
     the camera frames to reach the model)
   - Name, description, opening line, icon: as you like
4. Save and test from the Lingzhu debug pane, then publish to your glasses.

## Device commands (AR actions)

The system prompt teaches the LLM to emit a fenced JSON action block at
the end of any reply that should trigger a device action. Supported
commands (per Rokid spec):

| Command | Example fence body |
|---|---|
| Take a photo | `{"command":"take_photo"}` |
| Open navigation | `{"command":"take_navigation","action":"open","poi_name":"Bahnhofstrasse 1, Zürich","navi_type":"0"}` |
| Create calendar event | `{"command":"control_calendar","action":"create","title":"Dentist","start_time":"2026-05-22T09:00:00","end_time":"2026-05-22T09:30:00"}` |
| Exit the agent | `{"command":"notify_agent_off"}` |

The shim strips the fence from the streamed text (the user doesn't hear
the JSON spoken aloud) and emits a separate `type:"tool_call"` SSE event
which Rokid then dispatches to the glasses.

## Local dev (without Docker)

```bash
cd shim
python -m venv .venv
source .venv/bin/activate  # or .venv/Scripts/activate on Windows
pip install -r requirements.txt
export $(grep -v '^#' ../.env | xargs)
uvicorn app.main:app --reload --port 8024
```

## Integration test

`integration_test.py` at the repo root spins up four mock servers in-process
(OpenWebUI, LiteLLM, Rokid callback, and a fake camera-frame CDN), runs the
shim against them, and asserts the full v2 flow:

1. Short text → fast path hits LiteLLM with the fast model
2. Image present → vision path hits LiteLLM with the vision model, and the
   image URL is rewritten to the local `/photos/<sha>` cache
3. Tool-keyword text → full path hits OpenWebUI
4. Long text without keywords → full path
5. Fenced `rokid_action` extraction works end-to-end on the full path
6. Bad AK is rejected (401)
7. `POST /push` proxies correctly to Rokid's callback with the sk token
8. `POST /push` without the shared secret is rejected (401)

```bash
python integration_test.py
# ALL TESTS PASSED
```

It uses only the deps already in `shim/requirements.txt` — no extra install.
