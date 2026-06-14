# aiglasses

Bridges the **Rokid AI Glasses** (Lingzhu / 灵珠 custom-agent slot) to the local
**Plexus stack** (LiteLLM + the MCP tool fleet behind `mcp-hub`, deployed via
`mcp-servers/`). Rokid POSTs an SSE request; the shim routes it, drives the
agent loop directly against LiteLLM (calling MCP tools via `mcp-hub` as
needed), streams the reply back in Rokid's expected event format, and can emit
AR device commands
(take_photo / take_navigation / control_calendar / notify_agent_off).

## Why a shim is needed

Rokid's Lingzhu platform is a fork of ByteDance **Coze Studio**, not
OpenAI-compatible. Its custom-agent contract uses:

- `POST` to your URL, `Authorization: Bearer <AK>`
- Body: `{ message_id, agent_id, message: [{role,type,text|image_url}], metadata }`
- Response: SSE with `event: message` (chunks) and `event: done` (end),
  each carrying `{ role:"agent", message_id, agent_id, answer_stream,
  is_finish, type:"answer"|"tool_call"|"follow_up", ... }`

OpenWebUI speaks plain OpenAI chat-completions; LiteLLM is the OpenAI-compatible
proxy in front of every model (Swiss/Infomaniak, OpenRouter, …). The shim is the
adapter between Rokid's contract and that OpenAI surface.

The Rokid contract is documented inline in
[`shim/app/main.py`](shim/app/main.py) and
[`shim/app/rokid_types.py`](shim/app/rokid_types.py); the upstream source is the
official Rokid Yuque doc
(<https://rokid.yuque.com/ub8h5n/hth52o/qq4gs616xz4ellh1>).

## Architecture

```
Rokid Glasses ──POST /rokid/agent──► rokid-shim (FastAPI, :8024)
                                         │
                                         │  router.pick_path(req)
                                         │     ├─ image in payload   ──► VISION ──► LiteLLM (ROKID_VISION_MODEL)
                                         │     ├─ short + no keyword ──► FAST   ──► LiteLLM (ROKID_FAST_MODEL)
                                         │     └─ otherwise           ──► FULL   ──► agent_loop ──► LiteLLM (ROKID_FULL_MODEL)
                                         │                                                            │
                                         │                                                            └─► mcp-hub tools (mcp_tools.py)
                                         │
                                         ├──► POST /push   (async push back to glasses via Rokid /metis/callback/message)
                                         └──► GET  /photos/{hash}  (serves cached camera frames)
```

All three paths hit **LiteLLM** directly — OpenWebUI is no longer in the
request path. The "full" path runs the agent loop inside the shim
([`shim/app/agent_loop.py`](shim/app/agent_loop.py)): it streams the model's
text, dispatches any `tool_calls` against `mcp-hub`
([`shim/app/mcp_tools.py`](shim/app/mcp_tools.py)), feeds the results back, and
loops until the model stops requesting tools (`max_rounds` cap).

The shim is deployed on the same Proxmox LXC as `mcp-servers/` and joins the
existing `mcp-network` Docker network. Caddy on the LXC routes
`glasses.rohrbach.app` to the shim; Cloudflare Tunnel exposes that hostname
publicly with valid TLS so Rokid accepts it.

### Routing rules

| Trigger | Path | Model | Why |
|---|---|---|---|
| Any `image_url` item in `message[]` | vision | `ROKID_VISION_MODEL` (default `purpose-vision`) | Direct LiteLLM vision call — no tool fanout needed for "what is this?" |
| Last user text ≤ `ROKID_FAST_MAX_CHARS` chars AND no tool-hint keyword | fast | `ROKID_FAST_MODEL` (default `infomaniak-ministral`) | Sub-second TTFT for voice via Swiss-AI |
| Anything else | full | `ROKID_FULL_MODEL` (default `claude-haiku-4-5`) | Agent loop dispatches MCP tools via `mcp-hub` as needed |

Tool-hint keywords are in [`shim/app/router.py`](shim/app/router.py)
(`_TOOL_HINT_KEYWORDS`) — French + English mix covering mail, calendar,
contacts, smart home, knowledge, github, tasks, web search.

### Authorization (per-user allowlist)

The Rokid `ROKID_AK` is a **shared** secret between Lingzhu and the shim, not
per-user: any Rokid user who installs your published agent would reach your MCP
fleet (Gmail, Office, casasmooth) with the same AK. `ROKID_ALLOWED_USER_IDS`
([`shim/app/main.py`](shim/app/main.py), `_check_user_id`) is a comma-separated
allowlist of Rokid `user_id`s permitted to invoke the agent:

- **Empty / unset (default)** = allowlist DISABLED, every authenticated caller
  is allowed. Safe **only** while the agent stays in 草稿/private on Lingzhu.
- Before publishing (提审/发布) you **MUST** populate it. Find your own
  `user_id` after a real glasses call with
  `docker logs rokid-shim | grep INCOMING`.

Rejected callers get `403`; a request with no `user_id` while the allowlist is
enforced also gets `403`.

### MCP tools (mcp-hub profiles)

The full path's tools come from `mcp-hub`'s per-profile OpenAPI surface
([`shim/app/mcp_tools.py`](shim/app/mcp_tools.py)): it fetches
`GET /profiles/<name>/openapi.json` (TTL-cached) and dispatches via
`POST /profiles/<name>/tools/<tool>`. `ROKID_MCP_PROFILES` is a comma-separated
list of profiles to merge into one flat tool list (default
`mail,personal,knowledge`). Available profiles: `mail`, `personal`,
`knowledge`, `dev`, `agents`. (The legacy singular `ROKID_MCP_PROFILE` is still
honored if `ROKID_MCP_PROFILES` is unset.)

### Per-user identity (Plexus principal propagation)

Per-user MCP tools (Gmail, Office, …) don't take a username argument: each
Plexus backend resolves the **caller** from gateway-injected `X-Plexus-*`
headers, then looks up that principal's stored OAuth token in the credential
broker. On the public path a gateway injects those headers from the SSO
session; the shim talks to `mcp-hub` directly, so it must inject them
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
- Configurable retention (`PHOTO_RETENTION_HOURS`, default 48h) and a max size
  guard (`PHOTO_MAX_BYTES`, default 10 MiB).
- When `PHOTO_INLINE_BASE64=true` (default) the cached frame is inlined as a
  `data:` URL in the LLM request, so cloud backends (e.g. OpenRouter) that
  can't reach internal URLs still see the image; the disk copy is kept for
  downstream tool re-use. Set it to `false` to send `${PHOTOS_PUBLIC_URL}`
  links instead (only viable when that URL is internet-reachable).

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
    ├── main.py              FastAPI app — /rokid/agent + /push + /photos/{name} + /health, AK + allowlist auth
    ├── rokid_types.py       Pydantic models for the Rokid contract
    ├── translator.py        Rokid message[] -> OpenAI messages[] + system prompt
    ├── router.py            Decide fast / vision / full per request
    ├── agent_loop.py        FULL path: multi-round LLM ⇄ MCP-tool loop, streams text
    ├── mcp_tools.py         Fetch tool defs from mcp-hub profiles + dispatch tool calls
    ├── litellm_client.py    Async streaming client for LiteLLM (all 3 paths)
    ├── identity.py          Map Rokid user_id -> Plexus principal, emit X-Plexus-* headers
    ├── photo_cache.py       Download Rokid camera frames, expose locally / inline base64
    ├── rokid_callback.py    POST to Rokid /metis/callback/message
    └── tool_extractor.py    Parse fenced ```rokid_action JSON from LLM output

deploy/
└── glasses_hostname.caddyfile   Caddy site block, drop into mcp-servers/deploy-local/local-caddy/sites/

docker-compose.yml          Joins mcp-network external, exposes :8024, photo_cache volume
.env.example                ROKID_*, LITELLM_*, MCP_HUB_*, PHOTO_*, PUSH_*
```

## Deploy (LXC 500)

Assumes `mcp-servers/` is already running on LXC 500 with the
`mcp-network` Docker network up and Caddy fronting it.

1. **On your dev machine** — generate the AK and fill `.env`:

   ```bash
   cp .env.example .env
   # ROKID_AK: a long random string (paste this back into Lingzhu later)
   openssl rand -hex 24
   # LITELLM_API_KEY: the key your LiteLLM proxy accepts (sk-anything if open)
   # ROKID_PRINCIPAL_EMAIL: the email you connected Gmail/Office with in Plexus
   # Before publishing: set ROKID_ALLOWED_USER_IDS to your own Rokid user_id
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
   # {"status":"ok","mcp_profile":"mail,personal,knowledge","mcp_tool_count":117,"mcp_tool_error":null}

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
(LiteLLM with a multi-round agent, mcp-hub serving an OpenAPI spec + tool
dispatch, Rokid callback, and a fake camera-frame CDN), runs the shim against
them, and asserts the full v3 flow:

1. Short text → fast path hits LiteLLM with the fast model
2. Image present → vision path hits LiteLLM with the vision model, and the
   image URL is rewritten / inlined from the local `/photos/<sha>` cache
3. Tool-keyword text → full path runs the agent loop (LiteLLM ⇄ mcp-hub tool
   dispatch over multiple rounds)
4. Fenced `rokid_action` extraction works end-to-end and is emitted as a
   `tool_call` SSE event
5. Bad AK is rejected (401)
6. `POST /push` proxies correctly to Rokid's callback with the sk token, and is
   rejected (401) without the shared secret
7. `ROKID_ALLOWED_USER_IDS` allowlist: disabled = all allowed; enforced =
   listed user 200, unlisted/missing user 403
8. Plexus principal resolution: `ROKID_USER_PRINCIPAL_MAP` maps a user to their
   own principal, otherwise `ROKID_PRINCIPAL_EMAIL` is the default

```bash
python integration_test.py
# ALL TESTS PASSED
```

It uses only the deps already in `shim/requirements.txt` — no extra install.
