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
                                          │ POST /api/chat/completions (stream)
                                          ▼
                                     OpenWebUI ──► LiteLLM ──► Infomaniak / OpenRouter
                                          │
                                          └──► 12 MCP tool servers (Office, Gmail, Knowledge, ...)
```

The shim is deployed on the same Proxmox LXC as `mcp-servers/` and joins the
existing `mcp-network` Docker network. Caddy on the LXC routes
`glasses.rohrbach.app` to the shim; Cloudflare Tunnel exposes that hostname
publicly with valid TLS so Rokid accepts it.

## Files

```
shim/
├── Dockerfile
├── requirements.txt
└── app/
    ├── main.py              FastAPI app, /rokid/agent + /health
    ├── rokid_types.py       Pydantic models for the Rokid contract
    ├── translator.py        Rokid message[] → OpenAI messages[] + system prompt
    ├── openwebui_client.py  Async streaming client for OpenWebUI
    └── tool_extractor.py    Parse fenced ```rokid_action JSON from LLM output

deploy/
└── glasses_hostname.caddyfile   Caddy site block, drop into mcp-servers/deploy-local/local-caddy/sites/

docker-compose.yml          Joins mcp-network external, exposes :8024
.env.example                ROKID_AK, OPENWEBUI_*, MODEL
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

`integration_test.py` at the repo root spins up an in-process mock OpenWebUI,
runs the shim against it, and asserts the SSE flow (plain text, photo
tool_call, multimodal image input, bad-AK rejection, device-context
injection). Run it after any change to the shim:

```bash
python integration_test.py
# ALL TESTS PASSED
```

It uses only the deps already in `shim/requirements.txt` — no extra install.
