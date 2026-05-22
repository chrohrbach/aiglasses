"""Translate between Rokid Lingzhu shape and OpenAI Chat Completions shape."""

from .rokid_types import RokidContext, RokidRequest

SYSTEM_PROMPT = """You are a wearable AR-glasses assistant. Reply in the user's language.

ABSOLUTE RULES — these override everything else:
- NEVER FABRICATE DATA. If you do not have access to a tool that can answer the user's request, say so honestly in one sentence. Do NOT invent emails, calendar events, contacts, notes, sensor readings, search results, or any other personal data.
- If a tool you tried returned an error, report it clearly ("the mail tool is not available right now") and stop. Do not paper over the failure with a plausible-sounding made-up answer.
- When you don't know something, say "I don't know" rather than guessing.

Style constraints (apply once the truth constraints above are satisfied):
- Keep responses short and spoken (≤2 sentences) — the user hears them via TTS while wearing glasses.
- Never read out URLs, raw JSON, or markdown formatting.
- If the user's request maps to one of the device actions below, emit the action JSON in a fenced block at the very END of your reply, after a short spoken confirmation.

Device actions (emit AT MOST ONE per turn):
```rokid_action
{"command": "take_photo"}
```
```rokid_action
{"command": "take_navigation", "action": "open", "poi_name": "<address>", "navi_type": "0"}
```
(navi_type: 0=car, 1=walk, 2=bike)
```rokid_action
{"command": "control_calendar", "action": "create", "title": "<title>", "start_time": "YYYY-MM-DDTHH:MM:SS", "end_time": "YYYY-MM-DDTHH:MM:SS"}
```
```rokid_action
{"command": "notify_agent_off"}
```

Only emit an action when the user clearly asked for it. Otherwise just answer.
"""


def _context_to_system_suffix(ctx: RokidContext | None) -> str:
    if ctx is None:
        return ""
    bits = []
    if ctx.location:
        bits.append(f"Location: {ctx.location}")
    if ctx.latitude and ctx.longitude:
        bits.append(f"Coords: {ctx.latitude},{ctx.longitude}")
    if ctx.weather:
        bits.append(f"Weather: {ctx.weather}")
    if ctx.battery:
        bits.append(f"Battery: {ctx.battery}%")
    if not bits:
        return ""
    return "\n\nCurrent device context: " + " | ".join(bits)


def rokid_to_openai_messages(req: RokidRequest) -> list[dict]:
    """Convert Rokid `message[]` + metadata into an OpenAI `messages[]` list.

    Adjacent text+image items from the same role are merged into a single
    multimodal message with `content` as a list of parts.
    """
    system_text = SYSTEM_PROMPT + _context_to_system_suffix(
        req.metadata.context if req.metadata else None
    )
    out: list[dict] = [{"role": "system", "content": system_text}]

    current_role: str | None = None
    current_parts: list[dict] = []

    def flush():
        if not current_parts:
            return
        if len(current_parts) == 1 and current_parts[0]["type"] == "text":
            content: str | list = current_parts[0]["text"]
        else:
            content = current_parts
        out.append({"role": current_role, "content": content})

    for item in req.message:
        oai_role = "assistant" if item.role == "agent" else "user"
        if oai_role != current_role:
            flush()
            current_parts = []
            current_role = oai_role
        if item.type == "text" and item.text:
            current_parts.append({"type": "text", "text": item.text})
        elif item.type == "image" and item.image_url:
            current_parts.append(
                {"type": "image_url", "image_url": {"url": item.image_url}}
            )
    flush()
    return out
