"""Drive a multi-turn LLM ⇄ MCP-tool conversation, streaming the user-visible text.

The shim's FULL path enters this loop instead of calling the model once. Each
iteration streams whatever text the model produces (so the user hears
"checking your inbox…" immediately) while collecting any tool_call deltas in
the background. When the round ends:

  * If the model emitted tool_calls, dispatch them via the McpToolCatalog,
    append the assistant message + tool results to the conversation, and
    loop.
  * Otherwise this was the final round — the answer was already streamed.

A safety cap of `max_rounds` prevents runaway loops.
"""

import json
import logging
from collections.abc import AsyncIterator

from .litellm_client import Finish, TextDelta, ToolCallDelta, stream_events
from .mcp_tools import McpToolCatalog

logger = logging.getLogger(__name__)


def _assemble_tool_calls(deltas: list[ToolCallDelta]) -> list[dict]:
    """Merge streaming tool_call deltas into complete {id,name,arguments} dicts."""
    by_index: dict[int, dict] = {}
    for d in deltas:
        slot = by_index.setdefault(d.index, {"id": None, "name": None, "arguments": ""})
        if d.id:
            slot["id"] = d.id
        if d.name:
            slot["name"] = d.name
        if d.arguments_delta:
            slot["arguments"] += d.arguments_delta
    return [by_index[i] for i in sorted(by_index)]


async def stream_with_tools(
    messages: list[dict],
    *,
    model: str,
    catalog: McpToolCatalog,
    max_rounds: int = 4,
) -> AsyncIterator[str]:
    """Yield text deltas across N agent rounds, invoking MCP tools as needed."""
    tools = await catalog.get_tools()
    if not tools:
        logger.warning("no tools fetched from %s, falling back to plain chat", catalog.profile)

    convo = list(messages)

    for round_idx in range(max_rounds):
        tc_deltas: list[ToolCallDelta] = []
        any_text = False
        try:
            async for ev in stream_events(convo, model=model, tools=tools or None):
                if isinstance(ev, TextDelta):
                    any_text = True
                    yield ev.text
                elif isinstance(ev, ToolCallDelta):
                    tc_deltas.append(ev)
                elif isinstance(ev, Finish):
                    break
        except Exception as e:
            logger.exception("agent_loop round %d upstream error", round_idx)
            yield f"\n[shim error: {type(e).__name__}]"
            return

        if not tc_deltas:
            return  # final round, model returned text only

        tool_calls = _assemble_tool_calls(tc_deltas)
        assistant_msg = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": tc["id"] or f"call_{round_idx}_{i}",
                    "type": "function",
                    "function": {"name": tc["name"] or "", "arguments": tc["arguments"] or "{}"},
                }
                for i, tc in enumerate(tool_calls)
            ],
        }
        convo.append(assistant_msg)
        logger.info("round %d tool_calls=%s", round_idx, [tc["name"] for tc in tool_calls])

        # Dispatch each tool call and append its result as a tool message.
        for assistant_tc, raw in zip(assistant_msg["tool_calls"], tool_calls, strict=True):
            name = raw["name"] or ""
            try:
                args = json.loads(raw["arguments"]) if raw["arguments"] else {}
                if not isinstance(args, dict):
                    args = {}
            except json.JSONDecodeError:
                args = {}
                logger.warning("round %d tool %s sent malformed JSON args: %r", round_idx, name, raw["arguments"])
            result = await catalog.dispatch(name, args)
            convo.append({
                "role": "tool",
                "tool_call_id": assistant_tc["id"],
                "content": result,
            })
        # loop into next round to let the model react to the tool results

    # Hit the round cap — emit one apology so the user isn't stuck silent.
    logger.warning("agent_loop hit max_rounds=%d", max_rounds)
    yield "\n[The assistant reached its tool-call limit before finishing.]"
