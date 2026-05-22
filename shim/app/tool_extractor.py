"""Extract `rokid_action` fenced JSON blocks from the streaming LLM output.

System prompt instructs the LLM to emit AT MOST ONE fenced block at the END
of its reply. The extractor:

  - emits the text *before* the fence as normal `answer_stream` chunks
  - swallows the fence itself (never sent to the user as text)
  - emits the parsed JSON as a separate `tool_call` event after the answer is done
"""

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass

FENCE_OPEN = "```rokid_action"
FENCE_CLOSE = "```"


@dataclass
class StreamPart:
    """One output unit from the extractor."""

    kind: str  # "text" or "tool_call"
    text: str | None = None
    tool_call: dict | None = None


async def split_stream(chunks: AsyncIterator[str]) -> AsyncIterator[StreamPart]:
    """Read text chunks, yield text parts + at most one tool_call.

    Uses a small accumulator so a fence marker split across chunk boundaries
    is still detected.
    """
    buf = ""
    inside_fence = False
    fence_body = ""

    # Once any fence is seen we stop emitting plain text — even text that
    # arrives after the closing fence is suppressed (the LLM was asked to
    # put the fence at the very end).
    fence_seen = False

    async for chunk in chunks:
        if fence_seen and not inside_fence:
            # Trailing text after the action — drop it.
            continue

        buf += chunk

        # Loop in case multiple state transitions live inside one buf.
        while True:
            if not inside_fence:
                idx = buf.find(FENCE_OPEN)
                if idx == -1:
                    # No fence start in buf. Emit all text EXCEPT the trailing
                    # window that might be the prefix of a fence marker (so we
                    # don't leak a half "```rokid_act" into the user output).
                    safe_len = max(0, len(buf) - len(FENCE_OPEN) + 1)
                    if safe_len:
                        yield StreamPart(kind="text", text=buf[:safe_len])
                        buf = buf[safe_len:]
                    break  # need more chunks
                # Emit text before the fence
                if idx > 0:
                    yield StreamPart(kind="text", text=buf[:idx])
                buf = buf[idx + len(FENCE_OPEN) :]
                inside_fence = True
                fence_seen = True
                fence_body = ""
                continue
            else:
                idx = buf.find(FENCE_CLOSE)
                if idx == -1:
                    fence_body += buf
                    buf = ""
                    break
                fence_body += buf[:idx]
                buf = buf[idx + len(FENCE_CLOSE) :]
                inside_fence = False
                tool_json = _parse_json_relaxed(fence_body)
                if tool_json is not None:
                    yield StreamPart(kind="tool_call", tool_call=tool_json)
                fence_body = ""
                # Don't continue parsing buf — anything after the close fence
                # is post-action chatter to be dropped.
                break

    # End of stream — flush any pending safe text
    if not fence_seen and buf:
        yield StreamPart(kind="text", text=buf)


def _parse_json_relaxed(s: str) -> dict | None:
    s = s.strip()
    # Sometimes the LLM puts a language tag or newline right after the fence
    # open even though the prompt said no tag. Strip a leading word.
    if s and s[0].isalpha():
        nl = s.find("\n")
        if nl != -1:
            s = s[nl + 1 :].strip()
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        return None
    if isinstance(obj, dict) and "command" in obj:
        return obj
    return None
