"""Pydantic models for the Rokid Lingzhu custom-agent SSE contract.

Source of truth: https://rokid.yuque.com/ub8h5n/hth52o/qq4gs616xz4ellh1
"""

from typing import Literal

from pydantic import BaseModel, Field


class RokidMessageItem(BaseModel):
    role: Literal["user", "agent"]
    type: Literal["text", "image"]
    text: str | None = None
    image_url: str | None = None


class RokidContext(BaseModel):
    location: str | None = None
    latitude: str | None = None
    longitude: str | None = None
    weather: str | None = None
    battery: str | None = None


class RokidMetadata(BaseModel):
    context: RokidContext | None = None


class RokidRequest(BaseModel):
    message_id: str
    agent_id: str
    message: list[RokidMessageItem]
    user_id: str | None = None
    metadata: RokidMetadata | None = None


class RokidToolCall(BaseModel):
    command: Literal[
        "take_photo",
        "take_navigation",
        "notify_agent_off",
        "control_calendar",
    ]
    action: str | None = None
    poi_name: str | None = None
    navi_type: str | None = None
    title: str | None = None
    start_time: str | None = None
    end_time: str | None = None


class RokidEventPayload(BaseModel):
    """JSON body of a Rokid SSE event (data: ...)."""

    role: Literal["agent"] = "agent"
    message_id: str
    agent_id: str
    is_finish: bool
    type: Literal["answer", "tool_call", "follow_up"]
    answer_stream: str | None = None
    follow_up: list[str] | None = None
    tool_call: RokidToolCall | None = None

    def to_sse(self, event: Literal["message", "done"]) -> str:
        body = self.model_dump_json(exclude_none=True)
        return f"event: {event}\ndata: {body}\n\n"
