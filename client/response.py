from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

# we are defining a model schema for our messages, to receive structured messagesy


@dataclass
class TextDelta:
    content: str

    def __str__(self):
        return self.content


# text delta: any text llm gives you
# first set of tokens are text details
# message complete
# error
class StreamEventType(str, Enum):
    TEXT_DELTA = "text_delta"
    MESSAGE_COMPLETE = "message_complete"
    ERROR = "error"

    TOOL_CALL_START = "tool_call_start"
    TOOL_CALL_DELTA = "tool_call_delta"
    TOOL_CALL_COMPLETE = "tool_call_complete"


# Important as we can use it to determine whether to context compact or not, very IMportant**
@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0

    # a.__add_(b)
    def __add__(self, other: TokenUsage):
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            cached_tokens=self.cached_tokens + other.cached_tokens,
        )


@dataclass
class ToolCallDelta:
    call_id: str
    name: str | None = None
    arguments_delta: str = ""


@dataclass
class ToolCall:
    call_id: str
    name: str | None = None
    arguments: str = ""


@dataclass
class StreamEvent:
    type: StreamEventType
    text_delta: TextDelta | None = None
    error: str | None = None
    finish_reason: str | None = None
    tool_call_delta: ToolCallDelta | None = None
    tool_call: ToolCall | None = None
    usage: TokenUsage | None = None


@dataclass
class ToolResultMessage:
    tool_call_id: str
    content: str
    is_error: bool = False

    def to_openai_message(self) -> dict[str, Any]:
        return {
            "role": "tool",
            "content": self.content,
            "tool_call_id": self.tool_call_id,
            "is_error": str(self.is_error),
        }


def parse_tool_call_arguments(arguments_str: str) -> dict[str, str]:
    """
    Parse the tool call arguments from a string to a dictionary.
    The arguments are expected to be in the format of a JSON object.
    """
    import json

    if not arguments_str:
        return {}
    try:
        return json.loads(arguments_str)
    except json.JSONDecodeError:
        return {"raw_arguments": arguments_str}
