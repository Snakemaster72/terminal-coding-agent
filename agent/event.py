from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from client.response import TokenUsage
from tools.base import ToolResult


class AgentEventType(str, Enum):
    # agent lifecycle
    AGENT_START = "agent_start"
    AGENT_END = "agent_end"
    AGENT_ERROR = "agent_error"

    # text
    TEXT_DELTA = "text_delta"
    TEXT_COMPLETE = "text_complete"

    # agentevent: more details than streameventy

    # tool call
    TOOL_CALL_START = "tool_call_start"
    TOOL_CALL_COMPLETE = "tool_call_complete"

    # context management
    CONTEXT_PRUNED = "context_pruned"
    CONTEXT_COMPACTED = "context_compacted"
    MAX_TURNS_REACHED = "max_turns_reached"


@dataclass
class AgentEvent:
    type: AgentEventType
    data: dict[str, Any] = field(default_factory=dict)

    # cls is just self, no need to think
    @classmethod
    def agent_start(cls, message: str) -> AgentEvent:
        return cls(type=AgentEventType.AGENT_START, data={"messges": message})

    @classmethod
    def agent_end(
        cls, response: str | None = None, usage: TokenUsage | None = None
    ) -> AgentEvent:
        return cls(
            type=AgentEventType.AGENT_END,
            data={"response": response, "usage": usage.__dict__ if usage else None},
        )

    @classmethod
    def agent_error(
        cls, error: str, details: dict[str, Any] | None = None
    ) -> AgentEvent:
        return cls(
            type=AgentEventType.AGENT_ERROR,
            data={"error": error, "details": details or {}},
        )

    @classmethod
    def context_pruned(cls, reclaimed: int, total: int) -> AgentEvent:
        return cls(
            type=AgentEventType.CONTEXT_PRUNED,
            data={"reclaimed_tokens": reclaimed, "total_tokens": total},
        )

    @classmethod
    def context_compacted(cls, before: int, after: int) -> AgentEvent:
        return cls(
            type=AgentEventType.CONTEXT_COMPACTED,
            data={"before_tokens": before, "after_tokens": after},
        )

    @classmethod
    def max_turns_reached(cls, max_turns: int) -> AgentEvent:
        return cls(
            type=AgentEventType.MAX_TURNS_REACHED,
            data={"max_turns": max_turns},
        )

    @classmethod
    def text_delta(cls, content: str) -> AgentEvent:
        return cls(
            type=AgentEventType.TEXT_DELTA,
            data={"content": content},
        )

    @classmethod
    def text_complete(cls, content: str) -> AgentEvent:
        return cls(
            type=AgentEventType.TEXT_COMPLETE,
            data={"content": content},
        )

    @classmethod
    def tool_call_start(
        cls, call_id: str, name: str, arguments: dict[str, str]
    ) -> dict[str, Any]:
        return cls(
            type=AgentEventType.TOOL_CALL_START,
            data={"call_id": call_id, "name": name, "arguments": arguments},
        )

    @classmethod
    def tool_call_complete(cls, call_id: str, name: str, result: ToolResult):
        return cls(
            type=AgentEventType.TOOL_CALL_COMPLETE,
            data={
                "call_id": call_id,
                "name": name,
                "success": result.success,
                "output": result.output,
                "metadata": result.metadata,
                "truncated": result.truncated,
                "error": result.error,
                "diff": result.diff.create_diff() if result.diff else None,
                "exit_code": result.exit_code,
            },
        )
