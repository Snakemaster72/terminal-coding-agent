from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from config.config import Config
from prompts.system import get_system_prompt
from utils.text import count_tokens

if TYPE_CHECKING:
    from tools.base import Tool


@dataclass
class MessageItem:
    role: str
    content: str
    token_count: int | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    is_error: bool = False
    pruned: bool = False

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"role": self.role}

        if self.tool_call_id:
            result["tool_call_id"] = self.tool_call_id

        if self.tool_calls:
            result["tool_calls"] = self.tool_calls

        # assistant messages may legitimately have empty content (tool-call-only
        # turns), but "tool" and "user" messages must always carry a content key
        # or the API rejects/stalls on the malformed message
        if self.content or self.role in ("tool", "user"):
            result["content"] = self.content
        elif self.role == "assistant":
            result["content"] = None

        return result


class ContextManager:
    def __init__(
        self,
        config: Config,
        tools: list[Tool] | None = None,
        user_memory: str | None = None,
    ) -> None:
        self.config = config
        # the prompt describes the tools the model actually has; without this the
        # "Available Tools" section renders as "you have NO tools available"
        self._system_prompt = get_system_prompt(
            config=config, user_memory=user_memory, tools=tools
        )
        self._model_name = self.config.model_name
        self._messages: list[MessageItem] = []
        # the system prompt is rebuilt only on a new session, so its cost is
        # fixed for the lifetime of this manager
        self._system_tokens = count_tokens(self._system_prompt, self._model_name)

    def add_user_message(self, content: str) -> None:
        item = MessageItem(
            role="user",
            content=content,
            token_count=count_tokens(content, self._model_name),
        )

        self._messages.append(item)

    def add_assistant_message(
        self, content: str, tool_calls: list[dict[str, Any]] | None = None
    ) -> None:
        item = MessageItem(
            role="assistant",
            content=content or "",
            token_count=count_tokens(content or "", self._model_name),
            tool_calls=tool_calls or [],
        )

        self._messages.append(item)

    # this is a new method to add tool result messages to the context manager, which will be used to keep track of the results of tool calls made by the agent.
    def add_tool_result(self, tool_call_id: str, content: str, is_error: bool) -> None:
        item = MessageItem(
            role="tool",
            content=content,
            tool_call_id=tool_call_id,
            token_count=count_tokens(content, self._model_name),
            is_error=is_error,
        )
        self._messages.append(item)

    def total_tokens(self) -> int:
        """Approximate size of the next request, system prompt included."""
        return self._system_tokens + sum(m.token_count or 0 for m in self._messages)

    def usage_ratio(self) -> float:
        window = self.config.context_window
        if window <= 0:
            return 0.0
        return self.total_tokens() / window

    def should_compact(self) -> bool:
        return self.usage_ratio() >= self.config.compaction_threshold

    def prune_stale_tool_outputs(self) -> int:
        """Sliding-window pruning.

        A tool result is *stale* once `tool_output_window` newer tool results
        exist. Stale results keep their `tool_call_id` - dropping the message
        outright would orphan the assistant tool_call that references it and
        the API would reject the request - but their body is replaced with a
        placeholder. Small results are left alone; they cost less than the
        placeholder is worth. Returns the number of tokens reclaimed.
        """
        window = self.config.tool_output_window
        min_chars = self.config.prune_min_chars

        tool_indices = [i for i, m in enumerate(self._messages) if m.role == "tool"]
        if window and len(tool_indices) <= window:
            return 0

        stale = tool_indices[:-window] if window else tool_indices
        reclaimed = 0

        for i in stale:
            item = self._messages[i]
            if item.pruned or len(item.content) < min_chars:
                continue

            before = item.token_count or 0
            marker = "failed tool output" if item.is_error else "tool output"
            item.content = (
                f"[{marker} pruned to reclaim context: "
                f"{len(item.content)} chars omitted]"
            )
            item.token_count = count_tokens(item.content, self._model_name)
            item.pruned = True
            reclaimed += before - item.token_count

        return reclaimed

    def replace_with_summary(self, summary: str) -> None:
        """Swap the entire history for one summary message.

        This is deliberately all-or-nothing. Keeping a partial tail risks
        leaving a `tool` message whose matching assistant `tool_calls` entry
        was summarized away, which the API rejects outright.
        """
        self._messages = [
            MessageItem(
                role="user",
                content=summary,
                token_count=count_tokens(summary, self._model_name),
            )
        ]

    def reset(self) -> None:
        """Drop the conversation, keeping the system prompt."""
        self._messages = []

    def message_count(self) -> int:
        return len(self._messages)

    def get_messages(self) -> list[dict[str, Any]]:
        messages = []

        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})

        for item in self._messages:
            messages.append(item.to_dict())

        return messages
