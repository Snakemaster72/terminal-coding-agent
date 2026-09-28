from typing import Any

from client.llm_client import LLMClient
from client.response import StreamEventType, TokenUsage
from context.manager import ContextManager
from prompts.system import get_compression_prompt


def _clip(text: str, limit: int, note: str) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + note


class ChatCompactor:
    """Summarizes a conversation into a structured continuation prompt.

    The summary keeps the goal, what is already done, and the next step; it
    deliberately discards raw tool output, which is the bulk of the tokens and
    the least useful thing to carry forward.
    """

    TOOL_RESULT_LIMIT = 2000
    ASSISTANT_LIMIT = 3000
    TOOL_ARGS_LIMIT = 500
    USER_LIMIT = 1500
    MIN_MESSAGES = 3

    def __init__(self, client: LLMClient):
        self.client = client

    def _format_history_for_compression(self, messages: list[dict[str, Any]]) -> str:

        output = ["Here is the conversation that needs to be continued: \n"]
        for msg in messages:
            role = msg.get("role")
            # an assistant turn that only carries tool_calls has content=None
            content = msg.get("content") or ""
            if role == "system":
                continue
            elif role == "tool":
                tool_id = msg.get("tool_call_id", "unknown")
                truncated = _clip(
                    content, self.TOOL_RESULT_LIMIT, "\n.... [truncated output]"
                )
                output.append(f"[Tool Result ({tool_id})]: \n{truncated}")

            elif role == "assistant":
                if content:
                    truncated = _clip(
                        content, self.ASSISTANT_LIMIT, "\n... [response truncated]"
                    )
                    output.append(f"Assistant:\n{truncated}")
                tool_details = []
                if msg.get("tool_calls"):
                    for tc in msg["tool_calls"]:
                        func = tc.get("function", {})
                        name = func.get("name", "unknown")
                        args = func.get("arguments", "{}")
                        args = _clip(args, self.TOOL_ARGS_LIMIT, " ...")
                        tool_details.append(f"  - {name}({args})")

                    output.append(
                        "Assistant called tools: \n" + "\n".join(tool_details)
                    )

            else:
                if content:
                    truncated = _clip(
                        content, self.USER_LIMIT, "\n... [message truncated]"
                    )
                    output.append(f"User responded with:\n{truncated}")

        return "\n\n---\n\n".join(output)

    async def compress(
        self, context_manager: ContextManager
    ) -> tuple[str | None, TokenUsage | None]:
        """
        Compresses the context using the LLM client and returns a list of compressed messages.
        """
        messages = context_manager.get_messages()

        if len(messages) < self.MIN_MESSAGES:
            return None, None

        request = [
            {"role": "system", "content": get_compression_prompt()},
            {"role": "user", "content": self._format_history_for_compression(messages)},
        ]

        try:
            summary = ""
            usage = None
            async for event in self.client.chat_completion(
                request,
                stream=False,
            ):
                if event.type == StreamEventType.ERROR:
                    return None, None
                if event.type == StreamEventType.MESSAGE_COMPLETE:
                    usage = event.usage
                    # a provider may return usage but no text
                    if event.text_delta:
                        summary += event.text_delta.content

            # usage is optional - plenty of providers omit it - but a summary
            # is not: replacing history with an empty string loses the session
            if not summary.strip():
                return None, None
        except Exception:
            return None, None

        return summary, usage
