from __future__ import annotations

from collections.abc import AsyncGenerator

from agent.event import AgentEvent, AgentEventType
from client.llm_client import LLMClient
from client.response import StreamEventType, ToolCall, ToolResultMessage
from config.config import Config
from context.manager import ContextManager
from tools.builtin.registry import create_default_registry


class Agent:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.client = LLMClient(config=config)
        self.context_manager = ContextManager(self.config)
        self.tool_registry = create_default_registry()

    # currently run and loop for just one single message,
    async def run(self, message: str):

        yield AgentEvent.agent_start(message)
        self.context_manager.add_user_message(message)
        # add user message to context

        final_response: str | None = None
        async for event in self._agentic_loop():
            yield event

            if event.type == AgentEventType.TEXT_COMPLETE:
                final_response = event.data.get("content")

        yield AgentEvent.agent_end(final_response)

    async def _agentic_loop(self) -> AsyncGenerator[AgentEvent, None]:

        max_turns = self.config.max_turns

        for turn in range(max_turns):
            response_text = ""
            tool_schemas = self.tool_registry.get_schemas()

            tool_calls: list[ToolCall] = []

            async for event in self.client.chat_completion(
                self.context_manager.get_messages(),
                tools=tool_schemas if tool_schemas else None,
                stream=True,
            ):
                # print(event)
                if event.type == StreamEventType.TEXT_DELTA:
                    if event.text_delta:
                        content = event.text_delta.content
                        response_text += content
                        yield AgentEvent.text_delta(content)
                elif event.type == StreamEventType.TOOL_CALL_COMPLETE:
                    if event.tool_call:
                        tool_calls.append(event.tool_call)
                elif event.type == StreamEventType.ERROR:
                    yield AgentEvent.agent_error(event.error or "unknown error")

            self.context_manager.add_assistant_message(
                response_text or None,
                [
                    {
                        "id": tc.call_id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": tc.arguments},
                    }
                    for tc in tool_calls
                ]
                if tool_calls
                else None,
            )
            if response_text:
                yield AgentEvent.text_complete(response_text)

            if not tool_calls:
                return
            tool_call_results: list[ToolResultMessage] = []

            for tool_call in tool_calls:
                yield AgentEvent.tool_call_start(
                    call_id=tool_call.call_id,
                    name=tool_call.name or "",
                    arguments=tool_call.arguments,
                )

                result = await self.tool_registry.invoke(
                    tool_call.name, tool_call.arguments, self.config.cwd
                )

                yield AgentEvent.tool_call_complete(
                    tool_call.call_id,
                    tool_call.name,
                    result,
                )

                tool_call_results.append(
                    ToolResultMessage(
                        tool_call_id=tool_call.call_id,
                        content=result.to_model_output(),
                        is_error=not result.success,
                    )
                )

            for tool_result in tool_call_results:
                self.context_manager.add_tool_result(
                    tool_result.tool_call_id, tool_result.content, tool_result.is_error
                )

    async def __aenter__(self) -> Agent:
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.client:
            await self.client.close()
            self.client = None
