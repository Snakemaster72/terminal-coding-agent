from __future__ import annotations

import json
from collections.abc import AsyncGenerator

# from requests import Session
from agent.event import AgentEvent, AgentEventType
from agent.session import Session
from client.response import StreamEventType, ToolCall, ToolResultMessage
from config.config import Config


class Agent:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.session: Session | None = Session(self.config)

    # currently run and loop for just one single message,
    async def run(self, message: str):

        yield AgentEvent.agent_start(message)
        self.session.context_manager.add_user_message(message)
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
            self.session.increment_turn()
            response_text = ""

            # keep the next request inside the model's context window. Safe to
            # do here and only here: every assistant tool_calls message from the
            # previous turn already has its matching tool results appended, so
            # the history is balanced and nothing can be left orphaned
            async for context_event in self._manage_context():
                yield context_event

            tool_schemas = self.session.tool_registry.get_schemas()

            tool_calls: list[ToolCall] = []
            stream_failed = False

            async for event in self.session.client.chat_completion(
                self.session.context_manager.get_messages(),
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
                    stream_failed = True

            # a failed request produced no assistant turn - don't append an empty
            # message (it corrupts the context for every later turn) and don't
            # loop again, or we burn through max_turns replaying the same failure
            if stream_failed and not response_text and not tool_calls:
                return

            self.session.context_manager.add_assistant_message(
                response_text or None,
                [
                    {
                        "id": tc.call_id,
                        "type": "function",
                        # the API expects `arguments` as a JSON *string*, but
                        # parse_tool_call_arguments() already decoded it to a dict
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
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

                result = await self.session.tool_registry.invoke(
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
                self.session.context_manager.add_tool_result(
                    tool_result.tool_call_id, tool_result.content, tool_result.is_error
                )

        # falling out of the loop means the turn budget was spent mid-task.
        # Say so - silently returning stale text reads like a finished answer
        yield AgentEvent.max_turns_reached(max_turns)

    async def _manage_context(self) -> AsyncGenerator[AgentEvent, None]:
        """Two-stage context control, cheapest lever first.

        Stage 1 drops the bodies of stale tool outputs, which costs nothing.
        Stage 2 summarizes the whole history into a continuation prompt, which
        costs an extra model call, so it only runs if stage 1 left us still
        over budget.
        """
        context_manager = self.session.context_manager

        if context_manager.usage_ratio() >= self.config.prune_threshold:
            reclaimed = context_manager.prune_stale_tool_outputs()
            if reclaimed > 0:
                yield AgentEvent.context_pruned(
                    reclaimed, context_manager.total_tokens()
                )

        if not context_manager.should_compact():
            return

        # The system prompt is never summarized, so it is a hard floor. If the
        # history is already just one summary message and we are still over
        # budget, compacting again cannot help - it would only fail on every
        # remaining turn and spam the user with errors
        if context_manager.message_count() < 2:
            return

        before = context_manager.total_tokens()
        summary, _usage = await self.session.compactor.compress(context_manager)

        if not summary:
            # better to run at full context and let the provider complain than
            # to throw away history we could not replace
            yield AgentEvent.agent_error(
                "Context compaction failed; continuing with the full history."
            )
            return

        context_manager.replace_with_summary(summary)
        yield AgentEvent.context_compacted(before, context_manager.total_tokens())

    async def __aenter__(self) -> Agent:
        await self.session.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.session:
            await self.session.client.close()
            await self.session.mcp_manager.shutdown()
            self.session = None
