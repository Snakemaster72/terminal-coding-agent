import asyncio
from collections.abc import AsyncGenerator
from typing import Any

from openai import APIConnectionError, APIError, AsyncOpenAI, RateLimitError

from client.response import (
    StreamEvent,
    StreamEventType,
    TextDelta,
    TokenUsage,
    ToolCall,
    ToolCallDelta,
    parse_tool_call_arguments,
)
from config.config import Config


def _build_usage(raw: Any) -> TokenUsage:
    """Providers vary in which usage fields they populate; none are required."""
    details = getattr(raw, "prompt_tokens_details", None)
    return TokenUsage(
        prompt_tokens=getattr(raw, "prompt_tokens", 0) or 0,
        completion_tokens=getattr(raw, "completion_tokens", 0) or 0,
        total_tokens=getattr(raw, "total_tokens", 0) or 0,
        cached_tokens=getattr(details, "cached_tokens", 0) or 0,
    )


class LLMClient:
    def __init__(self, config: Config) -> None:
        # _client is a private member
        self._client: AsyncOpenAI | None = None
        self._max_retries: int = 3
        self.config = config

    def get_client(self) -> AsyncOpenAI:
        if self._client is None:
            # not selecting model, that depends on when we send a message, user can change model anytime,
            # cursor has an auto model feature
            self._client = AsyncOpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url,  # e.g. https://openrouter.ai/api/v1
            )
        return self._client

    async def close(self) -> None:
        if self._client:
            await self._client.close()
            self._client = None

    def _build_tools(self, tools: list[dict[str, Any]]):
        return [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get(
                        "parameters",
                        {"type": "object", "properties": {}},
                    ),
                },
            }
            for tool in tools
        ]

    # AsyncGenerator[..(what type you want to return)],
    async def chat_completion(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        stream: bool = True,
    ) -> AsyncGenerator[StreamEvent, None]:

        client = self.get_client()

        # kwargs = keyword arguments?
        kwargs = {
            "model": self.config.model_name,
            "messages": messages,
            "stream": stream,
            "temperature": self.config.temperature,
        }

        if self.config.max_tokens is not None:
            kwargs["max_tokens"] = self.config.max_tokens

        if tools:
            kwargs["tools"] = self._build_tools(tools)
            kwargs["tool_choice"] = "auto"

        for attempt in range(self._max_retries + 1):
            # a retry replays the request from the start, so it is only safe
            # while nothing has reached the caller yet - otherwise the caller
            # would see the first partial response followed by a full one
            emitted = False
            try:
                if stream:
                    async for event in self._stream_response(client, kwargs):
                        emitted = True
                        yield event
                else:
                    event = await self._non_stream_response(client, kwargs)
                    emitted = True
                    yield event
                return

            except (RateLimitError, APIConnectionError) as e:
                label = (
                    "Rate limit exceeded"
                    if isinstance(e, RateLimitError)
                    else "Connection error"
                )
                if attempt < self._max_retries and not emitted:
                    # 1s, 2s, 4s
                    await asyncio.sleep(2**attempt)
                    continue

                if emitted:
                    label = f"{label} after a partial response"

                yield StreamEvent(
                    type=StreamEventType.ERROR,
                    error=f"{label}: {e}",
                )
                return

            except APIError as e:
                # not retried: 4xx-class failures (bad request, context length
                # exceeded, auth) will fail identically on every attempt
                yield StreamEvent(
                    type=StreamEventType.ERROR,
                    error=f"API error: {e}",
                )
                return

    async def _stream_response(
        self, client: AsyncOpenAI, kwargs: dict[str, Any]
    ) -> AsyncGenerator[StreamEvent, None]:
        response = await client.chat.completions.create(**kwargs)

        finish_reason: str | None = None
        usage: TokenUsage | None = None
        tool_calls: dict[int, dict[str, Any]] = {}
        async for chunk in response:
            if hasattr(chunk, "usage") and chunk.usage:
                usage = _build_usage(chunk.usage)

            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            delta = choice.delta

            if choice.finish_reason:
                finish_reason = choice.finish_reason

            if delta.content:
                yield StreamEvent(
                    type=StreamEventType.TEXT_DELTA, text_delta=TextDelta(delta.content)
                )

            # if llm calls for a tool call, than we identify
            if delta.tool_calls:
                for tool_call_delta in delta.tool_calls:
                    idx = tool_call_delta.index

                    if idx not in tool_calls:
                        tool_calls[idx] = {
                            "id": tool_call_delta.id or "",
                            "name": "",
                            "arguments": "",
                        }

                    if tool_call_delta.id:
                        tool_calls[idx]["id"] = tool_call_delta.id

                    if tool_call_delta.function:
                        if tool_call_delta.function.name:
                            tool_calls[idx]["name"] = tool_call_delta.function.name
                            yield StreamEvent(
                                type=StreamEventType.TOOL_CALL_START,
                                tool_call_delta=ToolCallDelta(
                                    call_id=tool_calls[idx]["id"],
                                    name=tool_call_delta.function.name,
                                ),
                            )

                        if tool_call_delta.function.arguments:
                            tool_calls[idx]["arguments"] += (
                                tool_call_delta.function.arguments
                            )
                            yield StreamEvent(
                                type=StreamEventType.TOOL_CALL_DELTA,
                                tool_call_delta=ToolCallDelta(
                                    call_id=tool_calls[idx]["id"],
                                    name=tool_calls[idx]["name"],
                                    arguments_delta=tool_call_delta.function.arguments,
                                ),
                            )

        for idx, tc in tool_calls.items():
            yield StreamEvent(
                type=StreamEventType.TOOL_CALL_COMPLETE,
                tool_call=ToolCall(
                    call_id=tc["id"],
                    name=tc["name"],
                    arguments=parse_tool_call_arguments(tc["arguments"]),
                ),
            )

        yield StreamEvent(
            type=StreamEventType.MESSAGE_COMPLETE,
            finish_reason=finish_reason,
            usage=usage,
        )

    async def _non_stream_response(
        self, client: AsyncOpenAI, kwargs: dict[str, Any]
    ) -> StreamEvent:
        response = await client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        message = choice.message

        text_delta = None
        if message.content:
            text_delta = TextDelta(content=message.content)

        tool_calls: list[ToolCall] = []

        if message.tool_calls:
            for tc in message.tool_calls:
                tool_calls.append(
                    ToolCall(
                        call_id=tc.id or "",
                        name=tc.function.name if tc.function else None,
                        arguments=parse_tool_call_arguments(
                            tc.function.arguments if tc.function else ""
                        ),
                    )
                )
        usage = _build_usage(response.usage) if response.usage else None
        return StreamEvent(
            type=StreamEventType.MESSAGE_COMPLETE,
            text_delta=text_delta,
            finish_reason=choice.finish_reason,
            usage=usage,
        )
