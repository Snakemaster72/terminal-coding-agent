import asyncio
import sys

import click

from agent.agent import Agent
from agent.event import AgentEventType
from ui.tui import TUI, get_console

console = get_console()


class CLI:
    def __init__(self):
        self.agent: Agent | None = None
        self.tui = TUI(console)

    async def run_single(self, message: str) -> str | None:
        # agent = Agent()
        # async with handles the setup and cleanup of the Agent instance, ensuring that resources are properly managed.
        # resources like network connections or file handles are released when the block is exited, even if an error occurs.
        async with Agent() as agent:
            self.agent = agent
            return await self._process_message(message)

    async def _process_message(self, message: str) -> str | None:
        if not self.agent:
            return None

        assistant_streaming = False
        final_response: str | None = None

        async for event in self.agent.run(message):
            # print(event)
            # text delta jab saara output ek saath nhi , ruk ke aa raha hai, jaise ki streaming output
            # hum wait nhi karte, jaise hi output aata hai, hum usko turant print kar dete hai
            if event.type == AgentEventType.TEXT_DELTA:
                content = event.data.get("content", "")
                if not assistant_streaming:
                    self.tui.begin_assistant()
                    assistant_streaming = True
                self.tui.stream_assistant_delta(content)

            elif event.type == AgentEventType.TEXT_COMPLETE:
                final_response = event.data.get("content")
                if assistant_streaming:
                    self.tui.end_assistant()
                    assistant_streaming = False

            elif event.type == AgentEventType.AGENT_ERROR:
                error = event.data.get("error", "Unknown error")
                console.print(f"\n[error] error: {error}[error]")

            elif event.type == AgentEventType.TOOL_CALL_START:
                tool_name = event.data.get("name", "Unknown tool")
                tool_kind = None
                tool = self.agent.tool_registry.get(tool_name)
                if not tool:
                    tool_kind = None

                tool_kind = tool.kind.value
                self.tui.tool_call_start(
                    call_id=event.data.get("call_id", ""),
                    name=tool_name,
                    arguments=event.data.get("arguments", {}),
                    tool_kind=tool_kind,
                )

        return final_response


@click.command()
@click.argument("prompt", required=False)
def main(
    prompt: str | None,
):
    cli = CLI()
    if prompt:
        result = asyncio.run(cli.run_single(prompt))
        if result is None:
            sys.exit(1)


main()
