import asyncio
import sys
from pathlib import Path

import click

from agent.agent import Agent
from agent.event import AgentEventType
from config.config import Config
from config.loader import load_config
from ui.tui import TUI, get_console

console = get_console()


class CLI:
    def __init__(self, config: Config):
        self.config = config
        self.agent: Agent | None = None
        self.tui = TUI(console, config)

    async def run_single(self, message: str) -> str | None:
        # agent = Agent()
        # async with handles the setup and cleanup of the Agent instance, ensuring that resources are properly managed.
        # resources like network connections or file handles are released when the block is exited, even if an error occurs.
        async with Agent(config=self.config) as agent:
            self.agent = agent
            return await self._process_message(message)

    async def run_interactive(self) -> None:
        self.tui.print_welcome(
            "AI Agent",
            lines=[
                f"model: {self.config.model_name}",
                f"cwd: {self.config.cwd}",
                "commands: /exit, /help, /config",
                "/model",
                "/approval",
            ],
        )
        async with Agent(self.config) as agent:
            self.agent = agent
            while True:
                try:
                    user_input = console.input("\n[user]>[/user]").strip()
                    if not user_input:
                        continue
                    await self._process_message(user_input)
                except KeyboardInterrupt:
                    console.print("\n[dim] Use /exit to quite[/dim]")
                    break
                except EOFError:
                    break

        console.print("\n[dim]Goodbye![/dim]")

    def _get_tool_kind(self, tool_name: str) -> str | None:
        tool_kind = None
        tool = self.agent.session.tool_registry.get(tool_name)
        if not tool:
            tool_kind = None
        else:
            tool_kind = tool.kind.value

        return tool_kind

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
                tool_kind = self._get_tool_kind(tool_name)
                self.tui.tool_call_start(
                    call_id=event.data.get("call_id", ""),
                    name=tool_name,
                    arguments=event.data.get("arguments", {}),
                    tool_kind=tool_kind,
                )

            elif event.type == AgentEventType.TOOL_CALL_COMPLETE:
                tool_name = event.data.get("name", "Unknown tool")
                tool_kind = self._get_tool_kind(tool_name)
                self.tui.tool_call_complete(
                    call_id=event.data.get("call_id", ""),
                    tool_kind=tool_kind,
                    name=event.data.get("name", "Unknown tool"),
                    success=event.data.get("success", False),
                    truncated=event.data.get("truncated", False),
                    metadata=event.data.get("metadata", {}),
                    error=event.data.get("error"),
                    output=event.data.get("output", ""),
                    diff=event.data.get("diff", None),
                    exit_code=event.data.get("exit_code", None),
                )

        return final_response


@click.command()
@click.argument("prompt", required=False)
@click.option(
    "--cwd",
    "--c",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    help="Current working directory",
)
def main(
    prompt: str | None,
    cwd: Path | None = None,
):
    try:
        config = load_config(cwd=cwd)
    except Exception as e:
        console.print(f"[error]Config Error: {e}[/error]")

    errors = config.validate()
    if errors:
        for error in errors:
            console.print(f"[error]{error}[/error]")
        sys.exit(1)

    cli = CLI(config)
    if prompt:
        result = asyncio.run(cli.run_single(prompt))
        if result is None:
            sys.exit(1)
    else:
        asyncio.run(cli.run_interactive())


main()
