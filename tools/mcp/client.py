import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from fastmcp import Client
from fastmcp.client.transports import SSETransport, StdioTransport

from config.config import MCPServerConfig


class MCPServerStatus(str, Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"


@dataclass
class MCPToolInfo:
    name: str
    description: str
    input_schema: dict[str, Any] = field(default_factory=dict)
    server_name: str = ""


class MCPClient:
    def __init__(
        self, name: str, config: MCPServerConfig, cwd: Path | None = None
    ) -> None:
        self.name = name
        self.config = config
        self.cwd = cwd
        self.status = MCPServerStatus.DISCONNECTED
        self._client: Client | None = None

        self._tools: dict[str, MCPToolInfo] = dict()

    @property
    def tools(self) -> list[MCPToolInfo]:
        return list(self._tools.values())

    def _create_transport(self) -> StdioTransport | SSETransport:
        if self.config.command:
            env = os.environ.copy()
            env.update(self.config.env)
            return StdioTransport(
                command=self.config.command,
                args=list(self.config.args),
                env=env,
                cwd=str(self.config.cwd or self.cwd),
                # the server's stderr would otherwise scribble over the TUI
                log_file=Path(os.devnull),
            )
        else:
            return SSETransport(url=self.config.url)

    async def connect(self) -> None:
        if self.status == MCPServerStatus.CONNECTED:
            return
        self.status = MCPServerStatus.CONNECTING

        try:
            self._client = Client(transport=self._create_transport())
            await self._client.__aenter__()

            # list_tools() returns the list of mcp.types.Tool directly
            for tool in await self._client.list_tools():
                self._tools[tool.name] = MCPToolInfo(
                    name=tool.name,
                    description=tool.description or "",
                    input_schema=getattr(tool, "inputSchema", None) or {},
                    server_name=self.name,
                )

            self.status = MCPServerStatus.CONNECTED
        except Exception as e:
            self.status = MCPServerStatus.ERROR
            raise RuntimeError(
                f"Error connecting to MCP server '{self.name}': {e}"
            ) from e
            # print(f"Error connecting to MCP server: {e}")

    async def disconnect(self) -> None:
        if self._client:
            await self._client.__aexit__(None, None, None)
            self._client = None

        self._tools.clear()
        self.status = MCPServerStatus.DISCONNECTED

    async def call_tool(self, tool_name: str, params: dict[str, Any]) -> dict[str, Any]:
        if self.status != MCPServerStatus.CONNECTED:
            raise RuntimeError(
                f"Cannot call tool '{tool_name}' because MCP server '{self.name}' is not connected."
            )

        if not self._client:
            raise RuntimeError(
                f"Cannot call tool '{tool_name}' because MCP client is not initialized."
            )

        # MCPTool.execute already turns failures into a ToolResult - wrapping
        # here too would nest the same message three deep
        result = await self._client.call_tool(tool_name, params)

        output = []
        for item in result.content:
            if hasattr(item, "text"):
                output.append(item.text)
            else:
                output.append(str(item))

        return {
            "output": "\n".join(output),
            "is_error": result.is_error,
        }
