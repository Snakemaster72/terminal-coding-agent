import asyncio

from config.config import Config
from tools.builtin.registry import ToolRegistry
from tools.mcp.client import MCPClient, MCPServerStatus
from tools.mcp.mcp_tool import MCPTool


class MCPManager:
    def __init__(self, config: Config):
        self.config = config
        self._clients: dict[str, MCPClient] = {}
        self._initialized = False

    async def initialize(self) -> None:
        if self._initialized:
            return

        mcp_configs = self.config.mcp_servers

        if not mcp_configs:
            return
        for name, server_config in mcp_configs.items():
            if not server_config.enabled:
                continue
            client = MCPClient(name=name, config=server_config, cwd=self.config.cwd)
            self._clients[name] = client

        # awaiting here would serialize the connects and blow up on the first
        # bad server - build the coroutines, then gather them
        connection_tasks = [
            asyncio.wait_for(
                client.connect(),
                timeout=client.config.startup_timeout_sec,
            )
            for client in self._clients.values()
        ]

        await asyncio.gather(*connection_tasks, return_exceptions=True)
        self._initialized = True

    def register_tools(self, registry: ToolRegistry) -> int:
        count = 0

        for client in self._clients.values():
            if client.status != MCPServerStatus.CONNECTED:
                continue

            for tool_info in client.tools:
                mcp_tool = MCPTool(
                    config=self.config,
                    tool_info=tool_info,
                    # "." is not allowed in an OpenAI function name
                    name=f"{client.name}__{tool_info.name}",
                    client=client,
                )
                registry.register_mcp_tool(mcp_tool)
                count += 1

        return count

    async def shutdown(self) -> None:
        disconnection_tasks = [client.disconnect() for client in self._clients.values()]
        await asyncio.gather(*disconnection_tasks, return_exceptions=True)

        self._clients.clear()
        self._initialized = False
