from typing import Any

from config.config import Config
from tools.base import Tool, ToolInvocation, ToolKind, ToolResult
from tools.mcp.client import MCPClient, MCPToolInfo

# why not just shell


class MCPTool(Tool):
    def __init__(
        self,
        config: Config,
        tool_info: MCPToolInfo,
        name: str,
        client: MCPClient,
    ) -> None:
        super().__init__(
            config=config,
        )
        self._tool_info = tool_info
        self.name = name
        self._client = client
        self.description = self._tool_info.description

    # Tool.schema is a read-only property - overriding it as a property is the
    # only way to attach the server-provided input schema
    @property
    def schema(self) -> dict[str, Any]:
        input_schema = self._tool_info.input_schema or {}
        return {
            "type": "object",
            "properties": input_schema.get("properties", {}),
            "required": input_schema.get("required", []),
        }

    def is_mutating(self, params) -> bool:
        return True

    kind = ToolKind.MCP

    async def execute(self, invocation: ToolInvocation) -> ToolResult:

        try:
            result = await self._client.call_tool(
                self._tool_info.name, invocation.params
            )
            output = result.get("output", "")
            is_error = result.get("is_error", False)

            if is_error:
                return ToolResult.error_result(
                    f"Error calling tool '{self._tool_info.name}' on MCP server '{self._tool_info.server_name}': {output}"
                )
            return ToolResult.success_result(output)
        except Exception as e:
            return ToolResult.error_result(
                f"Error calling tool '{self._tool_info.name}' on MCP server '{self._tool_info.server_name}': {e}"
            )
