from pydantic import BaseModel, Field

from tools.base import Tool, ToolInvocation, ToolKind, ToolResult
from utils.paths import resolve_path, workspace_violation

# why not just shell


class ListDirParams(BaseModel):
    path: str = Field(".", description="The path of the directory to list")
    # what about hidden files
    include_hidden: bool = Field(
        False,
        description="Whether to include hidden files in the listing (default: False)",
    )


class ListDirTool(Tool):
    name = "list_dir"
    description = "List the contents of a directory"
    kind = ToolKind.READ
    schema = ListDirParams

    async def execute(self, invocation: ToolInvocation) -> ToolResult:

        params = ListDirParams(**invocation.params)

        dir_path = resolve_path(invocation.cwd, params.path)

        denied = workspace_violation(
            dir_path, invocation.cwd, self.config.workspace_jail
        )
        if denied:
            return ToolResult.error_result(denied)

        if not dir_path.exists() or not dir_path.is_dir():
            return ToolResult.error_result(
                f"Directory '{dir_path}' does not exist or is not a directory."
            )

        try:
            items = sorted(
                dir_path.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())
            )
        except Exception as e:
            return ToolResult.error_result(
                f"Error listing directory '{dir_path}': {e!s}"
            )

        if not params.include_hidden:
            items = [item for item in items if not item.name.startswith(".")]

        if not items:
            return ToolResult.success_result(f"Directory '{dir_path}' is empty.")

        lines = []

        for item in items:
            if item.is_dir():
                lines.append(f"{item.name}/")
            else:
                lines.append(f"[FILE] {item.name}")

        return ToolResult.success_result(
            "\n".join(lines),
            metadata={
                "path": str(dir_path),
                "entries": len(items),
            },
        )
