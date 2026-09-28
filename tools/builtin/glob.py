# grep: content searching tool
# This tool searches for a specific pattern in the input text and returns the matching lines.


# glob -> grep -> read -> reason -> write -> execute -> list_dir


from pydantic import BaseModel, Field

from tools.base import Tool, ToolInvocation, ToolKind, ToolResult
from utils.paths import resolve_path, workspace_violation

# why not just shell


class GlobParams(BaseModel):
    pattern: str = Field(
        ..., description="Glob pattern to search for (supports ** for recursive search)"
    )
    path: str = Field(
        ".",
        description="Directory path to search in (default: current directory)",
    )


class GlobTool(Tool):
    name = "glob"
    description = "Search for files matching a specific pattern. Supports ** for recursive search."
    kind = ToolKind.READ
    schema = GlobParams

    MAX_RESULTS = 1000

    async def execute(self, invocation: ToolInvocation) -> ToolResult:

        params = GlobParams(**invocation.params)

        search_path = resolve_path(invocation.cwd, params.path)

        denied = workspace_violation(
            search_path, invocation.cwd, self.config.workspace_jail
        )
        if denied:
            return ToolResult.error_result(denied)

        if not search_path.exists() or not search_path.is_dir():
            return ToolResult.error_result(f"Directory '{search_path}' does not exist.")

        try:
            matches = sorted(p for p in search_path.glob(params.pattern) if p.is_file())
        except Exception as e:
            return ToolResult.error_result(f"Error searching for files: {e!s}")

        if not matches:
            return ToolResult.success_result(
                f"No files matching '{params.pattern}' in '{search_path}'.",
                metadata={"path": str(search_path), "matches": 0},
            )

        output_lines = []
        for file_path in matches[: self.MAX_RESULTS]:
            try:
                display = file_path.relative_to(invocation.cwd)
            except ValueError:
                display = file_path
            output_lines.append(str(display))

        truncated = len(matches) > self.MAX_RESULTS
        if truncated:
            output_lines.append(
                f"...and {len(matches) - self.MAX_RESULTS} more matches."
            )

        return ToolResult.success_result(
            "\n".join(output_lines),
            truncated=truncated,
            metadata={
                "path": str(search_path),
                "matches": len(matches),
            },
        )
