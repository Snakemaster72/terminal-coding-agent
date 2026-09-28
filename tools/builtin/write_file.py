from pydantic import BaseModel, Field

from tools.base import FileDiff, Tool, ToolInvocation, ToolKind, ToolResult
from utils.paths import ensure_parent_dir, resolve_path, workspace_violation


class WriteFileParams(BaseModel):
    path: str = Field(
        ...,
        description="The path to the file to write to (relative to the working directory or absolute).",
    )
    content: str = Field(..., description="The content to write to the file.")
    create_directories: bool = Field(
        default=True,
        description="Create parent directories if they do not exist. Defaults to True.",
    )


class WriteFileTool(Tool):
    name = "write_file"
    description = (
        "Create a new file with the given content. Parent directories are created "
        "automatically. If the file already exists its entire contents are "
        "destroyed and replaced, so prefer edit_file for anything that already "
        "exists - including changes that remove most of the file. Only overwrite "
        "an existing file when you are replacing all of its content at once and "
        "have already read it."
    )

    kind = ToolKind.WRITE
    schema = WriteFileParams

    async def execute(self, invocation: ToolInvocation) -> ToolResult:
        params = WriteFileParams(**invocation.params)
        # invocation.cwd: current working directory of the agent
        # params.path: path to the file to write to
        path = resolve_path(invocation.cwd, params.path)

        denied = workspace_violation(
            path, invocation.cwd, self.config.workspace_jail
        )
        if denied:
            return ToolResult.error_result(denied)

        is_new_file = not path.exists()
        old_content = ""

        if not is_new_file:
            try:
                old_content = path.read_text(encoding="utf-8")
            except:
                pass

        try:
            if params.create_directories:
                ensure_parent_dir(path)

            elif not path.parent.exists():
                return ToolResult.error_result(
                    f"Parent directory {path.parent} does not exist. Set create_directories=True to create it."
                )

            path.write_text(params.content, encoding="utf-8")

            action = "Created" if is_new_file else "Updated"
            line_count = len(params.content.splitlines())

            return ToolResult.success_result(
                f"{action} file {path} with {line_count} lines",
                diff=FileDiff(
                    path=path,
                    old_content=old_content,
                    new_content=params.content,
                    is_new_file=is_new_file,
                ),
                metadata={
                    "path": str(path),
                    "is_new_file": is_new_file,
                    "line": line_count,
                    "bytes": len(params.content.encode("utf-8")),
                },
            )

        except OSError as e:
            return ToolResult.error_result(f"Failed to write to {path}: {e}")

        content = params.content
        create_directories = params.create_directories

        if create_directories:
            import os

            os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

        return f"Successfully wrote to {path}"
