# grep: content searching tool
# This tool searches for a specific pattern in the input text and returns the matching lines.


# glob -> grep -> read -> reason -> write -> execute -> list_dir

import os
import re
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, Field

from tools.base import Tool, ToolInvocation, ToolKind, ToolResult
from utils.paths import (
    is_binary_file,
    resolve_path,
    resolve_path_rel_to_cwd,
    workspace_violation,
)

# why not just shell


class GrepParams(BaseModel):
    pattern: str = Field(..., description="Regular expression pattern to search for")
    path: str = Field(
        ".",
        description="File or directory path to search in (default: current directory)",
    )
    # what about hidden files
    case_sensitive: bool = Field(
        False,
        description="Whether the search should be case-sensitive (default: False)",
    )


class GrepTool(Tool):
    name = "grep"
    description = "Search for a specific pattern in the input text"
    kind = ToolKind.READ
    schema = GrepParams

    async def execute(self, invocation: ToolInvocation) -> ToolResult:

        params = GrepParams(**invocation.params)

        search_path = resolve_path(invocation.cwd, params.path)

        denied = workspace_violation(
            search_path, invocation.cwd, self.config.workspace_jail
        )
        if denied:
            return ToolResult.error_result(denied)

        if not search_path.exists():
            return ToolResult.error_result(
                f"Search path '{search_path}' does not exist."
            )

        try:
            flags = re.IGNORECASE if not params.case_sensitive else 0
            pattern = re.compile(params.pattern, flags=flags)
        except re.error as e:
            return ToolResult.error_result(
                f"Error compiling regex pattern '{params.pattern}': {e!s}"
            )

        if search_path.is_dir():
            files = self._find_files(search_path)
        else:
            files = [search_path]

        output_lines = []
        matches = 0
        for file_path in files:
            try:
                content = file_path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue

            file_matches = False

            lines = content.splitlines()
            for i, line in enumerate(lines):
                if pattern.search(line):
                    matches += 1
                    if not file_matches:
                        rel_path = resolve_path_rel_to_cwd(
                            str(file_path), Path(invocation.cwd)
                        )
                        output_lines.append(f"==={rel_path}===")
                        file_matches = True
                    output_lines.append(f"{i + 1}: {line}")

            if file_matches:
                output_lines.append("")  # Add a blank line after each file's matches

        if not output_lines:
            return ToolResult.success_result(
                f"No matches found for pattern '{params.pattern}' in '{search_path}'.",
                metadata={
                    "path": str(search_path),
                    "matches": matches,
                    "files_searched": len(files),
                },
            )

        return ToolResult.success_result(
            "\n".join(output_lines),
            metadata={
                "path": str(search_path),
                "matches": matches,
                "files_searched": len(files),
            },
        )

    MAX_FILES = 500
    SKIP_DIRS: ClassVar[frozenset[str]] = frozenset(
        {"node_modules", "__pycache__", "venv", ".git", ".venv"}
    )

    def _find_files(self, search_path: Path) -> list[Path]:
        files = []
        for root, dirs, filenames in os.walk(search_path):
            # Skip hidden and vendored directories
            dirs[:] = [
                d for d in dirs if d not in self.SKIP_DIRS and not d.startswith(".")
            ]
            for filename in filenames:
                if filename.startswith("."):
                    continue  # Skip hidden files
                files_path = Path(root) / filename
                if not is_binary_file(files_path):
                    files.append(files_path)
                    if len(files) >= self.MAX_FILES:
                        return files
        return files
