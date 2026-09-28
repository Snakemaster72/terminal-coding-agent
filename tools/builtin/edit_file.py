from pathlib import Path

from pydantic import BaseModel, Field

from tools.base import FileDiff, Tool, ToolInvocation, ToolKind, ToolResult
from utils.paths import ensure_parent_dir, resolve_path, workspace_violation


class EditParams(BaseModel):
    path: str = Field(
        ...,
        description="The path to the file to edit (relative to the working directory or absolute).",
    )
    old_string: str = Field(
        "",
        description="The exact string to be replaced in the file. Must match exactly, including whitespace and indentation, and must be unique in the file unless replace_all is true.",
    )
    new_string: str = Field(
        ...,
        description="The new string to replace the old string with.",
    )
    replace_all: bool = Field(
        default=False,
        description="If true, all occurrences of old_string will be replaced. If false, only the first occurrence will be replaced. Defaults to false.",
    )


class EditTool(Tool):
    name = "edit_file"
    description = (
        "Edit an existing file by replacing text. This is the default tool for "
        "modifying a file that already exists, including edits that delete most "
        "of it. The old_string must match exactly (including whitespace and "
        "indentation) and must be unique in the file unless replace_all is true. "
        "Call this once per region you are changing; multiple calls on the same "
        "file are expected. Use write_file only to create a brand new file."
    )
    kind = ToolKind.WRITE
    schema = EditParams

    async def execute(self, invocation: ToolInvocation) -> ToolResult:
        params = EditParams(**invocation.params)
        # invocation.cwd: current working directory of the agent
        # params.path: path to the file to edit
        # params.content: new content for the file
        # params.diff: diff to apply to the file
        # params.is_new_file: whether the file is new or existing
        # params.is_deletion: whether the file is being deleted

        path = resolve_path(invocation.cwd, params.path)

        denied = workspace_violation(
            path, invocation.cwd, self.config.workspace_jail
        )
        if denied:
            return ToolResult.error_result(denied)

        if not path.exists():
            if params.old_string:
                return ToolResult.error_result(
                    f"File {path} does not exist. Cannot replace text in a non-existent file.",
                    metadata={"tool_name": self.name, "path": str(path)},
                )

            ensure_parent_dir(path)
            path.write_text(params.new_string, encoding="utf-8")
            line_count = len(params.new_string.splitlines())
            return ToolResult.success_result(
                f"Created {path} {line_count} lines",
                diff=FileDiff(
                    path=path,
                    old_content="",
                    new_content=params.new_string,
                    is_new_file=True,
                ),
                metadata={
                    "path": str(path),
                    "is_new_file": True,
                    "line_count": line_count,
                },
            )

        old_content = path.read_text(encoding="utf-8")

        if not params.old_string:
            return ToolResult.error_result(
                f"File {path} already exists. Cannot create a new file without specifying old_string. Provide old_string to replace, or use write_file to overwrite the file.",
            )

        occurrences = old_content.count(params.old_string)
        # occurrences = 0 means old_string not found in the file
        if occurrences == 0:
            return self._no_match_error(params.old_string, old_content, path)

        if occurrences > 1 and not params.replace_all:
            return ToolResult.error_result(
                f"old_string occurs {occurrences} times in {path}."
                f"Either \n"
                f"1. Provide more context to make the match unique\n"
                f"2. Set replace_all=True to replace all occurrences.",
                metadata={
                    "tool_name": self.name,
                    "path": str(path),
                    "occurrences": occurrences,
                },
            )

        if params.replace_all:
            new_content = old_content.replace(params.old_string, params.new_string)
            replace_count = occurrences
        else:
            new_content = old_content.replace(params.old_string, params.new_string, 1)
            replace_count = 1

        if new_content == old_content:
            return ToolResult.error_result(
                f"No changes made to {path}. The new_string is identical to the old_string.",
                metadata={
                    "tool_name": self.name,
                    "path": str(path),
                    "replace_count": replace_count,
                },
            )
        try:
            path.write_text(new_content, encoding="utf-8")

        except OSError as e:
            return ToolResult.error_result(
                f"Failed to write to {path}: {e}",
                metadata={"tool_name": self.name, "path": str(path)},
            )

        old_lines = len(old_content.splitlines())
        new_lines = len(new_content.splitlines())
        line_diff = new_lines - old_lines
        diff_msg = ""

        if line_diff > 0:
            diff_msg = f"(+{line_diff} lines)"
        elif line_diff < 0:
            diff_msg = f"({line_diff} lines)"

        return ToolResult.success_result(
            f"Edited {path} with {replace_count} replacements {diff_msg}",
            diff=FileDiff(
                path=path,
                old_content=old_content,
                new_content=new_content,
            ),
            metadata={
                "path": str(path),
                "is_new_file": False,
                "replace_count": replace_count,
                "line_diff": line_diff,
            },
        )

    def _no_match_error(self, old_string: str, content: str, path: Path) -> ToolResult:
        lines = content.splitlines()

        partial_matches = []
        search_terms = old_string.split()[:5]

        if search_terms:
            first_term = search_terms[0]
            for i, line in enumerate(lines, 1):
                if first_term in line:
                    partial_matches.append((i, line.strip()[:80]))
                    if len(partial_matches) >= 3:
                        break

        error_msg = f"old_string not found in {path}."

        if partial_matches:
            error_msg += "\n\nPossible similar lines:"
            for line_num, line_preview in partial_matches:
                error_msg += f"\n  Line {line_num}: {line_preview}"
            error_msg += "\n\nMake sure old_string matches exactly (including whitespace and indentation)."
        else:
            error_msg += (
                " Make sure the text matches exactly, including:\n"
                "- All whitespace and indentation\n"
                "- Line breaks\n"
                "- Any invisible characters\n"
                "Try re-reading the file using read_file tool and then editing."
            )

        return ToolResult.error_result(error_msg)
