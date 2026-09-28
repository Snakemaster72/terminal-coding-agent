import asyncio
import fnmatch
import os
import re
import shlex
import signal
import sys
from pathlib import Path

from pydantic import BaseModel, Field

from tools.base import Tool, ToolInvocation, ToolKind, ToolResult
from utils.paths import workspace_violation

# Matched against the *command word* of each segment, never as a substring:
# a substring test blocks "npm run format" because it contains "rm".
BLOCKED_BINARIES = frozenset(
    {
        "rm",
        "sudo",
        "su",
        "shutdown",
        "reboot",
        "init",
        "halt",
        "poweroff",
        "mkfs",
        "dd",
        "fdisk",
        "parted",
        "chown",
        "chmod",
        "chgrp",
    }
)

# Substring checks, for shapes that are dangerous regardless of the command word
BLOCKED_PATTERNS = (
    ":(){",  # fork bomb
    "rm -rf /",
    "rm -rf ~",
    "dd if=/dev/zero",
    "dd if=/dev/random",
    "chmod 777 /",
    "chmod -r 777",
    "> /dev/sda",
)

# wrappers that delegate to the command word after them
_PASSTHROUGH = frozenset({"env", "nohup", "time", "command", "exec", "builtin"})

# ; && || | and newlines all start a new command word
_SEGMENT_SPLIT = re.compile(r"[;&|\n]+")


class ShellParams(BaseModel):
    command: str = Field(
        ...,
        description="The shell command to execute in the current working directory.",
    )
    timeout: int = Field(
        default=120,
        ge=1,
        le=600,
        description="The maximum time in seconds to allow the command to run before timing out. Defaults to 120 seconds.",
    )
    cwd: str | None = Field(
        default=None,
        description="The working directory in which to execute the command. If not provided, the current working directory of the agent will be used.",
    )


class ShellTool(Tool):
    name = "shell"
    description = (
        "Execute a shell command in the current working directory. "
        "Use this for running scripts, commands, or any other shell operations. "
        "Be cautious with commands that can modify the system or files."
    )
    kind = ToolKind.SHELL
    schema = ShellParams

    @staticmethod
    def _command_word(segment: str) -> str | None:
        """The binary a segment actually invokes, ignoring env assignments."""
        segment = segment.strip()
        if not segment:
            return None

        try:
            tokens = shlex.split(segment)
        except ValueError:
            # unbalanced quotes - fall back to whitespace splitting
            tokens = segment.split()

        for token in tokens:
            # leading VAR=value assignments precede the real command word
            if "=" in token and not token.startswith(("/", "-", ".")):
                continue
            word = os.path.basename(token).lower()
            if word in _PASSTHROUGH:
                continue
            return word

        return None

    def _blocked_reason(self, command: str) -> str | None:
        lowered = command.lower()

        for pattern in BLOCKED_PATTERNS:
            if pattern in lowered:
                return f"it matches the blocked pattern '{pattern}'"

        for segment in _SEGMENT_SPLIT.split(command):
            word = self._command_word(segment)
            if word and word in BLOCKED_BINARIES:
                return f"'{word}' is a blocked command"

        return None

    async def execute(self, invocation: ToolInvocation) -> ToolResult:
        params = ShellParams(**invocation.params)

        if not params.command.strip():
            return ToolResult.error_result("No command provided.")

        blocked = self._blocked_reason(params.command)
        if blocked:
            return ToolResult.error_result(
                f"Command '{params.command}' is blocked for safety reasons: {blocked}."
            )

        if params.cwd:
            cwd = Path(params.cwd)
            if not cwd.is_absolute():
                cwd = invocation.cwd / cwd
        else:
            cwd = invocation.cwd

        denied = workspace_violation(
            cwd, invocation.cwd, self.config.workspace_jail
        )
        if denied:
            return ToolResult.error_result(denied)

        if not cwd.exists():
            return ToolResult.error_result(f"Working directory '{cwd}' does not exist.")

        env = self._build_environment()
        if sys.platform == "win32":
            shell_cmd = ["cmd", "/c", params.command]
        else:
            shell_cmd = ["/bin/bash", "-c", params.command]

        # this will run the command in a subprocess and capture the output and error
        process = await asyncio.create_subprocess_exec(
            *shell_cmd,
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            start_new_session=True,  # Start a new session to prevent signals from propagating
        )
        try:
            stdout_data, stderr_data = await asyncio.wait_for(
                process.communicate(), timeout=params.timeout
            )
        except asyncio.TimeoutError:
            if sys.platform != "win32":
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            else:
                process.kill()
            await process.wait()
            return ToolResult.error_result(
                f"Command '{params.command}' timed out after {params.timeout} seconds."
            )

        stdout = stdout_data.decode("utf-8", errors="replace").strip()
        stderr = stderr_data.decode("utf-8", errors="replace").strip()
        exit_code = process.returncode

        output = ""

        if stdout.strip():
            output += stdout.rstrip()

        if stderr.strip():
            output += "\n --- stderr --- \n" + stderr.rstrip()

        if exit_code != 0:
            output += f"\nExit code: {exit_code}"

        if len(output) > 100 * 1024:  # 100 KB limit
            output = output[: 100 * 1024] + "\n[Output truncated due to size]"

        return ToolResult(
            success=exit_code == 0,
            output=output,
            exit_code=exit_code,
            metadata={
                "tool_name": self.name,
                "command": params.command,
                "cwd": str(cwd),
                "stdout": stdout,
                "stderr": stderr,
            },
        )

    def _build_environment(self) -> dict[str, str]:
        env = os.environ.copy()
        # Add any additional environment variables here if needed
        shell_environment = self.config.shell_environment

        if not shell_environment.ignore_default_excludes:
            for pattern in shell_environment.exclude_patterns:
                keys_to_remove = [
                    k for k in env.keys() if fnmatch.fnmatch(k.upper(), pattern.upper())
                ]
                for key in keys_to_remove:
                    del env[key]

        if shell_environment.set_vars:
            env.update(shell_environment.set_vars)

        return env
