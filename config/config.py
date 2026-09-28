from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field, model_validator


class ModelConfig(BaseModel):
    name: str = "nvidia/nemotron-3-super-120b-a12b:free"
    # how creative a model is,
    # 0.0 is deterministic, 1.0 is default, and 2.0 is the most creative
    temperature: float = Field(default=1, ge=0.0, le=2.0)
    context_window: int = 256_000  # how many tokens the model can see at once
    max_tokens: int | None = Field(
        default=None,
        ge=1,
        description="Cap on tokens the model may generate per turn. None lets the provider decide.",
    )


class ShellEnvironmentPolicy(BaseModel):
    ignore_default_excludes: bool = False
    exclude_patterns: list[str] = Field(
        default_factory=lambda: ["*KEY*", "*PASSWORD*", "*SECRET*", "*TOKEN*"]
    )
    set_vars: dict[str, str] = Field(
        default_factory=dict
    )  # NODE_ENV = "production", etc. overrides environment variables for the shell tool, but does not affect the agent's environment


class MCPServerConfig(BaseModel):
    enabled: bool = True
    startup_timeout_sec: int = 30

    # stdio transport
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    cwd: Path | None = None

    # http/sse
    url: str | None = None

    @model_validator(mode="after")
    def validate(self) -> MCPServerConfig:
        if not self.command and not self.url:
            raise ValueError("Either 'command' or 'url' must be provided.")

        if self.command and self.url:
            raise ValueError("Only one of 'command' or 'url' can be provided.")

        return self


class Config(BaseModel):
    model: ModelConfig = Field(default_factory=ModelConfig)
    cwd: Path = Field(default_factory=Path.cwd)
    shell_environment: ShellEnvironmentPolicy = Field(
        default_factory=ShellEnvironmentPolicy
    )

    max_turns: int = 100  # maximum number of turns in a conversation

    # --- context budgeting ---
    # fraction of the model's context window at which history is summarized
    compaction_threshold: float = Field(default=0.8, gt=0.0, le=1.0)
    # context usage at which stale tool outputs start being pruned. Lower than
    # compaction_threshold so the cheap, lossless-ish lever runs first
    prune_threshold: float = Field(default=0.5, gt=0.0, le=1.0)
    # how many of the most recent tool results keep their full output; older
    # ones are replaced with a placeholder (sliding-window pruning)
    tool_output_window: int = Field(default=3, ge=0)
    # tool outputs smaller than this are never worth pruning
    prune_min_chars: int = Field(default=1000, ge=0)

    # --- sandboxing ---
    # confine file and shell tools to the working directory
    workspace_jail: bool = True
    # turn budget for a sub-agent, independent of the parent's max_turns
    sub_agent_max_turns: int = Field(default=20, ge=1)

    mcp_servers: dict[str, MCPServerConfig] = Field(default_factory=dict)
    allowed_tools: list[str] | None = Field(
        None,
        description="If set, only these tools will be available to the agent",
    )
    developer_instructions: str | None = None
    user_instructions: str | None = None
    debug: bool = False

    @property
    def api_key(self) -> str | None:
        return os.environ.get("API_KEY")

    @property
    def base_url(self) -> str:
        return os.environ.get("BASE_URL")

    @property
    def model_name(self) -> str:
        return self.model.name

    @model_name.setter
    def model_name(self, value: str):
        self.model.name = value

    @property
    def max_tokens(self) -> int | None:
        return self.model.max_tokens

    @property
    def context_window(self) -> int:
        return self.model.context_window

    @property
    def temperature(self) -> float:
        return self.model.temperature

    @temperature.setter
    def temperature(self, value: float):
        self.model.temperature = value

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not self.api_key:
            errors.append("API_KEY environment variable is not set.")
        if not self.cwd.exists():
            errors.append(f"Working directory does not exist: {self.cwd}")
        return errors
