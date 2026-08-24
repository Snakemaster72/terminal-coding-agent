import os
from pathlib import Path

from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    name: str = "nvidia/nemotron-3.5-lightning:free"
    # how creative a model is,
    # 0.0 is deterministic, 1.0 is default, and 2.0 is the most creative
    temperature: float = Field(default=1, ge=0.0, le=2.0)
    context_window: int = 256_000  # how many tokens the model can see at once


class Config(BaseModel):
    model: ModelConfig = Field(default_factory=ModelConfig)
    cwd: Path = Field(default_factory=Path.cwd)

    max_turns: int = 100  # maximum number of turns in a conversation
    # max_tool_output_tokens: int = 50_000
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
