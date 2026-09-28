import datetime
import json
import uuid

from client.llm_client import LLMClient
from config.config import Config
from config.loader import get_data_dir
from context.compaction import ChatCompactor
from context.manager import ContextManager
from tools.builtin.registry import create_default_registry
from tools.mcp.mcp_manager import MCPManager


class Session:
    def __init__(self, config: Config):
        self.config = config
        self.client = LLMClient(config=config)
        # registry first: the context manager needs the registered tools to build
        # the system prompt's Available Tools and operational sections
        self.tool_registry = create_default_registry(config)
        self.context_manager: ContextManager | None = None
        self.compactor = ChatCompactor(client=self.client)
        self.mcp_manager = MCPManager(config=config)
        self.session_id = str(uuid.uuid4())
        self.created_at = datetime.datetime.now()
        self.updated_at = datetime.datetime.now()
        self._turn_count = 0

    async def initialize(self):
        await self.mcp_manager.initialize()
        self.mcp_manager.register_tools(self.tool_registry)
        self.context_manager = ContextManager(
            self.config,
            tools=self.tool_registry.get_tools(),
            user_memory=self._load_memory(),
        )

    def _load_memory(self) -> str | None:
        """Flatten the persisted user memory entries into a prompt section."""
        data_dir = get_data_dir()
        data_dir.mkdir(parents=True, exist_ok=True)
        path = data_dir / "user_memory.json"

        if not path.exists():
            return None

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            entries = data.get("entries")
            if not entries:
                return None

            lines = ["User preferences and notes:"]
            for key, value in entries.items():
                lines.append(f"- {key}: {value}")

            return "\n".join(lines)
        except Exception:
            return None

    def increment_turn(self):
        self._turn_count += 1
        self.updated_at = datetime.datetime.now()
        return self._turn_count
