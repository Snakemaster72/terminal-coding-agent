from pydantic import BaseModel, Field

from tools.base import Tool, ToolInvocation, ToolKind, ToolResult


class TaskParams(BaseModel):
    description: str = Field(
        ...,
        description="A short (3-5 word) description of the sub-task, for display.",
    )
    prompt: str = Field(
        ...,
        description=(
            "The complete instruction for the sub-agent. It starts with no "
            "knowledge of this conversation, so restate every fact it needs: "
            "paths, constraints, and exactly what to report back."
        ),
    )
    tools: list[str] | None = Field(
        default=None,
        description=(
            "Optional list of tool names the sub-agent may use. Defaults to "
            "every built-in tool except `task` itself."
        ),
    )


class TaskTool(Tool):
    name = "task"
    description = (
        "Run a focused sub-task in a sub-agent that has its own isolated "
        "context window. The sub-agent sees nothing of this conversation - only "
        "the prompt you give it - and only its final text answer comes back. "
        "Use it for wide searches or multi-step investigations whose "
        "intermediate tool output would otherwise flood this context. It cannot "
        "spawn further sub-agents."
    )
    kind = ToolKind.AGENT
    schema = TaskParams

    def _sub_agent_tools(self, requested: list[str] | None) -> tuple[list[str], str | None]:
        """Resolve the sub-agent's tool whitelist.

        `task` is always excluded: a sub-agent that can spawn sub-agents can
        recurse until the process dies.
        """
        # imported here, not at module scope: tools.builtin imports this module
        from tools.builtin import get_all_builtin_tools

        available = [t.name for t in get_all_builtin_tools() if t.name != self.name]

        if requested is None:
            return available, None

        unknown = [name for name in requested if name not in available]
        if unknown:
            return [], (
                f"Unknown tool(s) for sub-agent: {', '.join(unknown)}. "
                f"Available: {', '.join(available)}."
            )

        return requested, None

    async def execute(self, invocation: ToolInvocation) -> ToolResult:
        # deferred: agent.agent imports the registry, which imports this module
        from agent.agent import Agent
        from agent.event import AgentEventType

        params = TaskParams(**invocation.params)

        allowed, error = self._sub_agent_tools(params.tools)
        if error:
            return ToolResult.error_result(error)

        sub_config = self.config.model_copy(deep=True)
        sub_config.allowed_tools = allowed
        sub_config.max_turns = self.config.sub_agent_max_turns
        # the parent already holds the MCP connections; re-spawning every
        # server for a short sub-task is pure overhead
        sub_config.mcp_servers = {}

        final_text: str | None = None
        errors: list[str] = []
        turn_limit_hit = False

        try:
            # a brand-new Agent means a brand-new Session and ContextManager:
            # the sub-agent's history starts empty and is discarded on exit
            async with Agent(sub_config) as sub_agent:
                async for event in sub_agent.run(params.prompt):
                    if event.type == AgentEventType.TEXT_COMPLETE:
                        final_text = event.data.get("content")
                    elif event.type == AgentEventType.MAX_TURNS_REACHED:
                        turn_limit_hit = True
                    elif event.type == AgentEventType.AGENT_ERROR:
                        errors.append(str(event.data.get("error")))
        except Exception as e:
            return ToolResult.error_result(
                f"Sub-agent '{params.description}' failed: {e}"
            )

        if not final_text:
            detail = f" Errors: {'; '.join(errors)}" if errors else ""
            return ToolResult.error_result(
                f"Sub-agent '{params.description}' returned no answer.{detail}"
            )

        output = final_text
        if turn_limit_hit:
            output += (
                f"\n\n[sub-agent stopped after its {sub_config.max_turns}-turn "
                "limit; this answer may be incomplete]"
            )
        if errors:
            output += f"\n\n[sub-agent reported errors: {'; '.join(errors)}]"

        return ToolResult.success_result(
            output,
            metadata={
                "tool_name": self.name,
                "description": params.description,
                "tools_granted": len(allowed),
                "turn_limit_hit": turn_limit_hit,
            },
        )
