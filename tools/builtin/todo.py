import uuid

from pydantic import BaseModel, Field

from config.config import Config
from tools.base import Tool, ToolInvocation, ToolKind, ToolResult


class TodosParams(BaseModel):
    action: str = Field(
        ..., description="Action to perform: add, remove, list, clear, complete"
    )
    id: str = Field(
        None,
        description="ID of the todo item (required for remove and complete actions)",
    )
    content: str = Field(
        None, description="Content of the todo item (required for add action)"
    )


class TodosTool(Tool):
    name = "todos"
    description = "Manage a list of todos. You can add, remove, and list todos."
    kind = ToolKind.MEMORY
    schema = TodosParams

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.todos: dict[str, str] = {}

    async def execute(self, invocation: ToolInvocation) -> ToolResult:

        params = TodosParams(**invocation.params)

        # Manage todos
        if params.action.lower() == "add":
            if not params.content:
                return ToolResult.error_result("Content is required to add a todo.")
            todo_id = str(uuid.uuid4())[:8]
            self.todos[todo_id] = params.content
            return ToolResult.success_result(
                f"Todo added with ID {todo_id}: {params.content}.",
                metadata={"id": todo_id, "content": params.content},
            )
        elif params.action.lower() == "complete":
            if not params.id:
                return ToolResult.error_result("ID is required to complete a todo.")
            if params.id not in self.todos:
                return ToolResult.error_result(f"Todo not found: {params.id}.")
            content = self.todos.pop(params.id)
            return ToolResult.success_result(
                f"Todo completed [{params.id}]: {content}.", metadata={"id": params.id}
            )

        elif params.action.lower() == "remove":
            if not params.id:
                return ToolResult.error_result("ID is required to remove a todo.")
            if params.id not in self.todos:
                return ToolResult.error_result(f"Todo not found: {params.id}.")
            content = self.todos.pop(params.id)
            return ToolResult.success_result(
                f"Todo removed [{params.id}]: {content}.", metadata={"id": params.id}
            )

        elif params.action.lower() == "list":
            if not self.todos:
                return ToolResult.success_result("No todos found.")
            lines = ["Todos:"]
            for todo_id, content in self.todos.items():
                lines.append(f"[{todo_id}] {content}")
            return ToolResult.success_result("\n".join(lines))

        elif params.action.lower() == "clear":
            count = len(self.todos)
            self.todos.clear()
            return ToolResult.success_result(f"All {count} todos cleared.")
        else:
            return ToolResult.error_result(
                f"Unknown action: {params.action}. Valid actions are: add, remove, list, clear, complete."
            )
