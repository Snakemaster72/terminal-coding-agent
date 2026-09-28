"""Regression tests for the defects fixed in this pass.

Dependency-free: run with `python tests/test_fixes.py` (no pytest needed).
Nothing here touches the network - the LLM client is stubbed everywhere.
"""

import asyncio
import json
import sys
import tempfile
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.agent import Agent  # noqa: E402
from agent.event import AgentEventType  # noqa: E402
from client.response import (  # noqa: E402
    StreamEvent,
    StreamEventType,
    TextDelta,
    TokenUsage,
    ToolCall,
    parse_tool_call_arguments,
)
from config.config import Config  # noqa: E402
from context.compaction import ChatCompactor  # noqa: E402
from context.manager import ContextManager  # noqa: E402
from tools.base import ToolInvocation, ToolKind, ToolResult  # noqa: E402
from tools.builtin.glob import GlobTool  # noqa: E402
from tools.builtin.read_file import ReadFileTool  # noqa: E402
from tools.builtin.registry import create_default_registry  # noqa: E402
from tools.builtin.shell import ShellTool  # noqa: E402
from utils.paths import workspace_violation  # noqa: E402
from utils.text import truncate_text  # noqa: E402

_TESTS = []
_FAILURES = []


def test(fn):
    _TESTS.append(fn)
    return fn


def run(coro):
    return asyncio.run(coro)


def cfg(**kwargs) -> Config:
    c = Config(**kwargs)
    return c


# --------------------------------------------------------------------------
# stub LLM client
# --------------------------------------------------------------------------
class FakeClient:
    """Replays scripted turns in place of a real provider."""

    def __init__(self, turns):
        self.turns = list(turns)
        self.calls = []

    async def chat_completion(self, messages, tools=None, stream=True):
        self.calls.append(messages)
        if not self.turns:
            yield StreamEvent(
                type=StreamEventType.TEXT_DELTA, text_delta=TextDelta("done")
            )
            yield StreamEvent(type=StreamEventType.MESSAGE_COMPLETE)
            return

        for event in self.turns.pop(0):
            yield event

    async def close(self):
        pass


def text_turn(text):
    return [
        StreamEvent(type=StreamEventType.TEXT_DELTA, text_delta=TextDelta(text)),
        StreamEvent(type=StreamEventType.MESSAGE_COMPLETE),
    ]


def tool_turn(name, args, call_id="call_1"):
    return [
        StreamEvent(
            type=StreamEventType.TOOL_CALL_COMPLETE,
            tool_call=ToolCall(call_id=call_id, name=name, arguments=args),
        ),
        StreamEvent(type=StreamEventType.MESSAGE_COMPLETE),
    ]


# --------------------------------------------------------------------------
# workspace jail
# --------------------------------------------------------------------------
@test
def test_workspace_jail():
    root = Path(__file__).resolve().parent.parent
    assert workspace_violation(root / "main.py", root) is None
    assert workspace_violation("/etc/passwd", root) is not None
    assert workspace_violation(root / ".." / ".." / ".ssh", root) is not None
    assert workspace_violation("/etc/passwd", root, jail=False) is None
    assert workspace_violation(root, root) is None


@test
def test_file_tools_respect_jail():
    root = Path(__file__).resolve().parent.parent
    tool = ReadFileTool(cfg())
    result = run(
        tool.execute(ToolInvocation(params={"path": "/etc/passwd"}, cwd=root))
    )
    assert not result.success
    assert "outside the workspace" in result.error

    off = cfg()
    off.workspace_jail = False
    tool = ReadFileTool(off)
    result = run(
        tool.execute(ToolInvocation(params={"path": "/etc/hostname"}, cwd=root))
    )
    assert result.success, result.error


# --------------------------------------------------------------------------
# shell
# --------------------------------------------------------------------------
@test
def test_shell_denylist_precision():
    tool = ShellTool(cfg())
    allowed = [
        "npm run format",
        "make format",
        "git status",
        "ls | grep rm",
        "echo formatting",
        'python -c "print(1)"',
    ]
    blocked = [
        "rm file.txt",
        "rm -rf /",
        "sudo apt install x",
        "echo hi && rm -rf ~",
        "FOO=bar rm x",
        "/bin/rm x",
        "env sudo ls",
        "chmod -R 777 .",
    ]
    for command in allowed:
        assert tool._blocked_reason(command) is None, f"should allow: {command}"
    for command in blocked:
        assert tool._blocked_reason(command) is not None, f"should block: {command}"


@test
def test_shell_exit_code_is_an_int():
    root = Path(__file__).resolve().parent.parent
    tool = ShellTool(cfg())
    ok = run(tool.execute(ToolInvocation(params={"command": "true"}, cwd=root)))
    assert ok.success and ok.exit_code == 0, ok

    bad = run(tool.execute(ToolInvocation(params={"command": "exit 3"}, cwd=root)))
    assert not bad.success
    assert isinstance(bad.exit_code, int), type(bad.exit_code)
    assert bad.exit_code == 3, bad.exit_code


@test
def test_shell_timeout_kills_process_group():
    root = Path(__file__).resolve().parent.parent
    tool = ShellTool(cfg())
    result = run(
        tool.execute(
            ToolInvocation(
                params={"command": "sleep 30 & sleep 30", "timeout": 1}, cwd=root
            )
        )
    )
    assert not result.success
    assert "timed out" in result.error


@test
def test_shell_cwd_is_jailed():
    root = Path(__file__).resolve().parent.parent
    tool = ShellTool(cfg())
    result = run(
        tool.execute(ToolInvocation(params={"command": "ls", "cwd": "/etc"}, cwd=root))
    )
    assert not result.success
    assert "outside the workspace" in result.error


# --------------------------------------------------------------------------
# glob / read_file
# --------------------------------------------------------------------------
@test
def test_glob_returns_every_match_as_a_path():
    root = Path(__file__).resolve().parent.parent
    tool = GlobTool(cfg())
    result = run(
        tool.execute(
            ToolInvocation(params={"pattern": "tools/builtin/*.py", "path": "."}, cwd=root)
        )
    )
    assert result.success
    lines = result.output.splitlines()
    assert len(lines) == result.metadata["matches"], (len(lines), result.metadata)
    assert len(lines) > 5, lines
    assert "tools/builtin/glob.py" in lines
    # paths, not file contents
    assert all(line.endswith(".py") for line in lines), lines


@test
def test_read_file_truncation_does_not_raise():
    root = Path(__file__).resolve().parent.parent
    tool = ReadFileTool(cfg())
    tool.MAX_OUTPUT_TOKENS = 20  # force the truncation branch
    with tempfile.NamedTemporaryFile(
        "w", suffix=".txt", dir=root, delete=False
    ) as handle:
        handle.write("\n".join(f"line {i}" for i in range(500)))
        temp = Path(handle.name)
    try:
        result = run(tool.execute(ToolInvocation(params={"path": temp.name}, cwd=root)))
        assert result.success, result.error
        assert result.truncated
        assert "truncated" in result.output
    finally:
        temp.unlink()


@test
def test_truncate_text_signature_is_honoured():
    out = truncate_text("hello world " * 200, "gpt-4", 10)
    assert out.endswith("[truncated]")
    assert len(out) < 200


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------
@test
def test_registry_unknown_tool():
    root = Path(__file__).resolve().parent.parent
    registry = create_default_registry(cfg())
    result = run(registry.invoke("nope", {}, root))
    assert not result.success
    assert "Unknown tool" in result.error


@test
def test_registry_allowed_tools_gates_execution():
    root = Path(__file__).resolve().parent.parent
    config = cfg()
    config.allowed_tools = ["read_file"]
    registry = create_default_registry(config)
    names = [t.name for t in registry.get_tools()]
    assert names == ["read_file"]
    # previously the tool was only hidden, still invokable by name
    result = run(registry.invoke("shell", {"command": "echo hi"}, root))
    assert not result.success
    assert "Unknown tool" in result.error


@test
def test_registry_wraps_tool_exceptions_with_dict_metadata():
    root = Path(__file__).resolve().parent.parent
    registry = create_default_registry(cfg())

    class Boom(Exception):
        pass

    async def explode(invocation):
        raise Boom("kaboom")

    registry.get("read_file").execute = explode
    result = run(registry.invoke("read_file", {"path": "main.py"}, root))
    assert not result.success
    assert "Internal error" in result.error
    assert isinstance(result.metadata, dict), type(result.metadata)
    assert result.metadata["tool_name"] == "read_file"
    assert result.metadata["exception"] == "Boom"


@test
def test_malformed_tool_arguments_become_a_validation_error():
    root = Path(__file__).resolve().parent.parent
    parsed = parse_tool_call_arguments('{"path": "main.py"')  # truncated JSON
    assert parsed == {"raw_arguments": '{"path": "main.py"'}
    registry = create_default_registry(cfg())
    result = run(registry.invoke("read_file", parsed, root))
    assert not result.success
    assert "Invalid parameters" in result.error


# --------------------------------------------------------------------------
# context manager
# --------------------------------------------------------------------------
def _loaded_manager(config=None, tool_results=6, size=4000):
    config = config or cfg()
    manager = ContextManager(config, tools=[])
    manager.add_user_message("do the thing")
    for i in range(tool_results):
        manager.add_assistant_message(
            "",
            [
                {
                    "id": f"c{i}",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": "{}"},
                }
            ],
        )
        manager.add_tool_result(f"c{i}", "X" * size, is_error=False)
    return manager


@test
def test_token_accounting_includes_system_prompt():
    manager = ContextManager(cfg(), tools=[])
    empty = manager.total_tokens()
    assert empty > 0, "system prompt must be counted"
    manager.add_user_message("hello there")
    assert manager.total_tokens() > empty


@test
def test_pruning_preserves_tool_call_pairing():
    config = cfg()
    config.tool_output_window = 2
    config.prune_min_chars = 100
    manager = _loaded_manager(config)

    before = manager.total_tokens()
    reclaimed = manager.prune_stale_tool_outputs()
    assert reclaimed > 0
    assert manager.total_tokens() == before - reclaimed

    messages = manager.get_messages()
    tool_messages = [m for m in messages if m["role"] == "tool"]
    assert len(tool_messages) == 6, "tool messages must never be dropped"
    assert all(m["tool_call_id"] for m in tool_messages)

    pruned = [m for m in tool_messages if "pruned to reclaim context" in m["content"]]
    assert len(pruned) == 4, len(pruned)
    # the newest two keep their full body
    assert tool_messages[-1]["content"].startswith("XXXX")
    assert tool_messages[-2]["content"].startswith("XXXX")


@test
def test_pruning_is_idempotent():
    config = cfg()
    config.tool_output_window = 1
    config.prune_min_chars = 100
    manager = _loaded_manager(config)
    first = manager.prune_stale_tool_outputs()
    second = manager.prune_stale_tool_outputs()
    assert first > 0
    assert second == 0, "already-pruned messages must not be re-counted"


@test
def test_pruning_leaves_small_outputs_alone():
    config = cfg()
    config.tool_output_window = 0
    config.prune_min_chars = 1000
    manager = _loaded_manager(config, tool_results=3, size=10)
    assert manager.prune_stale_tool_outputs() == 0


@test
def test_should_compact_tracks_the_threshold():
    config = cfg()
    # must clear the ~2k-token system prompt floor, or the manager is over
    # budget before a single message is added
    config.model.context_window = 10_000
    config.compaction_threshold = 0.8
    manager = ContextManager(config, tools=[])
    assert not manager.should_compact()
    manager.add_user_message("W " * 9000)
    assert manager.should_compact()
    assert manager.usage_ratio() > 0.8


@test
def test_replace_with_summary_clears_orphans():
    manager = _loaded_manager()
    manager.replace_with_summary("## ORIGINAL GOAL\nship it")
    messages = manager.get_messages()
    assert [m["role"] for m in messages] == ["system", "user"]
    assert "ship it" in messages[-1]["content"]
    assert manager.message_count() == 1


@test
def test_reset_keeps_system_prompt():
    manager = _loaded_manager()
    manager.reset()
    assert manager.message_count() == 0
    assert manager.get_messages()[0]["role"] == "system"


# --------------------------------------------------------------------------
# compaction
# --------------------------------------------------------------------------
@test
def test_compaction_formatting_handles_tool_only_turns():
    compactor = ChatCompactor(client=None)
    out = compactor._format_history_for_compression(
        [
            {"role": "system", "content": "SYSTEM"},
            {"role": "user", "content": "fix the bug"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"function": {"name": "read_file", "arguments": '{"path":"a.py"}'}}
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "X" * 5000},
        ]
    )
    assert "SYSTEM" not in out
    assert 'read_file({"path":"a.py"})' in out
    assert "[truncated output]" in out


@test
def test_compress_returns_summary_and_usage():
    compactor = ChatCompactor(
        client=FakeClient(
            [
                [
                    StreamEvent(
                        type=StreamEventType.MESSAGE_COMPLETE,
                        text_delta=TextDelta("## ORIGINAL GOAL\nship it"),
                        usage=TokenUsage(prompt_tokens=10, total_tokens=12),
                    )
                ]
            ]
        )
    )
    summary, usage = run(compactor.compress(_loaded_manager()))
    assert summary and "ORIGINAL GOAL" in summary
    assert usage.total_tokens == 12


@test
def test_compress_survives_a_provider_error():
    compactor = ChatCompactor(
        client=FakeClient(
            [[StreamEvent(type=StreamEventType.ERROR, error="boom")]]
        )
    )
    summary, usage = run(compactor.compress(_loaded_manager()))
    assert summary is None and usage is None


@test
def test_compress_refuses_an_empty_summary():
    compactor = ChatCompactor(
        client=FakeClient([[StreamEvent(type=StreamEventType.MESSAGE_COMPLETE)]])
    )
    summary, _ = run(compactor.compress(_loaded_manager()))
    assert summary is None, "an empty summary must not replace the history"


@test
def test_compress_needs_a_minimum_history():
    manager = ContextManager(cfg(), tools=[])
    summary, usage = run(ChatCompactor(client=FakeClient([])).compress(manager))
    assert summary is None and usage is None


# --------------------------------------------------------------------------
# agent loop
# --------------------------------------------------------------------------
async def _drive(agent, message):
    events = []
    async for event in agent.run(message):
        events.append(event)
    return events


def _agent(config, turns):
    agent = Agent(config)
    agent.session.client = FakeClient(turns)
    agent.session.compactor.client = agent.session.client
    return agent


@test
def test_agent_runs_a_tool_and_finishes():
    root = Path(__file__).resolve().parent.parent
    config = cfg()
    config.cwd = root

    async def scenario():
        agent = _agent(
            config,
            [
                tool_turn("list_dir", {"path": "."}),
                text_turn("there are the files"),
            ],
        )
        async with agent:
            events = await _drive(agent, "list the files")
        return events

    events = run(scenario())
    kinds = [e.type for e in events]
    assert AgentEventType.TOOL_CALL_START in kinds
    assert AgentEventType.TOOL_CALL_COMPLETE in kinds
    assert AgentEventType.MAX_TURNS_REACHED not in kinds

    completed = [e for e in events if e.type == AgentEventType.TOOL_CALL_COMPLETE][0]
    assert completed.data["success"], completed.data
    final = [e for e in events if e.type == AgentEventType.TEXT_COMPLETE][-1]
    assert final.data["content"] == "there are the files"


@test
def test_agent_reports_hitting_the_turn_limit():
    root = Path(__file__).resolve().parent.parent
    config = cfg()
    config.cwd = root
    config.max_turns = 3

    async def scenario():
        agent = _agent(
            config,
            [tool_turn("list_dir", {"path": "."}, call_id=f"c{i}") for i in range(3)],
        )
        async with agent:
            return await _drive(agent, "loop forever")

    events = run(scenario())
    limit = [e for e in events if e.type == AgentEventType.MAX_TURNS_REACHED]
    assert len(limit) == 1, "the turn limit must be announced, not silent"
    assert limit[0].data["max_turns"] == 3


@test
def test_agent_compacts_when_over_budget():
    root = Path(__file__).resolve().parent.parent
    config = cfg()
    config.cwd = root
    config.model.context_window = 10_000
    config.compaction_threshold = 0.6
    config.prune_threshold = 0.4
    config.tool_output_window = 0
    config.prune_min_chars = 50
    config.max_turns = 3

    summary_event = StreamEvent(
        type=StreamEventType.MESSAGE_COMPLETE,
        text_delta=TextDelta("## ORIGINAL GOAL\ncontinue"),
        usage=TokenUsage(total_tokens=5),
    )

    async def scenario():
        # _manage_context runs at the *top* of the turn, so the compactor's
        # call is the first one the fake client serves
        agent = _agent(
            config,
            [
                [summary_event],  # consumed by the compactor
                text_turn("finished"),
            ],
        )
        async with agent:
            manager = agent.session.context_manager
            manager.add_user_message("W " * 4000)  # blow past the budget
            return await _drive(agent, "go"), manager

    events, manager = run(scenario())
    kinds = [e.type for e in events]
    assert AgentEventType.CONTEXT_COMPACTED in kinds, kinds
    compacted = [e for e in events if e.type == AgentEventType.CONTEXT_COMPACTED][0]
    assert compacted.data["after_tokens"] < compacted.data["before_tokens"]
    assert manager.usage_ratio() < config.compaction_threshold


@test
def test_agent_continues_when_compaction_fails():
    root = Path(__file__).resolve().parent.parent
    config = cfg()
    config.cwd = root
    config.model.context_window = 10_000
    config.compaction_threshold = 0.5
    config.max_turns = 2

    async def scenario():
        agent = _agent(
            config,
            [
                [StreamEvent(type=StreamEventType.ERROR, error="compaction down")],
                text_turn("still here"),
            ],
        )
        async with agent:
            agent.session.context_manager.add_user_message("W " * 4000)
            return await _drive(agent, "go")

    events = run(scenario())
    kinds = [e.type for e in events]
    assert AgentEventType.CONTEXT_COMPACTED not in kinds
    errors = [e for e in events if e.type == AgentEventType.AGENT_ERROR]
    assert any("compaction failed" in e.data["error"] for e in errors), errors
    # history must survive a failed compaction
    assert agent_history_survived(events)


def agent_history_survived(events):
    return any(e.type == AgentEventType.TEXT_COMPLETE for e in events)


@test
def test_agent_bails_out_on_a_failed_stream():
    root = Path(__file__).resolve().parent.parent
    config = cfg()
    config.cwd = root

    async def scenario():
        agent = _agent(
            config, [[StreamEvent(type=StreamEventType.ERROR, error="upstream 500")]]
        )
        async with agent:
            events = await _drive(agent, "go")
            return events, agent.session.context_manager.message_count()

    events, count = run(scenario())
    assert any(e.type == AgentEventType.AGENT_ERROR for e in events)
    # only the user message: no empty assistant turn appended
    assert count == 1, count


# --------------------------------------------------------------------------
# llm client
# --------------------------------------------------------------------------
@test
def test_sampling_params_reach_the_request():
    from client.llm_client import LLMClient

    config = cfg()
    config.model.temperature = 0.3
    config.model.max_tokens = 512
    client = LLMClient(config)
    captured = {}

    async def fake_stream(_client, kwargs):
        captured.update(kwargs)
        yield StreamEvent(type=StreamEventType.MESSAGE_COMPLETE)

    client._stream_response = fake_stream
    client.get_client = lambda: None

    async def scenario():
        async for _ in client.chat_completion([{"role": "user", "content": "hi"}]):
            pass

    run(scenario())
    assert captured["temperature"] == 0.3
    assert captured["max_tokens"] == 512


@test
def test_no_retry_after_a_partial_response():
    import httpx
    from openai import APIConnectionError

    from client.llm_client import LLMClient

    client = LLMClient(cfg())
    client.get_client = lambda: None
    attempts = {"n": 0}

    async def flaky(_client, kwargs):
        attempts["n"] += 1
        yield StreamEvent(type=StreamEventType.TEXT_DELTA, text_delta=TextDelta("par"))
        raise APIConnectionError(request=httpx.Request("POST", "http://x"))

    client._stream_response = flaky

    async def scenario():
        return [
            e
            async for e in client.chat_completion([{"role": "user", "content": "hi"}])
        ]

    events = run(scenario())
    assert attempts["n"] == 1, "must not replay a partially delivered stream"
    errors = [e for e in events if e.type == StreamEventType.ERROR]
    assert errors and "partial response" in errors[0].error


@test
def test_usage_parsing_tolerates_missing_details():
    from client.llm_client import _build_usage

    class Raw:
        prompt_tokens = 10
        completion_tokens = 5
        total_tokens = 15
        prompt_tokens_details = None

    usage = _build_usage(Raw())
    assert usage.total_tokens == 15
    assert usage.cached_tokens == 0


# --------------------------------------------------------------------------
# sub-agents
# --------------------------------------------------------------------------
@test
def test_sub_agent_cannot_spawn_sub_agents():
    from tools.builtin.task import TaskTool

    tool = TaskTool(cfg())
    allowed, error = tool._sub_agent_tools(None)
    assert error is None
    assert "task" not in allowed
    assert "read_file" in allowed

    allowed, error = tool._sub_agent_tools(["read_file", "grep"])
    assert error is None and allowed == ["read_file", "grep"]

    allowed, error = tool._sub_agent_tools(["task"])
    assert error is not None and "Unknown tool" in error


@test
def test_sub_agent_context_is_isolated():
    """The sub-agent must not see the parent's history, and vice versa."""
    from tools.builtin.task import TaskTool

    root = Path(__file__).resolve().parent.parent
    config = cfg()
    config.cwd = root

    seen = {}
    original_init = Agent.__init__

    def spy_init(self, cfg_in):
        original_init(self, cfg_in)
        self.session.client = FakeClient([text_turn("the answer is 42")])
        self.session.compactor.client = self.session.client
        seen["sub_config"] = cfg_in

    Agent.__init__ = spy_init
    try:
        tool = TaskTool(config)
        result = run(
            tool.execute(
                ToolInvocation(
                    params={"description": "find it", "prompt": "search the repo"},
                    cwd=root,
                )
            )
        )
    finally:
        Agent.__init__ = original_init

    assert result.success, result.error
    assert result.output == "the answer is 42"
    assert result.metadata["turn_limit_hit"] is False
    # isolation guarantees
    assert seen["sub_config"] is not config, "sub-agent must get its own config"
    assert seen["sub_config"].mcp_servers == {}
    assert "task" not in seen["sub_config"].allowed_tools
    assert seen["sub_config"].max_turns == config.sub_agent_max_turns


@test
def test_sub_agent_reports_an_empty_answer_as_failure():
    from tools.builtin.task import TaskTool

    root = Path(__file__).resolve().parent.parent
    original_init = Agent.__init__

    def spy_init(self, cfg_in):
        original_init(self, cfg_in)
        self.session.client = FakeClient(
            [[StreamEvent(type=StreamEventType.ERROR, error="upstream down")]]
        )
        self.session.compactor.client = self.session.client

    Agent.__init__ = spy_init
    try:
        result = run(
            TaskTool(cfg()).execute(
                ToolInvocation(
                    params={"description": "x", "prompt": "y"}, cwd=root
                )
            )
        )
    finally:
        Agent.__init__ = original_init

    assert not result.success
    assert "no answer" in result.error


@test
def test_task_tool_is_exposed_with_a_valid_schema():
    registry = create_default_registry(cfg())
    schemas = {s["name"]: s for s in registry.get_schemas()}
    assert "task" in schemas
    params = schemas["task"]["parameters"]
    assert set(params["required"]) == {"description", "prompt"}
    assert params["properties"]["prompt"]["type"] == "string"
    json.dumps(schemas)  # must be serialisable for the API payload

    assert registry.get("task").kind is ToolKind.AGENT


@test
def test_subagent_prompt_section_activates():
    from prompts.system import get_system_prompt

    registry = create_default_registry(cfg())
    prompt = get_system_prompt(config=cfg(), tools=registry.get_tools())
    assert "**Sub-Agents:**" in prompt, "the task tool must switch the section on"
    assert "`task`" in prompt


# --------------------------------------------------------------------------
@test
def test_every_tool_reports_a_kind_for_the_tui():
    registry = create_default_registry(cfg())
    for tool in registry.get_tools():
        assert isinstance(tool.kind, ToolKind), tool.name


def main():
    for fn in _TESTS:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except Exception:
            _FAILURES.append(fn.__name__)
            print(f"  FAIL  {fn.__name__}")
            traceback.print_exc()

    print(f"\n{len(_TESTS) - len(_FAILURES)}/{len(_TESTS)} passed")
    return 1 if _FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
