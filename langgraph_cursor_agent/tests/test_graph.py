"""Graph-level tests driven by a fake model (no API key, no network).

Skipped automatically when LangGraph is not installed, so the sandbox tests can
still run in minimal environments.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("langgraph")

from langchain_core.messages import AIMessage, ToolMessage  # noqa: E402
from langgraph.types import Command  # noqa: E402

from agent import build_graph  # noqa: E402
from sandbox import SafeWorkspace  # noqa: E402


class FakeModel:
    """Returns queued AIMessages in order; ignores the input messages."""

    def __init__(self, responses: list[AIMessage]) -> None:
        self._responses = list(responses)
        self.calls = 0

    def bind_tools(self, _tools):
        return self

    def invoke(self, _messages):
        response = self._responses[min(self.calls, len(self._responses) - 1)]
        self.calls += 1
        return response


def _write_call(path: str, content: str, call_id: str = "call-1") -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{
            "name": "write_file",
            "args": {"relative_path": path, "content": content},
            "id": call_id,
            "type": "tool_call",
        }],
    )


def test_full_loop_writes_file_and_tracks_state(tmp_path: Path) -> None:
    ws = SafeWorkspace(tmp_path, allowed_commands=["python"])
    model = FakeModel([
        _write_call("gen.py", "print('hi')\n"),
        AIMessage(content="已创建 gen.py 并完成任务。"),
    ])
    graph = build_graph(ws, model=model, require_approval=False)
    config = {"configurable": {"thread_id": "t1"}}

    result = graph.invoke(
        {"messages": [{"role": "user", "content": "创建 gen.py"}], "changed_files": [], "commands_run": []},
        config,
    )

    assert "print('hi')" in ws.read_file("gen.py")
    assert result["changed_files"] == ["gen.py"]
    assert result["messages"][-1].content == "已创建 gen.py 并完成任务。"
    assert model.calls == 2  # model -> tools -> model


def test_approval_denied_blocks_write(tmp_path: Path) -> None:
    ws = SafeWorkspace(tmp_path, allowed_commands=["python"])
    model = FakeModel([
        _write_call("should_not_exist.py", "print('nope')\n"),
        AIMessage(content="好的，已取消写入。"),
    ])
    graph = build_graph(ws, model=model, require_approval=True)
    config = {"configurable": {"thread_id": "t2"}}

    graph.invoke(
        {"messages": [{"role": "user", "content": "写文件"}], "changed_files": [], "commands_run": []},
        config,
    )
    assert graph.get_state(config).next == ("approval",)  # paused at the approval gate

    result = graph.invoke(Command(resume="n"), config)

    assert "文件不存在" in ws.read_file("should_not_exist.py")
    assert result["changed_files"] == []
    assert any(isinstance(m, ToolMessage) and "拒绝" in m.content for m in result["messages"])
    assert result["messages"][-1].content == "好的，已取消写入。"


def test_approval_approved_allows_write(tmp_path: Path) -> None:
    ws = SafeWorkspace(tmp_path, allowed_commands=["python"])
    model = FakeModel([
        _write_call("approved.py", "print('yes')\n"),
        AIMessage(content="已按审批写入。"),
    ])
    graph = build_graph(ws, model=model, require_approval=True)
    config = {"configurable": {"thread_id": "t3"}}

    graph.invoke(
        {"messages": [{"role": "user", "content": "写文件"}], "changed_files": [], "commands_run": []},
        config,
    )
    assert graph.get_state(config).next == ("approval",)
    result = graph.invoke(Command(resume="y"), config)

    assert "print('yes')" in ws.read_file("approved.py")
    assert result["changed_files"] == ["approved.py"]
