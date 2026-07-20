r"""A practical, workspace-scoped LangGraph coding agent.

Highlights (the "complete flow" the docs describe):
  * custom ``AgentState`` with reducers that accumulate changed files/commands
  * side-effecting tools that update state via ``Command`` + ``InjectedToolCallId``
  * an optional human-in-the-loop approval gate implemented with ``interrupt``
  * thread-scoped memory via a checkpointer, plus a streaming CLI

Run (PowerShell):
    pip install -r requirements.txt
    $env:OPENAI_API_KEY="sk-..."
    python agent.py --workspace .\workspace --thread-id demo
"""

from __future__ import annotations

import argparse
import operator
import os
from pathlib import Path
from typing import Annotated, Iterable

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_core.tools.base import InjectedToolCallId
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command, interrupt
from typing_extensions import TypedDict

from sandbox import SafeWorkspace

SENSITIVE_TOOLS = {"write_file", "run_command"}
DEFAULT_RECURSION_LIMIT = 40

SYSTEM_PROMPT = """你是一名谨慎、务实的代码工程 Agent，只能在受限工作区内工作。

工作方式：
1. 先用 list_files / read_file / search_text 了解现状，不要凭空猜测项目结构。
2. 做小而完整的改动，保持既有代码风格，不改动无关文件。
3. 写文件后用 run_command 运行测试或静态检查（如 pytest / ruff）来验证。
4. 工具返回失败时如实说明，不要虚构成功；分析报错并给出下一步。
5. 不读取或写入 .env、密钥、凭据；不在回复中索要或回显任何密钥。
6. 最终回答请包含：完成了什么、改了哪些文件、验证结果、以及剩余风险（如有）。
"""


class AgentState(TypedDict):
    """Shared blackboard for the graph.

    ``messages`` uses the message reducer; ``changed_files`` and ``commands_run``
    use ``operator.add`` so tool nodes append to them across the whole run.
    """

    messages: Annotated[list, add_messages]
    changed_files: Annotated[list[str], operator.add]
    commands_run: Annotated[list[str], operator.add]
    approved: bool


def create_tools(workspace: SafeWorkspace) -> list:
    """Wrap sandbox methods as LangGraph tools bound to one workspace."""

    @tool
    def list_files(pattern: str = "**/*") -> str:
        """列出工作区内匹配 glob pattern 的文件；适合先了解项目结构。"""
        return workspace.list_files(pattern)

    @tool
    def read_file(relative_path: str, start_line: int = 1, end_line: int = 400) -> str:
        """读取工作区内 UTF-8 文本文件的指定行区间，行号从 1 开始（含端点）。"""
        return workspace.read_file(relative_path, start_line, end_line)

    @tool
    def search_text(query: str, file_pattern: str = "**/*") -> str:
        """在工作区文本文件中按字面量搜索 query，返回“文件:行号: 内容”。"""
        return workspace.search_text(query, file_pattern)

    @tool
    def write_file(
        relative_path: str,
        content: str,
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        """将完整 UTF-8 content 写入工作区文件；会创建父目录并覆盖同名文件。"""
        message, changed = workspace.write_file(relative_path, content)
        update: dict[str, object] = {"messages": [ToolMessage(message, tool_call_id=tool_call_id)]}
        if changed is not None:
            update["changed_files"] = [changed]
        return Command(update=update)

    @tool
    def run_command(
        command: str,
        tool_call_id: Annotated[str, InjectedToolCallId],
        timeout_seconds: int = 60,
    ) -> Command:
        """在工作区运行无 shell 命令；仅允许白名单中的程序（如 python / pytest）。"""
        output, ran = workspace.run_command(command, timeout_seconds)
        update: dict[str, object] = {"messages": [ToolMessage(output, tool_call_id=tool_call_id)]}
        if ran is not None:
            update["commands_run"] = [ran]
        return Command(update=update)

    return [list_files, read_file, search_text, write_file, run_command]


def _make_model(tools: list):
    """Create the external model client from env and bind the tools to it."""
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("未设置 OPENAI_API_KEY；请先通过环境变量配置 API Key。")
    from langchain_openai import ChatOpenAI  # imported lazily so tests can inject a fake model

    model_kwargs: dict[str, object] = {
        "model": os.getenv("OPENAI_MODEL", "gpt-5-mini"),
        "timeout": 60,
        "max_retries": 2,
    }
    if os.getenv("OPENAI_BASE_URL"):
        model_kwargs["base_url"] = os.environ["OPENAI_BASE_URL"]
    return ChatOpenAI(**model_kwargs).bind_tools(tools)


def build_graph(
    workspace: SafeWorkspace,
    *,
    model=None,
    require_approval: bool | None = None,
    checkpointer=None,
):
    """Build and compile the agent graph.

    ``model`` may be injected (e.g. a fake) for testing; otherwise it is created
    from environment variables. ``require_approval`` defaults to the
    ``AGENT_REQUIRE_APPROVAL`` env var.
    """
    tools = create_tools(workspace)
    model_with_tools = model if model is not None else _make_model(tools)
    if require_approval is None:
        require_approval = os.getenv("AGENT_REQUIRE_APPROVAL", "0").strip().lower() in {"1", "true", "yes"}

    def call_model(state: AgentState):
        response = model_with_tools.invoke([SystemMessage(content=SYSTEM_PROMPT), *state["messages"]])
        return {"messages": [response]}

    def route_after_agent(state: AgentState):
        route = tools_condition(state)  # returns "tools" or END
        if route != "tools":
            return END
        last = state["messages"][-1]
        if require_approval and any(tc["name"] in SENSITIVE_TOOLS for tc in last.tool_calls):
            return "approval"
        return "tools"

    def approval_node(state: AgentState):
        last = state["messages"][-1]
        pending = [
            {"name": tc["name"], "args": tc["args"]}
            for tc in last.tool_calls
            if tc["name"] in SENSITIVE_TOOLS
        ]
        decision = interrupt({"action": "confirm_sensitive_tools", "pending": pending})
        approved = str(decision).strip().lower() in {"y", "yes", "同意", "approve", "true", "1"}
        if approved:
            return {"approved": True}
        # Deny: satisfy every pending tool call with a ToolMessage, then re-plan.
        denials = [
            ToolMessage(content="用户拒绝执行该操作，请改用其他方案或直接说明。", tool_call_id=tc["id"])
            for tc in last.tool_calls
        ]
        return {"approved": False, "messages": denials}

    def route_after_approval(state: AgentState):
        return "tools" if state.get("approved") else "agent"

    builder = StateGraph(AgentState)
    builder.add_node("agent", call_model)
    builder.add_node("tools", ToolNode(tools, handle_tool_errors=True))
    builder.add_node("approval", approval_node)
    builder.add_edge(START, "agent")
    builder.add_conditional_edges(
        "agent",
        route_after_agent,
        {"tools": "tools", "approval": "approval", END: END},
    )
    builder.add_conditional_edges("approval", route_after_approval, {"tools": "tools", "agent": "agent"})
    builder.add_edge("tools", "agent")

    return builder.compile(checkpointer=checkpointer or InMemorySaver())


def content_as_text(message: AIMessage | None) -> str:
    """Flatten string or block-style AIMessage content to plain text."""
    if message is None:
        return "[无最终回复]"
    if isinstance(message.content, str):
        return message.content
    parts: Iterable[str] = (
        block.get("text", "")
        for block in message.content
        if isinstance(block, dict) and block.get("type") == "text"
    )
    return "\n".join(part for part in parts if part) or "[无文本回复]"


def _stream_until_pause(graph, stream_input, config) -> AIMessage | None:
    """Stream a run to its next pause/end; return the last no-tool AI message."""
    final: AIMessage | None = None
    for event in graph.stream(stream_input, config, stream_mode="values"):
        last = event["messages"][-1]
        if isinstance(last, AIMessage) and not last.tool_calls:
            final = last
    return final


def _pending_interrupt(graph, config):
    """Return the first interrupt payload if the run is paused, else None."""
    snapshot = graph.get_state(config)
    if not snapshot.next:
        return None
    for task in snapshot.tasks:
        if task.interrupts:
            return task.interrupts[0].value
    return None


def run_task(graph, user_input: str, config) -> AIMessage | None:
    """Run one user turn, handling any approval interrupts interactively."""
    final = _stream_until_pause(graph, {"messages": [HumanMessage(content=user_input)]}, config)
    while (payload := _pending_interrupt(graph, config)) is not None:
        print("\n[需要审批] 模型请求执行以下敏感操作：")
        for item in payload.get("pending", []):
            print(f"  - {item['name']}: {item['args']}")
        answer = input("允许执行？(y/N)> ").strip() or "n"
        final = _stream_until_pause(graph, Command(resume=answer), config)
    return final


def run_cli(workspace_path: Path, thread_id: str) -> None:
    workspace = SafeWorkspace(workspace_path)
    graph = build_graph(workspace)
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": DEFAULT_RECURSION_LIMIT}
    print(f"Cursor Code Agent 已启动 | workspace={workspace.root} | thread={thread_id}")
    print("输入任务；/new 新建会话，/state 查看状态，/quit 退出。")

    while True:
        try:
            user_input = input("\n你> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n已退出。")
            return
        if not user_input:
            continue
        if user_input == "/quit":
            print("已退出。")
            return
        if user_input == "/new":
            thread_id = input("新 thread_id> ").strip() or "default"
            config = {"configurable": {"thread_id": thread_id}, "recursion_limit": DEFAULT_RECURSION_LIMIT}
            print(f"已切换到会话 {thread_id}")
            continue
        if user_input == "/state":
            snapshot = graph.get_state(config)
            values = snapshot.values
            print(
                f"messages={len(values.get('messages', []))}, "
                f"changed_files={values.get('changed_files', [])}, "
                f"commands_run={values.get('commands_run', [])}, next={snapshot.next}"
            )
            continue

        try:
            final = run_task(graph, user_input, config)
        except Exception as exc:  # CLI boundary: keep the session alive, surface the real error.
            print(f"Agent 调用失败：{type(exc).__name__}: {exc}")
            continue
        print(f"\nAgent> {content_as_text(final)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Workspace-scoped LangGraph coding agent")
    parser.add_argument("--workspace", type=Path, default=Path(__file__).parent / "workspace")
    parser.add_argument("--thread-id", default="demo-thread")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_cli(args.workspace, args.thread_id)
