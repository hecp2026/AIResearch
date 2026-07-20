"""A practical LangGraph coding agent with workspace-scoped tools.

Run:
    pip install -r requirements.txt
    set OPENAI_API_KEY=your-key       # PowerShell: $env:OPENAI_API_KEY="..."
    python agent.py --workspace ./workspace
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
from pathlib import Path
from typing import Iterable

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition


MAX_FILE_BYTES = 100_000
MAX_WRITE_CHARS = 300_000
MAX_TOOL_OUTPUT = 20_000
MAX_LIST_RESULTS = 200
DEFAULT_ALLOWED_COMMANDS = "python,pytest,ruff,mypy"

SYSTEM_PROMPT = """你是一名谨慎、务实的代码工程 Agent。
你的目标是在工具所限定的工作区内理解需求、检查现有代码、实施最小修改并验证结果。

工作规则：
1. 修改前先读取相关文件；不要猜测项目结构。
2. 优先做小而完整的改动，保持现有风格，不改无关文件。
3. 写入后使用可用工具做语法检查或测试；失败时分析输出并修复。
4. 不读取或写入 .env、密钥、凭据，不在回复中索要或回显密钥。
5. 不运行删除、提权、联网下载、包发布、Git 推送等高风险操作。
6. 工具报错时不要虚构成功；说明限制并给出用户可执行的下一步。
7. 最终回答列出：完成内容、修改文件、验证结果、剩余风险（如有）。
"""


def _truncate(text: str, limit: int = MAX_TOOL_OUTPUT) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [输出已截断，共 {len(text)} 个字符]"


def _is_sensitive(path: Path) -> bool:
    sensitive_names = {
        ".env",
        ".env.local",
        ".env.production",
        "credentials.json",
        "secrets.json",
        "id_rsa",
        "id_ed25519",
    }
    name = path.name.lower()
    return name in sensitive_names or name.endswith((".pem", ".key", ".p12", ".pfx"))


def create_tools(workspace: Path):
    """Create tools closed over a single, resolved workspace directory."""
    root = workspace.resolve()
    root.mkdir(parents=True, exist_ok=True)

    def safe_path(relative_path: str) -> Path:
        if not relative_path or "\x00" in relative_path:
            raise ValueError("路径不能为空或包含空字符")
        candidate = (root / relative_path).resolve()
        if candidate != root and root not in candidate.parents:
            raise ValueError("拒绝访问工作区之外的路径")
        if _is_sensitive(candidate):
            raise ValueError("拒绝访问可能包含密钥或凭据的文件")
        return candidate

    @tool
    def list_files(pattern: str = "**/*") -> str:
        """列出工作区内匹配 glob pattern 的文件；适合先了解项目结构。"""
        if Path(pattern).is_absolute() or ".." in Path(pattern).parts:
            return "错误：pattern 必须是工作区内的相对 glob，且不能包含 .."
        files: list[str] = []
        try:
            for item in root.glob(pattern):
                resolved = item.resolve()
                if root not in resolved.parents or not item.is_file() or _is_sensitive(item):
                    continue
                files.append(item.relative_to(root).as_posix())
                if len(files) >= MAX_LIST_RESULTS:
                    break
        except (OSError, ValueError) as exc:
            return f"列举失败：{exc}"
        return "\n".join(sorted(files)) or "没有匹配的文件"

    @tool
    def read_file(relative_path: str, start_line: int = 1, end_line: int = 400) -> str:
        """读取工作区内 UTF-8 文本文件的指定行区间，行号从 1 开始。"""
        try:
            path = safe_path(relative_path)
            if not path.is_file():
                return f"错误：文件不存在：{relative_path}"
            if path.stat().st_size > MAX_FILE_BYTES:
                return f"错误：文件超过 {MAX_FILE_BYTES} 字节限制"
            lines = path.read_text(encoding="utf-8").splitlines()
            start = max(1, start_line)
            end = min(len(lines), max(start, end_line))
            numbered = (f"{i:4d} | {lines[i - 1]}" for i in range(start, end + 1))
            return _truncate("\n".join(numbered) or "[空文件]")
        except (OSError, UnicodeError, ValueError) as exc:
            return f"读取失败：{exc}"

    @tool
    def search_text(query: str, file_pattern: str = "**/*") -> str:
        """在工作区文本文件中按字面量搜索 query，并返回文件、行号和内容。"""
        if not query:
            return "错误：query 不能为空"
        if Path(file_pattern).is_absolute() or ".." in Path(file_pattern).parts:
            return "错误：file_pattern 必须是安全的相对 glob"
        matches: list[str] = []
        try:
            for path in root.glob(file_pattern):
                resolved = path.resolve()
                if (
                    root not in resolved.parents
                    or not path.is_file()
                    or _is_sensitive(resolved)
                    or path.stat().st_size > MAX_FILE_BYTES
                ):
                    continue
                try:
                    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                        if query.lower() in line.lower():
                            rel = path.relative_to(root).as_posix()
                            matches.append(f"{rel}:{number}: {line.strip()}")
                            if len(matches) >= 100:
                                return _truncate("\n".join(matches) + "\n... [达到 100 条上限]")
                except (UnicodeError, OSError):
                    continue
        except (OSError, ValueError) as exc:
            return f"搜索失败：{exc}"
        return _truncate("\n".join(matches) or "没有找到匹配内容")

    @tool
    def write_file(relative_path: str, content: str) -> str:
        """将完整 UTF-8 content 写入工作区文件；会创建父目录并覆盖同名文件。"""
        try:
            if len(content) > MAX_WRITE_CHARS:
                return f"错误：写入内容超过 {MAX_WRITE_CHARS} 字符限制"
            path = safe_path(relative_path)
            if path == root:
                return "错误：目标必须是文件"
            path.parent.mkdir(parents=True, exist_ok=True)
            existed = path.exists()
            path.write_text(content, encoding="utf-8")
            action = "已更新" if existed else "已创建"
            return f"{action} {path.relative_to(root).as_posix()}（{len(content)} 个字符）"
        except (OSError, UnicodeError, ValueError) as exc:
            return f"写入失败：{exc}"

    @tool
    def run_command(command: str, timeout_seconds: int = 60) -> str:
        """在工作区运行无 shell 命令；仅允许环境变量 AGENT_ALLOWED_COMMANDS 中的程序。"""
        allowed = {
            item.strip().lower()
            for item in os.getenv("AGENT_ALLOWED_COMMANDS", DEFAULT_ALLOWED_COMMANDS).split(",")
            if item.strip()
        }
        try:
            args = shlex.split(command, posix=os.name != "nt")
        except ValueError as exc:
            return f"命令解析失败：{exc}"
        if not args:
            return "错误：命令不能为空"
        if any(Path(arg).is_absolute() or ".." in Path(arg).parts for arg in args[1:]):
            return "拒绝运行：命令参数不能包含绝对路径或 .."
        executable = Path(args[0]).name.lower()
        if executable.endswith(".exe"):
            executable = executable[:-4]
        if executable not in allowed:
            return f"拒绝运行 {executable!r}；允许：{', '.join(sorted(allowed))}"
        timeout = min(max(timeout_seconds, 1), 120)
        try:
            completed = subprocess.run(
                args,
                cwd=root,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                shell=False,
                env=os.environ.copy(),
            )
            output = (
                f"exit_code={completed.returncode}\n"
                f"--- stdout ---\n{completed.stdout}\n"
                f"--- stderr ---\n{completed.stderr}"
            )
            return _truncate(output)
        except subprocess.TimeoutExpired:
            return f"命令在 {timeout} 秒后超时并已终止"
        except OSError as exc:
            return f"命令执行失败：{exc}"

    return [list_files, read_file, search_text, write_file, run_command]


def build_graph(workspace: Path):
    """Build and compile the LangGraph ReAct loop."""
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("未设置 OPENAI_API_KEY；请先通过环境变量配置 API Key。")

    tools = create_tools(workspace)
    model_kwargs: dict[str, object] = {
        "model": os.getenv("OPENAI_MODEL", "gpt-5-mini"),
        "timeout": 60,
        "max_retries": 2,
    }
    if os.getenv("OPENAI_BASE_URL"):
        model_kwargs["base_url"] = os.environ["OPENAI_BASE_URL"]
    model_with_tools = ChatOpenAI(**model_kwargs).bind_tools(tools)

    def call_model(state: MessagesState):
        response = model_with_tools.invoke([SystemMessage(content=SYSTEM_PROMPT), *state["messages"]])
        return {"messages": [response]}

    builder = StateGraph(MessagesState)
    builder.add_node("model", call_model)
    builder.add_node("tools", ToolNode(tools, handle_tool_errors=True))
    builder.add_edge(START, "model")
    builder.add_conditional_edges("model", tools_condition, {"tools": "tools", END: END})
    builder.add_edge("tools", "model")

    return builder.compile(checkpointer=InMemorySaver())


def _content_as_text(message: AIMessage) -> str:
    if isinstance(message.content, str):
        return message.content
    parts: Iterable[str] = (
        block.get("text", "")
        for block in message.content
        if isinstance(block, dict) and block.get("type") == "text"
    )
    return "\n".join(part for part in parts if part)


def run_cli(workspace: Path, thread_id: str) -> None:
    graph = build_graph(workspace)
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 30}
    print(f"Code Agent 已启动 | workspace={workspace.resolve()} | thread={thread_id}")
    print("输入任务；/new 新建会话，/state 查看状态摘要，/quit 退出。")

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
            config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 30}
            print(f"已切换到会话 {thread_id}")
            continue
        if user_input == "/state":
            snapshot = graph.get_state(config)
            print(f"messages={len(snapshot.values.get('messages', []))}, next={snapshot.next}")
            continue

        final_message: AIMessage | None = None
        try:
            for event in graph.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config,
                stream_mode="values",
            ):
                last = event["messages"][-1]
                if isinstance(last, AIMessage) and not last.tool_calls:
                    final_message = last
        except Exception as exc:  # CLI boundary: keep the session alive and show the real failure.
            print(f"Agent 调用失败：{type(exc).__name__}: {exc}")
            continue
        print(f"\nAgent> {_content_as_text(final_message) if final_message else '[无最终回复]'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Workspace-scoped LangGraph coding agent")
    parser.add_argument("--workspace", type=Path, default=Path("workspace"))
    parser.add_argument("--thread-id", default="demo-thread")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_cli(args.workspace, args.thread_id)
