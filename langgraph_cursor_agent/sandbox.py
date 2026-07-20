"""Workspace-scoped sandbox for the LangGraph coding agent.

This module has ZERO third-party dependencies. All security-critical logic
(path confinement, sensitive-file blocking, command allowlisting) lives here so
it can be unit-tested with the standard library alone, independent of LangGraph
or any model provider.

The agent (``agent.py``) wraps these methods as LangGraph tools.
"""

from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path

MAX_FILE_BYTES = 100_000
MAX_WRITE_CHARS = 300_000
MAX_TOOL_OUTPUT = 20_000
MAX_LIST_RESULTS = 200
MAX_SEARCH_RESULTS = 100
DEFAULT_ALLOWED_COMMANDS = ("python", "pytest", "ruff", "mypy")

_SENSITIVE_NAMES = {
    ".env",
    ".env.local",
    ".env.production",
    "credentials.json",
    "secrets.json",
    "id_rsa",
    "id_ed25519",
}
_SENSITIVE_SUFFIXES = (".pem", ".key", ".p12", ".pfx")


def truncate(text: str, limit: int = MAX_TOOL_OUTPUT) -> str:
    """Cap tool output so a single result cannot flood the model context."""
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [输出已截断，共 {len(text)} 个字符]"


def is_sensitive(path: Path) -> bool:
    """Return True for files that may hold secrets and must never be touched."""
    name = path.name.lower()
    return name in _SENSITIVE_NAMES or name.endswith(_SENSITIVE_SUFFIXES)


class WorkspaceError(ValueError):
    """Raised when a request violates the workspace security boundary."""


class SafeWorkspace:
    """Confines every file and command operation to a single directory."""

    def __init__(self, root: str | os.PathLike[str], allowed_commands=None) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        if allowed_commands is None:
            env_value = os.getenv("AGENT_ALLOWED_COMMANDS")
            if env_value:
                allowed_commands = [item.strip() for item in env_value.split(",")]
            else:
                allowed_commands = DEFAULT_ALLOWED_COMMANDS
        self.allowed_commands = {item.lower() for item in allowed_commands if item}

    # -- internal helpers -------------------------------------------------
    def safe_path(self, relative_path: str) -> Path:
        """Resolve a user path and reject escapes or sensitive targets."""
        if not relative_path or "\x00" in relative_path:
            raise WorkspaceError("路径不能为空或包含空字符")
        candidate = (self.root / relative_path).resolve()
        if candidate != self.root and self.root not in candidate.parents:
            raise WorkspaceError("拒绝访问工作区之外的路径")
        if is_sensitive(candidate):
            raise WorkspaceError("拒绝访问可能包含密钥或凭据的文件")
        return candidate

    def _safe_glob(self, pattern: str):
        if Path(pattern).is_absolute() or ".." in Path(pattern).parts:
            raise WorkspaceError("pattern 必须是工作区内的相对 glob，且不能包含 ..")
        for item in self.root.glob(pattern):
            resolved = item.resolve()
            if resolved != self.root and self.root not in resolved.parents:
                continue
            if is_sensitive(resolved):
                continue
            yield item

    # -- read-only tools --------------------------------------------------
    def list_files(self, pattern: str = "**/*") -> str:
        """List workspace files matching a relative glob pattern."""
        try:
            files: list[str] = []
            for item in self._safe_glob(pattern):
                if not item.is_file():
                    continue
                files.append(item.relative_to(self.root).as_posix())
                if len(files) >= MAX_LIST_RESULTS:
                    break
        except (OSError, WorkspaceError) as exc:
            return f"列举失败：{exc}"
        return "\n".join(sorted(files)) or "没有匹配的文件"

    def read_file(self, relative_path: str, start_line: int = 1, end_line: int = 400) -> str:
        """Read a UTF-8 text file within an inclusive 1-based line range."""
        try:
            path = self.safe_path(relative_path)
            if not path.is_file():
                return f"错误：文件不存在：{relative_path}"
            if path.stat().st_size > MAX_FILE_BYTES:
                return f"错误：文件超过 {MAX_FILE_BYTES} 字节限制"
            lines = path.read_text(encoding="utf-8").splitlines()
            if not lines:
                return "[空文件]"
            start = max(1, start_line)
            end = min(len(lines), max(start, end_line))
            numbered = (f"{i:4d} | {lines[i - 1]}" for i in range(start, end + 1))
            return truncate("\n".join(numbered))
        except (OSError, UnicodeError, WorkspaceError) as exc:
            return f"读取失败：{exc}"

    def search_text(self, query: str, file_pattern: str = "**/*") -> str:
        """Case-insensitive literal search across workspace text files."""
        if not query:
            return "错误：query 不能为空"
        matches: list[str] = []
        try:
            for path in self._safe_glob(file_pattern):
                if not path.is_file() or path.stat().st_size > MAX_FILE_BYTES:
                    continue
                try:
                    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                        if query.lower() in line.lower():
                            rel = path.relative_to(self.root).as_posix()
                            matches.append(f"{rel}:{number}: {line.strip()}")
                            if len(matches) >= MAX_SEARCH_RESULTS:
                                return truncate("\n".join(matches) + f"\n... [达到 {MAX_SEARCH_RESULTS} 条上限]")
                except (UnicodeError, OSError):
                    continue
        except (OSError, WorkspaceError) as exc:
            return f"搜索失败：{exc}"
        return truncate("\n".join(matches) or "没有找到匹配内容")

    # -- side-effecting tools --------------------------------------------
    def write_file(self, relative_path: str, content: str) -> tuple[str, str | None]:
        """Write full UTF-8 content to a file. Returns (message, changed_path)."""
        try:
            if len(content) > MAX_WRITE_CHARS:
                return f"错误：写入内容超过 {MAX_WRITE_CHARS} 字符限制", None
            path = self.safe_path(relative_path)
            if path == self.root:
                return "错误：目标必须是文件", None
            path.parent.mkdir(parents=True, exist_ok=True)
            existed = path.exists()
            path.write_text(content, encoding="utf-8")
            rel = path.relative_to(self.root).as_posix()
            action = "已更新" if existed else "已创建"
            return f"{action} {rel}（{len(content)} 个字符）", rel
        except (OSError, UnicodeError, WorkspaceError) as exc:
            return f"写入失败：{exc}", None

    def run_command(self, command: str, timeout_seconds: int = 60) -> tuple[str, str | None]:
        """Run an allowlisted, shell-free command. Returns (output, command_run)."""
        try:
            args = shlex.split(command, posix=os.name != "nt")
        except ValueError as exc:
            return f"命令解析失败：{exc}", None
        if not args:
            return "错误：命令不能为空", None
        if any(Path(arg).is_absolute() or ".." in Path(arg).parts for arg in args[1:]):
            return "拒绝运行：命令参数不能包含绝对路径或 ..", None
        executable = Path(args[0]).name.lower()
        if executable.endswith(".exe"):
            executable = executable[:-4]
        if executable not in self.allowed_commands:
            allowed = ", ".join(sorted(self.allowed_commands))
            return f"拒绝运行 {executable!r}；允许：{allowed}", None
        timeout = min(max(int(timeout_seconds), 1), 120)
        try:
            completed = subprocess.run(
                args,
                cwd=self.root,
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
            return truncate(output), command
        except subprocess.TimeoutExpired:
            return f"命令在 {timeout} 秒后超时并已终止", None
        except OSError as exc:
            return f"命令执行失败：{exc}", None
