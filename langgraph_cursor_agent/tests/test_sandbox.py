"""Dependency-free tests for the security boundary in sandbox.py.

These run with only the standard library + pytest, so the most safety-critical
logic is verifiable even where LangGraph / model SDKs are not installed.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sandbox import SafeWorkspace, WorkspaceError, is_sensitive  # noqa: E402


@pytest.fixture()
def ws(tmp_path: Path) -> SafeWorkspace:
    return SafeWorkspace(tmp_path, allowed_commands=["python"])


def test_write_then_read_roundtrip(ws: SafeWorkspace) -> None:
    message, changed = ws.write_file("pkg/mod.py", "x = 1\ny = 2\n")
    assert changed == "pkg/mod.py"
    assert "已创建" in message
    out = ws.read_file("pkg/mod.py")
    assert "x = 1" in out and "y = 2" in out


def test_write_reports_update_on_second_write(ws: SafeWorkspace) -> None:
    ws.write_file("a.txt", "one")
    message, changed = ws.write_file("a.txt", "two")
    assert changed == "a.txt"
    assert "已更新" in message


def test_path_traversal_is_blocked(ws: SafeWorkspace) -> None:
    with pytest.raises(WorkspaceError):
        ws.safe_path("../escape.txt")
    message, changed = ws.write_file("../escape.txt", "nope")
    assert changed is None
    assert "写入失败" in message


def test_absolute_path_is_blocked(ws: SafeWorkspace, tmp_path: Path) -> None:
    outside = (tmp_path.parent / "outside.txt").resolve()
    with pytest.raises(WorkspaceError):
        ws.safe_path(str(outside))


def test_sensitive_files_are_rejected(ws: SafeWorkspace) -> None:
    assert is_sensitive(Path(".env"))
    assert is_sensitive(Path("server.key"))
    with pytest.raises(WorkspaceError):
        ws.safe_path(".env")


def test_read_missing_file(ws: SafeWorkspace) -> None:
    assert "文件不存在" in ws.read_file("missing.py")


def test_read_line_range(ws: SafeWorkspace) -> None:
    ws.write_file("multi.txt", "l1\nl2\nl3\nl4\n")
    out = ws.read_file("multi.txt", start_line=2, end_line=3)
    assert "l2" in out and "l3" in out
    assert "l1" not in out and "l4" not in out


def test_list_files_pattern(ws: SafeWorkspace) -> None:
    ws.write_file("a.py", "1")
    ws.write_file("b.txt", "2")
    ws.write_file("sub/c.py", "3")
    listed = ws.list_files("**/*.py")
    assert "a.py" in listed and "sub/c.py" in listed
    assert "b.txt" not in listed


def test_list_rejects_parent_traversal_pattern(ws: SafeWorkspace) -> None:
    assert "列举失败" in ws.list_files("../*")


def test_search_text_finds_matches(ws: SafeWorkspace) -> None:
    ws.write_file("code.py", "def foo():\n    return TODO\n")
    hits = ws.search_text("todo")
    assert "code.py:2" in hits


def test_run_command_rejects_non_allowlisted(ws: SafeWorkspace) -> None:
    output, ran = ws.run_command("rm -rf /")
    assert ran is None
    assert "拒绝运行" in output


def test_run_command_rejects_traversal_args(ws: SafeWorkspace) -> None:
    output, ran = ws.run_command("python ../evil.py")
    assert ran is None
    assert "拒绝运行" in output


def test_run_command_executes_allowlisted(ws: SafeWorkspace) -> None:
    ws.write_file("hello.py", "print('hi-from-script')\n")
    output, ran = ws.run_command("python hello.py")
    assert ran == "python hello.py"
    assert "exit_code=0" in output
    assert "hi-from-script" in output
