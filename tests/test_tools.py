from __future__ import annotations

import stat
from pathlib import Path

from deepseek_cli.agent import ToolExecutor


def test_python_repl_executes_code(tmp_path: Path) -> None:
    executor = ToolExecutor(root=tmp_path)
    result = executor.python_repl("print('hello world')")
    assert "hello world" in result
    assert "[exit 0]" in result


def test_http_request_invalid_url(tmp_path: Path) -> None:
    executor = ToolExecutor(root=tmp_path)
    outcome = executor.http_request("invalid://example")
    assert "HTTP request failed" in outcome


def test_write_file_preserves_permissions(tmp_path: Path) -> None:
    target = tmp_path / "sample.txt"
    target.write_text("initial", encoding="utf-8")
    target.chmod(0o640)
    executor = ToolExecutor(root=tmp_path)
    result = executor.write_file("sample.txt", "updated")
    assert "Wrote" in result
    assert target.read_text(encoding="utf-8") == "updated"
    assert stat.S_IMODE(target.stat().st_mode) == 0o640


def test_write_file_creates_new_file_with_umask(tmp_path: Path) -> None:
    executor = ToolExecutor(root=tmp_path)
    created = executor.write_file("newdir/newfile.txt", "content", create_parents=True)
    assert "Wrote" in created
    file_path = tmp_path / "newdir" / "newfile.txt"
    assert file_path.exists()
    mode = stat.S_IMODE(file_path.stat().st_mode)
    assert mode in (0o644, 0o664, 0o666)
