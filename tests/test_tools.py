from __future__ import annotations

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
