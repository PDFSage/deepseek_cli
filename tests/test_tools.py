from __future__ import annotations

import stat
from pathlib import Path
from unittest import mock

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


def test_tavily_search_requires_query(tmp_path: Path) -> None:
    executor = ToolExecutor(root=tmp_path)
    response = executor.tavily_search("")
    assert "must not be empty" in response


def test_move_path_moves_file(tmp_path: Path) -> None:
    source = tmp_path / "a.txt"
    source.write_text("hello", encoding="utf-8")
    executor = ToolExecutor(root=tmp_path)
    result = executor.move_path("a.txt", "nested/b.txt", create_parents=True)
    assert "Moved" in result
    destination = tmp_path / "nested" / "b.txt"
    assert destination.exists()
    assert not source.exists()
    assert destination.read_text(encoding="utf-8") == "hello"


def test_download_file_persists_bytes(tmp_path: Path) -> None:
    executor = ToolExecutor(root=tmp_path)
    payload = b"sample-bytes"
    mock_response = mock.MagicMock()
    mock_response.read.return_value = payload
    mock_response.__enter__.return_value = mock_response
    mock_response.__exit__.return_value = False
    with mock.patch("urllib.request.urlopen", return_value=mock_response):
        outcome = executor.download_file(
            "https://example.com/archive.bin",
            "artifacts/archive.bin",
            create_parents=True,
        )
    assert "Downloaded" in outcome
    target = tmp_path / "artifacts" / "archive.bin"
    assert target.exists()
    assert target.read_bytes() == payload
