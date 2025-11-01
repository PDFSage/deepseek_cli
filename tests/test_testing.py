from __future__ import annotations

import json
from pathlib import Path

from deepseek_cli.testing import build_test_followups, detect_test_commands


def test_detect_python_pytest(tmp_path: Path) -> None:
    (tmp_path / "tests").mkdir()
    (tmp_path / "pyproject.toml").write_text("[build-system]\nrequires = []\n", encoding="utf-8")
    commands = detect_test_commands(tmp_path)
    assert "pytest" in commands

    followups = build_test_followups(tmp_path)
    assert followups
    assert "pytest" in followups[0]


def test_detect_node_pnpm(tmp_path: Path) -> None:
    package_json = {
        "name": "demo",
        "version": "0.0.0",
        "scripts": {"test": "vitest run"},
    }
    (tmp_path / "package.json").write_text(json.dumps(package_json), encoding="utf-8")
    (tmp_path / "pnpm-lock.yaml").write_text("", encoding="utf-8")
    commands = detect_test_commands(tmp_path)
    assert "pnpm test" in commands


def test_detect_makefile(tmp_path: Path) -> None:
    (tmp_path / "Makefile").write_text("test:\n\tpytest\n", encoding="utf-8")
    commands = detect_test_commands(tmp_path)
    assert "make test" in commands
