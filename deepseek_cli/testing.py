"""Test command heuristics for guiding the agent."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, List


def detect_test_commands(root: Path) -> List[str]:
    """Return a list of likely test commands for the given project root."""

    def add_unique(commands: List[str], *candidates: Iterable[str]) -> None:
        for candidate in candidates:
            if not candidate:
                continue
            if candidate not in commands:
                commands.append(candidate)

    commands: List[str] = []

    # Python projects
    if _looks_like_python_project(root):
        add_unique(commands, "pytest")
    if (root / "manage.py").exists():
        add_unique(commands, "python manage.py test")
    if (root / "tox.ini").exists():
        add_unique(commands, "tox")
    if (root / "noxfile.py").exists():
        add_unique(commands, "nox")

    # Node / JavaScript projects
    package_json = root / "package.json"
    if package_json.exists():
        script = _read_package_json_test_script(package_json)
        if script and "jest" in script and " --watch" in script:
            add_unique(commands, "npm test -- --watch=false")
        add_unique(commands, _pick_node_test_runner(root, script))

    # Rust, Go, PHP, Java, etc.
    if (root / "Cargo.toml").exists():
        add_unique(commands, "cargo test")
    if (root / "go.mod").exists():
        add_unique(commands, "go test ./...")
    if (root / "composer.json").exists():
        add_unique(commands, "composer test")
    if (root / "pom.xml").exists():
        add_unique(commands, "mvn test")

    makefile = _find_makefile(root)
    if makefile is not None and _has_make_like_target(makefile, "test"):
        add_unique(commands, "make test")

    justfile = _find_justfile(root)
    if justfile is not None and _has_just_target(justfile, "test"):
        add_unique(commands, "just test")

    return [cmd for cmd in commands if cmd]


def build_test_followups(root: Path) -> List[str]:
    """Generate agent follow-up hints that list preferred test commands."""
    commands = detect_test_commands(root)
    if not commands:
        return []
    header = "Testing hints: based on the repository layout, prefer running tests with the following commands (try them in this order):"
    numbered = "\n".join(f"{index}. `{command}`" for index, command in enumerate(commands, start=1))
    footer = (
        "If a command fails, diagnose and fix issues before proceeding. "
        "Document any unimplemented test coverage in the final summary."
    )
    return [f"{header}\n{numbered}\n{footer}"]


def _looks_like_python_project(root: Path) -> bool:
    if (root / "pytest.ini").exists() or (root / "conftest.py").exists():
        return True
    if any((root / name).exists() for name in ("requirements.txt", "setup.cfg", "pyproject.toml")):
        tests_dir = root / "tests"
        if tests_dir.exists() and tests_dir.is_dir():
            return True
    return False


def _read_package_json_test_script(package_json: Path) -> str | None:
    try:
        data = json.loads(package_json.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    scripts = data.get("scripts")
    if not isinstance(scripts, dict):
        return None
    value = scripts.get("test")
    if not isinstance(value, str):
        return None
    if value.strip().startswith("echo \"Error: no test specified\""):
        return None
    return value.strip()


def _pick_node_test_runner(root: Path, script: str | None) -> str | None:
    if not script:
        return None
    if (root / "pnpm-lock.yaml").exists():
        return "pnpm test"
    if (root / "yarn.lock").exists():
        return "yarn test"
    if (root / "bun.lockb").exists():
        return "bun test"
    if script.startswith("jest") or " jest" in script:
        return "npm test"
    if "vitest" in script:
        return "npm run test -- --run"
    if script.startswith("pnpm"):
        return "pnpm test"
    if script.startswith("yarn"):
        return "yarn test"
    if script.startswith("npm run"):
        return script
    return "npm test"


def _find_makefile(root: Path) -> Path | None:
    for name in ("Makefile", "makefile"):
        candidate = root / name
        if candidate.exists():
            return candidate
    return None


def _find_justfile(root: Path) -> Path | None:
    for name in ("Justfile", "justfile"):
        candidate = root / name
        if candidate.exists():
            return candidate
    return None


def _has_make_like_target(path: Path, target: str) -> bool:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return False
    return bool(re.search(rf"^{re.escape(target)}\s*:", text, flags=re.MULTILINE))


def _has_just_target(path: Path, target: str) -> bool:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return False
    return bool(re.search(rf"^{re.escape(target)}\s*:", text, flags=re.MULTILINE))


__all__ = [
    "build_test_followups",
    "detect_test_commands",
]
