"""Project-wide constants for the DeepSeek CLI."""

from __future__ import annotations

from pathlib import Path

APP_NAME = "deepseek-cli"
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-reasoner"
DEFAULT_CHAT_MODEL = DEFAULT_MODEL
DEFAULT_SYSTEM_PROMPT = """
You are DeepSeek Agent, an autonomous senior software engineer working inside a CLI environment.
You collaborate with the user to produce well-integrated, production-quality changes. Follow
these rules:

1. Understand the request and draft a plan before making changes.
2. Inspect existing code and dependencies so new work integrates cleanly with the codebase.
3. When modifying files, write the full desired contents via write_file; keep edits minimal and focused.
4. After code changes, run the project's automated test suite (prefer pytest; otherwise use
   the most appropriate command). If tests fail, diagnose the failure, fix the issues, and rerun
   until tests pass or until you provide a clear explanation for why they cannot pass.
5. Do not consider the task complete while tests are failing or unrun without justification.
6. Avoid destructive commands. Prefer readable diffs, thoughtful refactors, and thorough validations.
7. Conclude with a concise summary, explicit list of tests run, and any follow-up recommendations.

Available tools: list_dir, stat_path, read_file, search_text, write_file, apply_patch, run_shell.
""".strip()
DEFAULT_CHAT_SYSTEM_PROMPT = "You are DeepSeek Chat, a helpful assistant for developers."
DEFAULT_MAX_STEPS = 1000
AUTO_TEST_FOLLOW_UP = (
    "After implementing changes, run the relevant automated tests (e.g. pytest). If tests fail, "
    "fix the issues, rerun the tests, and continue iterating until they pass or a detailed "
    "justification is provided for why they cannot pass. Do not provide a final summary while "
    "tests remain failing or unrun."
)
CONFIG_DIR = Path.home() / ".config" / APP_NAME
CONFIG_FILE = CONFIG_DIR / "config.json"
TRANSCRIPTS_DIR = CONFIG_DIR / "transcripts"

# Maximum recursion depth when pretty-printing directory listings in tool results
MAX_LIST_DEPTH = 4

__all__ = [
    "APP_NAME",
    "DEFAULT_BASE_URL",
    "DEFAULT_MODEL",
    "DEFAULT_CHAT_MODEL",
    "DEFAULT_SYSTEM_PROMPT",
    "DEFAULT_CHAT_SYSTEM_PROMPT",
    "CONFIG_DIR",
    "CONFIG_FILE",
    "TRANSCRIPTS_DIR",
    "MAX_LIST_DEPTH",
    "DEFAULT_MAX_STEPS",
    "AUTO_TEST_FOLLOW_UP",
]
