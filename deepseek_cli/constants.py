"""Project-wide constants for the DeepSeek CLI."""

from __future__ import annotations

from pathlib import Path

APP_NAME = "deepseek-cli"
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-reasoner"
DEFAULT_CHAT_MODEL = DEFAULT_MODEL
DEFAULT_COMPLETION_MODEL = DEFAULT_MODEL
DEFAULT_EMBEDDING_MODEL = DEFAULT_MODEL
DEFAULT_TAVILY_API_KEY = "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"
DEFAULT_SYSTEM_PROMPT = """
You are DeepSeek Agent, an autonomous senior software engineer working inside a CLI environment.
You collaborate with the user to produce well-integrated, production-quality changes. Follow
these rules:

1. Understand the request and draft a step-by-step plan. Revise the plan whenever new information
   or test results show that the current approach is insufficient.
2. Inspect existing code, dependencies, and project context so new work integrates cleanly.
3. When modifying files, write the full desired contents via write_file; keep edits minimal and focused.
4. After code changes, run the project's automated test suite (prefer pytest; otherwise use the most
   appropriate command). If tests fail, diagnose, fix, and rerun until the suite passes or a clear
   justification is provided for why it cannot pass right now.
5. Proactively search for bugs and regressions introduced by your changes. Apply fixes before finalising.
6. Avoid destructive commands. Prefer readable diffs, thoughtful refactors, and thorough validations.
7. Narrate your reasoning. For each step, explain what you are planning, which tools you are using,
   and why. Print intermediate observations, decisions, and next steps so the user can follow your
   thought process.
8. Conclude with a concise summary, explicit list of tests run, outstanding risks, and follow-up recommendations.

Available tools: list_dir, stat_path, read_file, search_text, write_file, apply_patch, run_shell.
""".strip()
DEFAULT_CHAT_SYSTEM_PROMPT = "You are DeepSeek Chat, a helpful assistant for developers."
DEFAULT_COMPLETION_SYSTEM_PROMPT = (
    "You are DeepSeek Codex, a code completion engine. "
    "Given surrounding context, return only the code the developer should insert next. "
    "Do not repeat the provided prefix or suffix, and avoid explanations.\n"
)
DEFAULT_CHAT_STREAM_STYLE = "plain"
DEFAULT_MAX_STEPS = 5000
AUTO_TEST_FOLLOW_UP = (
    "After implementing changes, run the relevant automated tests (e.g. pytest). If tests fail, "
    "fix the issues, rerun the tests, and continue iterating until they pass or a detailed "
    "justification is provided for why they cannot pass. Do not provide a final summary while "
    "tests remain failing or unrun."
)
AUTO_BUG_FOLLOW_UP = (
    "Actively look for bugs or regressions caused by recent changes. If any issues are detected, "
    "update the plan as needed, apply fixes, and re-run verification until the repository is stable."
)
MAX_TOOL_RESULT_CHARS = 12000
CONFIG_DIR = Path.home() / ".config" / APP_NAME
CONFIG_FILE = CONFIG_DIR / "config.json"
TRANSCRIPTS_DIR = CONFIG_DIR / "transcripts"

# Maximum recursion depth when pretty-printing directory listings in tool results
MAX_LIST_DEPTH = 4

STREAM_STYLE_CHOICES = ("plain", "markdown", "rich")

__all__ = [
    "APP_NAME",
    "DEFAULT_BASE_URL",
    "DEFAULT_MODEL",
    "DEFAULT_CHAT_MODEL",
    "DEFAULT_COMPLETION_MODEL",
    "DEFAULT_SYSTEM_PROMPT",
    "DEFAULT_CHAT_SYSTEM_PROMPT",
    "DEFAULT_COMPLETION_SYSTEM_PROMPT",
    "DEFAULT_TAVILY_API_KEY",
    "CONFIG_DIR",
    "CONFIG_FILE",
    "TRANSCRIPTS_DIR",
    "MAX_LIST_DEPTH",
    "DEFAULT_MAX_STEPS",
    "AUTO_TEST_FOLLOW_UP",
    "AUTO_BUG_FOLLOW_UP",
    "MAX_TOOL_RESULT_CHARS",
    "DEFAULT_CHAT_STREAM_STYLE",
    "DEFAULT_EMBEDDING_MODEL",
    "STREAM_STYLE_CHOICES",
]
