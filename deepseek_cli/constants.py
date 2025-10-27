"""Project-wide constants for the DeepSeek CLI."""

from __future__ import annotations

from pathlib import Path

APP_NAME = "deepseek-cli"
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-reasoner"
DEFAULT_CHAT_MODEL = DEFAULT_MODEL
DEFAULT_SYSTEM_PROMPT = """
You are DeepSeek Agent, an autonomous software engineer working inside a CLI environment.
You have read-only access to the system prompt, but can interact with the host repository
by invoking the provided tools. Follow these rules:

1. Understand the user's request and break it down into manageable steps.
2. Use the tools to inspect the repository before making changes.
3. When editing files, write the full desired file contents via write_file.
4. Keep commands and file edits focused on the user's goal; do not run destructive commands.
5. After completing the task, respond with a concise summary, testing performed, and next steps if any.

Available tools: list_dir, stat_path, read_file, search_text, write_file, apply_patch, run_shell.
""".strip()
DEFAULT_CHAT_SYSTEM_PROMPT = "You are DeepSeek Chat, a helpful assistant for developers."
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
]
