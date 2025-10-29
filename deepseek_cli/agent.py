"""Agent execution primitives for the DeepSeek CLI."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

from rich.console import Console

from openai import OpenAI

from .constants import MAX_LIST_DEPTH, MAX_TOOL_RESULT_CHARS

ToolResult = str


@dataclass
class AgentOptions:
    """Options controlling the agent orchestration loop."""

    model: str
    system_prompt: str
    user_prompt: str
    follow_up: List[str]
    workspace: Path
    read_only: bool
    allow_global_access: bool
    max_steps: int
    verbose: bool
    transcript_path: Optional[Path]


@dataclass
class ToolExecutor:
    """Executes tool calls on behalf of the agent."""

    root: Path
    encoding: str = "utf-8"
    read_only: bool = False
    allow_global_access: bool = False

    def list_dir(self, path: str = ".", recursive: bool = False) -> ToolResult:
        target = _ensure_within_root(self.root, path, self.allow_global_access)
        if not target.exists():
            return f"Path '{path}' does not exist."

        def iter_entries(base: Path, depth: int = 0) -> Iterable[str]:
            if depth > MAX_LIST_DEPTH:
                yield "    " * depth + "… (max depth reached)"
                return
            entries = sorted(base.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
            for entry in entries:
                marker = "/" if entry.is_dir() else ""
                yield "    " * depth + entry.name + marker
                if recursive and entry.is_dir():
                    yield from iter_entries(entry, depth + 1)

        lines = [f"Listing for {target.relative_to(self.root) if target != self.root else '.'}:"]
        lines.extend(iter_entries(target))
        return "\n".join(lines)

    def read_file(self, path: str, offset: int = 0, limit: Optional[int] = None) -> ToolResult:
        target = _ensure_within_root(self.root, path, self.allow_global_access)
        if not target.exists():
            return f"File '{path}' does not exist."
        if not target.is_file():
            return f"Path '{path}' is not a file."

        text = target.read_text(encoding=self.encoding)
        if offset:
            text = text[offset:]
        if limit is not None:
            text = text[:limit]
        return text

    def write_file(self, path: str, content: str, create_parents: bool = False) -> ToolResult:
        if self.read_only:
            return "Write operations are disabled (read-only mode)."
        target = _ensure_within_root(self.root, path, self.allow_global_access)
        if create_parents:
            target.parent.mkdir(parents=True, exist_ok=True)
        if not target.parent.exists():
            return (
                f"Cannot write '{path}': parent directory does not exist. "
                "Pass create_parents=true to create it."
            )
        target.write_text(content, encoding=self.encoding)
        return f"Wrote {len(content)} characters to '{path}'."

    def stat_path(self, path: str = ".") -> ToolResult:
        target = _ensure_within_root(self.root, path, self.allow_global_access)
        if not target.exists():
            return f"Path '{path}' does not exist."
        stats = target.stat()
        info = {
            "path": str(target.relative_to(self.root)),
            "type": "directory" if target.is_dir() else "file" if target.is_file() else "other",
            "size": stats.st_size,
            "modified": datetime.fromtimestamp(stats.st_mtime).isoformat(),
        }
        if target.is_symlink():
            info["symlink_target"] = os.readlink(target)
        return json.dumps(info, indent=2)

    def search_text(
        self,
        pattern: str,
        path: str = ".",
        case_sensitive: bool = True,
        max_results: int = 200,
    ) -> ToolResult:
        target = _ensure_within_root(self.root, path, self.allow_global_access)
        if not target.exists():
            return f"Search path '{path}' does not exist."
        if not pattern:
            return "Search pattern must not be empty."
        use_rg = shutil.which("rg") is not None
        if use_rg:
            cmd = ["rg", "--line-number", "--color", "never"]
            if not case_sensitive:
                cmd.append("-i")
            cmd.extend(["--max-count", str(max_results), pattern, str(target)])
        else:
            cmd = ["grep", "-R", "-n", "-I"]
            if not case_sensitive:
                cmd.append("-i")
            cmd.extend([pattern, str(target)])
        try:
            proc = subprocess.run(
                cmd,
                text=True,
                capture_output=True,
                cwd=self.root,
            )
        except FileNotFoundError:
            return "Neither ripgrep nor grep is available on this system."
        stdout = proc.stdout.strip()
        stderr = proc.stderr.strip()
        if proc.returncode not in (0, 1):
            return f"Search command failed (exit {proc.returncode}).\n{stderr}"
        if not stdout:
            return "No matches found."
        lines = stdout.splitlines()
        truncated = ""
        if len(lines) > max_results:
            lines = lines[:max_results]
            truncated = f"\n… truncated to {max_results} results."
        result = "\n".join(lines) + truncated
        if stderr:
            result += f"\n[stderr]\n{stderr}"
        return result

    def apply_patch(self, patch: str) -> ToolResult:
        if self.read_only:
            return "Patch operations are disabled (read-only mode)."
        if not patch.strip():
            return "Patch content is empty."

        def _safe_path(text: str) -> bool:
            if self.allow_global_access:
                return True
            text = text.strip()
            if text in {"/dev/null", "a/", "b/"}:
                return True
            prefixes = ("a/", "b/", "c/")
            for prefix in prefixes:
                if text.startswith(prefix):
                    text = text[len(prefix):]
                    break
            if text.startswith("/"):
                return False
            parts = Path(text).parts
            return ".." not in parts

        for line in patch.splitlines():
            if line.startswith(("+++", "---")):
                tokens = line.split(maxsplit=1)
                if len(tokens) == 2 and not _safe_path(tokens[1]):
                    return f"Unsafe path detected in patch header: {tokens[1]}"
        patch_cmd = shutil.which("patch")
        patch_level = 1 if any(line.startswith("diff --git") for line in patch.splitlines()) else 0
        if patch_cmd:
            proc = subprocess.run(
                [patch_cmd, f"-p{patch_level}", "--batch", "--silent"],
                input=patch,
                text=True,
                capture_output=True,
                cwd=self.root,
            )
        else:
            git_cmd = shutil.which("git")
            if not git_cmd:
                return "No patch utility available (missing both patch and git)."
            proc = subprocess.run(
                [git_cmd, "apply", "--whitespace=nowarn", f"-p{patch_level}"],
                input=patch,
                text=True,
                capture_output=True,
                cwd=self.root,
            )
        stdout = proc.stdout.strip()
        stderr = proc.stderr.strip()
        if proc.returncode != 0:
            message = stderr or "Patch command failed"
            return f"Patch failed (exit {proc.returncode}).\n{message}"
        response_lines = ["Patch applied successfully."]
        if stdout:
            response_lines.append(stdout)
        if stderr:
            response_lines.append(f"[stderr]\n{stderr}")
        return "\n".join(response_lines)

    def run_shell(self, command: str, timeout: int = 120) -> ToolResult:
        if not command.strip():
            return "Command is empty."
        try:
            proc = subprocess.run(
                ["/bin/bash", "-lc", command],
                cwd=self.root,
                text=True,
                capture_output=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return f"Command timed out after {timeout} seconds."
        stdout = proc.stdout.strip()
        stderr = proc.stderr.strip()
        lines = [f"$ {command}"]
        if stdout:
            lines.append(stdout)
        if stderr:
            lines.append("[stderr]\n" + stderr)
        lines.append(f"[exit {proc.returncode}]")
        return "\n".join(lines)


def _ensure_within_root(root: Path, path: str, allow_global: bool) -> Path:
    return _resolve_path(root, path, allow_global=allow_global)


def _resolve_path(root: Path, path: str, allow_global: bool) -> Path:
    raw = Path(path).expanduser()
    if raw.is_absolute():
        candidate = raw.resolve()
    else:
        candidate = (root / raw).resolve()
    if not allow_global:
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"Path '{path}' escapes the workspace root") from exc
    return candidate


def tool_schemas() -> List[Dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "list_dir",
                "description": "List files and directories relative to the workspace root.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "default": "."},
                        "recursive": {"type": "boolean", "default": False},
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read file contents from the repository.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "offset": {"type": "integer", "minimum": 0, "default": 0},
                        "limit": {"type": "integer", "minimum": 1},
                    },
                    "required": ["path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "write_file",
                "description": "Write full file contents to a path within the repository.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                        "create_parents": {"type": "boolean", "default": False},
                    },
                    "required": ["path", "content"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "stat_path",
                "description": "Return metadata about a file or directory.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "default": "."},
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_text",
                "description": "Search for text within the repository using ripgrep or grep.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string"},
                        "path": {"type": "string", "default": "."},
                        "case_sensitive": {"type": "boolean", "default": True},
                        "max_results": {"type": "integer", "default": 200, "minimum": 1},
                    },
                    "required": ["pattern"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "apply_patch",
                "description": "Apply a unified diff patch to workspace files.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "patch": {"type": "string"},
                    },
                    "required": ["patch"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "run_shell",
                "description": "Execute a shell command from the workspace root.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string"},
                        "timeout": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 600,
                            "default": 120,
                        },
                    },
                    "required": ["command"],
                },
            },
        },
    ]


def execute_tool(executor: ToolExecutor, name: str, arguments: Dict[str, Any]) -> ToolResult:
    func: Callable[..., ToolResult]
    try:
        func = getattr(executor, name)
    except AttributeError as exc:
        raise ValueError(f"Unknown tool '{name}'.") from exc
    return func(**arguments)


def build_messages(system_prompt: str, user_prompt: str, follow_up: List[str]) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    for text in follow_up:
        messages.append({"role": "user", "content": text})
    return messages


def agent_loop(client: OpenAI, options: AgentOptions) -> None:
    messages = build_messages(
        options.system_prompt,
        options.user_prompt,
        options.follow_up,
    )
    specs = tool_schemas()
    thought_console = Console(stderr=True, highlight=False)

    def thought(message: str, *, style: str = "bright_blue") -> None:
        if not options.verbose:
            return
        thought_console.print(f"[bold bright_blue]▌[/] [{style}]{message}[/{style}]")

    transcript_path = options.transcript_path
    executor = ToolExecutor(
        options.workspace,
        read_only=options.read_only,
        allow_global_access=options.allow_global_access,
    )

    if transcript_path:
        transcript_path.parent.mkdir(parents=True, exist_ok=True)

    def log_to_transcript(message: Dict[str, Any], step_index: int) -> None:
        if not transcript_path:
            return
        entry = {"step": step_index, "message": message}
        with transcript_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    if transcript_path:
        for seed in messages:
            log_to_transcript(seed, step_index=0)

    for step in range(1, options.max_steps + 1):
        if options.verbose:
            thought_console.print()
            last_message = messages[-1]
            thought(f"Step {step}: requesting model reasoning…")
            thought(
                f"Last message {last_message.get('role')} · {len(str(last_message.get('content', '')))} characters",
                style="dim",
            )
        response = client.chat.completions.create(
            model=options.model,
            messages=messages,
            tools=specs,
            tool_choice="auto",
        )
        message = response.choices[0].message
        if message.tool_calls:
            tool_payload = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in message.tool_calls
            ]
            assistant_tool_message = {
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": tool_payload,
            }
            messages.append(assistant_tool_message)
            log_to_transcript(assistant_tool_message, step_index=step)
            for tool_call in message.tool_calls:
                name = tool_call.function.name
                try:
                    arguments = json.loads(tool_call.function.arguments or "{}")
                except json.JSONDecodeError as exc:
                    result = f"Failed to decode arguments for {name}: {exc}"
                else:
                    thought(f"Tool request: {name}({arguments})", style="magenta")
                    thought(f"Executing {name} to advance step {step}…", style="dim")
                    try:
                        result = execute_tool(executor, name, arguments)
                    except Exception as exc:  # pragma: no cover
                        result = f"Tool '{name}' raised an error: {exc}"
                if isinstance(result, str) and len(result) > MAX_TOOL_RESULT_CHARS:
                    original_len = len(result)
                    result = (
                        result[:MAX_TOOL_RESULT_CHARS]
                        + "\n… output truncated to "
                        + str(MAX_TOOL_RESULT_CHARS)
                        + f" characters (original length {original_len})."
                    )
                tool_message = {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                }
                messages.append(tool_message)
                log_to_transcript(tool_message, step_index=step)
                if isinstance(result, str):
                    thought(f"{name} completed · {len(result)} characters captured.", style="dim")
        else:
            content = message.content or ""
            assistant_message = {"role": "assistant", "content": content}
            messages.append(assistant_message)
            log_to_transcript(assistant_message, step_index=step)
            print(content)
            thought("Assistant produced final answer; ending loop.", style="green")
            return
    if transcript_path:
        try:
            location_str = str(transcript_path.relative_to(options.workspace))
        except ValueError:
            location_str = str(transcript_path)
        message = (
            "Max steps reached without a final response. "
            f"Transcript saved to '{location_str}'."
        )
    else:
        message = (
            "Max steps reached without a final response. "
            "Re-run with a higher --max-steps or provide --transcript to inspect the conversation."
        )
    print(message, file=sys.stderr)
    thought("Reached maximum steps without completion.", style="red")


__all__ = [
    "AgentOptions",
    "ToolExecutor",
    "tool_schemas",
    "agent_loop",
]
