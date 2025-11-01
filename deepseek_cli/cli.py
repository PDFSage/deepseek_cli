"""Main CLI entry point."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
import textwrap
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

from packaging.version import Version

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

from openai import OpenAI

from . import __version__
from .agent import AgentOptions, agent_loop
from .chat import ChatOptions, run_chat
from .completions import CompletionOptions, run_completion
from .config import (
    ResolvedConfig,
    ensure_config_dir,
    load_config,
    pretty_config,
    resolve_runtime_config,
    save_config,
    update_config,
    ENV_API_KEY,
)
from .constants import (
    AUTO_BUG_FOLLOW_UP,
    AUTO_TEST_FOLLOW_UP,
    CONFIG_FILE,
    DEFAULT_MAX_STEPS,
    STREAM_STYLE_CHOICES,
    TRANSCRIPTS_DIR,
)
from .embeddings import EmbeddingOptions, run_embeddings
from .models import ModelListOptions, list_models
from .testing import build_test_followups

COMMAND_PREFIXES = (":", "/", "@")
MAIN_CONSOLE = Console()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="deepseek",
        description="DeepSeek command line interface for coding and chat workflows.",
    )
    parser.add_argument("--version", action="store_true", help="Show version and exit")

    subparsers = parser.add_subparsers(dest="command")

    chat_parser = subparsers.add_parser(
        "chat", help="Chat with DeepSeek in a developer-friendly shell",
    )
    chat_parser.add_argument("prompt", nargs="?", help="Initial user prompt")
    chat_parser.add_argument("--model", help="Override chat model")
    chat_parser.add_argument("--system", help="Override system prompt")
    chat_parser.add_argument("--temperature", type=float, default=0.1, help="Sampling temperature")
    chat_parser.add_argument("--top-p", type=float, default=1.0, help="Nucleus sampling parameter")
    chat_parser.add_argument("--max-tokens", type=int, help="Limit the assistant reply tokens")
    chat_parser.add_argument("--no-stream", action="store_true", help="Disable streaming responses")
    chat_parser.add_argument(
        "--stream-style",
        choices=STREAM_STYLE_CHOICES,
        help="Adjust live streaming rendering (default from config).",
    )
    chat_parser.add_argument(
        "--interactive",
        action="store_true",
        help="Enter an interactive multi-turn chat session",
    )
    chat_parser.add_argument(
        "--transcript",
        help="Path to save a JSONL transcript (defaults under ~/.config/deepseek-cli)",
    )
    add_shared_connection_options(chat_parser)

    completion_parser = subparsers.add_parser(
        "completions",
        help="Request Codex-style text completions.",
    )
    completion_parser.add_argument("prompt", nargs="?", help="Prompt text (falls back to stdin)")
    completion_parser.add_argument(
        "--input-file",
        type=Path,
        help="Read prompt text from a file.",
    )
    completion_parser.add_argument("--suffix", help="Optional suffix appended after the insertion point.")
    completion_parser.add_argument("--model", help="Override completion model.")
    completion_parser.add_argument("--temperature", type=float, default=0.1, help="Sampling temperature.")
    completion_parser.add_argument("--top-p", type=float, default=1.0, help="Nucleus sampling parameter.")
    completion_parser.add_argument("--max-tokens", type=int, help="Limit completion tokens.")
    completion_parser.add_argument(
        "--stop",
        action="append",
        default=[],
        help="Stop sequence (repeat for multiple).",
    )
    completion_parser.add_argument("--n", type=int, default=1, help="Number of completions to request (default 1).")
    completion_parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable streaming output (default uses streaming).",
    )
    completion_parser.add_argument(
        "--stream-style",
        choices=STREAM_STYLE_CHOICES,
        help="Streaming renderer to use (default from config).",
    )
    completion_parser.add_argument(
        "--output",
        type=Path,
        help="Optional file path to save the primary completion.",
    )
    add_shared_connection_options(completion_parser)

    embeddings_parser = subparsers.add_parser(
        "embeddings",
        help="Generate embedding vectors for text snippets.",
    )
    embeddings_parser.add_argument("text", nargs="*", help="Text snippets to embed (repeatable).")
    embeddings_parser.add_argument("--input-file", type=Path, help="Read additional inputs from a file.")
    embeddings_parser.add_argument("--model", help="Override embedding model.")
    embeddings_parser.add_argument(
        "--format",
        choices=("table", "json", "plain"),
        default="table",
        help="Render format for embeddings output.",
    )
    embeddings_parser.add_argument(
        "--output",
        type=Path,
        help="Write embeddings payload to a JSON file.",
    )
    embeddings_parser.add_argument(
        "--show-dimensions",
        action="store_true",
        help="Include vector dimensionality in the table view.",
    )
    add_shared_connection_options(embeddings_parser)

    models_parser = subparsers.add_parser(
        "models",
        help="List available models from the DeepSeek API.",
    )
    models_parser.add_argument("--filter", help="Filter models containing this substring.")
    models_parser.add_argument("--limit", type=int, help="Limit the number of rows shown.")
    models_parser.add_argument("--json", action="store_true", help="Output raw JSON instead of a table.")
    add_shared_connection_options(models_parser)

    agent_parser = subparsers.add_parser(
        "agent",
        help="Run the agentic developer assistant with repository tools",
    )
    agent_parser.add_argument("prompt", help="User instruction for the agent")
    agent_parser.add_argument(
        "--follow-up",
        action="append",
        default=[],
        help="Additional user inputs appended after the initial prompt (repeatable)",
    )
    agent_parser.add_argument("--system", help="Override system prompt")
    agent_parser.add_argument("--model", help="Override model name")
    agent_parser.add_argument(
        "--workspace",
        default=Path.cwd(),
        help="Workspace directory (default: current directory)",
    )
    agent_parser.add_argument(
        "--max-steps",
        type=int,
        default=DEFAULT_MAX_STEPS,
        help="Maximum reasoning steps before aborting",
    )
    agent_parser.add_argument(
        "--transcript",
        help="Optional transcript path (default stored under ~/.config/deepseek-cli)",
    )
    agent_parser.add_argument("--read-only", action="store_true", help="Disable write operations")
    agent_parser.add_argument(
        "--global",
        action="store_true",
        dest="allow_global",
        help="Allow edits outside the workspace root (use with caution)",
    )
    agent_parser.add_argument("--quiet", action="store_true", help="Suppress progress logs")
    add_shared_connection_options(agent_parser)

    config_parser = subparsers.add_parser("config", help="Manage configuration")
    config_sub = config_parser.add_subparsers(dest="config_command")

    config_show = config_sub.add_parser("show", help="Display current configuration")
    config_show.add_argument("--raw", action="store_true", help="Do not redact the API key")

    config_set = config_sub.add_parser("set", help="Update a configuration value")
    config_set.add_argument(
        "key",
        choices=[
            "api_key",
            "base_url",
            "model",
            "chat_model",
            "completion_model",
            "embedding_model",
            "system_prompt",
            "chat_system_prompt",
            "chat_stream_style",
        ],
    )
    config_set.add_argument("value", help="Configuration value (wrap in quotes for spaces)")

    config_unset = config_sub.add_parser("unset", help="Remove a configuration value")
    config_unset.add_argument(
        "key",
        choices=[
            "api_key",
            "base_url",
            "model",
            "chat_model",
            "completion_model",
            "embedding_model",
            "system_prompt",
            "chat_system_prompt",
            "chat_stream_style",
        ],
    )

    config_init = config_sub.add_parser("init", help="Interactive configuration wizard")

    return parser


def add_shared_connection_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--api-key", help="DeepSeek API key (overrides env/config)")
    parser.add_argument("--base-url", help="DeepSeek API base URL")


def create_client(config: ResolvedConfig) -> OpenAI:
    return OpenAI(api_key=config.api_key, base_url=config.base_url)


def notify_if_update_available() -> None:
    url = "https://pypi.org/pypi/deepseek-agent/json"
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            payload = json.load(response)
    except Exception:  # pragma: no cover - best effort
        return
    releases = payload.get("releases") or {}
    versions = sorted(
        (Version(v) for v in releases.keys() if releases.get(v)),
        reverse=True,
    )
    if not versions:
        return
    latest = versions[0]
    current = Version(__version__)
    if latest > current:
        print(
            textwrap.dedent(
                f"""
                [update] A newer deepseek-agent release is available: {latest} (current {current}).
                Update with: python -m pip install --upgrade deepseek-agent
                """
            ).strip(),
            file=sys.stderr,
        )


@dataclass
class InteractiveSessionState:
    workspace: Path
    model: str
    system_prompt: str
    max_steps: int = DEFAULT_MAX_STEPS
    read_only: bool = False
    allow_global_access: bool = False
    verbose: bool = True
    transcript_path: Optional[Path] = None
    default_workspace: Path = field(init=False)
    default_model: str = field(init=False)
    default_system_prompt: str = field(init=False)
    default_max_steps: int = field(init=False)
    default_read_only: bool = field(init=False)
    default_allow_global_access: bool = field(init=False)
    default_verbose: bool = field(init=False)
    default_transcript_path: Optional[Path] = field(init=False)

    def __post_init__(self) -> None:
        self.default_workspace = self.workspace
        self.default_model = self.model
        self.default_system_prompt = self.system_prompt
        self.default_max_steps = self.max_steps
        self.default_read_only = self.read_only
        self.default_allow_global_access = self.allow_global_access
        self.default_verbose = self.verbose
        self.default_transcript_path = self.transcript_path


def _mask_api_key(value: Optional[str]) -> str:
    if not value:
        return "not set"
    if len(value) <= 8:
        return value[:2] + "…" + value[-2:]
    return value[:4] + "…" + value[-4:]


def _set_runtime_api_key(value: Optional[str]) -> None:
    if value:
        os.environ[ENV_API_KEY] = value
    else:
        os.environ.pop(ENV_API_KEY, None)


def _store_api_key(value: Optional[str]) -> bool:
    config = load_config()
    config["api_key"] = value
    try:
        save_config(config)
    except RuntimeError as exc:
        MAIN_CONSOLE.print(f"[red]Unable to persist API key:[/] {exc}")
        return False
    _set_runtime_api_key(value)
    if value:
        MAIN_CONSOLE.print(f"[green]Saved API key ({_mask_api_key(value)}) to config.[/]")
    else:
        MAIN_CONSOLE.print("[yellow]Cleared stored API key.[/]")
    return True


def _prompt_for_api_key(
    *,
    allow_empty: bool = False,
    prompt_text: str = "Enter DeepSeek API key",
) -> Optional[str]:
    try:
        entered = Prompt.ask(
            f"[bold yellow]{prompt_text}[/]",
            console=MAIN_CONSOLE,
            password=True,
        ).strip()
    except (KeyboardInterrupt, EOFError):
        MAIN_CONSOLE.print("[red]API key entry cancelled by user.[/]")
        return None
    if not entered and not allow_empty:
        MAIN_CONSOLE.print("[red]No API key entered.[/]")
        return None
    return entered or None


def _command_reference_table() -> Table:
    table = Table(
        title="Interactive Commands",
        box=box.ROUNDED,
        title_style="bold magenta",
        show_header=False,
        pad_edge=True,
    )
    table.add_column("Command", style="bold cyan", no_wrap=True)
    table.add_column("Description", style="white")
    rows = [
        ("@help /help :help", "Show this command palette"),
        ("@quit /quit :quit", "Exit the interactive shell"),
        ("@workspace [PATH]", "Show or change the active workspace"),
        ("@model [NAME]", "Display or update the active model"),
        ("@system [TEXT]", "Show or set the system prompt"),
        ("@max-steps [N]", "Display or update max reasoning steps"),
        ("@read-only [on|off|toggle]", "Toggle workspace write access"),
        ("@global [on|off|toggle]", "Allow edits outside the workspace root"),
        ("@transcript [PATH]", "Log transcripts to a file"),
        ("@clear-transcript", "Disable transcript logging"),
        ("@settings", "Display current session status"),
        ("@reset", "Restore defaults from config"),
        ("@api", "Update the stored DeepSeek API key"),
        ("@verbose", "Enable detailed thought process logging"),
        ("@quiet", "Disable detailed thought process logging"),
    ]
    for command, description in rows:
        table.add_row(command, description)
    return table


def _session_status_panel(state: InteractiveSessionState) -> Panel:
    grid = Table.grid(padding=(0, 1))
    grid.add_column(style="bold cyan", justify="right")
    grid.add_column(style="white")
    grid.add_row("Workspace", str(state.workspace))
    grid.add_row("Model", state.model)
    grid.add_row(
        "System",
        "custom prompt" if state.system_prompt != state.default_system_prompt else "default prompt",
    )
    grid.add_row("Read-only", "on" if state.read_only else "off")
    grid.add_row("Max steps", str(state.max_steps))
    grid.add_row("Global ops", "on" if state.allow_global_access else "off")
    grid.add_row("Verbose", "on" if state.verbose else "off")
    grid.add_row(
        "Transcript",
        str(state.transcript_path) if state.transcript_path else "disabled",
    )
    return Panel(
        grid,
        title="Session Status",
        border_style="bright_blue",
        expand=False,
    )


def _print_interactive_help() -> None:
    MAIN_CONSOLE.print(_command_reference_table())
    MAIN_CONSOLE.print(
        Panel(
            Text(
                "Enter your request at the prompt. Use a trailing '\\' to extend across lines.\n"
                "Commands can start with @, /, or :.\n"
                "Press Enter on an empty line to send the prompt together with automated test and bug checks.",
                style="bright_white",
            ),
            border_style="bright_magenta",
        )
    )


def handle_chat(args: argparse.Namespace, resolved: ResolvedConfig) -> int:
    client = create_client(resolved)
    transcript_path = _resolve_transcript_path(args.transcript)
    options = ChatOptions(
        prompt=args.prompt,
        system_prompt=args.system or resolved.chat_system_prompt,
        model=args.model or resolved.chat_model,
        stream=not args.no_stream,
        stream_style=args.stream_style or resolved.chat_stream_style,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        interactive=args.interactive,
        transcript_path=transcript_path,
    )
    return run_chat(client, options)


def handle_completions(args: argparse.Namespace, resolved: ResolvedConfig) -> int:
    client = create_client(resolved)
    options = CompletionOptions(
        prompt=args.prompt,
        input_file=args.input_file,
        suffix=args.suffix,
        model=args.model or resolved.completion_model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        n=args.n,
        stop=args.stop or [],
        stream=not args.no_stream,
        stream_style=args.stream_style or resolved.chat_stream_style,
        output_path=args.output,
    )
    return run_completion(client, options)


def handle_embeddings(args: argparse.Namespace, resolved: ResolvedConfig) -> int:
    client = create_client(resolved)
    options = EmbeddingOptions(
        texts=args.text,
        input_file=args.input_file,
        model=args.model or resolved.embedding_model,
        output_path=args.output,
        fmt=args.format,
        show_dimensions=args.show_dimensions,
    )
    return run_embeddings(client, options)


def handle_models(args: argparse.Namespace, resolved: ResolvedConfig) -> int:
    client = create_client(resolved)
    options = ModelListOptions(
        filter=args.filter,
        json_output=args.json,
        limit=args.limit,
    )
    return list_models(client, options)


def _resolve_transcript_path(value: Optional[str]) -> Optional[Path]:
    if value is None:
        return None
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return candidate
    ensure_config_dir()
    TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    return TRANSCRIPTS_DIR / candidate


def _handle_interactive_command(
    raw: str,
    state: InteractiveSessionState,
    *,
    on_api_command: Optional[Callable[[List[str]], None]] = None,
) -> bool:
    command_line = raw[1:].strip()
    if not command_line:
        _print_interactive_help()
        return True
    try:
        parts = shlex.split(command_line)
    except ValueError as exc:
        MAIN_CONSOLE.print(f"[red]Unable to parse command:[/] {exc}")
        return True
    if not parts:
        return True
    name = parts[0].lower()
    args = parts[1:]

    if name in {"quit", "exit", "q"}:
        MAIN_CONSOLE.print("[bold magenta]Goodbye![/] Exiting interactive agent.")
        return False
    if name in {"help", "?"}:
        _print_interactive_help()
        return True
    if name in {"settings", "status"}:
        MAIN_CONSOLE.print(_session_status_panel(state))
        return True
    if name == "workspace":
        if not args:
            MAIN_CONSOLE.print(f"[cyan]Workspace:[/] {state.workspace}")
            return True
        raw_path = " ".join(args)
        candidate = Path(raw_path).expanduser()
        if not candidate.is_absolute():
            candidate = (state.workspace / raw_path).expanduser()
        try:
            candidate = candidate.resolve()
        except FileNotFoundError:
            MAIN_CONSOLE.print(f"[red]Workspace '{candidate}' does not exist.[/]")
            return True
        if not candidate.exists() or not candidate.is_dir():
            MAIN_CONSOLE.print(f"[red]Workspace '{candidate}' is not a directory.[/]")
            return True
        state.workspace = candidate
        MAIN_CONSOLE.print(f"[green]Workspace set to[/] {state.workspace}")
        return True
    if name == "model":
        if not args:
            MAIN_CONSOLE.print(f"[cyan]Model:[/] {state.model}")
            return True
        state.model = " ".join(args)
        MAIN_CONSOLE.print(f"[green]Model set to[/] {state.model}")
        return True
    if name == "system":
        if not args:
            MAIN_CONSOLE.print(Panel(state.system_prompt or "(empty)", title="System Prompt"))
            return True
        state.system_prompt = " ".join(args)
        MAIN_CONSOLE.print("[green]System prompt updated.[/]")
        return True
    if name == "max-steps":
        if not args:
            MAIN_CONSOLE.print(f"[cyan]Max steps:[/] {state.max_steps}")
            return True
        try:
            value = int(args[0])
        except ValueError:
            MAIN_CONSOLE.print("[red]max-steps requires an integer value.[/]")
            return True
        if value < 1:
            MAIN_CONSOLE.print("[red]max-steps must be at least 1.[/]")
            return True
        state.max_steps = value
        MAIN_CONSOLE.print(f"[green]Max steps set to[/] {value}")
        return True
    if name == "read-only":
        if not args:
            MAIN_CONSOLE.print(f"[cyan]Read-only:[/] {'on' if state.read_only else 'off'}")
            return True
        setting = args[0].lower()
        if setting in {"on", "true", "1"}:
            state.read_only = True
        elif setting in {"off", "false", "0"}:
            state.read_only = False
        elif setting == "toggle":
            state.read_only = not state.read_only
        else:
            MAIN_CONSOLE.print("[red]Use on/off/toggle to control read-only mode.[/]")
            return True
        MAIN_CONSOLE.print(f"[green]Read-only mode {'enabled' if state.read_only else 'disabled'}.[/]")
        return True
    if name == "global":
        if not args:
            MAIN_CONSOLE.print(f"[cyan]Global operations:[/] {'on' if state.allow_global_access else 'off'}")
            return True
        setting = args[0].lower()
        if setting in {"on", "true", "1"}:
            state.allow_global_access = True
        elif setting in {"off", "false", "0"}:
            state.allow_global_access = False
        elif setting == "toggle":
            state.allow_global_access = not state.allow_global_access
        else:
            MAIN_CONSOLE.print("[red]Use on/off/toggle to control global operations.[/]")
            return True
        message = "enabled (paths may escape workspace)" if state.allow_global_access else "disabled"
        MAIN_CONSOLE.print(f"[green]Global operations {message}.[/]")
        return True
    if name == "verbose":
        state.verbose = True
        MAIN_CONSOLE.print("[green]Verbose tool logging enabled.[/]")
        return True
    if name == "quiet":
        state.verbose = False
        MAIN_CONSOLE.print("[yellow]Verbose tool logging disabled.[/]")
        return True
    if name == "transcript":
        if not args:
            if state.transcript_path:
                MAIN_CONSOLE.print(f"[cyan]Transcript logging to[/] {state.transcript_path}")
            else:
                MAIN_CONSOLE.print("[yellow]Transcript logging is disabled.[/]")
            return True
        raw_path = " ".join(args)
        candidate = Path(raw_path).expanduser()
        if not candidate.is_absolute():
            candidate = (state.workspace / raw_path).expanduser()
        state.transcript_path = candidate
        MAIN_CONSOLE.print(f"[green]Transcript logging set to[/] {state.transcript_path}")
        return True
    if name == "clear-transcript":
        state.transcript_path = None
        MAIN_CONSOLE.print("[yellow]Transcript logging disabled.[/]")
        return True
    if name == "reset":
        state.workspace = state.default_workspace
        state.model = state.default_model
        state.system_prompt = state.default_system_prompt
        state.max_steps = state.default_max_steps
        state.read_only = state.default_read_only
        state.allow_global_access = state.default_allow_global_access
        state.verbose = state.default_verbose
        state.transcript_path = state.default_transcript_path
        MAIN_CONSOLE.print("[green]Session settings reset to defaults.[/]")
        MAIN_CONSOLE.print(_session_status_panel(state))
        return True
    if name == "api":
        if on_api_command:
            on_api_command(args)
        return True
    MAIN_CONSOLE.print(f"[red]Unknown command '{name}'. Type /help for options.[/]")
    return True


def _collect_follow_ups() -> List[str]:
    # Follow-ups are no longer collected via an additional prompt; return empty list.
    return []


def _run_interactive_agent_prompt(
    client: OpenAI,
    state: InteractiveSessionState,
    prompt: str,
    follow_ups: List[str],
) -> None:
    workspace = state.workspace
    if not workspace.exists():
        MAIN_CONSOLE.print(
            f"[red]Workspace '{workspace}' does not exist.[/] Use /workspace to choose another."
        )
        return
    options = AgentOptions(
        model=state.model,
        system_prompt=state.system_prompt,
        user_prompt=prompt,
        follow_up=follow_ups,
        workspace=workspace,
        read_only=state.read_only,
        allow_global_access=state.allow_global_access,
        max_steps=state.max_steps,
        verbose=state.verbose,
        transcript_path=state.transcript_path,
    )
    try:
        agent_loop(client, options)
    except Exception as exc:  # pragma: no cover
        MAIN_CONSOLE.print(f"[red]Agent error:[/] {exc}")


def run_interactive_agent_shell(resolved: ResolvedConfig) -> int:
    current_config = resolved
    client = create_client(current_config)
    state = InteractiveSessionState(
        workspace=Path.cwd().resolve(),
        model=current_config.model,
        system_prompt=current_config.system_prompt,
    )

    MAIN_CONSOLE.print(
        Panel(
            Text(
                "DeepSeek Agent\nInteractive coding workspace",
                justify="center",
                style="bold bright_cyan",
            ),
            subtitle="Try /help for the command palette",
            border_style="bright_magenta",
            padding=(1, 2),
        )
    )
    MAIN_CONSOLE.print(f"[cyan]API key:[/] {_mask_api_key(current_config.api_key)}")
    MAIN_CONSOLE.print(_session_status_panel(state))
    _print_interactive_help()

    def handle_api_command(args: List[str]) -> None:
        nonlocal client
        MAIN_CONSOLE.print(f"[cyan]Current API key:[/] {_mask_api_key(current_config.api_key)}")
        if args and args[0].lower() == "show":
            return
        new_key = _prompt_for_api_key(
            allow_empty=True,
            prompt_text="Enter new DeepSeek API key (leave blank to cancel)",
        )
        if not new_key:
            MAIN_CONSOLE.print("[yellow]API key unchanged.[/]")
            return
        if _store_api_key(new_key):
            current_config.api_key = new_key
            client = create_client(current_config)
            MAIN_CONSOLE.print("[green]API key updated and reloaded for this session.[/]")

    while True:
        try:
            raw = MAIN_CONSOLE.input("[bold bright_cyan]Prompt ▸ [/]")
        except EOFError:
            MAIN_CONSOLE.line()
            return 0
        except KeyboardInterrupt:
            MAIN_CONSOLE.line()
            return 130
        prompt = raw.strip()
        if not prompt:
            continue
        if prompt[0] in COMMAND_PREFIXES:
            should_continue = _handle_interactive_command(
                prompt,
                state,
                on_api_command=handle_api_command,
            )
            if not should_continue:
                return 0
            continue

        prompt_lines = [prompt]
        while prompt_lines[-1].endswith("\\"):
            prompt_lines[-1] = prompt_lines[-1].rstrip("\\")
            try:
                continuation = MAIN_CONSOLE.input("[bold bright_cyan]… [/]")
            except EOFError:
                MAIN_CONSOLE.line()
                break
            except KeyboardInterrupt:
                MAIN_CONSOLE.line()
                break
            prompt_lines.append(continuation.rstrip())
        final_prompt = "\n".join(line.strip() for line in prompt_lines if line.strip())
        follow_ups = _collect_follow_ups()
        follow_ups.extend(build_test_followups(state.workspace))
        follow_ups.extend([AUTO_TEST_FOLLOW_UP, AUTO_BUG_FOLLOW_UP])
        if not final_prompt:
            continue
        _run_interactive_agent_prompt(client, state, final_prompt, follow_ups)

def handle_agent(args: argparse.Namespace, resolved: ResolvedConfig) -> int:
    client = create_client(resolved)
    workspace = Path(args.workspace).expanduser().resolve()
    if not workspace.exists():
        print(f"Workspace '{workspace}' does not exist.", file=sys.stderr)
        return 1

    transcript_path: Optional[Path]
    if args.transcript:
        transcript_path = Path(args.transcript).expanduser()
        if not transcript_path.is_absolute():
            transcript_path = workspace / transcript_path
    else:
        transcript_path = None

    options = AgentOptions(
        model=args.model or resolved.model,
        system_prompt=args.system or resolved.system_prompt,
        user_prompt=args.prompt,
        follow_up=(args.follow_up or [])
        + build_test_followups(workspace)
        + [AUTO_TEST_FOLLOW_UP, AUTO_BUG_FOLLOW_UP],
        workspace=workspace,
        read_only=args.read_only,
        allow_global_access=getattr(args, "allow_global", False),
        max_steps=args.max_steps,
        verbose=not args.quiet,
        transcript_path=transcript_path,
    )
    agent_loop(client, options)
    return 0


def handle_config(args: argparse.Namespace) -> int:
    if args.config_command is None:
        print("Select a config subcommand (show, set, unset, init).", file=sys.stderr)
        return 1
    if args.config_command == "show":
        config = load_config()
        print(pretty_config(config, redact=not args.raw))
        print(f"Config file: {CONFIG_FILE}")
        return 0
    if args.config_command == "set":
        try:
            update_config([(args.key, args.value)])
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        print(f"Updated '{args.key}'.")
        return 0
    if args.config_command == "unset":
        config = load_config()
        config[args.key] = None
        try:
            save_config(config)
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        print(f"Cleared '{args.key}'.")
        return 0
    if args.config_command == "init":
        ensure_config_dir()
        config = load_config()
        try:
            api_key = input("Enter DeepSeek API key: ")
        except EOFError:
            print("Input aborted.", file=sys.stderr)
            return 1
        config["api_key"] = api_key.strip() or None
        try:
            save_config(config)
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        print(f"Configuration saved to {CONFIG_FILE}")
        return 0
    return 1


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.version:
        print(__version__)
        return 0

    MAIN_CONSOLE.print(f"[cyan]deepseek-agent v{__version__}[/]")

    notify_if_update_available()

    if args.command == "config":
        return handle_config(args)

    config_kwargs = {
        "api_key": getattr(args, "api_key", None),
        "base_url": getattr(args, "base_url", None),
    }
    if args.command == "chat":
        config_kwargs.update(
            {
                "chat_model": getattr(args, "model", None),
                "chat_system_prompt": getattr(args, "system", None),
                "chat_stream_style": getattr(args, "stream_style", None),
            }
        )
    elif args.command == "completions":
        config_kwargs["completion_model"] = getattr(args, "model", None)
    elif args.command == "embeddings":
        config_kwargs["embedding_model"] = getattr(args, "model", None)
    else:
        config_kwargs.update(
            {
                "model": getattr(args, "model", None),
                "system_prompt": getattr(args, "system", None),
            }
        )
    try:
        resolved = resolve_runtime_config(**config_kwargs)
    except RuntimeError as exc:
        missing_api_key = "No DeepSeek API key found" in str(exc)
        if args.command is None and missing_api_key:
            MAIN_CONSOLE.print(
                Panel(
                    Text(
                        "A DeepSeek API key is required before the interactive shell can start.\n"
                        "Create one at https://platform.deepseek.com/api_keys and paste it below.",
                        justify="center",
                        style="bright_white",
                    ),
                    title="API key required",
                    border_style="red",
                    padding=(1, 2),
                )
            )
            api_key = _prompt_for_api_key(prompt_text="Enter DeepSeek API key to continue")
            if not api_key:
                return 1
            if not _store_api_key(api_key):
                return 1
            config_kwargs["api_key"] = api_key
            try:
                resolved = resolve_runtime_config(**config_kwargs)
            except RuntimeError as inner_exc:  # pragma: no cover
                MAIN_CONSOLE.print(f"[red]{inner_exc}[/]")
                return 1
        else:
            print(str(exc), file=sys.stderr)
            return 1

    if args.command is None:
        return run_interactive_agent_shell(resolved)

    if args.command == "chat":
        return handle_chat(args, resolved)
    if args.command == "completions":
        return handle_completions(args, resolved)
    if args.command == "embeddings":
        return handle_embeddings(args, resolved)
    if args.command == "models":
        return handle_models(args, resolved)
    if args.command == "agent":
        return handle_agent(args, resolved)

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
