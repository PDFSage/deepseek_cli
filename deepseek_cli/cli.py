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

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.styles import Style
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
    ENV_TAVILY_API_KEY,
)
from .constants import (
    AUTO_BUG_FOLLOW_UP,
    AUTO_TEST_FOLLOW_UP,
    CONFIG_FILE,
    DEFAULT_TAVILY_API_KEY,
    DEFAULT_MAX_STEPS,
    STREAM_STYLE_CHOICES,
    TRANSCRIPTS_DIR,
)
from .embeddings import EmbeddingOptions, run_embeddings
from .models import ModelListOptions, list_models
from .testing import build_test_followups

COMMAND_PREFIXES = (":", "/", "@")
MAIN_CONSOLE = Console()
PROMPT_STYLE = Style.from_dict({"prompt": "ansibrightcyan bold"})
PROMPT_MESSAGE = FormattedText([("class:prompt", "Prompt ▸ ")])
PROMPT_CONTINUATION = FormattedText([("class:prompt", "… ")])


def _prompt_continuation(width: int, line_number: int, is_soft_wrap: bool) -> FormattedText:
    return PROMPT_CONTINUATION


def _build_prompt_session() -> PromptSession:
    """Create a prompt session that supports multiline editing and history."""
    bindings = KeyBindings()

    @bindings.add("enter")
    def _(event) -> None:
        buffer = event.app.current_buffer
        event.app.exit(result=buffer.text)

    @bindings.add("s-enter")
    @bindings.add("c-enter")
    @bindings.add("escape", "enter")
    def _(event) -> None:
        event.app.current_buffer.insert_text("\n")

    return PromptSession(
        history=InMemoryHistory(),
        key_bindings=bindings,
        multiline=True,
        prompt_continuation=_prompt_continuation,
        style=PROMPT_STYLE,
        reserve_space_for_menu=0,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="deepseek",
        description="DeepSeek command line interface for unified agent workflows.",
    )
    parser.add_argument("--version", action="store_true", help="Show version and exit")
    parser.add_argument(
        "--prompt",
        help="Run a single agent instruction and exit (omit to launch the interactive shell).",
    )
    parser.add_argument(
        "--follow-up",
        action="append",
        default=[],
        help="Additional user inputs appended after --prompt (repeatable).",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path.cwd(),
        help="Workspace directory for agent operations (default: current directory).",
    )
    parser.add_argument("--model", help="Override the default agent model.")
    parser.add_argument("--system", help="Override the active system prompt.")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=DEFAULT_MAX_STEPS,
        help="Maximum reasoning steps before aborting.",
    )
    parser.add_argument(
        "--transcript",
        help="Optional transcript path (default stored under ~/.config/deepseek-cli).",
    )
    parser.add_argument("--read-only", action="store_true", help="Disable write operations.")
    parser.add_argument(
        "--no-global",
        action="store_false",
        dest="allow_global",
        help="Restrict edits to the workspace directory only.",
    )
    parser.add_argument(
        "--global",
        action="store_true",
        dest="allow_global",
        help="Allow edits outside the workspace root.",
    )
    parser.set_defaults(allow_global=True)
    parser.add_argument("--quiet", action="store_true", help="Suppress detailed progress logs.")

    add_shared_connection_options(parser)

    subparsers = parser.add_subparsers(dest="command")

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
            "tavily_api_key",
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
            "tavily_api_key",
        ],
    )

    config_sub.add_parser("init", help="Interactive configuration wizard")

    return parser


def add_shared_connection_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--api-key", help="DeepSeek API key (overrides env/config)")
    parser.add_argument("--base-url", help="DeepSeek API base URL")
    parser.add_argument(
        "--tavily-api-key",
        help="Tavily API key used for @tavily search commands and the tavily_search tool.",
    )


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
    tavily_api_key: str
    max_steps: int = DEFAULT_MAX_STEPS
    read_only: bool = False
    allow_global_access: bool = True
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
    default_tavily_api_key: str = field(init=False)

    def __post_init__(self) -> None:
        self.default_workspace = self.workspace
        self.default_model = self.model
        self.default_system_prompt = self.system_prompt
        self.default_tavily_api_key = self.tavily_api_key
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


def _set_runtime_tavily_api_key(value: Optional[str]) -> None:
    if value:
        os.environ[ENV_TAVILY_API_KEY] = value
    else:
        os.environ.pop(ENV_TAVILY_API_KEY, None)


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


def _store_tavily_api_key(value: Optional[str]) -> bool:
    config = load_config()
    config["tavily_api_key"] = value
    try:
        save_config(config)
    except RuntimeError as exc:
        MAIN_CONSOLE.print(f"[red]Unable to persist Tavily API key:[/] {exc}")
        return False
    _set_runtime_tavily_api_key(value)
    if value:
        MAIN_CONSOLE.print(
            f"[green]Saved Tavily API key ({_mask_api_key(value)}) to config.[/]"
        )
    else:
        MAIN_CONSOLE.print(
            "[yellow]Reverted to the bundled Tavily developer key.[/]"
        )
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
        ("@chat <TEXT>", "Send a lightweight chat request without invoking the agent"),
        ("@complete <TEXT>", "Request a single completion using the completion model"),
        ("@embed <TEXT…>", "Generate embeddings for one or more snippets"),
        ("@models [--filter X] [--limit N] [--json]", "List available API models"),
        ("@reset", "Restore defaults from config"),
        ("@api", "Update the stored DeepSeek API key"),
        ("@tavily", "Update the Tavily API key used for web search"),
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
    tavily_display = _mask_api_key(state.tavily_api_key)
    if state.tavily_api_key == DEFAULT_TAVILY_API_KEY:
        tavily_display += " (default)"
    grid.add_row("Tavily key", tavily_display)
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
                "Enter your request at the prompt. Use Shift+Enter to insert new lines.\n"
                "Commands can start with @, /, or :.\n"
                "Press Enter to send the prompt together with automated test and bug checks.",
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
    on_tavily_command: Optional[Callable[[List[str]], None]] = None,
    on_chat_command: Optional[Callable[[List[str]], None]] = None,
    on_completion_command: Optional[Callable[[List[str]], None]] = None,
    on_embedding_command: Optional[Callable[[List[str]], None]] = None,
    on_models_command: Optional[Callable[[List[str]], None]] = None,
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
    if name == "chat":
        if on_chat_command:
            on_chat_command(args)
        else:
            MAIN_CONSOLE.print("[red]Chat command unavailable in this session.[/]")
        return True
    if name in {"complete", "completion"}:
        if on_completion_command:
            on_completion_command(args)
        else:
            MAIN_CONSOLE.print("[red]Completion command unavailable in this session.[/]")
        return True
    if name in {"embed", "embedding", "embeddings"}:
        if on_embedding_command:
            on_embedding_command(args)
        else:
            MAIN_CONSOLE.print("[red]Embedding command unavailable in this session.[/]")
        return True
    if name in {"models", "model-list"}:
        if on_models_command:
            on_models_command(args)
        else:
            MAIN_CONSOLE.print("[red]Model listing is unavailable in this session.[/]")
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
        state.tavily_api_key = state.default_tavily_api_key
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
    if name == "tavily":
        if on_tavily_command:
            on_tavily_command(args)
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
        tavily_api_key=state.tavily_api_key,
    )
    try:
        agent_loop(client, options)
    except Exception as exc:  # pragma: no cover
        MAIN_CONSOLE.print(f"[red]Agent error:[/] {exc}")


def run_interactive_agent_shell(
    resolved: ResolvedConfig,
    *,
    workspace_override: Optional[Path] = None,
    model_override: Optional[str] = None,
    system_override: Optional[str] = None,
    max_steps_override: Optional[int] = None,
    read_only: bool = False,
    allow_global: bool = True,
    transcript_override: Optional[Path] = None,
    verbose: bool = True,
) -> int:
    current_config = resolved
    client = create_client(current_config)
    workspace = (workspace_override or Path.cwd()).resolve()
    state = InteractiveSessionState(
        workspace=workspace,
        model=model_override or current_config.model,
        system_prompt=system_override or current_config.system_prompt,
        tavily_api_key=current_config.tavily_api_key,
        max_steps=max_steps_override if max_steps_override is not None else DEFAULT_MAX_STEPS,
        read_only=read_only,
        allow_global_access=allow_global,
        verbose=verbose,
        transcript_path=transcript_override,
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

    def handle_tavily_command(args: List[str]) -> None:
        MAIN_CONSOLE.print(
            f"[cyan]Current Tavily key:[/] {_mask_api_key(state.tavily_api_key)}"
        )
        if args:
            keyword = args[0].lower()
            if keyword == "show":
                return
            if keyword in {"default", "reset"}:
                if _store_tavily_api_key(None):
                    current_config.tavily_api_key = DEFAULT_TAVILY_API_KEY
                    state.tavily_api_key = DEFAULT_TAVILY_API_KEY
                    MAIN_CONSOLE.print(
                        "[green]Reverted to the bundled Tavily developer key.[/]"
                    )
                return
        new_key = None
        if args:
            new_key = args[0]
        else:
            new_key = _prompt_for_api_key(
                allow_empty=True,
                prompt_text="Enter Tavily API key (press Enter to cancel)",
            )
            if not new_key:
                MAIN_CONSOLE.print("[yellow]Tavily API key unchanged.[/]")
                return
        if _store_tavily_api_key(new_key):
            current_config.tavily_api_key = new_key
            state.tavily_api_key = new_key
            MAIN_CONSOLE.print(
                "[green]Tavily API key updated and will be used going forward.[/]"
            )

    def handle_chat_command(args: List[str]) -> None:
        if not args:
            MAIN_CONSOLE.print("[red]Usage: @chat <message>[/]")
            return
        prompt_text = " ".join(args)
        options = ChatOptions(
            prompt=prompt_text,
            system_prompt=state.system_prompt,
            model=state.model,
            stream=True,
            temperature=0.1,
            top_p=1.0,
            max_tokens=None,
            interactive=False,
            transcript_path=state.transcript_path,
            stream_style=current_config.chat_stream_style,
        )
        run_chat(client, options)

    def handle_completion_command(args: List[str]) -> None:
        if not args:
            MAIN_CONSOLE.print("[red]Usage: @complete <prompt>[/]")
            return
        prompt_text = " ".join(args)
        options = CompletionOptions(
            prompt=prompt_text,
            input_file=None,
            suffix=None,
            model=current_config.completion_model or state.model,
            max_tokens=None,
            temperature=0.1,
            top_p=1.0,
            n=1,
            stop=[],
            stream=False,
            stream_style=current_config.chat_stream_style,
            output_path=None,
        )
        run_completion(client, options, console=MAIN_CONSOLE)

    def handle_embedding_command(args: List[str]) -> None:
        if not args:
            MAIN_CONSOLE.print("[red]Usage: @embed <text>[/]")
            return
        options = EmbeddingOptions(
            texts=args,
            input_file=None,
            model=current_config.embedding_model,
            output_path=None,
            fmt="table",
            show_dimensions=False,
        )
        run_embeddings(client, options, console=MAIN_CONSOLE)

    def handle_models_command(args: List[str]) -> None:
        filter_text: Optional[str] = None
        json_output = False
        limit: Optional[int] = None
        idx = 0
        tokens = list(args)
        while idx < len(tokens):
            token = tokens[idx]
            if token == "--json":
                json_output = True
            elif token == "--limit" and idx + 1 < len(tokens):
                try:
                    limit = int(tokens[idx + 1])
                except ValueError:
                    MAIN_CONSOLE.print("[red]--limit expects an integer value.[/]")
                    return
                idx += 1
            elif token == "--filter" and idx + 1 < len(tokens):
                filter_text = tokens[idx + 1]
                idx += 1
            else:
                remaining = tokens[idx:]
                filter_text = " ".join(remaining)
                break
            idx += 1
        options = ModelListOptions(
            filter=filter_text,
            json_output=json_output,
            limit=limit,
        )
        list_models(client, options, console=MAIN_CONSOLE)

    session = _build_prompt_session()

    while True:
        try:
            with patch_stdout():
                raw = session.prompt(PROMPT_MESSAGE)
        except EOFError:
            MAIN_CONSOLE.line()
            return 0
        except KeyboardInterrupt:
            MAIN_CONSOLE.line()
            return 130
        stripped = raw.strip()
        if not stripped:
            continue
        if stripped[0] in COMMAND_PREFIXES and "\n" not in stripped:
            should_continue = _handle_interactive_command(
                stripped,
                state,
                on_api_command=handle_api_command,
                on_tavily_command=handle_tavily_command,
                on_chat_command=handle_chat_command,
                on_completion_command=handle_completion_command,
                on_embedding_command=handle_embedding_command,
                on_models_command=handle_models_command,
            )
            if not should_continue:
                return 0
            continue

        final_prompt = "\n".join(line.strip() for line in raw.splitlines() if line.strip())
        if not final_prompt:
            continue
        follow_ups = _collect_follow_ups()
        follow_ups.extend(build_test_followups(state.workspace))
        follow_ups.extend([AUTO_TEST_FOLLOW_UP, AUTO_BUG_FOLLOW_UP])
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
        allow_global_access=args.allow_global,
        max_steps=args.max_steps,
        verbose=not args.quiet,
        transcript_path=transcript_path,
        tavily_api_key=resolved.tavily_api_key,
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
            tavily_key = input(
                "Enter Tavily API key (optional, press Enter to use default): "
            )
        except EOFError:
            tavily_key = ""
        config["tavily_api_key"] = tavily_key.strip() or None
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
        "tavily_api_key": getattr(args, "tavily_api_key", None),
        "model": getattr(args, "model", None),
        "system_prompt": getattr(args, "system", None),
    }
    try:
        resolved = resolve_runtime_config(**config_kwargs)
    except RuntimeError as exc:
        missing_api_key = "No DeepSeek API key found" in str(exc)
        if getattr(args, "prompt", None) is None and args.command is None and missing_api_key:
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

    if getattr(args, "prompt", None):
        return handle_agent(args, resolved)

    workspace = Path(args.workspace).expanduser().resolve()
    transcript_path = _resolve_transcript_path(args.transcript) if args.transcript else None

    return run_interactive_agent_shell(
        resolved,
        workspace_override=workspace,
        model_override=args.model,
        system_override=args.system,
        max_steps_override=args.max_steps,
        read_only=args.read_only,
        allow_global=args.allow_global,
        transcript_override=transcript_path,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    raise SystemExit(main())
