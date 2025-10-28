"""Main CLI entry point."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
import textwrap
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from packaging.version import Version

from openai import OpenAI

from . import __version__
from .agent import AgentOptions, agent_loop
from .chat import ChatOptions, run_chat
from .config import (
    ResolvedConfig,
    ensure_config_dir,
    load_config,
    pretty_config,
    resolve_runtime_config,
    save_config,
    update_config,
)
from .constants import AUTO_TEST_FOLLOW_UP, CONFIG_FILE, DEFAULT_MAX_STEPS, TRANSCRIPTS_DIR


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
        "--interactive",
        action="store_true",
        help="Enter an interactive multi-turn chat session",
    )
    chat_parser.add_argument(
        "--transcript",
        help="Path to save a JSONL transcript (defaults under ~/.config/deepseek-cli)",
    )
    add_shared_connection_options(chat_parser)

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
            "system_prompt",
            "chat_system_prompt",
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
            "system_prompt",
            "chat_system_prompt",
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


def handle_chat(args: argparse.Namespace, resolved: ResolvedConfig) -> int:
    client = create_client(resolved)
    transcript_path = _resolve_transcript_path(args.transcript)
    options = ChatOptions(
        prompt=args.prompt,
        system_prompt=args.system or resolved.chat_system_prompt,
        model=args.model or resolved.chat_model,
        stream=not args.no_stream,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        interactive=args.interactive,
        transcript_path=transcript_path,
    )
    return run_chat(client, options)


def _resolve_transcript_path(value: Optional[str]) -> Optional[Path]:
    if value is None:
        return None
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return candidate
    ensure_config_dir()
    TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    return TRANSCRIPTS_DIR / candidate


def _format_session_status(state: InteractiveSessionState) -> str:
    lines = [
        f"Workspace : {state.workspace}",
        f"Model     : {state.model}",
        f"System    : {('custom' if state.system_prompt != state.default_system_prompt else 'default')} prompt",
        f"Read-only : {'on' if state.read_only else 'off'}",
        f"Max steps : {state.max_steps}",
        f"Global ops: {'on' if state.allow_global_access else 'off'}",
        f"Verbose   : {'on' if state.verbose else 'off'}",
    ]
    if state.transcript_path:
        lines.append(f"Transcript: {state.transcript_path}")
    return "\n".join(lines)


def _print_interactive_help() -> None:
    help_text = textwrap.dedent(
        """
        Commands:
          :help or :?           Show this help message
          :quit or :exit        Leave the interactive agent
          :workspace [PATH]     Show or change the active workspace directory
          :model [NAME]         Show or change the model
          :system [TEXT]        Show or replace the system prompt
          :max-steps [N]        Show or change the max reasoning steps
          :read-only [on|off]   Toggle write access to the workspace
          :verbose / :quiet     Enable or disable tool logging
          :global [on|off]      Allow editing outside the workspace root
          :transcript [PATH]    Write transcripts to PATH (relative to workspace)
          :clear-transcript     Stop writing transcripts
          :settings             Display the current session settings
          :reset                Restore defaults from your configuration

        Enter your request at the prompt. After a prompt you can add optional
        follow-ups; press Enter on an empty line to run the agent.
        """
    ).strip()
    print(help_text)


def _handle_interactive_command(
    raw: str,
    state: InteractiveSessionState,
) -> bool:
    command_line = raw[1:].strip()
    if not command_line:
        _print_interactive_help()
        return True
    try:
        parts = shlex.split(command_line)
    except ValueError as exc:
        print(f"Unable to parse command: {exc}", file=sys.stderr)
        return True
    if not parts:
        return True
    name = parts[0].lower()
    args = parts[1:]

    if name in {"quit", "exit", "q"}:
        print("Exiting interactive agent.")
        return False
    if name in {"help", "?"}:
        _print_interactive_help()
        return True
    if name in {"settings", "status"}:
        print(_format_session_status(state))
        return True
    if name == "workspace":
        if not args:
            print(f"Workspace: {state.workspace}")
            return True
        raw_path = " ".join(args)
        candidate = Path(raw_path).expanduser()
        if not candidate.is_absolute():
            candidate = (state.workspace / raw_path).expanduser()
        try:
            candidate = candidate.resolve()
        except FileNotFoundError:
            print(f"Workspace '{candidate}' does not exist.", file=sys.stderr)
            return True
        if not candidate.exists() or not candidate.is_dir():
            print(f"Workspace '{candidate}' is not a directory.", file=sys.stderr)
            return True
        state.workspace = candidate
        print(f"Workspace set to {state.workspace}")
        return True
    if name == "model":
        if not args:
            print(f"Model: {state.model}")
            return True
        state.model = " ".join(args)
        print(f"Model set to {state.model}")
        return True
    if name == "system":
        if not args:
            print(state.system_prompt)
            return True
        state.system_prompt = " ".join(args)
        print("System prompt updated.")
        return True
    if name == "max-steps":
        if not args:
            print(f"Max steps: {state.max_steps}")
            return True
        try:
            value = int(args[0])
        except ValueError:
            print("max-steps requires an integer value.", file=sys.stderr)
            return True
        if value < 1:
            print("max-steps must be at least 1.", file=sys.stderr)
            return True
        state.max_steps = value
        print(f"Max steps set to {value}")
        return True
    if name == "read-only":
        if not args:
            print(f"Read-only: {'on' if state.read_only else 'off'}")
            return True
        setting = args[0].lower()
        if setting in {"on", "true", "1"}:
            state.read_only = True
        elif setting in {"off", "false", "0"}:
            state.read_only = False
        elif setting == "toggle":
            state.read_only = not state.read_only
        else:
            print("Use on/off/toggle to control read-only mode.", file=sys.stderr)
            return True
        print(f"Read-only mode {'enabled' if state.read_only else 'disabled'}.")
        return True
    if name == "global":
        if not args:
            print(f"Global operations: {'on' if state.allow_global_access else 'off'}")
            return True
        setting = args[0].lower()
        if setting in {"on", "true", "1"}:
            state.allow_global_access = True
        elif setting in {"off", "false", "0"}:
            state.allow_global_access = False
        elif setting == "toggle":
            state.allow_global_access = not state.allow_global_access
        else:
            print("Use on/off/toggle to control global operations.", file=sys.stderr)
            return True
        message = "enabled (paths may escape workspace)" if state.allow_global_access else "disabled"
        print(f"Global operations {message}.")
        return True
    if name == "verbose":
        state.verbose = True
        print("Verbose tool logging enabled.")
        return True
    if name == "quiet":
        state.verbose = False
        print("Verbose tool logging disabled.")
        return True
    if name == "transcript":
        if not args:
            if state.transcript_path:
                print(f"Transcript logging to {state.transcript_path}")
            else:
                print("Transcript logging is disabled.")
            return True
        raw_path = " ".join(args)
        candidate = Path(raw_path).expanduser()
        if not candidate.is_absolute():
            candidate = (state.workspace / raw_path).expanduser()
        state.transcript_path = candidate
        print(f"Transcript logging set to {state.transcript_path}")
        return True
    if name == "clear-transcript":
        state.transcript_path = None
        print("Transcript logging disabled.")
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
        print("Session settings reset to defaults.")
        print(_format_session_status(state))
        return True

    print(f"Unknown command '{name}'. Type :help for options.", file=sys.stderr)
    return True


def _collect_follow_ups() -> List[str]:
    follow_ups: List[str] = []
    while True:
        try:
            line = input("  Follow-up (press Enter to run) › ")
        except EOFError:
            print()
            break
        except KeyboardInterrupt:
            print()
            break
        stripped = line.strip()
        if not stripped:
            break
        follow_ups.append(stripped)
    return follow_ups


def _run_interactive_agent_prompt(
    client: OpenAI,
    state: InteractiveSessionState,
    prompt: str,
    follow_ups: List[str],
) -> None:
    workspace = state.workspace
    if not workspace.exists():
        print(
            f"Workspace '{workspace}' does not exist. Use :workspace to choose another.",
            file=sys.stderr,
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
        print(f"Agent error: {exc}", file=sys.stderr)


def run_interactive_agent_shell(resolved: ResolvedConfig) -> int:
    client = create_client(resolved)
    state = InteractiveSessionState(
        workspace=Path.cwd().resolve(),
        model=resolved.model,
        system_prompt=resolved.system_prompt,
    )

    print("Initializing DeepSeek agent…", file=sys.stderr)
    banner = textwrap.dedent(
        """
        DeepSeek Agent • Interactive coding workspace
        Commands start with ':'. Type :help for assistance, :quit to exit.
        """
    ).strip()
    print(banner)
    print(_format_session_status(state))

    while True:
        try:
            raw = input("Prompt › ")
        except EOFError:
            print()
            return 0
        except KeyboardInterrupt:
            print()
            return 130
        prompt = raw.strip()
        if not prompt:
            continue
        if prompt.startswith(":"):
            should_continue = _handle_interactive_command(prompt, state)
            if not should_continue:
                return 0
            continue

        prompt_lines = [prompt]
        while prompt_lines[-1].endswith("\\"):
            prompt_lines[-1] = prompt_lines[-1].rstrip("\\")
            try:
                continuation = input("… ")
            except EOFError:
                print()
                break
            except KeyboardInterrupt:
                print()
                break
            prompt_lines.append(continuation.rstrip())
        final_prompt = "\n".join(line.strip() for line in prompt_lines if line.strip())
        follow_ups = _collect_follow_ups()
        follow_ups.append(AUTO_TEST_FOLLOW_UP)
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
        follow_up=(args.follow_up or []) + [AUTO_TEST_FOLLOW_UP],
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

    notify_if_update_available()

    if args.command == "config":
        return handle_config(args)

    try:
        resolved = resolve_runtime_config(
            api_key=getattr(args, "api_key", None),
            base_url=getattr(args, "base_url", None),
            model=getattr(args, "model", None),
            system_prompt=getattr(args, "system", None),
        )
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if args.command is None:
        return run_interactive_agent_shell(resolved)

    if args.command == "chat":
        return handle_chat(args, resolved)
    if args.command == "agent":
        return handle_agent(args, resolved)

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
