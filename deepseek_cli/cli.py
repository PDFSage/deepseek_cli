"""Main CLI entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

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
from .constants import CONFIG_FILE, TRANSCRIPTS_DIR


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
        default=20,
        help="Maximum reasoning steps before aborting",
    )
    agent_parser.add_argument(
        "--transcript",
        help="Optional transcript path (default stored under ~/.config/deepseek-cli)",
    )
    agent_parser.add_argument("--read-only", action="store_true", help="Disable write operations")
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
        follow_up=args.follow_up,
        workspace=workspace,
        read_only=args.read_only,
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

    if args.command is None:
        parser.print_help()
        return 1

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

    if args.command == "chat":
        return handle_chat(args, resolved)
    if args.command == "agent":
        return handle_agent(args, resolved)

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
