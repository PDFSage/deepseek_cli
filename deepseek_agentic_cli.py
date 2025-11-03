#!/usr/bin/env python3
"""Backward compatible shim for the legacy DeepSeek agent CLI."""

from __future__ import annotations

import sys

from deepseek_cli.cli import main as cli_main

_VALUE_FLAGS = {
    "--api-key",
    "--base-url",
    "--follow-up",
    "--system",
    "--model",
    "--workspace",
    "--max-steps",
    "--transcript",
    "--tavily-api-key",
    "--prompt",
}


def _rewrite_args(argv: list[str]) -> list[str]:
    converted: list[str] = []
    prompt: str | None = None
    pending_flag: str | None = None
    prompt_flag_seen = False

    for token in argv:
        if pending_flag is not None:
            converted.append(token)
            if pending_flag == "--prompt":
                prompt_flag_seen = True
            pending_flag = None
            continue
        if token in _VALUE_FLAGS:
            converted.append(token)
            pending_flag = token
            continue
        if token in {"--no-global", "--global", "--read-only", "--quiet"}:
            converted.append(token)
            continue
        if token.startswith("-"):
            converted.append(token)
            continue
        if prompt is None and not prompt_flag_seen:
            prompt = token
        else:
            converted.extend(["--follow-up", token])

    if prompt is not None and not prompt_flag_seen:
        converted = ["--prompt", prompt, *converted]

    return converted


if __name__ == "__main__":
    print(
        "`deepseek_agentic_cli.py` is deprecated. Use `deepseek` (with --prompt) instead.",
        file=sys.stderr,
    )
    forwarded = _rewrite_args(sys.argv[1:])
    raise SystemExit(cli_main(forwarded))
