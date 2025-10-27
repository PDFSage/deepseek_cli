#!/usr/bin/env python3
"""Backward compatible shim for the legacy DeepSeek agent CLI."""

from __future__ import annotations

import sys

from deepseek_cli.cli import main as cli_main

if __name__ == "__main__":
    print(
        "`deepseek_agentic_cli.py` is deprecated. Use `deepseek agent` instead.",
        file=sys.stderr,
    )
    raise SystemExit(cli_main(["agent", *sys.argv[1:]]))
