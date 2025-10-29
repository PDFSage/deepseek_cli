"""Codex-style completion helpers."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from openai import OpenAI
from rich.console import Console
from rich.table import Table

from .streaming import stream_completion_response


@dataclass
class CompletionOptions:
    """Options for issuing text/code completion requests."""

    prompt: Optional[str]
    input_file: Optional[Path]
    suffix: Optional[str]
    model: str
    max_tokens: Optional[int]
    temperature: float
    top_p: float
    n: int
    stop: Sequence[str]
    stream: bool
    stream_style: str
    output_path: Optional[Path]


def _read_prompt_from_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"Unable to read prompt file '{path}': {exc}") from exc


def _gather_prompt(options: CompletionOptions) -> str:
    prompt = options.prompt
    if options.input_file is not None:
        prompt = _read_prompt_from_file(options.input_file)
    if prompt is not None:
        return prompt
    if not sys.stdin.isatty():
        return sys.stdin.read()
    raise RuntimeError("No prompt provided. Pass text, --input-file, or pipe content via stdin.")


def _save_output(path: Path, text: str) -> None:
    try:
        path.write_text(text, encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"Failed to write completion to '{path}': {exc}") from exc


def _format_choices(choices: Sequence[str]) -> Table:
    table = Table(title="Completion Choices", show_header=True, header_style="bold magenta")
    table.add_column("#", style="cyan", justify="right")
    table.add_column("Output", style="white")
    for idx, choice in enumerate(choices, start=1):
        table.add_row(str(idx), choice)
    return table


def run_completion(client: OpenAI, options: CompletionOptions, *, console: Optional[Console] = None) -> int:
    """Run a completion request and render results."""

    console = console or Console()
    try:
        prompt = _gather_prompt(options)
    except RuntimeError as exc:
        console.print(f"[red]{exc}[/]")
        return 1
    prompt = prompt.rstrip("\n")
    if not prompt:
        console.print("[red]Prompt is empty after trimming input.[/]")
        return 1

    stop = list(options.stop)
    if options.stream and options.n != 1:
        console.print("[red]Streaming is only supported with n=1 completions.[/]")
        return 1

    kwargs = {
        "model": options.model,
        "prompt": prompt,
        "temperature": options.temperature,
        "top_p": options.top_p,
        "n": options.n,
    }
    if options.max_tokens is not None:
        kwargs["max_tokens"] = options.max_tokens
    if options.suffix:
        kwargs["suffix"] = options.suffix
    if stop:
        kwargs["stop"] = list(stop)

    if options.stream:
        stream = client.completions.create(stream=True, **kwargs)
        completion_text = stream_completion_response(
            stream,
            style=options.stream_style,
            console=console,
        )
        choices = [completion_text]
    else:
        response = client.completions.create(**kwargs)
        choices = [choice.text or "" for choice in response.choices]
        console.print(_format_choices(choices))

    if options.output_path:
        _save_output(options.output_path, choices[0])
        console.print(f"[green]Saved primary completion to {options.output_path}[/]")

    return 0


__all__ = ["CompletionOptions", "run_completion"]
