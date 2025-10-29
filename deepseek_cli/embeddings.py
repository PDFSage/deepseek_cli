"""Embedding utility helpers."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

from openai import OpenAI
from rich.console import Console
from rich.table import Table


@dataclass
class EmbeddingOptions:
    """Options for generating embeddings across one or more inputs."""

    texts: Sequence[str]
    input_file: Optional[Path]
    model: str
    output_path: Optional[Path]
    fmt: str
    show_dimensions: bool


def _collect_inputs(options: EmbeddingOptions) -> List[str]:
    collected: List[str] = list(options.texts)
    if options.input_file:
        try:
            file_text = options.input_file.read_text(encoding="utf-8")
        except OSError as exc:
            raise RuntimeError(f"Unable to read input file '{options.input_file}': {exc}") from exc
        if file_text:
            collected.extend(line for line in file_text.splitlines() if line.strip())
    if not collected and not sys.stdin.isatty():
        stdin_text = sys.stdin.read()
        if stdin_text:
            collected.append(stdin_text.strip())
    cleaned = [text for text in (text.strip() for text in collected) if text]
    if not cleaned:
        raise RuntimeError("Provide at least one text snippet or input file for embeddings.")
    return cleaned


def _build_table(texts: Sequence[str], vectors: Sequence[Sequence[float]], *, show_dimensions: bool) -> Table:
    table = Table(title="Embedding Vectors", show_header=True, header_style="bold magenta")
    table.add_column("#", style="cyan", justify="right")
    table.add_column("Input", style="white")
    if show_dimensions:
        table.add_column("Dimensions", style="green", justify="right")
    table.add_column("Preview", style="yellow")
    for idx, (text, vector) in enumerate(zip(texts, vectors), start=1):
        preview = ", ".join(f"{value:.3f}" for value in vector[:6])
        if len(vector) > 6:
            preview += ", …"
        columns = [str(idx), text]
        if show_dimensions:
            columns.append(str(len(vector)))
        columns.append(preview)
        table.add_row(*columns)
    return table


def run_embeddings(client: OpenAI, options: EmbeddingOptions, *, console: Optional[Console] = None) -> int:
    """Generate embeddings and render the output."""

    console = console or Console()
    try:
        inputs = _collect_inputs(options)
    except RuntimeError as exc:
        console.print(f"[red]{exc}[/]")
        return 1

    response = client.embeddings.create(model=options.model, input=list(inputs))
    vectors = [data.embedding for data in response.data]

    if options.fmt == "json":
        payload = {
            "model": response.model,
            "embeddings": [
                {"index": item.index, "embedding": item.embedding}
                for item in response.data
            ],
        }
        serialized = json.dumps(payload, indent=2)
        console.print(serialized)
    elif options.fmt == "plain":
        for vector in vectors:
            console.print(" ".join(f"{value:.6f}" for value in vector))
    else:
        table = _build_table(inputs, vectors, show_dimensions=options.show_dimensions)
        console.print(table)

    if options.output_path:
        payload = {
            "model": response.model,
            "inputs": inputs,
            "embeddings": vectors,
        }
        try:
            options.output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except OSError as exc:
            console.print(f"[red]Failed to persist embeddings to {options.output_path}: {exc}[/]")
            return 1
        console.print(f"[green]Saved embeddings to {options.output_path}[/]")

    return 0


__all__ = ["EmbeddingOptions", "run_embeddings"]
