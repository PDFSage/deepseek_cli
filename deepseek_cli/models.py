"""Model listing utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

from openai import OpenAI
from rich.console import Console
from rich.table import Table


@dataclass
class ModelListOptions:
    """Options for retrieving models from the API."""

    filter: Optional[str]
    json_output: bool
    limit: Optional[int]


def _render_table(models, *, console: Console, limit: Optional[int], filter_text: Optional[str]) -> None:
    table = Table(title="Available Models", show_header=True, header_style="bold magenta")
    table.add_column("#", style="cyan", justify="right")
    table.add_column("ID", style="white")
    table.add_column("Owner", style="green")
    table.add_column("Created", style="yellow", justify="right")

    items = list(models)
    if filter_text:
        items = [model for model in items if filter_text.lower() in model.id.lower()]
    if limit is not None:
        items = items[:limit]

    for idx, model in enumerate(items, start=1):
        owner = getattr(model, "owned_by", getattr(model, "organization", "n/a"))
        created = getattr(model, "created", None)
        table.add_row(str(idx), model.id, owner or "n/a", str(created or "unknown"))
    console.print(table)


def list_models(client: OpenAI, options: ModelListOptions, *, console: Optional[Console] = None) -> int:
    """Fetch and display available models."""

    console = console or Console()
    response = client.models.list()
    if options.json_output:
        payload = []
        for model in response.data:
            if hasattr(model, "model_dump"):
                payload.append(model.model_dump())
            elif hasattr(model, "dict"):
                payload.append(model.dict())  # type: ignore[call-arg]
            else:
                payload.append(getattr(model, "__dict__", {}))
        console.print(json.dumps(payload, indent=2))
    else:
        _render_table(response.data, console=console, limit=options.limit, filter_text=options.filter)
    return 0


__all__ = ["ModelListOptions", "list_models"]
