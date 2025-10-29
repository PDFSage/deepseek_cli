"""Developer-friendly chat helpers."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from openai import OpenAI
from rich.console import Console

from .streaming import stream_chat_response


@dataclass
class ChatOptions:
    """Options for an interactive or single-turn chat session."""

    prompt: Optional[str]
    system_prompt: str
    model: str
    stream: bool
    temperature: float
    top_p: float
    max_tokens: Optional[int]
    interactive: bool
    transcript_path: Optional[Path]
    stream_style: str


def _log_to_transcript(path: Optional[Path], payload: dict) -> None:
    if not path:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _collect_user_turn(initial_prompt: Optional[str], interactive: bool) -> Optional[str]:
    if initial_prompt is not None:
        return initial_prompt
    if not interactive:
        return None
    try:
        return input("You ▸ ")
    except EOFError:
        return None


def run_chat(client: OpenAI, options: ChatOptions) -> int:
    console = Console()
    messages: List[dict] = []
    if options.system_prompt:
        messages.append({"role": "system", "content": options.system_prompt})
        _log_to_transcript(options.transcript_path, {"role": "system", "content": options.system_prompt})

    prompt = _collect_user_turn(options.prompt, options.interactive)
    if prompt is None:
        print("No prompt provided. Pass text as an argument or use interactive mode.", file=sys.stderr)
        return 1

    while prompt is not None:
        messages.append({"role": "user", "content": prompt})
        _log_to_transcript(options.transcript_path, {"role": "user", "content": prompt})

        kwargs = {
            "model": options.model,
            "messages": messages,
            "temperature": options.temperature,
            "top_p": options.top_p,
        }
        if options.max_tokens is not None:
            kwargs["max_tokens"] = options.max_tokens

        if options.stream:
            stream = client.chat.completions.create(stream=True, **kwargs)
            reply = stream_chat_response(stream, style=options.stream_style, console=console)
        else:
            completion = client.chat.completions.create(**kwargs)
            reply = completion.choices[0].message.content or ""
            console.print(f"Assistant ▸ {reply}")

        messages.append({"role": "assistant", "content": reply})
        _log_to_transcript(options.transcript_path, {"role": "assistant", "content": reply})

        if not options.interactive:
            break
        try:
            prompt = input("You ▸ ")
        except EOFError:
            break
    return 0


__all__ = ["ChatOptions", "run_chat"]
