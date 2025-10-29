"""Shared streaming renderers for DeepSeek CLI responses."""

from __future__ import annotations

from typing import Any, Callable, Iterable, List, Optional

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

StreamExtractor = Callable[[Any], str]


def _render_with_live(
    renderable_factory: Callable[[str], Any],
    pieces: List[str],
    chunks: Iterable[Any],
    *,
    console: Optional[Console],
) -> None:
    with Live(renderable_factory(""), refresh_per_second=12, console=console) as live:
        for chunk in chunks:
            text = chunk or ""
            if not text:
                continue
            pieces.append(text)
            live.update(renderable_factory("".join(pieces)))


def _stream_text(
    chunks: Iterable[Any],
    *,
    prefix: str,
    style: str,
    extractor: StreamExtractor,
    console: Optional[Console] = None,
) -> str:
    console = console or Console()
    pieces: List[str] = []
    normalized = (style or "plain").lower()
    if normalized == "markdown":
        console.print(prefix)
        def build_markdown(text: str) -> Markdown:
            return Markdown(text or "")
        _render_with_live(build_markdown, pieces, (extractor(chunk) for chunk in chunks), console=console)
        console.line()
    elif normalized == "rich":
        console.print(prefix)
        def build_panel(text: str) -> Panel:
            body = Text(text or "", no_wrap=False)
            return Panel(body, border_style="cyan")
        _render_with_live(build_panel, pieces, (extractor(chunk) for chunk in chunks), console=console)
        console.line()
    else:
        console.print(prefix, end="", soft_wrap=True, highlight=False)
        for chunk in chunks:
            text = extractor(chunk)
            if not text:
                continue
            pieces.append(text)
            console.print(text, end="", soft_wrap=True, highlight=False)
        console.print()
    return "".join(pieces)


def stream_chat_response(stream: Iterable[Any], *, style: str, console: Optional[Console] = None) -> str:
    """Stream chat completions to the console using the requested rendering style."""

    def extractor(chunk: Any) -> str:
        choice = getattr(chunk, "choices", None)
        if not choice:
            return ""
        choice = choice[0]
        delta = getattr(choice, "delta", None)
        if delta:
            return getattr(delta, "content", "") or ""
        return getattr(choice, "message", {}).get("content", "") if hasattr(choice, "message") else ""

    return _stream_text(
        stream,
        prefix="Assistant ▸ ",
        style=style,
        extractor=extractor,
        console=console,
    )


def stream_completion_response(stream: Iterable[Any], *, style: str, console: Optional[Console] = None) -> str:
    """Stream text completions to the console using the requested rendering style."""

    def extractor(chunk: Any) -> str:
        choice = getattr(chunk, "choices", None)
        if not choice:
            return ""
        choice = choice[0]
        if hasattr(choice, "text"):
            return getattr(choice, "text", "") or ""
        delta = getattr(choice, "delta", None)
        if delta:
            return getattr(delta, "content", "") or ""
        return ""

    return _stream_text(
        stream,
        prefix="Completion ▸ ",
        style=style,
        extractor=extractor,
        console=console,
    )


__all__ = ["stream_chat_response", "stream_completion_response"]
