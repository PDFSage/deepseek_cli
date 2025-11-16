from __future__ import annotations

import io

from rich.console import Console

from deepseek_cli.agent import BreadcrumbLogger
from deepseek_cli.cli import PromptAwarePrinter


def _console_capture() -> tuple[Console, io.StringIO]:
    buffer = io.StringIO()
    console = Console(
        file=buffer,
        force_terminal=False,
        markup=False,
        highlight=False,
        width=120,
    )
    return console, buffer


def test_plan_update_limits_and_deduplicates() -> None:
    console, buffer = _console_capture()
    logger = BreadcrumbLogger(console, max_steps=5, plan_item_limit=2)

    logger.plan_update(3, "- alpha\n- beta\n- gamma")

    output = buffer.getvalue()
    assert "[PLAN step 3] Plan refreshed with 3 items:" in output
    assert "    1. alpha" in output
    assert "    2. beta" in output
    assert "    … (1 more items)" in output

    frozen = buffer.getvalue()
    logger.plan_update(3, "- alpha\n- beta\n- gamma")
    assert buffer.getvalue() == frozen


def test_worker_and_tool_breadcrumbs() -> None:
    console, buffer = _console_capture()
    logger = BreadcrumbLogger(console, max_steps=7)

    logger.worker_step(2)
    first = buffer.getvalue()
    assert "[WORKER] Executing step 2/7" in first

    logger.worker_step(2)
    assert buffer.getvalue() == first

    arguments = {"path": ".", "command": "ls -la"}
    logger.tool_started(2, "run_shell", arguments)
    assert "    -> run_shell (path=., command=ls -la)" in buffer.getvalue()

    logger.tool_completed(2, "run_shell", arguments, 0.5, "Completed action")
    tool_output = buffer.getvalue()
    assert "    ✓ run_shell (path=., command=ls -la) (0.50s) – Completed action" in tool_output

    logger.tool_failed(2, "run_shell", "No output")
    assert "    x run_shell – No output" in buffer.getvalue()

    logger.final_response(2)
    assert "[DONE] Step 2 returned the final response." in buffer.getvalue()


def test_breadcrumb_logger_uses_custom_sink() -> None:
    lines: list[str] = []
    console, _ = _console_capture()
    logger = BreadcrumbLogger(console, max_steps=2, line_sink=lines.append)

    logger.worker_step(1)
    logger.tool_failed(1, "run_shell", "Oops")

    assert lines[0] == "[WORKER] Executing step 1/2"
    assert lines[1].startswith("    x run_shell")


def test_prompt_aware_printer_serializes_lines() -> None:
    class DummyApp:
        def __init__(self) -> None:
            self.is_running = False

        def run_in_terminal(self, func) -> None:
            func()

    class DummySession:
        def __init__(self) -> None:
            self.app = DummyApp()

    console, buffer = _console_capture()
    printer = PromptAwarePrinter(DummySession(), console)
    try:
        printer("first")
        printer("second")
    finally:
        printer.close()

    assert buffer.getvalue().splitlines() == ["first", "second"]
