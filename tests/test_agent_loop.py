from __future__ import annotations

import json
import io
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace

from deepseek_cli.agent import AgentOptions, agent_loop


def _dummy_response(text: str) -> SimpleNamespace:
    message = SimpleNamespace(content=text, tool_calls=None)
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(choices=[choice])


def test_agent_loop_records_transcript(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    transcript_path = tmp_path / "transcript.jsonl"

    fake_client = SimpleNamespace()
    fake_client.chat = SimpleNamespace()
    fake_client.chat.completions = SimpleNamespace()
    fake_client.chat.completions.create = lambda **_: _dummy_response("All done.")

    options = AgentOptions(
        model="dummy-model",
        system_prompt="System guidance",
        user_prompt="Do the thing",
        follow_up=[],
        workspace=workspace,
        read_only=False,
        allow_global_access=True,
        max_steps=3,
        verbose=False,
        transcript_path=transcript_path,
        tavily_api_key="",
    )

    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        agent_loop(fake_client, options)

    assert transcript_path.exists()
    entries = [json.loads(line) for line in transcript_path.read_text(encoding="utf-8").splitlines()]
    assert len(entries) == 3
    assert entries[0]["step"] == 0
    assert entries[0]["message"]["role"] == "system"
    assert entries[1]["message"]["role"] == "user"
    assert entries[2]["step"] == 1
    assert entries[2]["message"]["role"] == "assistant"
    assert entries[2]["message"]["content"] == "All done."
