from __future__ import annotations

from deepseek_agentic_cli import _rewrite_args


def test_rewrite_args_inserts_prompt() -> None:
    result = _rewrite_args(["fix", "this"])
    assert result == ["--prompt", "fix", "--follow-up", "this"]


def test_rewrite_args_preserves_flag_order() -> None:
    result = _rewrite_args(["--model", "coder", "Implement feature"])
    assert result == ["--prompt", "Implement feature", "--model", "coder"]


def test_rewrite_args_respects_explicit_follow_up() -> None:
    result = _rewrite_args(["--follow-up", "later", "main task"])
    assert result == ["--prompt", "main task", "--follow-up", "later"]


def test_rewrite_args_does_not_duplicate_prompt_flag() -> None:
    result = _rewrite_args(["--prompt", "already", "extra"])
    assert result == ["--prompt", "already", "--follow-up", "extra"]
