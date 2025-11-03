from __future__ import annotations

from pathlib import Path

import pytest

from deepseek_cli import config as config_module
from deepseek_cli.constants import DEFAULT_CHAT_STREAM_STYLE, DEFAULT_TAVILY_API_KEY


@pytest.fixture()
def isolated_config_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    config_path = tmp_path / "config.json"
    monkeypatch.setattr(config_module, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(config_module, "CONFIG_FILE", config_path)
    return config_path


def test_resolve_runtime_config_includes_enhanced_defaults(isolated_config_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(config_module.ENV_API_KEY, "test-key")
    resolved = config_module.resolve_runtime_config()
    assert resolved.completion_model
    assert resolved.embedding_model
    assert resolved.chat_stream_style == DEFAULT_CHAT_STREAM_STYLE


def test_chat_stream_style_env_override(isolated_config_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(config_module.ENV_API_KEY, "test-key")
    monkeypatch.setenv(config_module.ENV_CHAT_STREAM_STYLE, "markdown")
    resolved = config_module.resolve_runtime_config()
    assert resolved.chat_stream_style == "markdown"

    monkeypatch.setenv(config_module.ENV_CHAT_STREAM_STYLE, "unsupported")
    resolved = config_module.resolve_runtime_config()
    assert resolved.chat_stream_style == DEFAULT_CHAT_STREAM_STYLE


def test_tavily_key_default_and_override(isolated_config_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(config_module.ENV_API_KEY, "test-key")
    resolved = config_module.resolve_runtime_config()
    assert resolved.tavily_api_key == DEFAULT_TAVILY_API_KEY

    monkeypatch.setenv(config_module.ENV_TAVILY_API_KEY, "tvly-custom-key")
    resolved = config_module.resolve_runtime_config()
    assert resolved.tavily_api_key == "tvly-custom-key"
