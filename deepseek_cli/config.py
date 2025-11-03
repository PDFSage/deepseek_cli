"""Configuration helpers for the DeepSeek CLI."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional

from .constants import (
    CONFIG_DIR,
    CONFIG_FILE,
    DEFAULT_BASE_URL,
    DEFAULT_CHAT_MODEL,
    DEFAULT_CHAT_STREAM_STYLE,
    DEFAULT_CHAT_SYSTEM_PROMPT,
    DEFAULT_COMPLETION_MODEL,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_MODEL,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_TAVILY_API_KEY,
)

ENV_API_KEY = "DEEPSEEK_API_KEY"
ENV_BASE_URL = "DEEPSEEK_BASE_URL"
ENV_MODEL = "DEEPSEEK_MODEL"
ENV_SYSTEM_PROMPT = "DEEPSEEK_SYSTEM_PROMPT"
ENV_CHAT_MODEL = "DEEPSEEK_CHAT_MODEL"
ENV_COMPLETION_MODEL = "DEEPSEEK_COMPLETION_MODEL"
ENV_EMBEDDING_MODEL = "DEEPSEEK_EMBEDDING_MODEL"
ENV_CHAT_SYSTEM_PROMPT = "DEEPSEEK_CHAT_SYSTEM_PROMPT"
ENV_CHAT_STREAM_STYLE = "DEEPSEEK_CHAT_STREAM_STYLE"
ENV_TAVILY_API_KEY = "TAVILY_API_KEY"

_DEFAULTS: Dict[str, Any] = {
    "api_key": None,
    "base_url": DEFAULT_BASE_URL,
    "model": DEFAULT_MODEL,
    "chat_model": DEFAULT_CHAT_MODEL,
    "completion_model": DEFAULT_COMPLETION_MODEL,
    "embedding_model": DEFAULT_EMBEDDING_MODEL,
    "system_prompt": DEFAULT_SYSTEM_PROMPT,
    "chat_system_prompt": DEFAULT_CHAT_SYSTEM_PROMPT,
    "chat_stream_style": DEFAULT_CHAT_STREAM_STYLE,
    "tavily_api_key": DEFAULT_TAVILY_API_KEY,
}


@dataclass
class ResolvedConfig:
    """Fully resolved runtime configuration."""

    api_key: str
    base_url: str
    model: str
    system_prompt: str
    chat_model: str
    chat_system_prompt: str
    completion_model: str
    embedding_model: str
    chat_stream_style: str
    tavily_api_key: str


def ensure_config_dir() -> None:
    """Create the configuration directory if it doesn't already exist."""

    try:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        # Best effort: allow runtime usage without persistent config storage.
        pass


def load_config() -> Dict[str, Any]:
    """Load configuration values from disk, falling back to defaults."""

    ensure_config_dir()
    if not CONFIG_FILE.exists():
        return dict(_DEFAULTS)
    try:
        data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return dict(_DEFAULTS)
    result = dict(_DEFAULTS)
    result.update({k: v for k, v in data.items() if k in result})
    return result


def save_config(values: Mapping[str, Any]) -> None:
    """Persist configuration values to disk."""

    ensure_config_dir()
    payload = dict(_DEFAULTS)
    payload.update({k: v for k, v in values.items() if k in payload})
    try:
        CONFIG_FILE.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except PermissionError as exc:  # pragma: no cover
        raise RuntimeError(
            f"Unable to write configuration to '{CONFIG_FILE}'. "
            "Set environment variables instead or adjust permissions."
        ) from exc


def update_config(pairs: Iterable[tuple[str, Any]]) -> Dict[str, Any]:
    """Update the configuration file with the provided key/value pairs."""

    config = load_config()
    for key, value in pairs:
        if key not in config:
            raise KeyError(f"Unknown configuration key '{key}'.")
        config[key] = value
    save_config(config)
    return config


def resolve_runtime_config(
    *,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    system_prompt: Optional[str] = None,
    chat_model: Optional[str] = None,
    chat_system_prompt: Optional[str] = None,
    completion_model: Optional[str] = None,
    embedding_model: Optional[str] = None,
    chat_stream_style: Optional[str] = None,
    tavily_api_key: Optional[str] = None,
) -> ResolvedConfig:
    """Resolve runtime config using CLI options, environment variables, and stored config."""

    stored = load_config()
    resolved_api_key = (
        api_key
        or os.environ.get(ENV_API_KEY)
        or stored.get("api_key")
    )
    if not resolved_api_key:
        raise RuntimeError(
            "No DeepSeek API key found. Visit https://platform.deepseek.com/api_keys "
            "to create one, then run 'deepseek config set api_key YOUR_KEY' or set "
            "the DEEPSEEK_API_KEY environment variable."
        )

    resolved = ResolvedConfig(
        api_key=resolved_api_key,
        base_url=base_url or os.environ.get(ENV_BASE_URL) or stored.get("base_url", DEFAULT_BASE_URL),
        model=model or os.environ.get(ENV_MODEL) or stored.get("model", DEFAULT_MODEL),
        system_prompt=system_prompt or os.environ.get(ENV_SYSTEM_PROMPT) or stored.get("system_prompt", DEFAULT_SYSTEM_PROMPT),
        chat_model=chat_model or os.environ.get(ENV_CHAT_MODEL) or stored.get("chat_model", DEFAULT_CHAT_MODEL),
        chat_system_prompt=chat_system_prompt or os.environ.get(ENV_CHAT_SYSTEM_PROMPT) or stored.get("chat_system_prompt", DEFAULT_CHAT_SYSTEM_PROMPT),
        completion_model=completion_model or os.environ.get(ENV_COMPLETION_MODEL) or stored.get("completion_model", DEFAULT_COMPLETION_MODEL),
        embedding_model=embedding_model or os.environ.get(ENV_EMBEDDING_MODEL) or stored.get("embedding_model", DEFAULT_EMBEDDING_MODEL),
        chat_stream_style=(chat_stream_style or os.environ.get(ENV_CHAT_STREAM_STYLE) or stored.get("chat_stream_style", DEFAULT_CHAT_STREAM_STYLE)).lower(),
        tavily_api_key=tavily_api_key
        or os.environ.get(ENV_TAVILY_API_KEY)
        or stored.get("tavily_api_key")
        or DEFAULT_TAVILY_API_KEY,
    )
    if resolved.chat_stream_style not in {"plain", "markdown", "rich"}:
        resolved.chat_stream_style = DEFAULT_CHAT_STREAM_STYLE
    return resolved


def pretty_config(config: Mapping[str, Any], *, redact: bool = True) -> str:
    """Format configuration for terminal display."""

    payload = dict(config)
    if redact:
        for key in ("api_key", "tavily_api_key"):
            value = payload.get(key)
            if value:
                payload[key] = value[:4] + "…" + value[-4:]
    return json.dumps(payload, indent=2, sort_keys=True)


__all__ = [
    "ResolvedConfig",
    "ENV_API_KEY",
    "ENV_BASE_URL",
    "ENV_MODEL",
    "ENV_SYSTEM_PROMPT",
    "ENV_CHAT_MODEL",
    "ENV_COMPLETION_MODEL",
    "ENV_EMBEDDING_MODEL",
    "ENV_CHAT_SYSTEM_PROMPT",
    "ENV_CHAT_STREAM_STYLE",
    "ENV_TAVILY_API_KEY",
    "load_config",
    "save_config",
    "update_config",
    "resolve_runtime_config",
    "ensure_config_dir",
    "pretty_config",
]
