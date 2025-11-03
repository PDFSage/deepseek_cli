# DeepSeek CLI Internal Architecture

This document explains how the DeepSeek command line interface (CLI) works under the hood. It covers
package layout, execution flow, configuration resolution, and the implementation details for the unified
agent-first experience (including inline chat/completion helpers) and the `config` command group. Paths
below are relative to the repository root.

## Package Layout and Entry Points
- `pyproject.toml` registers two console scripts (`deepseek` and `deepseek-cli`) that both call
  `deepseek_cli.cli:main`. Invoking `python -m deepseek_cli` goes through `deepseek_cli/__main__.py`
  and ultimately hits the same `main()` function.
- `deepseek_agentic_cli.py` is a legacy shim that prints a warning and forwards arguments to the unified
  `deepseek` entry point (coercing legacy positional prompts into `--prompt`) so existing automation keeps working.
- `deepseek_cli/__init__.py` exposes the package version (`__version__`), used for `--version` output
  and update checks.

The package depends on `openai` ≥ 1.13.3 for Chat Completions API access and `packaging` for version
comparison. All other functionality is implemented locally.

## Configuration Resolution (`deepseek_cli/config.py`)
Configuration values come from three layers, resolved at runtime in the following order (highest
priority first):
1. Command line flags such as `--api-key` and `--model`.
2. Environment variables (`DEEPSEEK_API_KEY`, `DEEPSEEK_BASE_URL`, `DEEPSEEK_MODEL`,
   `DEEPSEEK_SYSTEM_PROMPT`, `DEEPSEEK_CHAT_MODEL`, `DEEPSEEK_COMPLETION_MODEL`,
   `DEEPSEEK_EMBEDDING_MODEL`, `DEEPSEEK_CHAT_STREAM_STYLE`).
3. The JSON config file at `~/.config/deepseek-cli/config.json`.

`ensure_config_dir()` creates the config directory on demand. `load_config()` merges the persisted JSON
with built-in defaults. `resolve_runtime_config()` validates that an API key is available, then builds a
`ResolvedConfig` dataclass that includes both agent and chat defaults. Saving and updating values flow
through `save_config()` and `update_config()`, which both filter unknown keys and persist the combined
(default + overrides) payload.

## CLI Wiring and Command Dispatch (`deepseek_cli/cli.py`)
`build_parser()` constructs a single `argparse.ArgumentParser` that defaults to the unified agent
experience. Top-level flags control the session (`--prompt`, `--workspace`, `--model`, `--system`,
`--max-steps`, `--read-only`, `--no-global`, `--quiet`) while shared connection options
(`--api-key`, `--base-url`, `--tavily-api-key`) funnel into configuration resolution. The only remaining
subcommand is `config`, which retains its nested `show`, `set`, `unset`, and `init` actions for managing
stored preferences.

`main()` is the top-level dispatcher:
1. Parse arguments and return immediately if `--version` is requested.
2. Perform a best-effort release check against PyPI (`notify_if_update_available()`).
3. If the `config` subcommand is selected, route to `handle_config()` without contacting the API.
4. Otherwise, call `resolve_runtime_config()` to assemble a `ResolvedConfig` from CLI options,
   environment variables, and stored JSON.
5. When `--prompt` is supplied the CLI executes a one-off agent run via `handle_agent()`; otherwise it
   launches `run_interactive_agent_shell()` for the full-screen interactive workflow.

## Transcript Handling
Both chat and agent modes support transcripts. When the user supplies `--transcript`:
- Chat mode writes each turn as a JSON object appended to the selected file. Relative paths default to
  `~/.config/deepseek-cli/transcripts/`.
- Agent mode resolves relative transcript paths inside the workspace so outputs stay co-located with
  project files. Each logged message stores the step index along with the raw OpenAI message payload.

## Chat Workflow (`deepseek_cli/chat.py`)
`ChatOptions` captures all user-configurable parameters, including the preferred streaming style, and
`run_chat()` powers the `@chat` shortcut inside the interactive shell. It builds the initial message set
(system + user) and calls
`client.chat.completions.create`:
- In streaming mode (`--no-stream` absent), responses are rendered via
  `stream_chat_response()` which supports the `plain`, `markdown`, and `rich`
  presentation modes configured via CLI, environment, or stored config.
- Otherwise it issues a standard chat completion and prints the full reply.

Interactive chats loop, prompting the user (`You ▸`) until EOF. Non-interactive or single-turn calls exit
after the first assistant reply. Every turn is optionally logged to the transcript file.

## Completion Workflow (`deepseek_cli/completions.py`)
`CompletionOptions` mirrors the classic Codex parameters (prompt, suffix, max tokens, stop sequences,
sampling controls, streaming). `run_completion()` backs the `@complete` command: it gathers input from
the provided text (or legacy file/stdin routes), then issues `client.completions.create`. Streaming flows through
`stream_completion_response()` so completions share the same renderer families as chat. Non-streaming
requests print a Rich table summarising each returned choice. Optional `--output` persistence writes the
primary completion to disk.

## Embedding Workflow (`deepseek_cli/embeddings.py`)
`EmbeddingOptions` aggregates positional snippets, optional file inputs, or stdin to build a list of
texts. `run_embeddings()` powers the `@embed` command, calling `client.embeddings.create`, then rendering
the output as a Rich table
(previewing the first few vector components), JSON, or a plain numeric dump depending on `--format`.
When `--output` is supplied, the function persists a JSON payload containing the model name, inputs, and
vectors for reuse.

## Model Listing (`deepseek_cli/models.py`)
`ModelListOptions` carries optional filtering, JSON output, and row limits. `list_models()` (surfaced via
the `@models` command) wraps `client.models.list()` and produces a Rich table of IDs, owners, and creation
timestamps unless the user requests raw JSON for automation pipelines.

## Agent Workflow Overview (`deepseek_cli/agent.py`)
### Input Preparation
`handle_agent()` (non-interactive) and `run_interactive_agent_shell()` (interactive) both build an
`AgentOptions` instance containing:
- Model, system prompt, initial user prompt, and any follow-up prompts.
- Workspace path, read-only flag, and `allow_global_access` flag (now enabled by default so the agent can operate anywhere on disk).
- Max reasoning steps, verbosity toggle, and transcript path.

Agent invocations append two hidden follow-ups (`AUTO_TEST_FOLLOW_UP` and `AUTO_BUG_FOLLOW_UP`) so the
model automatically plans to run tests and regression checks before finishing.

### Interactive Shell
Running `deepseek` with no arguments enters an interactive agent shell. The shell maintains an
`InteractiveSessionState` that tracks mutable session settings (workspace, model, system prompt,
max steps, read-only, transcript destination, etc.). Lines beginning with `:`, `/`, or `@` are treated as
control commands (`:workspace`, `:model`, `:system`, `:max-steps`, `:read-only`, `:global`, `:transcript`,
`:reset`, `:help`, `:quit`). Additional shortcuts expose API primitives directly: `@chat` issues a single
chat completion, `@complete` performs a text/code completion, `@embed` generates embeddings, and
`@models` lists available models. Regular prompts are sent to the agent loop.

### Tool Execution
`ToolExecutor` exposes the functions the language model can call during reasoning. Each tool validates
paths against the workspace root unless `allow_global_access` is enabled (the default behaviour):
- `list_dir(path=".", recursive=False)`: hierarchical directory listings capped by `MAX_LIST_DEPTH`.
- `read_file(path, offset=0, limit=None)`: returns file contents or slices.
- `write_file(path, content, create_parents=False)`: writes text unless the session is read-only.
- `move_path(source, destination, overwrite=False, create_parents=False)`: move or rename files/directories with optional parent creation and overwrite.
- `stat_path(path=".")`: JSON metadata describing file size, type, and timestamps.
- `search_text(pattern, path=".", case_sensitive=True, max_results=200)`: wraps ripgrep (`rg`) when
  available, otherwise falls back to `grep`.
- `apply_patch(patch)`: applies unified diffs using `patch` or `git apply`, with guards against paths
  escaping the workspace.
- `run_shell(command, timeout=120)`: executes `/bin/bash -lc` inside the workspace and returns stdout,
  stderr, and exit status. Long outputs are truncated to `MAX_TOOL_RESULT_CHARS`.
- `python_repl(code, timeout=120)`: executes inline Python snippets via the active interpreter and
  captures stdout/stderr for iterative experimentation.
- `http_request(url, method="GET", headers=None, body=None, timeout=30)`: performs simple HTTP requests
  so the agent can retrieve remote references or API payloads during reasoning.
- `download_file(url, destination, overwrite=False, create_parents=False, mode="binary", timeout=120)`: fetches remote resources and stores them on disk, supporting binary or text decoding.

### Agent Loop
`agent_loop()` orchestrates the conversation with the API:
1. Build the initial message list (`system`, `user`, and follow-up prompts) via `build_messages()`.
2. Register tool schemas (`tool_schemas()`) so the model may choose function calls.
3. For each step up to `max_steps`:
   - Request a chat completion with `tool_choice="auto"`.
   - If the assistant returns tool calls, decode arguments (JSON), run each tool through the
     `ToolExecutor`, append the tool results to the message stack, and log them to transcripts. Tool
     invocations that do not execute a shell command surface as Rich progress spinners with elapsed time,
     mirroring modern coding CLIs.
   - If the assistant returns plain text, echo it to stdout and terminate successfully.
4. If `max_steps` is reached without a final reply, print guidance suggesting a rerun or transcript
   inspection.

Verbose mode (`--quiet` absent) prints debug information to stderr describing the active step, last
message size, and tool invocations to help users understand the agent’s reasoning process.

## Configuration Subcommands (`handle_config`)
`deepseek config show` reads the stored JSON and optionally redacts the API key before printing.
`config set` and `config unset` update individual fields via `update_config()` / `save_config()`.
`config init` prompts for a key, writes it to the config file, and gracefully handles EOF/permission
errors.

## Update Notifications
At startup (except for `deepseek config`), `notify_if_update_available()` fetches metadata from PyPI
(`https://pypi.org/pypi/deepseek-agent/json`), parses known releases with `packaging.version.Version`,
and compares the highest version against the bundled `__version__`. When a newer release exists, the CLI
prints upgrade instructions to stderr but continues executing.

## Error Handling and Safeguards
- Missing API keys abort early with a descriptive error.
- Workspace paths are resolved and validated before invoking the agent. Non-existent directories cause a
  non-zero exit.
- Tool operations include explicit guardrails for read-only sessions and workspace escape attempts.
- Long tool outputs are truncated to keep conversation context manageable.
- Shell commands include timeouts to avoid runaway processes.

## Distribution Notes
- `Formula/` and `build/` house packaging artifacts (e.g., Homebrew formulas) but do not affect runtime.
- The CLI targets Python ≥ 3.9 and carries an MIT license (`LICENSE`).

Together, these components implement a developer-focused interface over the DeepSeek models that
balances ease of use with the guardrails required for safe repository automation.
