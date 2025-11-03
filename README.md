# DeepSeek CLI

> This CLI is open source at https://github.com/PDFSage/deepseek_cli – collaborators and maintainers are welcome! Submit ideas, issues, or pull requests to help the project grow.

https://pypi.org/project/deepseek-agent/

Developer-focused command line tools for working with DeepSeek models. The CLI
packages both an interactive chat shell and an agentic coding assistant with
repository-aware tooling, plus configuration helpers and transcript logging.

## Features
- Unified agent shell orchestrates tool-aware coding sessions with the DeepSeek
  API, optional read-only mode, transcripts, and workspace controls. It launches
  by default when you run `deepseek`.
- Inline shortcuts (`@chat`, `@complete`, `@embed`, `@models`) surface chat,
  completion, embedding, and model-list primitives without leaving the agent
  shell.
- Auto-detects likely project test commands and reminds the agent to run them,
  keeping changes validated against the repo's real workflows.
- Config mode (`deepseek config`) manages stored defaults while respecting
  environment variable overrides.
- Interactive mode now launches a colourful Rich-powered shell that surfaces
  `/` and `@` command shortcuts, streams progress spinners for non-shell tool
  calls, and lets you update the stored API key without leaving the session.
- Built-in Tavily web search support supplements the agent with live research,
  using a bundled developer key that you can override per session or in config.
- Ships as a Python package with an executable entry point and Homebrew formula
  for distribution flexibility.

## Requirements
- Python 3.9 or newer.
- A DeepSeek API key exported as `DEEPSEEK_API_KEY` or stored via
  `deepseek config`.
- `pip` 23+ is recommended. Create a virtual environment for isolated installs.

## Installation

### From PyPI (recommended once released)
```bash
python -m pip install --upgrade pip
python -m pip install deepseek-agent
```

To update later, run `python -m pip install --upgrade deepseek-agent`.

### From GitHub
Install the latest commit directly from GitHub:
```bash
python -m pip install "git+https://github.com/PDFSage/deepseek_cli.git@main"
```
Specify a tag (for example `v0.2.0`) to pin a release:
```bash
python -m pip install "git+https://github.com/PDFSage/deepseek_cli.git@v0.2.0"
```

### From a local clone
```bash
git clone https://github.com/PDFSage/deepseek_cli.git
cd deepseek_cli
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\\Scripts\\activate
python -m pip install --upgrade pip
python -m pip install -e .  # or `python -m pip install .` for a standard install
```

The editable install (`-e`) keeps the CLI synced with local source changes while
developing.

## Configuration
The CLI resolves settings in the following order:
1. Command line flags (`--api-key`, `--base-url`, `--model`, etc.).
2. Environment variables: `DEEPSEEK_API_KEY`, `DEEPSEEK_BASE_URL`,
   `DEEPSEEK_MODEL`, `DEEPSEEK_SYSTEM_PROMPT`, `DEEPSEEK_CHAT_MODEL`,
   `DEEPSEEK_COMPLETION_MODEL`, `DEEPSEEK_EMBEDDING_MODEL`,
   `DEEPSEEK_CHAT_STREAM_STYLE`, `TAVILY_API_KEY`.
3. Stored configuration file at `~/.config/deepseek-cli/config.json`.

Helpful commands:
```bash
deepseek config init        # Guided prompt to store your API key
deepseek config show        # Display the current configuration (API key redacted)
deepseek config show --raw  # Show the API key in plain text
deepseek config set model deepseek-reasoner  # Update an individual field
deepseek config set completion_model deepseek-coder
deepseek config set chat_stream_style markdown
deepseek config set tavily_api_key tvly-live-xxxxxxx
deepseek config unset model
```

If the config directory is unwritable, fall back to environment variables.

## Usage

### Interactive agent (default)
Running `deepseek` with no arguments launches the interactive coding agent,
now presented through a colourful Rich-powered shell. A command palette is
displayed on start so you can see the available `/`, `@`, or `:` shortcuts at a
glance (for example `@workspace`, `@model`, `@read-only`, `@transcript`,
`@help`, `@api`, and `@tavily`). Exit with `@quit` or `Ctrl+C`. Each request runs as soon
as you press Enter—include follow-up guidance in your initial prompt. The
assistant appends internal follow-ups that run automated tests and regression
checks until they succeed or a clear justification is provided.
Use `@global on` when you need to edit files outside the active workspace.
During execution the shell streams the agent's thought process (`▌` lines) while
non-shell tools render as bright spinners with elapsed time, mirroring modern
coding CLIs. Tool outputs
are still truncated if they exceed the configured limits; narrow the scope or
request additional reads for more detail.

If no API key is detected, the CLI now prompts you to paste one on launch and
safely stores it. You can update the stored key at any time with `@api` or via
`deepseek config set api_key`.

### Verify installation
```bash
deepseek --version
# or use the legacy alias if preferred
deepseek-cli --version
```

Get help for the CLI:
```bash
deepseek --help
```

### One-off agent run
Invoke a single task without entering the interactive shell:
```bash
deepseek --prompt "Refactor the HTTP client" \
  --workspace ~/code/project --max-steps 30 \
  --transcript transcript.jsonl --no-global
```
- Combine `--follow-up "Also add tests"` to append extra instructions.
- Pass `--read-only` to prevent write operations and `--quiet` to suppress
  verbose progress logs.

### Inline shortcuts
Inside the interactive shell, use the new `@` commands for quick tasks:
- `@chat "Summarise the last commit"` streams a single chat response using the
  current session model and system prompt.
- `@complete "def fib(n):"` issues a Codex-style completion (results print
  immediately in the shell).
- `@embed "vectorize me" "and me"` generates embeddings for one or more snippets.
- `@models --filter coder --limit 5` lists available models, with `--json` for raw
  output.
All other `/`/`:` controls (`@workspace`, `@model`, `@transcript`, etc.) continue
to work as before.

### Transcripts and workspaces
- Relative transcript paths under agent mode are resolved within the selected
  workspace.
- Chat transcripts default to `~/.config/deepseek-cli/transcripts/` when a file
  name (not path) is supplied.

### Legacy shim
Running `python deepseek_agentic_cli.py` prints a compatibility notice and
forwards the call to `deepseek --prompt …`, so existing automation keeps working.

## Publishing to PyPI
1. Update the version in `pyproject.toml` and commit your changes.
2. Remove old build artifacts:
   ```bash
   rm -rf build dist *.egg-info
   ```
3. Install packaging tooling:
   ```bash
   python -m pip install --upgrade build twine
   ```
4. Build the source and wheel distributions:
   ```bash
   python -m build
   ```
5. Verify the archives:
   ```bash
   python -m twine check dist/*
   ```
6. Upload to TestPyPI (optional but recommended):
   ```bash
   python -m twine upload --repository testpypi dist/*
   ```
7. Upload to PyPI:
   ```bash
   python -m twine upload dist/*
   ```

After publishing, users can install with `pip install deepseek-agent`.

## Development
- `python -m deepseek_cli --version` exercises the module entry point.
- `python -m deepseek_cli --help` displays the unified CLI options.
- `python -m deepseek_cli config --help` shows configuration helpers.
- Run `ruff`, `pytest`, or other tooling as required by your workflow.

Contributions welcome! Open issues or pull requests to extend functionality.
