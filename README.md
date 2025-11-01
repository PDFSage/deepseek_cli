# DeepSeek CLI

> This CLI is open source at https://github.com/PDFSage/deepseek_cli – collaborators and maintainers are welcome! Submit ideas, issues, or pull requests to help the project grow.

https://pypi.org/project/deepseek-agent/

Developer-focused command line tools for working with DeepSeek models. The CLI
packages both an interactive chat shell and an agentic coding assistant with
repository-aware tooling, plus configuration helpers and transcript logging.

## Features
- Agent mode (`deepseek agent`) orchestrates tool-aware coding sessions with the
  DeepSeek API, optional read-only mode, transcripts, and workspace controls.
- Auto-detects likely project test commands and reminds the agent to run them,
  keeping changes validated against the repo's real workflows.
- Chat mode (`deepseek chat`) supports single-response or interactive
  conversations with streaming output and transcript capture.
- Completions mode (`deepseek completions`) mirrors Codex-style code/text
  completions with optional streaming renderers and file output.
- Embeddings mode (`deepseek embeddings`) batches text snippets and renders
  vectors as JSON, tabular previews, or plain values for quick inspection.
- Models view (`deepseek models`) enumerates the available DeepSeek API models
  with rich filtering or JSON export for automation.
- Config mode (`deepseek config`) manages stored defaults while respecting
  environment variable overrides.
- Interactive mode now launches a colourful Rich-powered shell that surfaces
  `/` and `@` command shortcuts, streams the agent's thought process, and lets
  you update the stored API key without leaving the session.
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
   `DEEPSEEK_CHAT_STREAM_STYLE`.
3. Stored configuration file at `~/.config/deepseek-cli/config.json`.

Helpful commands:
```bash
deepseek config init        # Guided prompt to store your API key
deepseek config show        # Display the current configuration (API key redacted)
deepseek config show --raw  # Show the API key in plain text
deepseek config set model deepseek-reasoner  # Update an individual field
deepseek config set completion_model deepseek-coder
deepseek config set chat_stream_style markdown
deepseek config unset model
```

If the config directory is unwritable, fall back to environment variables.

## Usage

### Interactive agent (default)
Running `deepseek` with no arguments launches the interactive coding agent,
now presented through a colourful Rich-powered shell. A command palette is
displayed on start so you can see the available `/`, `@`, or `:` shortcuts at a
glance (for example `@workspace`, `@model`, `@read-only`, `@transcript`,
`@help`, and `@api`). Exit with `@quit` or `Ctrl+C`. Each request runs as soon
as you press Enter—include follow-up guidance in your initial prompt. The
assistant appends internal follow-ups that run automated tests and regression
checks until they succeed or a clear justification is provided.
Use `@global on` when you need to edit files outside the active workspace.
During execution the shell streams the agent's thought process prefixed with
`▌`, so you can follow what tools are being invoked in real time. Tool outputs
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

Get help for any subcommand:
```bash
deepseek --help
```

### Chat mode
```bash
deepseek chat "Summarise the last commit"

deepseek chat --interactive --model deepseek-reasoner \
  --transcript ~/Desktop/session.jsonl
```
- `--no-stream` disables live token streaming.
- `--temperature`, `--top-p`, and `--max-tokens` mirror the OpenAI Chat
  Completions API.
- `--stream-style` swaps between `plain`, `markdown`, or `rich` streaming
  renderers (defaults to the stored configuration).
- Provide `--transcript` to log each exchange to JSONL for later review.

### Completions mode
```bash
deepseek completions "def fib(n):" --max-tokens 128

deepseek completions --input-file snippet.py --stream-style rich --output completion.txt
```
- Matches Codex-style behaviour with `--stream-style` support and optional
  file-backed prompts (`--input-file`).
- `--stop` may be supplied multiple times to register stop sequences.
- Pipe prompts via stdin when omitting the positional `prompt`.

### Embeddings mode
```bash
deepseek embeddings "vectorize me" "and me"

deepseek embeddings --input-file sentences.txt --format json --output embeddings.json
```
- Outputs tabular previews by default, with optional JSON (`--format json`) or
  whitespace-delimited vectors (`--format plain`).
- Supply repeated positional text values, a newline-delimited file, or stdin.
- `--show-dimensions` reveals the embedding length alongside previews.

### Model listing
```bash
deepseek models
deepseek models --filter coder --limit 5
deepseek models --json > models.json
```
- Displays a rich table of available models, mirroring the Gemini/OpenAI/Claude
  CLIs for quick discovery.

### Agent mode
```bash
deepseek agent "Refactor the HTTP client" \
  --workspace ~/code/project --max-steps 30 --transcript transcript.jsonl
```
- The agent uses repository-aware tools: list directories, read/write files,
  apply patches, run shell commands, and search text.
- Additional integrations now include a Python REPL tool and a lightweight
  HTTP client for fetching external resources during reasoning.
- Pass `--global` to permit edits outside the workspace root when you need
  system-wide changes.
- The agent automatically watches for regressions, replans on the fly, and
  fixes issues before finishing.
- Tool outputs are truncated to keep prompts within context limits; refine
  commands or request additional detail if necessary.
- Add `--follow-up "Also add tests"` for additional prompts.
- Use `--read-only` to prevent write operations and `--quiet` to suppress
  progress logs.

### Transcripts and workspaces
- Relative transcript paths under agent mode are resolved within the selected
  workspace.
- Chat transcripts default to `~/.config/deepseek-cli/transcripts/` when a file
  name (not path) is supplied.

### Legacy shim
Running `python deepseek_agentic_cli.py` prints a compatibility notice and
forwards the call to `deepseek agent`, so existing automation keeps working.

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
- `python -m deepseek_cli chat --help` shows chat-specific flags.
- `python -m deepseek_cli agent --help` lists agent options.
- Run `ruff`, `pytest`, or other tooling as required by your workflow.

Contributions welcome! Open issues or pull requests to extend functionality.
