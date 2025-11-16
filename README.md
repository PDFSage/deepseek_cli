
• DeepSeek CLI Stack

  - Entry and configuration flow stay entirely inside this repo: both console
    scripts land in deepseek_cli.cli:main, which resolves config via CLI
    flags, env vars, then JSON in ~/.config/deepseek-cli/config.json, so the
    orchestrator always runs locally before it ever talks to DeepSeek’s API
    (DOCUMENTATION.md:8-51).
  - The default experience is a PromptToolkit/Rich shell that exposes /, @,
    and : shortcuts; every run launches a planner, then iterates plan steps
    with automated test/flow-attempt follow-ups, Tavily-assisted research,
    and @global workspace overrides, mirroring “Claude/Codex-style” status
    breadcrumbs the whole time (README.md:94-118).
  - LLM actions are defined up front as OpenAI-style JSON tool schemas (list/
    read/write/move/stat/search/apply_patch/run_shell/python_repl/http_request/
    download_file, etc.) and implemented by a single ToolExecutor, so adding a
    new executor module is as simple as registering another Python method and
    schema entry (deepseek_cli/agent.py:856-1041, DOCUMENTATION.md:125-144).
  - agent_loop() keeps the Worker conversation state, repeatedly calling
    client.chat.completions.create(..., tool_choice="auto"), decoding tool
    arguments, executing them locally, truncating oversized tool output,
    persisting transcripts, and stopping only when the LLM emits a final
    assistant reply or the max-step guard trips (deepseek_cli/agent.py:1138-
    1274, DOCUMENTATION.md:145-164).
  - State updates are first-class: transcript files log every OpenAI
    payload with the step index, stderr streams per-step ▌ reasoning, stdout
    BreadcrumbLogger reports planner refreshes / worker steps / tool start-
    stop, and users can replay the JSONL traces later (README.md:120-129,
    DOCUMENTATION.md:52-67, deepseek_cli/agent.py:1148-1252).

  Cross-CLI Differences

  - Control plane location: DeepSeek’s Python process is the engine—it owns
    tool execution inside the user’s repo and only asks the remote LLM for
    the next JSON-coded step. Claude Code runs Anthropic’s planner/worker in
    the cloud and mirrors a remote container (with “Computer Use” actions like
    bash, editor operations, UI clicks) back through its client; Codex CLI
    (the harness we’re in now) gives the remote LLM direct access to sandboxed
    primitives such as shell, apply_patch, plan, etc., so there is no local
    orchestrator beyond the thin relay; Gemini CLI (Gemini Code Assist) pipes
    prompts to Google’s hosted agent service and typically executes file edits/
    tests in managed Cloud Workstations or a gcloud-connected repo.
  - Action schema granularity: DeepSeek uses strictly-defined OpenAI function
    specs that map 1:1 to Python methods and returns UTF-8 strings; Claude Code
    encodes Anthropic tool_use events that can emit both structured payloads
    (e.g., JSON for write_file) and binary screen diffs; Codex CLI calls each
    tool with bespoke RPC envelopes ({"command": ["bash","-lc", ...]} or lark-
    patched diffs) and enforces sandbox/approval rules instead of JSON schemas;
    Gemini CLI routes higher-level tasks (code_assist.apply_patch, test.run,
    dependency.update) through Google Cloud APIs that stream structured “event”
    JSON rather than simple UTF-8 blobs.
  - Execution & trust boundaries: DeepSeek runs every file system write/test
    under the user’s UID, honoring --read-only/--no-global flags to fence
    the workspace; Claude Code executes inside Anthropic-managed sandboxes,
    uploading/downloading diffs rather than touching your disk directly; Codex
    CLI executes commands in a controllable sandbox on your machine (with
    explicit approval modes and optional escalation) so untrusted steps can
    be denied; Gemini CLI often routes edits through Cloud-hosted mirrors, so
    credentials and source code sit inside Google’s tenancy during a session.
  - State, telemetry, and user feedback: DeepSeek exposes JSONL transcripts,
    verbose ▌ traces, and breadcrumb feeds the user can version-control; Claude
    Code emphasizes a live TUI with panes for plan/command/output but does not
    yet export the full JSON stream; Codex CLI shows tool calls implicitly via
    our conversation but retains approval logs and sandbox reports rather than
    producing end-user transcripts; Gemini CLI integrates with Google Cloud
    logging/issue tracking, surfacing status in Cloud consoles or IDE panes
    rather than local JSON files.
  - Extensibility/offline posture: DeepSeek is MIT-licensed Python—new tools,
    new planner logic, or alternative vendors can be dropped in by editing
    this repo; Claude Code, Codex CLI, and Gemini CLI are closed, vendor-hosted
    stacks—you can’t extend their action schemas beyond what the providers
    expose, and offline use is impossible because each requires authenticated
    access to its respective cloud.

  Differences Among Claude Code, Codex CLI, Gemini CLI

  - Claude Code focuses on an Anthropic-run remote workstation with “Computer
    Use” super-actions (cursor control, OS windowing, browser fetches) layered
    on top of simpler bash/edit tools, so it can operate entire GUIs but at the
    cost of higher latency and no direct local repo access.
  - Codex CLI (this harness) is intentionally minimal: the LLM itself drives
    plan creation, tool sequencing, and approvals; actions are limited to a
    curated set (shell, apply_patch, file reads via MCP, optional plan tool)
    to preserve determinism, and the environment enforces sandbox/approval
    policies instead of project-configured guardrails.
  - Gemini CLI (Gemini Code Assist / gcloud integration) leans into Google
    Cloud infrastructure: tasks are routed through project/workspace metadata,
    commands can spin up Cloud builds/tests, and the CLI exchanges structured
    “operation” objects with the Gemini backend so that actions appear
    alongside other Google Cloud operations (identity, IAM, audit logging).

  Natural next steps: if you need parity docs for stakeholders, consider
  dropping this comparison into DOCUMENTATION.md or publishing a README
  appendix so contributors understand how DeepSeek CLI’s open-source executor
  differs from the closed vendor CLIs.