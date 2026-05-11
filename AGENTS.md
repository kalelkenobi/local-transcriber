# AGENTS.md

## Scope

- Keep this file short. If prose docs conflict with code or config, trust
  `pyproject.toml`, `ContainerFile`, and `local_transcriber/*.py`.

## Repo Facts

- Python packaging requires `>=3.13` in `pyproject.toml`. Container uses
  `python:3.13-slim-bookworm`.
- Single dependency on `ffmpeg` in the container image (audio decoding).
- Tests use stdlib `unittest`. Run with
  `.venv/bin/python -m unittest discover -s tests -t .` from this directory.
- End-to-end tests live in `tests_e2e/` and use `pytest` against a built
  container plus a mock ASR server. Run with `.venv/bin/pytest tests_e2e -q`.

## Code Map

- `local_transcriber/cli.py` — `click`-based entry point. Resolves flags +
  env vars, builds VAD + backend, drives `transcribe_session()` over each
  discovered session, prints summary, sets exit code.
- `local_transcriber/session.py` — manifest + per-participant metadata
  loaders, timeline offset math, recursive session discovery.
- `local_transcriber/audio.py` — `ffmpeg`-subprocess decode to 16 kHz mono
  int16 PCM, PCM slicing into WAV bytes.
- `local_transcriber/vad.py` — Silero VAD v5 ONNX wrapper with a clean
  state machine that emits speech segments.
- `local_transcriber/backend.py` — `RemoteASRBackend`: OpenAI-compatible
  `/v1/audio/transcriptions` HTTP client with retries on 5xx + transport
  errors.
- `local_transcriber/output.py` — `transcript.json` + `transcript.txt`
  writers, `format_timestamp` (HH:MM:SS.ss).
- `local_transcriber/pipeline.py` — orchestrator: decode → VAD → slice →
  backend (bounded concurrency) → merge on the global timeline → write
  outputs. Returns `SessionResult`.

## Runtime Gotchas

- `ffmpeg` must be on `$PATH`. The container ships it; local dev needs it
  installed (`brew install ffmpeg`, `apt-get install ffmpeg`, etc.).
- `TRANSCRIBE_URL` and `TRANSCRIBE_MODEL` are required (or `--api-url` and
  `--model`). Missing → exit 1.
- The container is short-lived: it transcribes the given path and exits.
  Mount the recordings dir at any read-write path and pass it as the CLI
  argument.
- Silero VAD ONNX path is configurable via `SILERO_VAD_PATH` (default
  `/app/models/silero_vad.onnx`). Dev fallback: `models/silero_vad.onnx`
  next to the package.
- Per-segment failures during ASR are logged and skipped — they do not
  fail the whole session. Pre-ASR failures (manifest parse, decode,
  VAD load) do fail the session.

## CLI Surface

- `local-transcriber <PATH>` — transcribe one session (must contain
  `manifest.json`).
- `local-transcriber <PATH> --recursive` — transcribe every immediate
  child of `<PATH>` that contains a `manifest.json`.
- Exit codes: `0` all sessions succeeded, `1` config / discovery error,
  `2` at least one session failed.

## Implementation Plan Convention

For any non-trivial change (multi-file, new feature, significant
refactor), create a plan document **before** writing code. **You can
write this even if you are in planning mode.** Trivial fixes
(single-line, typo, simple config) do not need a plan.

### Plan File

- **Location:** `.opencode/plans/PLAN-<short-slug>.md`
- **Git-ignored:** Yes (`.opencode/` is already in `.gitignore`).

### Plan File Format

```markdown
# Plan: <brief title>
**Created:** YYYY-MM-DD
**Status:** in-progress

## Overview
<what is being done and why — 2-4 sentences>

## Changes
Detailed file-by-file breakdown of what needs to change and how.

### `local_transcriber/<FileA>.py`
- **Lines L1-L40:** <what to change and why>
- **Line L15:** Change `foo` to `bar` to fix <reason>
- **After line L30:** Add new function `do_x()` that <purpose>

### `local_transcriber/<FileB>.py`
- **Lines L50-L80:** <description of change>
- ...

## Todo
- [ ] Task 1 — maps to items in Changes above
- [/] Task 2 (in progress)
- [x] Task 3 (done)
```

### During Implementation

- The `todowrite` tool is the **canonical** todo tracker during an active
  session.
- Sync the plan file's checkbox list to match `todowrite` state at session
  boundaries (before ending or when the agent detects potential
  interruption).
- On starting a new session, scan `.opencode/plans/` for any `in-progress`
  plans and resume from the last synced state.
- The `Changes` section must be detailed enough (files, line numbers,
  intent) that even a smaller coding model can follow it with minimal
  mistakes.

### Plan Lifecycle

- **Created** → `Status: in-progress` with initial todo.
- **During work** → Update checkboxes as tasks progress.
- **Completed** → Delete the plan file after the work is verified
  (unit + e2e tests pass).
- **Abandoned** → Set `Status: abandoned` with a brief note explaining
  why; keep the file for future reference.

## Working Rules

- Every code change must include a documentation change in the same task.
- Default documentation destination is `docs/`, not `AGENTS.md` and not
  code comments.
- Use `AGENTS.md` only for agent workflow, repo commands, and durable repo
  gotchas.
- If you change CLI flags, env vars, install or run steps, or
  container/runtime behavior, update `docs/cli.md` and `README.md` in the
  same change.
- **All READMEs are documentation surfaces**: `README.md` (project root)
  and `tests_e2e/README.md` must be updated in the same change when the
  corresponding area is modified. Treat them as part of the docs
  requirement — not optional extras.
- If you change a public class signature, function signature, or
  constructor kwarg in `local_transcriber/*.py`, update the matching test
  doubles in `tests/` and run
  `.venv/bin/python -m unittest discover -s tests -t .` in the same task.
  New env vars or CLI flags should also gain coverage in `tests/test_cli.py`.
- `pytest` is allowed for `tests_e2e/` only. The offline `tests/` suite
  continues to use stdlib `unittest`. Do not invent `ruff`, `mypy`, or CI
  commands. If you add a real verification command, document the exact
  command here and in `docs/`.
- Every completed task must end with the following ordered sequence.
  **Do not reorder these steps.** Do not commit before the `/review`
  step has been run and its findings addressed.
  1. **Run the full test suite** (unit + e2e). All tests must pass.
  2. **Bump the `version`** in `pyproject.toml` using semver (patch for
     fixes, minor for features, major for breaking changes).
  3. **Invoke the opencode `/review` agent** for code review. The
     `/review` agent evaluates each diff against:
     (1) **OWASP compliance** — no injection, path traversal, secrets
     leakage, or missing input validation;
     (2) **Performance** — no gratuitous I/O, efficient data handling,
     proper async usage, and awareness of OOM risk when processing large
     files in memory (stream or chunk instead);
     (3) **Clarity** — intention-revealing names, minimal cyclomatic
     complexity, no clever tricks;
     (4) **Quality** — robust error handling, edge-case coverage, no dead
     code, adequate test assertions;
     (5) **Test coverage** — new or changed logic must have corresponding
     unit tests in `tests/` and, where applicable, e2e coverage in
     `tests_e2e/`.
  4. **Address every blocking finding** from the `/review` agent. If the
     fixes change behavior, re-run step 1 (tests) before continuing. The
     version bumped in step 2 stays as-is unless review-driven changes
     alter the semver classification.
  5. **Commit** all changes with a descriptive message that references the
     plan slug and the version.
  6. **Tag** `v{version}`.
  7. **Push** the commit + tag to remote. The tag push triggers CI to
     build and publish the container image.
- Exception: if a task only changes documentation (`docs/`, `README.md`,
  `AGENTS.md`) or local-only files, skip the test suite run, version
  bump, `/review` invocation, and remote push. Just commit locally.
- Keep `README.md` up to date with the current project state. If a task
  changes the CLI surface, output format, configuration, or project
  capabilities, update `README.md` in the same change.

## Verification

- Always use the local virtual environment for Python commands: prefix
  with `.venv/bin/` (e.g. `.venv/bin/python`, `.venv/bin/pytest`). Do not
  use the system Python.
- Run the offline unit test suite with
  `.venv/bin/python -m unittest discover -s tests -t .`.
- Run the end-to-end suite with `.venv/bin/pytest tests_e2e -q`. Requires
  Apple `container` ≥ 0.12.0, Docker, or Podman on the host; auto-skips if
  none are available. Pin a backend with
  `E2E_RUNTIME=container|docker|podman`. See `tests_e2e/README.md`.
- Manually sweep stranded e2e artifacts with `scripts/e2e_cleanup.sh`
  (defaults to the `lt-e2e-` name prefix; honours `E2E_RUNTIME` and falls
  back to the same auto-detection order).
