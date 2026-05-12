# End-to-end tests

These tests build (or reuse) the `local-transcriber` container image and
run it against `tests_e2e/fixtures/` with a mock OpenAI-compatible ASR
server running on the host. They verify the full pipeline end-to-end:

- ffmpeg decode of the bundled `.opus` file,
- Silero VAD segmenting,
- per-segment HTTP requests to the mock server,
- output `transcript.json` + `transcript.txt` with correct shape,
- recursive mode discovers each session,
- exit-code semantics (1 when no sessions, 2 when any session fails).

## Requirements

- One of:
  - **Docker** (Linux 20.10+, or Docker Desktop on Mac/Windows)
  - **Podman** (3.0+)
  - **Apple `container`** (≥ 0.12.0)
- `python -m pytest` with the project's dev deps installed
  (`pip install -r requirements-dev.txt -e .`).

The suite auto-skips when no container runtime is present.

## Running

```bash
.venv/bin/pytest tests_e2e -q
```

Pin a specific runtime:

```bash
E2E_RUNTIME=docker .venv/bin/pytest tests_e2e -q
```

Reuse a pre-built image instead of rebuilding for every run:

```bash
docker build -f ContainerFile -t local-transcriber:dev .
LOCAL_TRANSCRIBER_IMAGE=local-transcriber:dev .venv/bin/pytest tests_e2e -q
```

## Cleanup

The test suite automatically cleans up after itself during session
teardown:

- `conftest.py` removes any `lt-e2e-` containers and the built
  `local-transcriber:e2e` image (unless `LOCAL_TRANSCRIBER_IMAGE` was
  provided, in which case the prebuilt image is left alone).
- The CI workflow (`ci.yml`) runs an explicit cleanup step (`if: always()`)
  that force-removes `lt-e2e-*` containers and the `local-transcriber:test`
  image built by the e2e job.

To sweep any stranded artifacts manually (e.g. after a test interruption):

```bash
scripts/e2e_cleanup.sh
```

The cleanup script honors `E2E_RUNTIME` and matches both the `lt-e2e-`
prefix (override with `E2E_PREFIX=...`) and the standard
`local-transcriber:(e2e|test|dev)` image tags.

## What the mock ASR server returns

`tests_e2e/mock_asr/server.py` implements
`POST /v1/audio/transcriptions` and responds with
`{"text": "mock-<N>"}` where `N` is an in-process call counter. The
e2e assertions don't depend on the specific text — only on the shape
of the response and the merged transcript.

## Network notes

Containers reach the host via `host.docker.internal`. For Docker and
Podman on Linux the conftest passes
`--add-host=host.docker.internal:host-gateway`, which is a no-op on
Docker Desktop where the name already resolves. Apple `container`
support for this hostname is best-effort — file a ticket if you hit
issues there.
