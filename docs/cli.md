# CLI Reference

The container's entrypoint is the `local-transcriber` CLI. It accepts a
single positional `PATH` (a session directory or a parent dir of session
directories) and a set of flags described below.

## Positional argument

| Argument | Description |
|----------|-------------|
| `PATH`   | Directory to process. Either a session directory containing `manifest.json`, or — with `--recursive` — a parent dir whose immediate children are session directories. |

## Flags

| Flag                    | Short | Env var                    | Default                           | Description |
|-------------------------|-------|----------------------------|-----------------------------------|-------------|
| `--recursive`           | `-r`  |                            | off                               | Scan immediate subdirs of `PATH` for sessions. |
| `--output-dir`          | `-o`  |                            | session dir                       | Directory for transcripts. With multiple sessions, outputs are nested under `<output-dir>/<session-name>/`. |
| `--language`            | `-l`  |                            | `en`                              | Language code passed to the ASR server. |
| `--api-url`             |       | `TRANSCRIBE_URL`           | _required_                        | Base URL of the OpenAI-compatible ASR server. The CLI appends `/v1/audio/transcriptions`. |
| `--api-key`             |       | `TRANSCRIBE_API_KEY`       | _(none)_                          | Bearer token. Omit for unauthenticated local servers. |
| `--model`               |       | `TRANSCRIBE_MODEL`         | _required_                        | Model name passed to the server in form data. |
| `--vad-threshold`       |       | `TRANSCRIBE_VAD_THRESHOLD` | `0.5`                             | Speech probability threshold (0.0–1.0). Lower = more sensitive. |
| `--vad-min-speech-ms`   |       |                            | `250`                             | Minimum continuous speech duration to open a segment. |
| `--vad-min-silence-ms`  |       |                            | `500`                             | Minimum continuous silence duration to close a segment. |
| `--concurrency`         |       |                            | `4`                               | Max in-flight ASR requests per session. |
| `--timeout`             |       |                            | `300.0`                           | Per-request HTTP timeout in seconds. |
| `--log-level`           |       | `LOG_LEVEL`                | `INFO`                            | Python logging level. |
| `--silero-vad-path`     |       | `SILERO_VAD_PATH`          | `/app/models/silero_vad.onnx`     | Path to Silero VAD ONNX model. Dev fallback: repo-local `models/silero_vad.onnx`. |

## Exit Codes

- `0` — every discovered session succeeded.
- `1` — configuration error or no sessions found.
- `2` — at least one session failed.

## Examples

### One session against a local vLLM

```bash
local-transcriber ./recordings/2026-05-11_19-26-38 \
  --api-url http://localhost:8000 \
  --model openai/whisper-large-v3 \
  --language en
```

### Every session under a recordings folder

```bash
local-transcriber ./recordings --recursive \
  --api-url http://localhost:8000 \
  --model openai/whisper-large-v3
```

### Apple Silicon, mlx-whisper-server

```bash
local-transcriber ./recordings/session-1 \
  --api-url http://localhost:8000 \
  --model whisper-large-v3-mlx \
  --concurrency 2
```

### Hosted OpenAI-compatible server with auth

```bash
local-transcriber ./recordings/session-1 \
  --api-url https://asr.example.com \
  --api-key sk-... \
  --model whisper-1
```

### Containerized run (Apple container)

```bash
scripts/run.sh --url http://192.168.64.1:8000 \
  --model whisper-large-v3-mlx \
  --recursive "$PWD/recordings"
```

The `scripts/run.sh` wrapper launches the container via Apple's native `container` CLI. Accepts the same flags as
`docker run` examples: `--url`, `--model`, `--api-key`, `--recursive`,
`--tag`, `--language`, `--log-level`, `--vad-threshold`, `--timeout`,
`--concurrency`.

### Tighter VAD for quiet recordings

```bash
local-transcriber ./recordings/quiet \
  --api-url http://localhost:8000 \
  --model openai/whisper-large-v3 \
  --vad-threshold 0.3 \
  --vad-min-speech-ms 150
```

## Output

For each session, two files are written:

- `transcript.json` — structured JSON. Schema documented in
  `docs/architecture.md` and shown in `README.md`.
- `transcript.txt` — human-readable plain text:
  ```
  HH:MM:SS.ss <Speaker>
  <text>

  HH:MM:SS.ss <Speaker>
  <text>
  ```

Times are in seconds relative to `manifest.start_epoch`. Overlapping
speech from multiple speakers appears as adjacent blocks with the same
(or near-equal) timestamps.

## ASR server contract

Each detected speech segment becomes a `POST` to
`{api-url}/v1/audio/transcriptions` with:

- Multipart form field `file` — WAV bytes (`audio/wav`, 16 kHz mono int16).
- Form field `model` — string passed verbatim from `--model`.
- Form field `language` — string passed verbatim from `--language`.
- Optional `Authorization: Bearer <api-key>` header when `--api-key` is
  set.

The server is expected to respond with JSON containing a `text` field:

```json
{"text": "..."}
```

Any other top-level fields are ignored. HTTP 5xx and transport errors are
retried up to two times with exponential backoff (1s, 2s). HTTP 4xx is
non-recoverable and fails the segment (logged, skipped — the rest of the
session continues).
