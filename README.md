# local-transcriber

CLI that transcribes multi-participant audio recordings into a single,
speaker-labeled, time-stamped transcript. Designed to run as a one-shot
Docker container that you point at a recording directory, transcribe, and
exit.

The transcription model is never bundled. The CLI talks to any
OpenAI-compatible `/v1/audio/transcriptions` server — remote (hosted) or
local (your own vLLM, mlx-whisper-server, ollama-compatible shim, etc.).

## Recording format

A recording is a directory shaped like `example_recording/`:

```
my-session/
├── manifest.json
└── <Identity>_<shortid>/
    ├── metadata.json
    └── <Identity>_<shortid>.opus
```

### `manifest.json`

```json
{
  "session_id": "2026-05-11_19-26-38",
  "room_name": "...",
  "start_epoch": 1778520398.5128,
  "end_epoch":   1778520436.8609,
  "participants": ["Riccardo"],
  "format": "opus",
  "bitrate": 128000
}
```

`start_epoch` defines the global timeline zero. Everything in the merged
transcript is measured relative to it.

### Per-participant `metadata.json`

```json
{
  "identity": "Riccardo",
  "sample_rate": 48000,
  "channels": 1,
  "format": "opus",
  "start_epoch": 1778520398.5128,
  "audio_file": "Riccardo_f8ac3e.opus",
  "events": [
    { "type": "start_receiving", "epoch": 1778520398.6809 },
    { "type": "finalized",       "epoch": 1778520436.8606 }
  ]
}
```

`start_receiving` (when present) is used as the participant's true audio
start, so a participant who joined late shows up correctly aligned on the
global timeline.

Any audio format `ffmpeg` can read is supported; the example uses Opus.

## Output

Two files are written to the session directory (or to `--output-dir`):

### `transcript.txt`

```
00:00:00.17 Riccardo
Welcome to the session.

00:00:02.40 Riccardo
Roll for initiative.

00:00:02.55 Player1
I jump forward!
```

Segments are sorted by absolute start time; overlapping speech renders as
adjacent blocks with the same timestamp.

### `transcript.json`

Structured form of the same data, plus session metadata:

```json
{
  "session_id": "...",
  "room_name": "...",
  "language": "en",
  "model": "Systran/faster-whisper-large-v3",
  "start_epoch": 1778520398.5128,
  "end_epoch":   1778520436.8609,
  "segments": [
    {
      "speaker": "Riccardo",
      "start": 0.168,
      "end":   1.842,
      "start_absolute": "00:00:00.17",
      "text": "Welcome to the session."
    }
  ]
}
```

## Quick start (Docker)

```bash
docker run --rm \
  -e TRANSCRIBE_URL=http://host.docker.internal:8000 \
  -e TRANSCRIBE_MODEL=Systran/faster-whisper-large-v3 \
  -v "$PWD/recordings:/in" \
  kalelkenobi/local-transcriber:latest \
  /in/my-session
```

Process every session under a parent directory:

```bash
docker run --rm \
  -e TRANSCRIBE_URL=http://host.docker.internal:8000 \
  -e TRANSCRIBE_MODEL=Systran/faster-whisper-large-v3 \
  -v "$PWD/recordings:/in" \
  kalelkenobi/local-transcriber:latest \
  /in --recursive
```

## CLI

| Flag / arg               | Env var                    | Default                            |
|--------------------------|----------------------------|------------------------------------|
| `PATH` (positional)      |                            | _required_                         |
| `--recursive`, `-r`      |                            | off                                |
| `--output-dir`, `-o`     |                            | session dir                        |
| `--language`, `-l`       |                            | `en`                               |
| `--api-url`              | `TRANSCRIBE_URL`           | _required_                         |
| `--api-key`              | `TRANSCRIBE_API_KEY`       | _(none)_                           |
| `--model`                | `TRANSCRIBE_MODEL`         | _required_                         |
| `--vad-threshold`        | `TRANSCRIBE_VAD_THRESHOLD` | `0.5`                              |
| `--vad-min-speech-ms`    |                            | `250`                              |
| `--vad-min-silence-ms`   |                            | `500`                              |
| `--concurrency`          |                            | `4`                                |
| `--timeout`              |                            | `300.0`                            |
| `--log-level`            | `LOG_LEVEL`                | `INFO`                             |
| `--silero-vad-path`      | `SILERO_VAD_PATH`          | `/app/models/silero_vad.onnx`      |

See `docs/cli.md` for the full reference and examples.

## Choosing an ASR backend

The CLI sends each detected speech segment as a WAV to
`POST {api-url}/v1/audio/transcriptions` (multipart `file`, form `model`,
form `language`). Anything that implements that endpoint works.

Examples:

- **vLLM** — `--api-url http://vllm.example.com --model openai/whisper-large-v3`
- **mlx-whisper-server** (Apple Silicon) — `--api-url http://localhost:8000 --model whisper-large-v3-mlx`
- **Hosted OpenAI-compatible** — `--api-url https://api.example.com --api-key sk-... --model whisper-1`

## Development

```bash
python3.13 -m venv .venv
.venv/bin/pip install -r requirements.txt -r requirements-dev.txt
.venv/bin/pip install -e .

# Offline unit tests
.venv/bin/python -m unittest discover -s tests -t .

# End-to-end (requires Docker, Podman, or Apple container; auto-detects)
.venv/bin/pytest tests_e2e -q
```

See `AGENTS.md` for the contribution workflow and `docs/architecture.md`
for how the pipeline is wired internally.
