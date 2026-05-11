# LiveKit Transcriber

Async transcription service that accepts audio files (WAV or ZIP session archives from [livekit-recorder](../README.md)) and produces speaker-labeled transcripts.

## Features

- **Job-based API** — submit files, poll for status, retrieve transcripts
- **Multi-backend ASR** — local faster-whisper or remote vLLM server
- **Silero VAD** — voice activity detection to skip silence
- **RNNoise denoising** — neural noise suppression for 48kHz audio
- **Webhook notifications** — optional callback on job completion
- **ZIP session support** — transcribe multi-participant recordings from livekit-recorder
- **Single WAV support** — transcribe any standalone WAV file

## Quick Start

```bash
pip install -r requirements.txt
pip install .
livekit-transcriber
```

The service starts on port 8091 by default.

## API

### Submit a transcription job

```
POST /transcribe
Content-Type: multipart/form-data

file: <.wav or .zip file>
language: en          (optional, default "en")
webhook_url: <url>    (optional)
```

Response: `{"job_id": "abc123", "status": "queued"}`

### Check job status

```
GET /transcribe/{job_id}
```

Returns job metadata including transcript when complete.

### Download transcript

```
GET /transcribe/{job_id}/transcript       (JSON)
GET /transcribe/{job_id}/transcript.txt   (plain text)
```

### List jobs

```
GET /jobs?limit=50
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `HOST` | `0.0.0.0` | Bind address |
| `PORT` | `8091` | HTTP port |
| `LOG_LEVEL` | `INFO` | Logging level |
| `JOBS_DIR` | `/jobs` | Job storage directory |
| `TRANSCRIBE_BACKEND` | `local` | ASR backend: `local` or `vllm` |
| `TRANSCRIBE_URL` | `http://localhost:8000` | vLLM server URL (when backend=vllm) |
| `TRANSCRIBE_MODEL_SIZE` | `large-v3` | Whisper model size (when backend=local) |
| `WHISPER_CACHE_DIR` | `/models/whisper` | Model download cache |
| `TRANSCRIBE_BEAM_SIZE` | `5` | Beam search width |
| `TRANSCRIBE_DENOISE` | `true` | Enable RNNoise denoising |
| `TRANSCRIBE_VAD_THRESHOLD` | `0.5` | VAD speech probability threshold |
| `SILERO_VAD_PATH` | `/app/models/silero_vad.onnx` | Path to Silero VAD ONNX model |

## Container

```bash
# Build
docker build -f ContainerFile -t livekit-transcriber .

# Run
docker run -p 8091:8091 -v ./jobs:/jobs -v ./models:/models/whisper livekit-transcriber
```

## Development

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt -r requirements-dev.txt
.venv/bin/pip install -e .
.venv/bin/python -m unittest discover -s tests -t .
```
