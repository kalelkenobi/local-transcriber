# Architecture

The LiveKit Transcriber is a standalone async job-based service that accepts
audio files and produces speaker-labeled transcripts. It is fully decoupled
from the recorder — no shared runtime, communication only via files.

## High-Level Flow

```
Client                 Transcriber                    ASR Backend
  |                        |                              |
  |  POST /transcribe      |                              |
  |  (upload .zip/.wav)    |                              |
  |----------------------->|                              |
  |  {job_id, "queued"}    |                              |
  |<-----------------------|                              |
  |                        |  background task starts      |
  |                        |  1. Extract ZIP / read WAV   |
  |                        |  2. Resample to 16kHz mono   |
  |                        |  3. RNNoise denoise (opt)    |
  |                        |  4. Silero VAD               |
  |                        |  5. ASR on speech segments   |
  |                        |----------------------------->|
  |                        |  {text, timestamps}          |
  |                        |<-----------------------------|
  |                        |  6. Write transcript files   |
  |                        |  7. Send webhook (optional)  |
  |                        |                              |
  |  GET /transcribe/{id}  |                              |
  |----------------------->|                              |
  |  {status: "completed", |                              |
  |   transcript: [...]}   |                              |
  |<-----------------------|                              |
```

## Job Lifecycle

1. **Queued** — file uploaded, job registered in `JOBS_DIR/{job_id}/job.json`
2. **Processing** — background task running VAD + ASR pipeline
3. **Completed** — transcript files written to job directory
4. **Failed** — error recorded in job metadata

## Pipeline Stages

### 1. Input Handling

- **ZIP** (from livekit-recorder): extracted, manifest.json located, multi-participant
  PCM segments reassembled per participant with timeline-preserving silence.
- **WAV** (standalone): read directly as single-speaker input.

### 2. Audio Preprocessing

- Resample to 16kHz mono PCM (required by both VAD and Whisper).
- Optional RNNoise denoising at 48kHz before downsampling (controlled by `TRANSCRIBE_DENOISE`).

### 3. Voice Activity Detection

- Silero VAD ONNX model classifies 30ms frames.
- Frames below `TRANSCRIBE_VAD_THRESHOLD` probability are dropped.
- Contiguous speech frames are grouped into segments with padding.

### 4. ASR Transcription

Two backends:

- **local** — faster-whisper running in-process (GPU or CPU). Configurable model
  size and beam width.
- **vllm** — remote vLLM-compatible server via OpenAI-compatible `/v1/audio/transcriptions`
  endpoint.

### 5. Output

Transcript files written to job directory:

- `transcript.json` — structured JSON with speaker labels, timestamps, segments
- `transcript.txt` — plain-text formatted transcript

## Storage Layout

```
JOBS_DIR/
└── {job_id}/
    ├── job.json              # Job metadata and status
    ├── {uploaded_file}       # Original upload
    ├── session/              # (ZIP only) extracted session contents
    ├── transcript.json       # Structured transcript
    └── transcript.txt        # Plain-text transcript
```

## Configuration

All configuration is via environment variables. See README.md for the full table.

Key architectural decisions:

- **Backend selection** (`TRANSCRIBE_BACKEND`): `local` for self-contained deployments,
  `vllm` for shared GPU infrastructure.
- **Denoise toggle** (`TRANSCRIBE_DENOISE`): can distort synthetic audio; disable for
  testing with TTS signals.
- **VAD threshold** (`TRANSCRIBE_VAD_THRESHOLD`): lower = more sensitive (catches
  quieter speech), higher = fewer false positives.
