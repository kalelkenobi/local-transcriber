# Architecture

`local-transcriber` is a single-purpose CLI: read a multi-participant
recording, produce one merged speaker-labeled transcript, exit. It is
designed to run as a short-lived container that the user spins up
whenever a recording is ready.

## High-Level Flow

```
                       local-transcriber container
  ┌───────────────────────────────────────────────────────────────┐
  │                                                               │
  │   manifest.json        ffmpeg          Silero VAD             │
  │       │                  │                 │                  │
  │       ▼                  ▼                 ▼                  │
  │   parse session ──▶  decode per      ──▶  speech              │
  │                      participant          segments            │
  │                      to 16 kHz mono                           │
  │                                            │                  │
  │                                            ▼                  │
  │                                       slice + WAV-wrap        │
  │                                            │                  │
  │                                            ▼                  │
  │                                       Semaphore(N)            │
  │                                            │                  │
  │                                            ▼                  │
  │                                       POST /v1/audio/         │  ◀──┐
  │                                       transcriptions          │     │
  │                                            │                  │     │
  │                                            ▼                  │     │
  │                                       merge + sort            │     │
  │                                       by global time          │     │
  │                                            │                  │     │
  │                                            ▼                  │     │
  │                                  transcript.json + .txt       │     │
  └───────────────────────────────────────────────────────────────┘     │
                                                                        │
                              external ASR server ─────────────────────┘
                              (OpenAI-compatible)
```

There is no embedded model. The container ships:

- `ffmpeg` for audio decoding
- `onnxruntime` + `silero_vad.onnx` for VAD
- the CLI

…and nothing else. All ASR work happens over HTTP.

## Pipeline Stages

### 1. Session discovery

The CLI accepts either a single session directory (containing
`manifest.json`) or, with `--recursive`, a parent dir whose immediate
children are session directories. `iter_sessions()` in `session.py`
implements this.

### 2. Session loading

`load_session()` parses `manifest.json`, then for every identity listed
in `manifest.participants` it scans the session for a subdirectory whose
`metadata.json` matches that identity. Each match becomes a `Participant`
dataclass with audio path + sample rate + epoch metadata + event list.

### 3. Timeline alignment

Global zero = `manifest.start_epoch`. Per-participant offset:

```
audio_epoch  = participant.start_receiving_epoch
                  if start_receiving event exists
               else participant.start_epoch

offset       = max(0.0, audio_epoch - manifest.start_epoch)
```

`start_receiving` (when present) is preferred because it is closer to the
actual first byte of audio than the recorder-side `start_epoch`. The
clamp at zero handles the corner case where a participant's recording
began before the manifest's `start_epoch`.

### 4. Audio decoding

`audio.decode_to_pcm16_mono()` invokes `ffmpeg -i <file> -ac 1 -ar 16000
-f s16le -` and reads stdout. The output is 16 kHz mono int16 PCM —
suitable for both Silero VAD and every OpenAI-compatible ASR server we
care about. Any format `ffmpeg` can read is supported automatically.

### 5. Voice activity detection

`vad.SileroVAD` runs Silero VAD v5 ONNX on 512-sample windows (32 ms at
16 kHz), preserving the 64-sample context required by the model and the
`(2, 1, 128)` state. Per-frame probabilities feed a deterministic state
machine:

- **silence**: a contiguous run of frames with `p ≥ threshold` lasting
  at least `min_speech_ms` opens a segment at the first crossing frame.
- **speech**: a contiguous run of frames with `p < threshold` lasting
  at least `min_silence_ms` closes the segment at the first
  below-threshold frame.

Each segment is then padded by `speech_pad_ms` on both sides (clamped to
buffer bounds) and returned as `(start_s, end_s)` tuples in seconds.

### 6. Segment slicing

Each speech segment is sliced out of the participant's PCM buffer with
`audio.pcm_slice_to_wav_bytes()`, which prepends a WAV header so the ASR
server receives a fully-formed `audio/wav` upload.

### 7. ASR over HTTP

`backend.RemoteASRBackend.transcribe()` POSTs each WAV to
`{api_url}/v1/audio/transcriptions` with form fields `model` and
`language`. The `Authorization: Bearer <key>` header is set when
`--api-key`/`TRANSCRIBE_API_KEY` is provided.

Retries: up to two attempts on transport errors and HTTP 5xx, with
exponential backoff (`1s`, `2s`). HTTP 4xx is non-recoverable and surfaces
immediately. Per-segment failures during the ASR phase are logged and
skipped — they do not fail the session.

### 8. Merge + write

All transcribed segments from all participants are sorted by their
absolute start time (`offset + segment_start`) and written:

- `transcript.json` — structured form with session metadata, model name,
  language, and the segment list (`speaker`, `start`, `end`,
  `start_absolute`, `text`).
- `transcript.txt` — human-readable form, one block per segment:
  ```
  HH:MM:SS.ss <Speaker>
  <text>
  ```

Overlapping speech renders naturally as consecutive blocks sharing (or
near-sharing) a timestamp.

## Concurrency Model

Decode + VAD are sequential per participant. Pending WAV bytes are scoped
to one participant — each participant is decoded, VAD-filtered, and sent to
ASR before the next participant is processed. Within a single participant,
ASR calls run through `asyncio.Semaphore(concurrency)` (default: four
in-flight requests). Final `TranscriptSegment` objects are collected for
the whole session, then sorted and written at the end.

Across sessions (when `--recursive`), sessions are processed sequentially so
that the ASR server is never hit with more than one session's worth of
concurrent requests.

One full participant track is still decoded at a time. Extremely long
single-participant tracks may still need a higher container memory limit
(`scripts/run.sh --memory 4G`).

## Configuration

All knobs are CLI flags with `TRANSCRIBE_*` / `LOG_LEVEL` /
`SILERO_VAD_PATH` env-var equivalents. See `docs/cli.md` for the full
reference.

## Exit Codes

- `0` — every discovered session succeeded.
- `1` — configuration error (missing required flag, no sessions found,
  Silero model unavailable).
- `2` — at least one session failed (other sessions may have succeeded;
  per-session results are still printed to stdout).
