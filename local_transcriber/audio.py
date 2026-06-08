"""Audio decoding and PCM slicing utilities.

Uses ffmpeg as a subprocess so any format ffmpeg can read is supported.
The pipeline always decodes to 16 kHz mono int16 PCM — both Silero VAD
and most OpenAI-compatible ASR backends operate at 16 kHz internally.
"""

from __future__ import annotations

import io
import logging
import subprocess
import wave
from pathlib import Path

logger = logging.getLogger(__name__)

SAMPLE_WIDTH = 2  # 16-bit
DEFAULT_SAMPLE_RATE = 16000


class AudioDecodeError(RuntimeError):
    """Raised when ffmpeg fails to decode an audio file."""


def decode_to_pcm16_mono(
    audio_path: Path,
    target_sr: int = DEFAULT_SAMPLE_RATE,
) -> tuple[bytes, int]:
    """Decode any ffmpeg-readable audio file to 16-bit mono PCM at target_sr.

    Returns (pcm_bytes, sample_rate). Raises AudioDecodeError on failure.
    """
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-nostdin",
        "-loglevel", "error",
        "-i", str(audio_path),
        "-ac", "1",
        "-ar", str(target_sr),
        "-f", "s16le",
        "-",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, check=False)
    except FileNotFoundError as exc:
        raise AudioDecodeError("ffmpeg not found on PATH") from exc

    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        raise AudioDecodeError(
            f"ffmpeg failed for {audio_path.name} "
            f"(exit {result.returncode}): {stderr}"
        )
    return result.stdout, target_sr


def probe_audio_duration(audio_path: Path) -> float:
    """Return audio duration in seconds using ffprobe without decoding."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(audio_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, check=False)
    except FileNotFoundError as exc:
        raise AudioDecodeError("ffprobe not found on PATH") from exc

    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        raise AudioDecodeError(
            f"ffprobe failed for {audio_path.name} "
            f"(exit {result.returncode}): {stderr}"
        )

    raw = result.stdout.decode("utf-8", errors="replace").strip()
    try:
        duration = float(raw)
    except ValueError as exc:
        raise AudioDecodeError(
            f"ffprobe returned invalid duration for {audio_path.name}: {raw!r}"
        ) from exc
    if duration < 0:
        raise AudioDecodeError(
            f"ffprobe returned negative duration for {audio_path.name}: {duration}"
        )
    return duration


def decode_range_to_pcm16_mono(
    audio_path: Path,
    start_s: float,
    duration_s: float,
    target_sr: int = DEFAULT_SAMPLE_RATE,
) -> tuple[bytes, int]:
    """Decode a time range to 16-bit mono PCM at target_sr."""
    if start_s < 0:
        raise ValueError("start_s must be non-negative")
    if duration_s <= 0:
        raise ValueError("duration_s must be positive")

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-nostdin",
        "-loglevel", "error",
        "-ss", f"{start_s:.3f}",
        "-t", f"{duration_s:.3f}",
        "-i", str(audio_path),
        "-ac", "1",
        "-ar", str(target_sr),
        "-f", "s16le",
        "-",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, check=False)
    except FileNotFoundError as exc:
        raise AudioDecodeError("ffmpeg not found on PATH") from exc

    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        raise AudioDecodeError(
            f"ffmpeg failed for {audio_path.name} range "
            f"{start_s:.3f}-{start_s + duration_s:.3f}s "
            f"(exit {result.returncode}): {stderr}"
        )
    return result.stdout, target_sr


def pcm_duration_seconds(
    pcm: bytes,
    sample_rate: int,
    sample_width: int = SAMPLE_WIDTH,
) -> float:
    """Return the total duration in seconds of a PCM buffer."""
    if sample_rate <= 0 or sample_width <= 0:
        raise ValueError("sample_rate and sample_width must be positive")
    return len(pcm) / (sample_rate * sample_width)


def pcm_slice_to_wav_bytes(
    pcm: bytes,
    sample_rate: int,
    start_s: float,
    end_s: float,
    sample_width: int = SAMPLE_WIDTH,
) -> bytes:
    """Slice PCM by time range and wrap in a WAV header.

    Returns complete WAV file bytes.
    """
    if end_s <= start_s:
        raise ValueError("end_s must be greater than start_s")

    bytes_per_second = sample_rate * sample_width
    start_byte = int(start_s * bytes_per_second)
    end_byte = int(end_s * bytes_per_second)

    # Align to sample boundary and clamp.
    start_byte -= start_byte % sample_width
    end_byte -= end_byte % sample_width
    start_byte = max(0, start_byte)
    end_byte = min(end_byte, len(pcm))
    segment = pcm[start_byte:end_byte]

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(segment)
    return buf.getvalue()
