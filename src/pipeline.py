"""
Transcription orchestrator — runs RNNoise denoising and VAD on
per-participant PCM files, sends speech segments to the ASR backend,
and assembles the final speaker-labeled transcript.
"""

import io
import json
import logging
import struct
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .backends import create_backend
from .rnnoise import denoise_pcm, is_available as rnnoise_available

# Re-export for convenience
__all__ = ["transcribe_session", "transcribe_wav", "TranscriptSegment", "_pcm_slice_to_wav"]

logger = logging.getLogger(__name__)

import os

# Silero VAD ONNX model path — configurable via env var
_SILERO_ONNX_PATH = Path(
    os.environ.get("SILERO_VAD_PATH", "/app/models/silero_vad.onnx")
)
# Fallback for local development
_SILERO_ONNX_DEV_PATH = Path(__file__).parent.parent / "models" / "silero_vad.onnx"


@dataclass
class TranscriptSegment:
    """A single transcribed speech segment."""

    speaker: str
    channel: int
    start: float
    end: float
    text: str


def _get_vad_model_path() -> Path:
    """Resolve the Silero VAD ONNX model path."""
    if _SILERO_ONNX_PATH.exists():
        return _SILERO_ONNX_PATH
    if _SILERO_ONNX_DEV_PATH.exists():
        return _SILERO_ONNX_DEV_PATH
    raise FileNotFoundError(
        "Silero VAD ONNX model not found. Expected at "
        f"{_SILERO_ONNX_PATH} or {_SILERO_ONNX_DEV_PATH}. "
        "See docs/transcription.md for setup instructions."
    )


def _run_vad(pcm_data: bytes, sample_rate: int, sample_width: int, threshold: float = 0.5) -> list[tuple[float, float]]:
    """
    Run Silero VAD on raw PCM data and return speech segments.

    Args:
        pcm_data: Raw PCM bytes (mono).
        sample_rate: Sample rate in Hz (e.g. 48000).
        sample_width: Bytes per sample (e.g. 2 for 16-bit).

    Returns:
        List of (start_seconds, end_seconds) tuples for voiced segments.
    """
    import onnxruntime as ort

    model_path = _get_vad_model_path()

    # Convert PCM bytes to float32 numpy array
    num_samples = len(pcm_data) // sample_width
    if sample_width == 2:
        audio_int16 = np.frombuffer(pcm_data, dtype=np.int16)
        audio_f32 = audio_int16.astype(np.float32) / 32768.0
    else:
        raise ValueError(f"Unsupported sample width: {sample_width}")

    # Silero VAD expects 16kHz — resample if needed
    if sample_rate != 16000:
        ratio = sample_rate // 16000
        if sample_rate % 16000 == 0 and ratio > 1:
            # Integer ratio: apply simple averaging anti-alias filter then decimate
            # Truncate to multiple of ratio
            truncated = len(audio_f32) - (len(audio_f32) % ratio)
            audio_f32 = audio_f32[:truncated].reshape(-1, ratio).mean(axis=1)
        else:
            # Non-integer ratio: linear interpolation
            new_length = int(len(audio_f32) * 16000 / sample_rate)
            x_old = np.linspace(0, 1, len(audio_f32))
            x_new = np.linspace(0, 1, new_length)
            audio_f32 = np.interp(x_new, x_old, audio_f32).astype(np.float32)
        effective_sr = 16000
    else:
        effective_sr = sample_rate

    # Run Silero VAD in chunks
    session = ort.InferenceSession(str(model_path))

    window_size = 512  # 32ms at 16kHz — Silero v5 expects 512 samples
    context_size = 64  # Silero requires 64-sample context prepended to each chunk
    min_speech_duration = 0.25  # seconds
    min_silence_duration = 0.5  # seconds

    speeches: list[tuple[float, float]] = []
    is_speaking = False
    speech_start = 0
    silence_start = 0

    # Silero VAD v5 state: shape [2, 1, 128]
    state = np.zeros((2, 1, 128), dtype=np.float32)
    # Context: last `context_size` samples from previous chunk (zeros initially)
    context = np.zeros(context_size, dtype=np.float32)

    num_windows = len(audio_f32) // window_size
    for i in range(num_windows):
        chunk = audio_f32[i * window_size: (i + 1) * window_size]
        # Prepend context to chunk (model expects context_size + window_size samples)
        chunk_with_context = np.concatenate([context, chunk]).reshape(1, -1)
        sr_input = np.array(effective_sr, dtype=np.int64)

        ort_inputs = {
            "input": chunk_with_context,
            "state": state,
            "sr": sr_input,
        }
        ort_outputs = session.run(None, ort_inputs)
        prob = ort_outputs[0][0][0]
        state = ort_outputs[1]
        # Save last context_size samples for next iteration
        context = chunk[-context_size:]

        current_time = (i * window_size) / effective_sr

        if prob >= threshold and not is_speaking:
            is_speaking = True
            speech_start = current_time
        elif prob < threshold and is_speaking:
            silence_duration = current_time - silence_start if silence_start > speech_start else 0
            if silence_start <= speech_start:
                silence_start = current_time
            else:
                silence_duration = current_time - silence_start
                if silence_duration >= min_silence_duration:
                    speech_end = silence_start
                    if (speech_end - speech_start) >= min_speech_duration:
                        speeches.append((speech_start, speech_end))
                    is_speaking = False

        if prob < threshold and is_speaking:
            if silence_start <= speech_start:
                silence_start = current_time

    # Flush any remaining speech
    if is_speaking:
        speech_end = (num_windows * window_size) / effective_sr
        if (speech_end - speech_start) >= min_speech_duration:
            speeches.append((speech_start, speech_end))

    logger.info("VAD found %d speech segments", len(speeches))
    return speeches


def _pcm_slice_to_wav(
    pcm_data: bytes,
    sample_rate: int,
    sample_width: int,
    start_s: float,
    end_s: float,
) -> bytes:
    """
    Slice PCM data by time range and wrap in a WAV header.

    Returns complete WAV file bytes.
    """
    bytes_per_second = sample_rate * sample_width
    start_byte = int(start_s * bytes_per_second)
    end_byte = int(end_s * bytes_per_second)

    # Align to sample boundary
    start_byte -= start_byte % sample_width
    end_byte -= end_byte % sample_width

    segment_pcm = pcm_data[start_byte:end_byte]

    # Wrap in WAV
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(segment_pcm)

    return buf.getvalue()


async def transcribe_session(
    session_dir: Path,
    backend_type: str = "local",
    language: str = "en",
    denoise: bool = True,
    vad_threshold: float = 0.5,
    **backend_kwargs,
) -> bool:
    """
    Transcribe a recording session using per-participant PCM files.

    Must be called BEFORE package_session (which deletes the PCM files).

    Args:
        session_dir: Path to the session directory.
        backend_type: "local" or "vllm".
        language: Language code for ASR.
        denoise: Whether to apply RNNoise denoising before VAD.
        vad_threshold: Speech probability threshold for VAD (0.0–1.0).
        **backend_kwargs: Passed to create_backend (e.g. base_url, model_size).

    Returns:
        True if transcription succeeded, False otherwise.
    """
    manifest_path = session_dir / "manifest.json"
    if not manifest_path.exists():
        logger.error("No manifest.json in %s", session_dir)
        return False

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    session_id = manifest.get("session_id", session_dir.name)
    participant_identities = manifest.get("participants", [])

    if not participant_identities:
        logger.warning("No participants to transcribe in session %s", session_id)
        return False

    # Create backend
    backend = create_backend(backend_type, **backend_kwargs)

    all_segments: list[TranscriptSegment] = []

    try:
        for channel_idx, identity in enumerate(participant_identities):
            safe_name = "".join(
                c if c.isalnum() or c in "-_" else "_" for c in identity
            )
            participant_dir = session_dir / safe_name
            metadata_file = participant_dir / "metadata.json"

            if not metadata_file.exists():
                logger.warning(
                    "No metadata.json for participant '%s', skipping transcription",
                    identity,
                )
                continue

            with open(metadata_file, "r") as f:
                meta = json.load(f)

            pcm_file = meta.get("pcm_file", f"{safe_name}.pcm")
            pcm_path = participant_dir / pcm_file
            sample_rate = meta.get("sample_rate", 48000)
            sample_width = meta.get("sample_width", 2)

            if not pcm_path.exists():
                logger.warning("PCM file not found for '%s': %s", identity, pcm_path)
                continue

            pcm_data = pcm_path.read_bytes()
            if not pcm_data:
                logger.warning("Empty PCM file for '%s'", identity)
                continue

            audio_duration = len(pcm_data) / (sample_rate * sample_width)

            # Denoise with RNNoise (48kHz only)
            if denoise and rnnoise_available() and sample_rate == 48000:
                logger.info(
                    "Denoising audio for participant '%s' (%.1f seconds)",
                    identity,
                    audio_duration,
                )
                pcm_data = denoise_pcm(pcm_data, sample_rate)
            elif denoise and not rnnoise_available():
                logger.warning(
                    "RNNoise not available, skipping denoising for '%s'",
                    identity,
                )

            logger.info(
                "Running VAD for participant '%s' (%.1f seconds of audio)",
                identity,
                audio_duration,
            )

            # Run VAD on denoised audio
            speech_segments = _run_vad(pcm_data, sample_rate, sample_width, vad_threshold)

            if not speech_segments:
                logger.info("No speech detected for participant '%s'", identity)
                continue

            logger.info(
                "Transcribing %d speech segments for '%s'",
                len(speech_segments),
                identity,
            )

            # Transcribe each speech segment
            for start_s, end_s in speech_segments:
                wav_bytes = _pcm_slice_to_wav(
                    pcm_data, sample_rate, sample_width, start_s, end_s
                )

                try:
                    text = await backend.transcribe(wav_bytes, language)
                except Exception:
                    logger.exception(
                        "Failed to transcribe segment [%.1f-%.1f] for '%s'",
                        start_s,
                        end_s,
                        identity,
                    )
                    continue

                if text:
                    all_segments.append(
                        TranscriptSegment(
                            speaker=identity,
                            channel=channel_idx,
                            start=round(start_s, 2),
                            end=round(end_s, 2),
                            text=text,
                        )
                    )

    finally:
        await backend.close()

    # If no speech detected at all, skip transcript creation entirely.
    if not all_segments:
        logger.info(
            "No speech detected in any participant for session %s — "
            "skipping transcript creation",
            session_id,
        )
        return False

    # Sort by start time
    all_segments.sort(key=lambda s: s.start)

    # Write transcript.json
    transcript_data = {
        "session_id": session_id,
        "language": language,
        "backend": backend_type,
        "segments": [
            {
                "speaker": seg.speaker,
                "channel": seg.channel,
                "start": seg.start,
                "end": seg.end,
                "text": seg.text,
            }
            for seg in all_segments
        ],
    }

    transcript_json_path = session_dir / "transcript.json"
    with open(transcript_json_path, "w") as f:
        json.dump(transcript_data, f, indent=2)

    # Write transcript.txt
    transcript_txt_path = session_dir / "transcript.txt"
    with open(transcript_txt_path, "w") as f:
        for i, seg in enumerate(all_segments):
            if i > 0:
                f.write("\n")
            f.write(f"{seg.speaker}\n")
            f.write(f"{seg.text}\n")

    # Update manifest
    manifest["transcript_file"] = "transcript.json"
    manifest["transcript_txt_file"] = "transcript.txt"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info(
        "Transcription complete for session %s: %d segments from %d speakers",
        session_id,
        len(all_segments),
        len(set(s.speaker for s in all_segments)),
    )

    return True


async def transcribe_wav(
    wav_path: Path,
    output_dir: Path,
    backend_type: str = "local",
    language: str = "en",
    denoise: bool = True,
    vad_threshold: float = 0.5,
    speaker: str = "speaker",
    **backend_kwargs,
) -> bool:
    """
    Transcribe a single WAV file as one speaker.

    Reads the WAV, converts to PCM, runs VAD, transcribes speech segments,
    and writes transcript.json + transcript.txt to output_dir.

    Args:
        wav_path: Path to the WAV file.
        output_dir: Directory where transcript files will be written.
        backend_type: "local" or "vllm".
        language: Language code for ASR.
        denoise: Whether to apply RNNoise denoising (48kHz only).
        vad_threshold: Speech probability threshold for VAD.
        speaker: Speaker label for the transcript.
        **backend_kwargs: Passed to create_backend.

    Returns:
        True if transcription succeeded, False otherwise.
    """
    import wave as wave_mod

    if not wav_path.exists():
        logger.error("WAV file not found: %s", wav_path)
        return False

    # Read WAV and extract PCM
    with wave_mod.open(str(wav_path), "rb") as wf:
        sample_rate = wf.getframerate()
        sample_width = wf.getsampwidth()
        n_channels = wf.getnchannels()
        pcm_data = wf.readframes(wf.getnframes())

    # If stereo/multi-channel, mix down to mono
    if n_channels > 1:
        samples = np.frombuffer(pcm_data, dtype=np.int16)
        samples = samples.reshape(-1, n_channels).mean(axis=1).astype(np.int16)
        pcm_data = samples.tobytes()

    audio_duration = len(pcm_data) / (sample_rate * sample_width)
    logger.info("Transcribing WAV: %.1f seconds, %d Hz", audio_duration, sample_rate)

    # Denoise
    if denoise and rnnoise_available() and sample_rate == 48000:
        pcm_data = denoise_pcm(pcm_data, sample_rate)
    elif denoise and not rnnoise_available():
        logger.warning("RNNoise not available, skipping denoising")

    # VAD
    speech_segments = _run_vad(pcm_data, sample_rate, sample_width, vad_threshold)
    if not speech_segments:
        logger.info("No speech detected in WAV file")
        return False

    logger.info("Transcribing %d speech segments", len(speech_segments))

    # Transcribe
    backend = create_backend(backend_type, **backend_kwargs)
    all_segments: list[TranscriptSegment] = []

    try:
        for start_s, end_s in speech_segments:
            wav_bytes = _pcm_slice_to_wav(pcm_data, sample_rate, sample_width, start_s, end_s)
            try:
                text = await backend.transcribe(wav_bytes, language)
            except Exception:
                logger.exception("Failed to transcribe segment [%.1f-%.1f]", start_s, end_s)
                continue
            if text:
                all_segments.append(
                    TranscriptSegment(
                        speaker=speaker,
                        channel=0,
                        start=round(start_s, 2),
                        end=round(end_s, 2),
                        text=text,
                    )
                )
    finally:
        await backend.close()

    if not all_segments:
        logger.info("No speech transcribed from WAV file")
        return False

    all_segments.sort(key=lambda s: s.start)

    # Write outputs
    output_dir.mkdir(parents=True, exist_ok=True)

    transcript_data = {
        "source": wav_path.name,
        "language": language,
        "backend": backend_type,
        "segments": [
            {
                "speaker": seg.speaker,
                "channel": seg.channel,
                "start": seg.start,
                "end": seg.end,
                "text": seg.text,
            }
            for seg in all_segments
        ],
    }

    with open(output_dir / "transcript.json", "w") as f:
        json.dump(transcript_data, f, indent=2)

    with open(output_dir / "transcript.txt", "w") as f:
        for i, seg in enumerate(all_segments):
            if i > 0:
                f.write("\n")
            f.write(f"{seg.speaker}\n")
            f.write(f"{seg.text}\n")

    logger.info("Transcription complete: %d segments", len(all_segments))
    return True
