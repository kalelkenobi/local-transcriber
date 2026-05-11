"""Transcription pipeline orchestrator.

`transcribe_session()` decodes every participant's audio, runs Silero VAD,
slices speech regions, sends them through the ASR backend (bounded
concurrency), merges all segments on the global timeline
(0.0 == manifest.start_epoch), and writes transcript.json +
transcript.txt to the chosen output directory.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path

from .audio import decode_to_pcm16_mono, pcm_slice_to_wav_bytes
from .backend import RemoteASRBackend
from .output import (
    TranscriptSegment,
    write_transcript_json,
    write_transcript_txt,
)
from .session import (
    Manifest,
    Participant,
    load_session,
    participant_offset,
)
from .vad import SileroVAD

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SessionResult:
    session_id: str
    ok: bool
    num_segments: int
    num_speakers: int
    output_dir: Path
    error: str | None = None


@dataclass
class _PendingSegment:
    speaker: str
    start_abs: float
    end_abs: float
    wav_bytes: bytes


async def transcribe_session(
    session_dir: Path,
    *,
    backend: RemoteASRBackend,
    vad: SileroVAD,
    language: str,
    output_dir: Path | None = None,
    concurrency: int = 4,
) -> SessionResult:
    """Transcribe one session directory and return a SessionResult.

    On any error before the ASR phase the result is `ok=False` with an
    `error` description. Individual segment failures during ASR are
    logged but do not fail the session.
    """
    out_dir = output_dir or session_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        manifest, participants = load_session(session_dir)
    except Exception as exc:
        logger.exception("Failed to load session %s", session_dir)
        return SessionResult(
            session_id=session_dir.name,
            ok=False,
            num_segments=0,
            num_speakers=0,
            output_dir=out_dir,
            error=f"load_session failed: {exc}",
        )

    if not participants:
        logger.warning("Session %s has no participants", manifest.session_id)
        return SessionResult(
            session_id=manifest.session_id,
            ok=False,
            num_segments=0,
            num_speakers=0,
            output_dir=out_dir,
            error="no participants",
        )

    pending: list[_PendingSegment] = []
    for participant in participants:
        try:
            pending.extend(
                _prepare_participant_segments(manifest, participant, vad)
            )
        except Exception as exc:
            logger.exception(
                "Failed to prepare segments for %s in %s",
                participant.identity,
                manifest.session_id,
            )
            return SessionResult(
                session_id=manifest.session_id,
                ok=False,
                num_segments=0,
                num_speakers=0,
                output_dir=out_dir,
                error=f"{participant.identity}: {exc}",
            )

    if not pending:
        logger.info("No speech detected in session %s", manifest.session_id)
        return SessionResult(
            session_id=manifest.session_id,
            ok=False,
            num_segments=0,
            num_speakers=0,
            output_dir=out_dir,
            error="no speech detected",
        )

    transcribed = await _run_transcriptions(
        pending, backend, language, concurrency
    )

    if not transcribed:
        logger.info("No transcribed text for session %s", manifest.session_id)
        return SessionResult(
            session_id=manifest.session_id,
            ok=False,
            num_segments=0,
            num_speakers=0,
            output_dir=out_dir,
            error="all segments empty",
        )

    transcribed.sort(key=lambda s: (s.start, s.speaker))

    write_transcript_json(
        out_dir / "transcript.json",
        manifest=manifest,
        model=backend.model,
        language=language,
        segments=transcribed,
    )
    write_transcript_txt(out_dir / "transcript.txt", transcribed)

    num_speakers = len({s.speaker for s in transcribed})
    logger.info(
        "Session %s complete: %d segments, %d speakers",
        manifest.session_id,
        len(transcribed),
        num_speakers,
    )
    return SessionResult(
        session_id=manifest.session_id,
        ok=True,
        num_segments=len(transcribed),
        num_speakers=num_speakers,
        output_dir=out_dir,
    )


def _prepare_participant_segments(
    manifest: Manifest,
    participant: Participant,
    vad: SileroVAD,
) -> list[_PendingSegment]:
    """Decode + VAD + slice for a single participant."""
    logger.info(
        "Decoding %s (%s)", participant.identity, participant.audio_path.name
    )
    pcm, sr = decode_to_pcm16_mono(participant.audio_path)
    if not pcm:
        logger.warning("Empty PCM after decode for %s", participant.identity)
        return []

    speech = vad.iter_speech_segments(pcm)
    if not speech:
        logger.info("No speech detected for %s", participant.identity)
        return []

    offset = participant_offset(manifest, participant)
    pending: list[_PendingSegment] = []
    for start_s, end_s in speech:
        wav_bytes = pcm_slice_to_wav_bytes(pcm, sr, start_s, end_s)
        pending.append(
            _PendingSegment(
                speaker=participant.identity,
                start_abs=offset + start_s,
                end_abs=offset + end_s,
                wav_bytes=wav_bytes,
            )
        )
    logger.info(
        "Prepared %d speech segments for %s (offset=%.3fs)",
        len(pending),
        participant.identity,
        offset,
    )
    return pending


async def _run_transcriptions(
    pending: list[_PendingSegment],
    backend: RemoteASRBackend,
    language: str,
    concurrency: int,
) -> list[TranscriptSegment]:
    """Send pending segments to the backend with bounded concurrency."""
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def _run(seg: _PendingSegment) -> TranscriptSegment | None:
        async with semaphore:
            try:
                text = await backend.transcribe(seg.wav_bytes, language)
            except Exception:
                logger.exception(
                    "Failed to transcribe %s [%.2f-%.2f]",
                    seg.speaker,
                    seg.start_abs,
                    seg.end_abs,
                )
                return None
        if not text:
            return None
        return TranscriptSegment(
            speaker=seg.speaker,
            start=seg.start_abs,
            end=seg.end_abs,
            text=text,
        )

    results = await asyncio.gather(*[_run(s) for s in pending])
    return [r for r in results if r is not None]
