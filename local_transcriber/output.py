"""Transcript output writers and shared TranscriptSegment dataclass."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .session import Manifest


@dataclass(frozen=True)
class TranscriptSegment:
    speaker: str
    start: float  # seconds relative to manifest.start_epoch
    end: float
    text: str


def format_timestamp(seconds: float) -> str:
    """Format seconds as HH:MM:SS.ss (two-digit centiseconds, zero-padded)."""
    if seconds < 0:
        seconds = 0.0
    total_cs = int(round(seconds * 100))
    cs = total_cs % 100
    total_s = total_cs // 100
    s = total_s % 60
    total_m = total_s // 60
    m = total_m % 60
    h = total_m // 60
    return f"{h:02d}:{m:02d}:{s:02d}.{cs:02d}"


def write_transcript_json(
    path: Path,
    manifest: Manifest,
    model: str,
    language: str,
    segments: list[TranscriptSegment],
) -> None:
    """Write structured JSON transcript."""
    payload = {
        "session_id": manifest.session_id,
        "room_name": manifest.room_name,
        "language": language,
        "model": model,
        "start_epoch": manifest.start_epoch,
        "end_epoch": manifest.end_epoch,
        "segments": [
            {
                "speaker": s.speaker,
                "start": round(s.start, 3),
                "end": round(s.end, 3),
                "start_absolute": format_timestamp(s.start),
                "text": s.text,
            }
            for s in segments
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def write_transcript_txt(path: Path, segments: list[TranscriptSegment]) -> None:
    """Write plain-text transcript.

    Format (per segment, blank-line separated):

        HH:MM:SS.ss <Speaker>
        <text>
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for i, seg in enumerate(segments):
        if i > 0:
            lines.append("")
        lines.append(f"{format_timestamp(seg.start)} {seg.speaker}")
        lines.append(seg.text)
    with open(path, "w") as f:
        f.write("\n".join(lines))
        if lines:
            f.write("\n")
