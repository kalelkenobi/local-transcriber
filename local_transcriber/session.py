"""Session manifest and per-participant metadata loaders.

Session layout:

    session_dir/
        manifest.json                  # session-wide info
        <Identity>_<shortid>/
            metadata.json              # per-participant info
            <Identity>_<shortid>.opus  # (or wav/flac/m4a/...)
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Manifest:
    session_id: str
    room_name: str
    start_epoch: float
    end_epoch: float | None
    participants: tuple[str, ...]
    format: str | None
    bitrate: int | None
    raw: dict


@dataclass(frozen=True)
class Participant:
    identity: str
    dir: Path
    audio_path: Path
    audio_format: str
    sample_rate: int
    channels: int
    start_epoch: float
    start_receiving_epoch: float | None
    events: tuple[dict, ...]
    raw: dict


def load_manifest(session_dir: Path) -> Manifest:
    """Load the manifest.json from a session directory."""
    manifest_path = session_dir / "manifest.json"
    with open(manifest_path) as f:
        data = json.load(f)
    return Manifest(
        session_id=data.get("session_id") or session_dir.name,
        room_name=data.get("room_name", ""),
        start_epoch=float(data["start_epoch"]),
        end_epoch=(
            float(data["end_epoch"])
            if data.get("end_epoch") is not None
            else None
        ),
        participants=tuple(data.get("participants", [])),
        format=data.get("format"),
        bitrate=data.get("bitrate"),
        raw=data,
    )


def _find_participant_dir(session_dir: Path, identity: str) -> Path | None:
    """Locate a participant subdir by matching its metadata.identity field."""
    for child in sorted(session_dir.iterdir()):
        if not child.is_dir():
            continue
        meta_path = child / "metadata.json"
        if not meta_path.exists():
            continue
        try:
            with open(meta_path) as f:
                meta = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        if meta.get("identity") == identity:
            return child
    return None


def load_participant(session_dir: Path, identity: str) -> Participant | None:
    """Load a participant's metadata + resolve audio path. Returns None if missing."""
    participant_dir = _find_participant_dir(session_dir, identity)
    if participant_dir is None:
        logger.warning(
            "No directory found for participant %r in %s", identity, session_dir
        )
        return None

    meta_path = participant_dir / "metadata.json"
    with open(meta_path) as f:
        meta = json.load(f)

    audio_file = meta.get("audio_file")
    if not audio_file:
        logger.warning("Participant %r metadata has no audio_file", identity)
        return None
    audio_path = participant_dir / audio_file
    if not audio_path.exists():
        logger.warning("Audio file missing for %r: %s", identity, audio_path)
        return None

    events = tuple(meta.get("events", []) or ())
    start_receiving: float | None = None
    for event in events:
        if event.get("type") == "start_receiving":
            try:
                start_receiving = float(event["epoch"])
            except (KeyError, TypeError, ValueError):
                logger.warning(
                    "Malformed start_receiving event for %r", identity
                )
            break

    return Participant(
        identity=identity,
        dir=participant_dir,
        audio_path=audio_path,
        audio_format=str(meta.get("format", "unknown")),
        sample_rate=int(meta.get("sample_rate", 48000)),
        channels=int(meta.get("channels", 1)),
        start_epoch=float(meta.get("start_epoch", 0.0)),
        start_receiving_epoch=start_receiving,
        events=events,
        raw=meta,
    )


def load_session(session_dir: Path) -> tuple[Manifest, list[Participant]]:
    """Load manifest + all known participants for a session directory."""
    manifest = load_manifest(session_dir)
    participants: list[Participant] = []
    for identity in manifest.participants:
        p = load_participant(session_dir, identity)
        if p is not None:
            participants.append(p)
    return manifest, participants


def participant_offset(manifest: Manifest, participant: Participant) -> float:
    """Return seconds between manifest start and the participant's audio start.

    Uses the start_receiving event when available (more precise),
    otherwise metadata.start_epoch. Clamped at zero — a participant that
    joined before the recording started is treated as starting at t=0
    (which is consistent with how the manifest defines the timeline).
    """
    audio_epoch = (
        participant.start_receiving_epoch
        if participant.start_receiving_epoch is not None
        else participant.start_epoch
    )
    return max(0.0, audio_epoch - manifest.start_epoch)


def iter_sessions(root: Path, recursive: bool) -> Iterator[Path]:
    """Enumerate session directories under root.

    - If `root/manifest.json` exists, yield `root` (single-session mode).
    - Else if `recursive=True`, yield each immediate child of `root` that
      contains a `manifest.json`.
    - Else yield nothing.
    """
    if not root.exists() or not root.is_dir():
        return
    if (root / "manifest.json").exists():
        yield root
        return
    if not recursive:
        return
    for child in sorted(root.iterdir()):
        if child.is_dir() and (child / "manifest.json").exists():
            yield child
