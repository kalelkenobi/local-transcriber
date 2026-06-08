"""Tests for the pipeline orchestrator.

`decode_to_pcm16_mono` and `SileroVAD.iter_speech_segments` are patched so
this suite does not require ffmpeg or the ONNX model — they have their own
tests. The backend is replaced with an in-memory fake.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from local_transcriber import pipeline as pipeline_mod
from local_transcriber.pipeline import SessionResult, transcribe_session


class _EventRecordingBackend:
    """Records call events for ordering assertions."""

    def __init__(self, events: list[str]):
        self.events = events
        self.model = "fake-model"

    async def transcribe(self, wav_bytes: bytes, language: str) -> str:
        self.events.append("asr")
        return "text"

    async def close(self) -> None:
        pass


class _SequenceVAD:
    """Returns a different segment list for each call."""

    def __init__(self, responses: list[list[tuple[float, float]]]):
        self._responses = list(responses)
        self.calls = 0

    def iter_speech_segments(self, pcm: bytes) -> list[tuple[float, float]]:
        self.calls += 1
        if self._responses:
            return self._responses.pop(0)
        return []


def _build_session(
    root: Path,
    *,
    manifest_start: float = 1000.0,
    participants: list[tuple[str, float, float | None]],
) -> Path:
    """Create a synthetic session directory matching the example layout."""
    session = root / "session"
    session.mkdir()
    (session / "manifest.json").write_text(
        json.dumps(
            {
                "session_id": "syn-1",
                "room_name": "syn-room",
                "start_epoch": manifest_start,
                "end_epoch": manifest_start + 60.0,
                "participants": [p[0] for p in participants],
                "format": "opus",
                "bitrate": 128000,
            }
        )
    )
    for identity, start_epoch, start_recv in participants:
        pdir = session / f"{identity}_aaaa"
        pdir.mkdir()
        audio = pdir / f"{identity}_aaaa.opus"
        audio.write_bytes(b"\x00" * 4)
        events: list[dict] = []
        if start_recv is not None:
            events.append({"type": "start_receiving", "epoch": start_recv})
        (pdir / "metadata.json").write_text(
            json.dumps(
                {
                    "identity": identity,
                    "sample_rate": 48000,
                    "channels": 1,
                    "format": "opus",
                    "start_epoch": start_epoch,
                    "audio_file": audio.name,
                    "events": events,
                }
            )
        )
    return session


class FakeBackend:
    """Records every call and returns canned text."""

    def __init__(self, responses: dict[tuple[str, int], str] | None = None):
        self.responses = responses or {}
        self.calls: list[dict] = []
        self.model = "fake-model"

    async def transcribe(self, wav_bytes: bytes, language: str) -> str:
        idx = len(self.calls)
        self.calls.append(
            {"language": language, "bytes": len(wav_bytes)}
        )
        return f"segment-{idx}"

    async def close(self) -> None:
        pass


class FakeVAD:
    """Returns a configurable list of (start, end) tuples per call."""

    def __init__(self, segments: list[tuple[float, float]]):
        self._segments = segments
        self.calls = 0

    def iter_speech_segments(self, pcm: bytes) -> list[tuple[float, float]]:
        self.calls += 1
        return list(self._segments)


class TestTranscribeSession(unittest.IsolatedAsyncioTestCase):
    async def test_writes_outputs_and_returns_ok(self):
        with tempfile.TemporaryDirectory() as td:
            session = _build_session(
                Path(td),
                participants=[
                    ("Alice", 1000.0, None),
                    ("Bob", 1005.0, None),
                ],
            )
            backend = FakeBackend()
            vad = FakeVAD([(0.0, 1.0), (2.0, 3.0)])
            with patch.object(
                pipeline_mod,
                "decode_to_pcm16_mono",
                return_value=(b"\x00" * 32000, 16000),
            ):
                result = await transcribe_session(
                    session,
                    backend=backend,
                    vad=vad,
                    language="en",
                    concurrency=2,
                )

            self.assertTrue(result.ok, msg=f"err={result.error}")
            # 2 participants * 2 segments = 4 segments
            self.assertEqual(result.num_segments, 4)
            self.assertEqual(result.num_speakers, 2)
            self.assertEqual(len(backend.calls), 4)
            self.assertEqual(vad.calls, 2)

            tjson = session / "transcript.json"
            ttxt = session / "transcript.txt"
            self.assertTrue(tjson.exists())
            self.assertTrue(ttxt.exists())
            data = json.loads(tjson.read_text())
            self.assertEqual(len(data["segments"]), 4)
            # Segments must be sorted by start time
            starts = [s["start"] for s in data["segments"]]
            self.assertEqual(starts, sorted(starts))
            # Bob's offset is +5s, so his earliest segment >= 5.0
            bob_starts = [
                s["start"] for s in data["segments"] if s["speaker"] == "Bob"
            ]
            self.assertTrue(all(s >= 5.0 for s in bob_starts))
            self.assertEqual(data["model"], "fake-model")

    async def test_no_participants(self):
        with tempfile.TemporaryDirectory() as td:
            session = Path(td) / "empty"
            session.mkdir()
            session.joinpath("manifest.json").write_text(
                json.dumps(
                    {
                        "session_id": "empty",
                        "start_epoch": 1.0,
                        "end_epoch": 2.0,
                        "participants": [],
                    }
                )
            )
            result = await transcribe_session(
                session,
                backend=FakeBackend(),
                vad=FakeVAD([]),
                language="en",
            )
            self.assertFalse(result.ok)
            self.assertIn("no participants", result.error or "")

    async def test_no_speech_detected(self):
        with tempfile.TemporaryDirectory() as td:
            session = _build_session(
                Path(td),
                participants=[("Alice", 1000.0, None)],
            )
            with patch.object(
                pipeline_mod,
                "decode_to_pcm16_mono",
                return_value=(b"\x00" * 32000, 16000),
            ):
                result = await transcribe_session(
                    session,
                    backend=FakeBackend(),
                    vad=FakeVAD([]),
                    language="en",
                )
            self.assertFalse(result.ok)
            self.assertIn("no speech", result.error or "")

    async def test_all_empty_responses(self):
        class EmptyBackend(FakeBackend):
            async def transcribe(self, wav_bytes: bytes, language: str) -> str:
                self.calls.append({})
                return ""

        with tempfile.TemporaryDirectory() as td:
            session = _build_session(
                Path(td),
                participants=[("Alice", 1000.0, None)],
            )
            with patch.object(
                pipeline_mod,
                "decode_to_pcm16_mono",
                return_value=(b"\x00" * 32000, 16000),
            ):
                result = await transcribe_session(
                    session,
                    backend=EmptyBackend(),
                    vad=FakeVAD([(0.0, 1.0)]),
                    language="en",
                )
            self.assertFalse(result.ok)
            self.assertIn("empty", result.error or "")

    async def test_writes_to_custom_output_dir(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            session = _build_session(
                root, participants=[("Alice", 1000.0, None)]
            )
            outdir = root / "out"
            with patch.object(
                pipeline_mod,
                "decode_to_pcm16_mono",
                return_value=(b"\x00" * 32000, 16000),
            ):
                result = await transcribe_session(
                    session,
                    backend=FakeBackend(),
                    vad=FakeVAD([(0.0, 1.0)]),
                    language="en",
                    output_dir=outdir,
                )
            self.assertTrue(result.ok)
            self.assertTrue((outdir / "transcript.json").exists())
            self.assertTrue((outdir / "transcript.txt").exists())
            # session dir must be untouched
            self.assertFalse((session / "transcript.json").exists())

    async def test_asr_runs_before_next_participant_decode(self):
        events: list[str] = []

        def fake_decode(path):
            events.append(f"decode:{path.name}")
            return b"\x00" * 32000, 16000

        with tempfile.TemporaryDirectory() as td:
            session = _build_session(
                Path(td),
                participants=[
                    ("Alice", 1000.0, None),
                    ("Bob", 1005.0, None),
                ],
            )
            backend = _EventRecordingBackend(events)
            vad = FakeVAD([(0.0, 1.0)])
            with patch.object(
                pipeline_mod,
                "decode_to_pcm16_mono",
                side_effect=fake_decode,
            ):
                result = await transcribe_session(
                    session,
                    backend=backend,
                    vad=vad,
                    language="en",
                )

        self.assertTrue(result.ok)
        self.assertLess(
            events.index("asr"),
            events.index("decode:Bob_aaaa.opus"),
            f"ASR should run before second participant decode, got {events}",
        )

    async def test_participant_no_speech_skipped_later_succeeds(self):
        with tempfile.TemporaryDirectory() as td:
            session = _build_session(
                Path(td),
                participants=[
                    ("Alice", 1000.0, None),
                    ("Bob", 1005.0, None),
                ],
            )
            backend = FakeBackend()
            vad = _SequenceVAD([[], [(0.0, 1.0)]])
            with patch.object(
                pipeline_mod,
                "decode_to_pcm16_mono",
                return_value=(b"\x00" * 32000, 16000),
            ):
                result = await transcribe_session(
                    session,
                    backend=backend,
                    vad=vad,
                    language="en",
                )

            self.assertTrue(result.ok, msg=f"err={result.error}")
            self.assertEqual(result.num_segments, 1)
            self.assertEqual(len(backend.calls), 1)
            data = json.loads((session / "transcript.json").read_text())
            self.assertEqual(data["segments"][0]["speaker"], "Bob")

    async def test_preparation_failure_fails_session(self):
        with tempfile.TemporaryDirectory() as td:
            session = _build_session(
                Path(td),
                participants=[("Alice", 1000.0, None)],
            )
            with patch.object(
                pipeline_mod,
                "decode_to_pcm16_mono",
                side_effect=RuntimeError("decode crash"),
            ):
                result = await transcribe_session(
                    session,
                    backend=FakeBackend(),
                    vad=FakeVAD([]),
                    language="en",
                )

            self.assertFalse(result.ok)
            self.assertIn("Alice", result.error or "")


if __name__ == "__main__":
    unittest.main()
