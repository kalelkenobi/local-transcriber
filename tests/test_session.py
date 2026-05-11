"""Tests for session manifest + participant loaders."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from local_transcriber.session import (
    Manifest,
    Participant,
    iter_sessions,
    load_manifest,
    load_participant,
    load_session,
    participant_offset,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLE = REPO_ROOT / "tests_e2e/fixtures"


class TestLoadExampleRecording(unittest.TestCase):
    """Drive the loaders against the committed fixture."""

    def test_load_manifest_fields(self):
        manifest = load_manifest(EXAMPLE)
        self.assertIsInstance(manifest, Manifest)
        self.assertEqual(manifest.session_id, "2026-05-11_21-47-37")
        self.assertEqual(
            manifest.room_name,
            "1xUzpG6EH3kAMjBbWfhjur1vLCGhQTW0",
        )
        self.assertEqual(manifest.participants, ("Kal",))
        self.assertEqual(manifest.format, "opus")
        self.assertAlmostEqual(manifest.start_epoch, 1778528857.7341638, places=4)
        self.assertIsNotNone(manifest.end_epoch)

    def test_load_session_resolves_participant(self):
        manifest, participants = load_session(EXAMPLE)
        self.assertEqual(len(participants), 1)
        p = participants[0]
        self.assertEqual(p.identity, "Kal")
        self.assertTrue(p.audio_path.exists())
        self.assertEqual(p.audio_path.suffix, ".opus")
        self.assertEqual(p.audio_format, "opus")
        self.assertEqual(p.sample_rate, 48000)
        self.assertEqual(p.channels, 1)
        # start_receiving is present in the fixture
        self.assertIsNotNone(p.start_receiving_epoch)

    def test_participant_offset_uses_start_receiving_when_later(self):
        manifest, participants = load_session(EXAMPLE)
        p = participants[0]
        offset = participant_offset(manifest, p)
        self.assertGreater(offset, 0.0)
        self.assertLess(offset, 1.0)
        expected = p.start_receiving_epoch - manifest.start_epoch
        self.assertAlmostEqual(offset, expected, places=4)


class TestSyntheticSessions(unittest.TestCase):
    """Loader behavior on synthetic in-memory sessions."""

    def _make_session(
        self,
        tmpdir: Path,
        *,
        manifest_start: float,
        participants: list[tuple[str, float, float | None]],
    ) -> Path:
        """Create a minimal session dir.

        participants: list of (identity, start_epoch, start_receiving_or_None).
        """
        session = tmpdir / "session"
        session.mkdir()
        manifest = {
            "session_id": "syn-1",
            "room_name": "syn-room",
            "start_epoch": manifest_start,
            "end_epoch": manifest_start + 30.0,
            "participants": [p[0] for p in participants],
            "format": "opus",
            "bitrate": 128000,
        }
        (session / "manifest.json").write_text(json.dumps(manifest))
        for identity, start_epoch, start_recv in participants:
            pdir = session / f"{identity}_aaaa"
            pdir.mkdir()
            audio = pdir / f"{identity}_aaaa.opus"
            audio.write_bytes(b"\x00" * 4)  # placeholder; not decoded here
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

    def test_offset_zero_when_audio_matches_manifest(self):
        with tempfile.TemporaryDirectory() as td:
            session = self._make_session(
                Path(td),
                manifest_start=1000.0,
                participants=[("Alice", 1000.0, None)],
            )
            manifest, participants = load_session(session)
            self.assertEqual(
                participant_offset(manifest, participants[0]), 0.0
            )

    def test_offset_positive_when_participant_joins_late(self):
        with tempfile.TemporaryDirectory() as td:
            session = self._make_session(
                Path(td),
                manifest_start=1000.0,
                participants=[("Alice", 1005.0, None)],
            )
            manifest, participants = load_session(session)
            self.assertAlmostEqual(
                participant_offset(manifest, participants[0]), 5.0, places=4
            )

    def test_offset_clamped_when_participant_earlier(self):
        with tempfile.TemporaryDirectory() as td:
            session = self._make_session(
                Path(td),
                manifest_start=1000.0,
                participants=[("Alice", 999.0, None)],
            )
            manifest, participants = load_session(session)
            self.assertEqual(
                participant_offset(manifest, participants[0]), 0.0
            )

    def test_start_receiving_overrides_start_epoch(self):
        with tempfile.TemporaryDirectory() as td:
            session = self._make_session(
                Path(td),
                manifest_start=1000.0,
                participants=[("Alice", 1000.0, 1002.5)],
            )
            manifest, participants = load_session(session)
            self.assertAlmostEqual(
                participant_offset(manifest, participants[0]), 2.5, places=4
            )

    def test_missing_participant_dir_is_skipped(self):
        with tempfile.TemporaryDirectory() as td:
            session = self._make_session(
                Path(td),
                manifest_start=1000.0,
                participants=[("Alice", 1000.0, None)],
            )
            # Remove the Alice dir to simulate missing data.
            for child in session.iterdir():
                if child.is_dir():
                    for sub in child.iterdir():
                        sub.unlink()
                    child.rmdir()
            _, participants = load_session(session)
            self.assertEqual(participants, [])


class TestIterSessions(unittest.TestCase):
    def test_single_session_dir(self):
        sessions = list(iter_sessions(EXAMPLE, recursive=False))
        self.assertEqual(sessions, [EXAMPLE])

    def test_parent_dir_without_recursive_yields_nothing(self):
        # tests_e2e/fixtures's parent has no manifest.json — without recursive
        # the iterator yields nothing.
        sessions = list(iter_sessions(EXAMPLE.parent, recursive=False))
        self.assertNotIn(EXAMPLE.parent, sessions)

    def test_recursive_finds_children(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for name in ("s1", "s2", "not-a-session"):
                d = root / name
                d.mkdir()
                if name != "not-a-session":
                    (d / "manifest.json").write_text(
                        json.dumps(
                            {
                                "session_id": name,
                                "start_epoch": 1.0,
                                "end_epoch": 2.0,
                                "participants": [],
                            }
                        )
                    )
            sessions = list(iter_sessions(root, recursive=True))
            self.assertEqual(
                sorted(s.name for s in sessions), ["s1", "s2"]
            )

    def test_nonexistent_path_yields_nothing(self):
        sessions = list(
            iter_sessions(Path("/this/does/not/exist"), recursive=True)
        )
        self.assertEqual(sessions, [])


if __name__ == "__main__":
    unittest.main()
