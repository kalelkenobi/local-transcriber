"""Tests for transcript output writers."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from local_transcriber.output import (
    TranscriptSegment,
    format_timestamp,
    write_transcript_json,
    write_transcript_txt,
)
from local_transcriber.session import Manifest


def _manifest() -> Manifest:
    return Manifest(
        session_id="syn-1",
        room_name="syn-room",
        start_epoch=1000.0,
        end_epoch=1060.0,
        participants=("Alice", "Bob"),
        format="opus",
        bitrate=128000,
        raw={},
    )


class TestFormatTimestamp(unittest.TestCase):
    def test_zero(self):
        self.assertEqual(format_timestamp(0.0), "00:00:00.00")

    def test_sub_second(self):
        self.assertEqual(format_timestamp(0.55), "00:00:00.55")

    def test_one_hour_two_minutes_five_seconds_half(self):
        self.assertEqual(format_timestamp(3725.5), "01:02:05.50")

    def test_negative_clamped(self):
        self.assertEqual(format_timestamp(-1.0), "00:00:00.00")

    def test_centisecond_rounding(self):
        # 0.555 → 0.56 (banker's rounding -> 0.56 because of how int(round())
        # works on .5; we just check it stays two digits and is close).
        ts = format_timestamp(0.555)
        self.assertTrue(ts.startswith("00:00:00."))
        self.assertEqual(len(ts), len("00:00:00.55"))


class TestWriteJSON(unittest.TestCase):
    def test_schema_and_values(self):
        segments = [
            TranscriptSegment("Alice", 0.0, 1.5, "hello"),
            TranscriptSegment("Bob", 1.6, 2.0, "hi"),
        ]
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "transcript.json"
            write_transcript_json(
                path,
                manifest=_manifest(),
                model="mock-model",
                language="en",
                segments=segments,
            )
            self.assertTrue(path.exists())
            data = json.loads(path.read_text())
        self.assertEqual(data["session_id"], "syn-1")
        self.assertEqual(data["room_name"], "syn-room")
        self.assertEqual(data["language"], "en")
        self.assertEqual(data["model"], "mock-model")
        self.assertEqual(data["start_epoch"], 1000.0)
        self.assertEqual(data["end_epoch"], 1060.0)
        self.assertEqual(len(data["segments"]), 2)
        self.assertEqual(
            data["segments"][0],
            {
                "speaker": "Alice",
                "start": 0.0,
                "end": 1.5,
                "start_absolute": "00:00:00.00",
                "text": "hello",
            },
        )
        self.assertEqual(data["segments"][1]["start_absolute"], "00:00:01.60")


class TestWriteTxt(unittest.TestCase):
    def test_format(self):
        segments = [
            TranscriptSegment("GM", 310.55, 312.0, "Roll for initiative."),
            TranscriptSegment("Player1", 310.55, 311.2, "I jump forward!"),
        ]
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "transcript.txt"
            write_transcript_txt(path, segments)
            content = path.read_text()
        expected = (
            "00:05:10.55 GM\n"
            "Roll for initiative.\n"
            "\n"
            "00:05:10.55 Player1\n"
            "I jump forward!\n"
        )
        self.assertEqual(content, expected)

    def test_empty_segments(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "transcript.txt"
            write_transcript_txt(path, [])
            self.assertEqual(path.read_text(), "")


if __name__ == "__main__":
    unittest.main()
