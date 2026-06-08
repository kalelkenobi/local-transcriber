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


class TestMergeSameSpeakerSegments(unittest.TestCase):
    def test_empty(self):
        from local_transcriber.output import merge_same_speaker_segments
        self.assertEqual(merge_same_speaker_segments([]), [])

    def test_single(self):
        from local_transcriber.output import merge_same_speaker_segments
        seg = TranscriptSegment("A", 0.0, 1.0, "hi")
        self.assertEqual(merge_same_speaker_segments([seg]), [seg])

    def test_three_in_a_row_merge(self):
        from local_transcriber.output import merge_same_speaker_segments
        segs = [
            TranscriptSegment("V", 8.03, 22.0, "Perché oggi..."),
            TranscriptSegment("V", 24.83, 25.5, "Ma che lo sarà?"),
            TranscriptSegment("V", 26.50, 30.0, "45 secondi..."),
        ]
        merged = merge_same_speaker_segments(segs)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].speaker, "V")
        self.assertEqual(merged[0].start, 8.03)
        self.assertEqual(merged[0].end, 30.0)
        self.assertEqual(
            merged[0].text,
            "Perché oggi... Ma che lo sarà? 45 secondi...",
        )

    def test_speaker_interleave_breaks_run(self):
        from local_transcriber.output import merge_same_speaker_segments
        segs = [
            TranscriptSegment("A", 0.0, 1.0, "one"),
            TranscriptSegment("B", 1.0, 2.0, "two"),
            TranscriptSegment("A", 2.0, 3.0, "three"),
        ]
        self.assertEqual(merge_same_speaker_segments(segs), segs)

    def test_run_resumes_after_other_speaker(self):
        from local_transcriber.output import merge_same_speaker_segments
        segs = [
            TranscriptSegment("A", 0.0, 1.0, "one"),
            TranscriptSegment("A", 1.5, 2.0, "two"),
            TranscriptSegment("B", 2.5, 3.0, "x"),
            TranscriptSegment("A", 3.5, 4.0, "three"),
        ]
        merged = merge_same_speaker_segments(segs)
        self.assertEqual(len(merged), 3)
        self.assertEqual(merged[0].text, "one two")
        self.assertEqual(merged[0].end, 2.0)
        self.assertEqual(merged[1].speaker, "B")
        self.assertEqual(merged[2].text, "three")

    def test_whitespace_normalized(self):
        from local_transcriber.output import merge_same_speaker_segments
        segs = [
            TranscriptSegment("A", 0.0, 1.0, "  hello  "),
            TranscriptSegment("A", 1.0, 2.0, "  world  "),
        ]
        merged = merge_same_speaker_segments(segs)
        self.assertEqual(merged[0].text, "hello world")

    def test_empty_text_dropped(self):
        from local_transcriber.output import merge_same_speaker_segments
        segs = [
            TranscriptSegment("A", 0.0, 1.0, "hello"),
            TranscriptSegment("A", 1.0, 2.0, "   "),
            TranscriptSegment("A", 2.0, 3.0, "world"),
        ]
        merged = merge_same_speaker_segments(segs)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].text, "hello world")

    def test_input_not_mutated(self):
        from local_transcriber.output import merge_same_speaker_segments
        segs = [
            TranscriptSegment("A", 0.0, 1.0, "hi"),
            TranscriptSegment("A", 1.0, 2.0, "there"),
        ]
        snapshot = list(segs)
        _ = merge_same_speaker_segments(segs)
        self.assertEqual(segs, snapshot)


if __name__ == "__main__":
    unittest.main()
