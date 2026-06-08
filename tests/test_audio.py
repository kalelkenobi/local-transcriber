"""Tests for audio helper subprocess wrappers."""

from __future__ import annotations

import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

from local_transcriber.audio import (
    AudioDecodeError,
    decode_range_to_pcm16_mono,
    probe_audio_duration,
)


class TestAudioHelpers(unittest.TestCase):
    def test_probe_audio_duration_success(self):
        completed = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=b"12.345\n", stderr=b""
        )
        with patch("local_transcriber.audio.subprocess.run", return_value=completed) as run:
            duration = probe_audio_duration(Path("sample.opus"))

        self.assertEqual(duration, 12.345)
        cmd = run.call_args.args[0]
        self.assertIn("ffprobe", cmd[0])
        self.assertIn("format=duration", cmd)

    def test_probe_audio_duration_invalid_output(self):
        completed = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=b"n/a\n", stderr=b""
        )
        with patch("local_transcriber.audio.subprocess.run", return_value=completed):
            with self.assertRaises(AudioDecodeError):
                probe_audio_duration(Path("sample.opus"))

    def test_probe_audio_duration_negative_output(self):
        completed = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=b"-1.0\n", stderr=b""
        )
        with patch("local_transcriber.audio.subprocess.run", return_value=completed):
            with self.assertRaisesRegex(AudioDecodeError, "negative duration"):
                probe_audio_duration(Path("sample.opus"))

    def test_probe_audio_duration_missing_ffprobe(self):
        with patch(
            "local_transcriber.audio.subprocess.run",
            side_effect=FileNotFoundError,
        ):
            with self.assertRaisesRegex(AudioDecodeError, "ffprobe not found"):
                probe_audio_duration(Path("sample.opus"))

    def test_probe_audio_duration_failure_includes_stderr(self):
        completed = subprocess.CompletedProcess(
            args=[], returncode=1, stdout=b"", stderr=b"bad input"
        )
        with patch("local_transcriber.audio.subprocess.run", return_value=completed):
            with self.assertRaisesRegex(AudioDecodeError, "bad input"):
                probe_audio_duration(Path("sample.opus"))

    def test_decode_range_to_pcm16_mono_success(self):
        completed = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=b"pcm", stderr=b""
        )
        with patch("local_transcriber.audio.subprocess.run", return_value=completed) as run:
            pcm, sr = decode_range_to_pcm16_mono(
                Path("sample.opus"), 1.25, 2.5
            )

        self.assertEqual(pcm, b"pcm")
        self.assertEqual(sr, 16000)
        cmd = run.call_args.args[0]
        self.assertEqual(cmd[0], "ffmpeg")
        self.assertIn("-nostdin", cmd)
        self.assertIn("-ss", cmd)
        self.assertIn("1.250", cmd)
        self.assertIn("-t", cmd)
        self.assertIn("2.500", cmd)
        self.assertIn("-ac", cmd)
        self.assertIn("1", cmd)
        self.assertIn("-ar", cmd)
        self.assertIn("16000", cmd)
        self.assertIn("s16le", cmd)

    def test_decode_range_to_pcm16_mono_failure_includes_range(self):
        completed = subprocess.CompletedProcess(
            args=[], returncode=1, stdout=b"", stderr=b"decode failed"
        )
        with patch("local_transcriber.audio.subprocess.run", return_value=completed):
            with self.assertRaisesRegex(AudioDecodeError, "2.000-5.000"):
                decode_range_to_pcm16_mono(Path("sample.opus"), 2.0, 3.0)


if __name__ == "__main__":
    unittest.main()
