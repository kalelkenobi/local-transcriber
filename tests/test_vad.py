"""Tests for the Silero VAD wrapper.

Focuses on the state machine that converts per-frame probabilities into
speech segments. The state machine is the part most likely to regress
during refactors and the part where the previous implementation had bugs.
A small end-to-end smoke test exercises the real ONNX model on synthetic
input and just checks that it returns a list.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from local_transcriber.vad import SileroVAD


REPO_ROOT = Path(__file__).resolve().parent.parent
SILERO_PATH = REPO_ROOT / "models" / "silero_vad.onnx"


@unittest.skipUnless(SILERO_PATH.exists(), "Silero VAD ONNX model not present")
class TestStateMachine(unittest.TestCase):
    """Drive `_probs_to_segments` with synthetic probability arrays."""

    def setUp(self):
        # Pick durations chosen so frame math is exact:
        # frame = 32 ms → 8 frames = 256 ms (≥ 250 ms speech threshold)
        # 16 frames = 512 ms (≥ 500 ms silence threshold)
        self.vad = SileroVAD(
            SILERO_PATH,
            threshold=0.5,
            min_speech_ms=250,
            min_silence_ms=500,
            speech_pad_ms=0,
        )

    def _segs(self, probs: list[float]):
        return self.vad._probs_to_segments(np.array(probs, dtype=np.float32))

    def test_empty_probs(self):
        self.assertEqual(self._segs([]), [])

    def test_all_silence(self):
        self.assertEqual(self._segs([0.0] * 40), [])

    def test_continuous_speech_single_segment(self):
        # 40 frames of speech → one segment from frame 0 to frame 40
        result = self._segs([0.9] * 40)
        self.assertEqual(len(result), 1)
        start_s, end_s = result[0]
        self.assertEqual(start_s, 0.0)
        # 40 frames * 512 samples / 16000 Hz = 1.28 s
        self.assertAlmostEqual(end_s, 40 * 512 / 16000, places=3)

    def test_short_burst_below_min_speech_is_discarded(self):
        # 4 frames of speech (~128 ms) < 250 ms threshold → no segment
        probs = [0.0] * 10 + [0.9] * 4 + [0.0] * 30
        self.assertEqual(self._segs(probs), [])

    def test_brief_dip_does_not_split_segment(self):
        # 20 frames speech, 5 frames silence (~160 ms < 500 ms), 20 frames speech
        # The dip is shorter than min_silence so it must NOT split the segment.
        probs = [0.9] * 20 + [0.0] * 5 + [0.9] * 20
        result = self._segs(probs)
        self.assertEqual(len(result), 1)

    def test_long_silence_splits_segments(self):
        # 20 speech, 20 silence (~640 ms > 500 ms), 20 speech → two segments
        probs = [0.9] * 20 + [0.0] * 20 + [0.9] * 20
        result = self._segs(probs)
        self.assertEqual(len(result), 2)
        self.assertLess(result[0][1], result[1][0])

    def test_segment_starts_at_first_above_threshold_frame(self):
        leading_silence = 12
        probs = [0.0] * leading_silence + [0.9] * 20
        result = self._segs(probs)
        self.assertEqual(len(result), 1)
        expected_start = leading_silence * 512 / 16000
        self.assertAlmostEqual(result[0][0], expected_start, places=3)

    def test_padding_extends_bounds(self):
        vad = SileroVAD(
            SILERO_PATH,
            threshold=0.5,
            min_speech_ms=250,
            min_silence_ms=500,
            speech_pad_ms=100,  # ~3 frames
        )
        probs = np.array([0.0] * 10 + [0.9] * 20 + [0.0] * 20, dtype=np.float32)
        result = vad._probs_to_segments(probs)
        self.assertEqual(len(result), 1)
        # Without padding the segment would start at frame 10. With 100 ms
        # padding (≈3 frames) it should start earlier.
        unpadded_start = 10 * 512 / 16000
        self.assertLess(result[0][0], unpadded_start)


@unittest.skipUnless(SILERO_PATH.exists(), "Silero VAD ONNX model not present")
class TestEndToEnd(unittest.TestCase):
    """Smoke test: real ONNX session against synthetic PCM."""

    def test_empty_pcm(self):
        vad = SileroVAD(SILERO_PATH)
        self.assertEqual(vad.iter_speech_segments(b""), [])

    def test_too_short_pcm(self):
        vad = SileroVAD(SILERO_PATH)
        # 10 samples of int16 → 20 bytes, less than one frame (512 samples)
        self.assertEqual(vad.iter_speech_segments(b"\x00" * 20), [])

    def test_pure_silence(self):
        vad = SileroVAD(SILERO_PATH)
        # 1 second of silence
        pcm = (np.zeros(16000, dtype=np.int16)).tobytes()
        result = vad.iter_speech_segments(pcm)
        self.assertEqual(result, [])

    def test_returns_list(self):
        vad = SileroVAD(SILERO_PATH)
        # 1 second of low-amplitude white noise — output may be empty or
        # may contain segments; we only assert the contract.
        rng = np.random.default_rng(42)
        samples = (rng.standard_normal(16000) * 500).astype(np.int16)
        result = vad.iter_speech_segments(samples.tobytes())
        self.assertIsInstance(result, list)
        for start_s, end_s in result:
            self.assertGreaterEqual(start_s, 0.0)
            self.assertGreater(end_s, start_s)


class TestModelNotFound(unittest.TestCase):
    def test_raises_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            SileroVAD(Path("/nonexistent/silero.onnx"))


if __name__ == "__main__":
    unittest.main()
