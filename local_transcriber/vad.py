"""Silero VAD v5 ONNX wrapper with a clean speech-segment state machine.

Operates on 16 kHz mono int16 PCM (matches the project's decode target).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_WINDOW = 512  # samples per frame at 16 kHz (32 ms)
_CONTEXT = 64  # context samples prepended to each frame
_STATE_SHAPE = (2, 1, 128)
_SAMPLE_RATE = 16000


class SileroVAD:
    """Silero VAD v5 runner.

    Run `iter_speech_segments(pcm)` to get a list of `(start_s, end_s)`
    tuples (relative to the start of `pcm`). The state machine emits a
    segment when speech has been continuous for at least `min_speech_ms`
    and closes it when silence has been continuous for at least
    `min_silence_ms`. Each segment is padded by `speech_pad_ms` on both
    sides, clamped to the buffer bounds.
    """

    def __init__(
        self,
        model_path: Path,
        threshold: float = 0.5,
        min_speech_ms: int = 250,
        min_silence_ms: int = 500,
        speech_pad_ms: int = 100,
    ) -> None:
        if not model_path.exists():
            raise FileNotFoundError(
                f"Silero VAD ONNX model not found at {model_path}"
            )
        # Lazy import so non-VAD code paths don't pay the onnxruntime cost.
        import onnxruntime as ort

        self._session = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
        )
        self._threshold = float(threshold)
        self._min_speech_ms = int(min_speech_ms)
        self._min_silence_ms = int(min_silence_ms)
        self._speech_pad_ms = int(speech_pad_ms)

    def iter_speech_segments(self, pcm: bytes) -> list[tuple[float, float]]:
        """Return speech segments found in pcm as (start_s, end_s) tuples."""
        if not pcm:
            return []
        audio_int16 = np.frombuffer(pcm, dtype=np.int16)
        if audio_int16.size < _WINDOW:
            return []
        audio_f32 = audio_int16.astype(np.float32) / 32768.0
        probs = self._frame_probabilities(audio_f32)
        return self._probs_to_segments(probs)

    def _frame_probabilities(self, audio_f32: np.ndarray) -> np.ndarray:
        """Run the model frame-by-frame and return per-frame speech probabilities."""
        num_frames = audio_f32.size // _WINDOW
        if num_frames == 0:
            return np.empty(0, dtype=np.float32)

        state = np.zeros(_STATE_SHAPE, dtype=np.float32)
        context = np.zeros(_CONTEXT, dtype=np.float32)
        sr_input = np.array(_SAMPLE_RATE, dtype=np.int64)
        probs = np.empty(num_frames, dtype=np.float32)

        for i in range(num_frames):
            chunk = audio_f32[i * _WINDOW : (i + 1) * _WINDOW]
            x = np.concatenate([context, chunk]).reshape(1, -1).astype(np.float32)
            outputs = self._session.run(
                None,
                {"input": x, "state": state, "sr": sr_input},
            )
            probs[i] = float(outputs[0][0][0])
            state = outputs[1]
            context = chunk[-_CONTEXT:]
        return probs

    def _probs_to_segments(self, probs: np.ndarray) -> list[tuple[float, float]]:
        """Convert per-frame probabilities into time-range speech segments.

        State machine over frames:
          - silence: when prob >= threshold for >= min_speech_frames in a row,
            open a segment at the first crossing frame.
          - speech: when prob <  threshold for >= min_silence_frames in a row,
            close the segment at the first below-threshold frame.
        Pads each segment by pad_frames on both sides, clamped to bounds.
        """
        if probs.size == 0:
            return []

        ms_per_frame = 1000.0 * _WINDOW / _SAMPLE_RATE  # 32 ms
        min_speech_frames = max(1, int(round(self._min_speech_ms / ms_per_frame)))
        min_silence_frames = max(1, int(round(self._min_silence_ms / ms_per_frame)))
        pad_frames = max(0, int(round(self._speech_pad_ms / ms_per_frame)))

        segments: list[tuple[int, int]] = []  # half-open [start_frame, end_frame)
        in_speech = False
        pending_start: int | None = None
        run_start: int | None = None  # start of current above/below run

        for i, p in enumerate(probs):
            above = bool(p >= self._threshold)
            if not in_speech:
                if above:
                    if run_start is None:
                        run_start = i
                    if i - run_start + 1 >= min_speech_frames:
                        in_speech = True
                        pending_start = run_start
                        run_start = None
                else:
                    run_start = None
            else:
                if not above:
                    if run_start is None:
                        run_start = i
                    if i - run_start + 1 >= min_silence_frames:
                        assert pending_start is not None
                        segments.append((pending_start, run_start))
                        in_speech = False
                        pending_start = None
                        run_start = None
                else:
                    run_start = None

        # Flush trailing speech to end-of-buffer.
        if in_speech and pending_start is not None:
            end_frame = run_start if run_start is not None else probs.size
            segments.append((pending_start, end_frame))

        result: list[tuple[float, float]] = []
        total_frames = probs.size
        for start_f, end_f in segments:
            start_f = max(0, start_f - pad_frames)
            end_f = min(total_frames, end_f + pad_frames)
            start_s = start_f * _WINDOW / _SAMPLE_RATE
            end_s = end_f * _WINDOW / _SAMPLE_RATE
            if end_s > start_s:
                result.append((round(start_s, 3), round(end_s, 3)))

        logger.debug(
            "VAD: %d segments from %d frames (%.2fs)",
            len(result), total_frames, total_frames * ms_per_frame / 1000.0,
        )
        return result
