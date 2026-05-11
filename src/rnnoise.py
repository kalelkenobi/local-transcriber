"""
RNNoise denoiser — ctypes wrapper around librnnoise.so.

Provides neural noise suppression for speech audio. Processes 48kHz
16-bit mono PCM in 10ms frames (480 samples).
"""

from __future__ import annotations

import ctypes
import ctypes.util
import logging
import struct
from pathlib import Path

logger = logging.getLogger(__name__)

# Frame size: 480 samples = 10ms at 48kHz
FRAME_SIZE = 480
FRAME_BYTES = FRAME_SIZE * 2  # 16-bit = 2 bytes per sample

# Library search paths (container path first, then system)
_LIB_PATHS = [
    "/usr/local/lib/librnnoise.so",
    "/usr/local/lib/librnnoise.so.0",
]


def _load_library() -> ctypes.CDLL | None:
    """Load librnnoise shared library, return None if unavailable."""
    for path in _LIB_PATHS:
        if Path(path).exists():
            try:
                return ctypes.CDLL(path)
            except OSError:
                continue

    # Try system-wide search
    name = ctypes.util.find_library("rnnoise")
    if name:
        try:
            return ctypes.CDLL(name)
        except OSError:
            pass

    return None


_lib = _load_library()


def is_available() -> bool:
    """Return True if librnnoise is available on this system."""
    return _lib is not None


class RNNoiseDenoiser:
    """
    Stateful RNNoise denoiser instance.

    Each instance maintains internal state for streaming denoising.
    Create one per audio channel/participant.
    """

    def __init__(self) -> None:
        if _lib is None:
            raise RuntimeError(
                "librnnoise.so not found. Install RNNoise or run in the "
                "container image which includes it."
            )

        # Set up function signatures
        _lib.rnnoise_create.restype = ctypes.c_void_p
        _lib.rnnoise_create.argtypes = [ctypes.c_void_p]
        _lib.rnnoise_destroy.argtypes = [ctypes.c_void_p]
        _lib.rnnoise_process_frame.restype = ctypes.c_float
        _lib.rnnoise_process_frame.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
        ]

        self._state = _lib.rnnoise_create(None)
        if not self._state:
            raise RuntimeError("rnnoise_create returned NULL")

    def process_frame(self, pcm_int16: bytes) -> tuple[float, bytes]:
        """
        Denoise a single 10ms frame of audio.

        Args:
            pcm_int16: Exactly 960 bytes (480 samples of 16-bit PCM).

        Returns:
            Tuple of (voice_activity_probability, denoised_pcm_int16).
            VAD probability is 0.0–1.0.
        """
        if len(pcm_int16) != FRAME_BYTES:
            raise ValueError(
                f"Expected {FRAME_BYTES} bytes, got {len(pcm_int16)}"
            )

        # Unpack int16 → float32 (RNNoise expects float scaled to int16 range)
        samples = struct.unpack(f"<{FRAME_SIZE}h", pcm_int16)
        in_buf = (ctypes.c_float * FRAME_SIZE)(*[float(s) for s in samples])
        out_buf = (ctypes.c_float * FRAME_SIZE)()

        vad_prob = _lib.rnnoise_process_frame(self._state, out_buf, in_buf)

        # Pack float → int16
        out_samples = [
            max(-32768, min(32767, int(out_buf[i]))) for i in range(FRAME_SIZE)
        ]
        out_bytes = struct.pack(f"<{FRAME_SIZE}h", *out_samples)

        return float(vad_prob), out_bytes

    def close(self) -> None:
        """Release the native state."""
        if self._state:
            _lib.rnnoise_destroy(self._state)
            self._state = None

    def __del__(self) -> None:
        self.close()


def denoise_pcm(pcm_data: bytes, sample_rate: int = 48000) -> bytes:
    """
    Denoise an entire PCM buffer using RNNoise.

    Args:
        pcm_data: Raw 16-bit mono PCM bytes at the given sample rate.
        sample_rate: Must be 48000 (RNNoise's native rate).

    Returns:
        Denoised PCM bytes of the same length.

    Raises:
        RuntimeError: If librnnoise is not available.
        ValueError: If sample_rate is not 48000.
    """
    if sample_rate != 48000:
        raise ValueError(
            f"RNNoise requires 48kHz audio, got {sample_rate}Hz. "
            "Resample before calling denoise_pcm()."
        )

    if not is_available():
        logger.warning(
            "librnnoise not available, skipping denoising"
        )
        return pcm_data

    denoiser = RNNoiseDenoiser()
    output = bytearray()

    # Process full frames
    num_frames = len(pcm_data) // FRAME_BYTES
    for i in range(num_frames):
        frame = pcm_data[i * FRAME_BYTES: (i + 1) * FRAME_BYTES]
        _, denoised_frame = denoiser.process_frame(frame)
        output.extend(denoised_frame)

    # Handle remainder (pad with zeros, process, then truncate)
    remainder = len(pcm_data) % FRAME_BYTES
    if remainder:
        padded = pcm_data[num_frames * FRAME_BYTES:] + b"\x00" * (FRAME_BYTES - remainder)
        _, denoised_frame = denoiser.process_frame(padded)
        output.extend(denoised_frame[:remainder])

    denoiser.close()
    return bytes(output)
