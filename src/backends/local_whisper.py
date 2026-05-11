"""
Local ASR backend using faster-whisper (CTranslate2).

Runs on CPU by default; supports GPU if available.
"""

import io
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Lazy-loaded to avoid import cost when using vllm backend
_model_cache: dict[tuple[str, str], Any] = {}


class LocalWhisperBackend:
    """ASR backend using faster-whisper for in-process transcription."""

    def __init__(
        self,
        model_size: str = "large-v3",
        cache_dir: str | None = None,
        beam_size: int = 5,
    ):
        self._model_size = model_size
        self._cache_dir = cache_dir or str(
            Path.home() / ".cache" / "livekit-recorder" / "whisper"
        )
        self._beam_size = beam_size
        self._model = None

    def _get_model(self):
        """Lazy-load the faster-whisper model (cached across instances)."""
        cache_key = (self._model_size, self._cache_dir)
        if cache_key not in _model_cache:
            from faster_whisper import WhisperModel

            Path(self._cache_dir).mkdir(parents=True, exist_ok=True)

            logger.info(
                "Loading faster-whisper model '%s' using cache dir '%s'...",
                self._model_size,
                self._cache_dir,
            )
            _model_cache[cache_key] = WhisperModel(
                self._model_size,
                device="auto",
                compute_type="int8",
                download_root=self._cache_dir,
            )
            logger.info("Model '%s' loaded.", self._model_size)

        return _model_cache[cache_key]

    async def transcribe(self, wav_bytes: bytes, language: str) -> str:
        """
        Transcribe WAV audio bytes using faster-whisper.

        Args:
            wav_bytes: Complete WAV file bytes (with header).
            language: Language code (e.g. "en", "fr").

        Returns:
            Transcribed text.
        """
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._transcribe_sync, wav_bytes, language)

    def _transcribe_sync(self, wav_bytes: bytes, language: str) -> str:
        """Synchronous transcription."""
        model = self._get_model()

        audio_file = io.BytesIO(wav_bytes)
        segments, _info = model.transcribe(
            audio_file,
            language=language,
            beam_size=self._beam_size,
            vad_filter=False,  # We already ran VAD externally
        )

        text_parts = []
        for segment in segments:
            text_parts.append(segment.text.strip())

        return " ".join(text_parts)

    async def close(self) -> None:
        """No-op for local backend (model stays cached)."""
        pass
