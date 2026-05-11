"""
ASR backend interface and factory.

Backends transcribe a single audio segment (WAV bytes) and return text.
"""

import logging
from typing import Protocol

logger = logging.getLogger(__name__)


class ASRBackend(Protocol):
    """Protocol for ASR backends."""

    async def transcribe(self, wav_bytes: bytes, language: str) -> str:
        """Transcribe a WAV audio segment and return the text."""
        ...

    async def close(self) -> None:
        """Release resources."""
        ...


def create_backend(backend_type: str, **kwargs) -> ASRBackend:
    """
    Factory to create the appropriate ASR backend.

    Args:
        backend_type: "local" or "vllm"
        **kwargs: Backend-specific configuration.
    """
    if backend_type == "local":
        from .local_whisper import LocalWhisperBackend

        return LocalWhisperBackend(
            model_size=kwargs.get("model_size", "large-v3"),
            cache_dir=kwargs.get("cache_dir"),
            beam_size=kwargs.get("beam_size", 5),
        )
    elif backend_type == "vllm":
        from .vllm_client import VLLMBackend

        return VLLMBackend(
            base_url=kwargs.get("base_url", "http://localhost:8000"),
        )
    else:
        raise ValueError(f"Unknown backend type: {backend_type!r}. Use 'local' or 'vllm'.")
