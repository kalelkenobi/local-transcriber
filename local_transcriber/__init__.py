"""local-transcriber — CLI for transcribing multi-participant recordings via an external ASR API."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("local-transcriber")
except PackageNotFoundError:  # pragma: no cover - not installed
    __version__ = "0.0.0"

__all__ = ["__version__"]
