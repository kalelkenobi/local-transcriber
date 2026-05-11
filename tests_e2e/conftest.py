"""E2E test fixtures.

Detects an available container runtime (`container`/`docker`/`podman`),
builds or reuses the `local-transcriber` image, and starts a mock ASR
server that the container can reach over the host network. Skips the
whole suite when no runtime is available.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path

import pytest

from tests_e2e.mock_asr.server import MockASRServer, start_mock_asr

REPO_ROOT = Path(__file__).resolve().parent.parent
CONTAINERFILE = REPO_ROOT / "ContainerFile"
EXAMPLE_RECORDING = REPO_ROOT / "example_recording"
DEFAULT_IMAGE_TAG = "local-transcriber:e2e"


def _detect_runtime() -> str | None:
    """Pick a container runtime, honoring $E2E_RUNTIME if set."""
    explicit = os.environ.get("E2E_RUNTIME")
    if explicit:
        return explicit if shutil.which(explicit) else None
    for candidate in ("container", "docker", "podman"):
        if shutil.which(candidate):
            return candidate
    return None


@pytest.fixture(scope="session")
def runtime() -> str:
    rt = _detect_runtime()
    if rt is None:
        pytest.skip("No container runtime (container/docker/podman) available")
    return rt


@pytest.fixture(scope="session")
def container_image(runtime: str) -> str:
    """Reuse $LOCAL_TRANSCRIBER_IMAGE if set, else build from ContainerFile."""
    prebuilt = os.environ.get("LOCAL_TRANSCRIBER_IMAGE")
    if prebuilt:
        return prebuilt

    image = DEFAULT_IMAGE_TAG
    cmd = [
        runtime,
        "build",
        "-f", str(CONTAINERFILE),
        "-t", image,
        str(REPO_ROOT),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        pytest.fail(
            f"Image build failed (exit {result.returncode}):\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
    return image


@pytest.fixture
def mock_asr() -> Iterator[tuple[str, int, MockASRServer]]:
    """Run a mock OpenAI-compatible ASR server in-process."""
    with start_mock_asr("0.0.0.0", 0) as bound:
        yield bound


def _container_host_address(runtime: str) -> tuple[str, list[str]]:
    """Return (host_addr, extra_run_args) for reaching the host from a container.

    - docker/podman: rely on `host.docker.internal` with the host-gateway
      add-host shortcut (works on Linux Docker 20.10+, Podman 3.0+, and is
      already present on Docker Desktop for Mac/Windows where the flag is
      a harmless no-op).
    - container (Apple): Apple's `container` CLI does not expose `--add-host`
      and does not auto-resolve `host.docker.internal`. The container's
      default gateway is the host, typically `192.168.64.1` for the vmnet
      bridge. Override via `E2E_HOST_ADDR` if your setup differs.
    """
    if runtime in ("docker", "podman"):
        return (
            "host.docker.internal",
            ["--add-host=host.docker.internal:host-gateway"],
        )
    if runtime == "container":
        host = os.environ.get("E2E_HOST_ADDR", "192.168.64.1")
        return (host, [])
    return ("host.docker.internal", [])
