"""
Entry point for the LiveKit Transcriber service.

Exposes an HTTP API to submit transcription jobs and retrieve results.
"""

import asyncio
import json
import logging
import os
import shutil
import sys
import tempfile
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from .jobs import Job, JobRegistry, JobStatus
from .pipeline import transcribe_session, transcribe_wav

# ---------------------------------------------------------------------------
# Configuration from environment
# ---------------------------------------------------------------------------
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8091"))
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
JOBS_DIR = Path(os.environ.get("JOBS_DIR", "/jobs"))

# ASR backend configuration
TRANSCRIBE_BACKEND = os.environ.get("TRANSCRIBE_BACKEND", "local")
TRANSCRIBE_URL = os.environ.get("TRANSCRIBE_URL", "http://localhost:8000")
TRANSCRIBE_MODEL_SIZE = os.environ.get("TRANSCRIBE_MODEL_SIZE", "large-v3")
WHISPER_CACHE_DIR = os.environ.get("WHISPER_CACHE_DIR", "/models/whisper")
try:
    TRANSCRIBE_BEAM_SIZE = max(1, int(os.environ.get("TRANSCRIBE_BEAM_SIZE", "5")))
except ValueError:
    TRANSCRIBE_BEAM_SIZE = 5
TRANSCRIBE_DENOISE = os.environ.get("TRANSCRIBE_DENOISE", "true").lower() in (
    "1", "true", "yes"
)
TRANSCRIBE_VAD_THRESHOLD = float(os.environ.get("TRANSCRIBE_VAD_THRESHOLD", "0.5"))

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
registry: JobRegistry | None = None


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global registry
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    registry = JobRegistry(JOBS_DIR)
    logger.info("Transcriber service starting")
    logger.info("Jobs dir: %s", JOBS_DIR)
    logger.info("Backend: %s", TRANSCRIBE_BACKEND)
    yield


app = FastAPI(title="LiveKit Transcriber", lifespan=lifespan)


def _get_backend_kwargs() -> dict:
    """Build backend kwargs from env config."""
    kwargs: dict = {}
    if TRANSCRIBE_BACKEND == "vllm":
        kwargs["base_url"] = TRANSCRIBE_URL
    elif TRANSCRIBE_BACKEND == "local":
        kwargs["model_size"] = TRANSCRIBE_MODEL_SIZE
        kwargs["cache_dir"] = WHISPER_CACHE_DIR
        kwargs["beam_size"] = TRANSCRIBE_BEAM_SIZE
    return kwargs


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.post("/transcribe")
async def submit_transcription(
    file: UploadFile = File(...),
    language: str = Form("en"),
    webhook_url: str | None = Form(None),
):
    """
    Submit a transcription job.

    Upload a ZIP (multi-participant session from livekit-recorder) or a single
    WAV file. Returns a job ID for polling status.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    filename = file.filename.lower()
    if filename.endswith(".zip"):
        source_type = "zip"
    elif filename.endswith(".wav"):
        source_type = "wav"
    else:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Upload a .zip or .wav file.",
        )

    # Create job
    job = registry.create_job(
        source_type=source_type,
        source_name=file.filename,
        language=language,
        webhook_url=webhook_url,
    )
    job_dir = registry.get_job_dir(job.job_id)

    # Save uploaded file
    upload_path = job_dir / file.filename
    content = await file.read()
    upload_path.write_bytes(content)

    # Start background processing
    asyncio.create_task(_process_job(job))

    return {"job_id": job.job_id, "status": "queued"}


@app.get("/transcribe/{job_id}")
async def get_job_status(job_id: str):
    """Get the status of a transcription job, including transcript if complete."""
    job = registry.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job.to_dict()


@app.get("/transcribe/{job_id}/transcript")
async def get_transcript_json(job_id: str):
    """Download transcript as JSON."""
    job = registry.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != JobStatus.COMPLETED:
        return JSONResponse(
            status_code=202,
            content={"status": job.status.value, "message": "Transcription not yet complete"},
        )

    transcript_path = registry.get_job_dir(job_id) / "transcript.json"
    if not transcript_path.exists():
        raise HTTPException(status_code=404, detail="Transcript file not found")

    with open(transcript_path) as f:
        return json.load(f)


@app.get("/transcribe/{job_id}/transcript.txt")
async def get_transcript_txt(job_id: str):
    """Download transcript as plain text."""
    job = registry.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != JobStatus.COMPLETED:
        return JSONResponse(
            status_code=202,
            content={"status": job.status.value, "message": "Transcription not yet complete"},
        )

    txt_path = registry.get_job_dir(job_id) / "transcript.txt"
    if not txt_path.exists():
        raise HTTPException(status_code=404, detail="Transcript text file not found")

    from fastapi.responses import PlainTextResponse

    return PlainTextResponse(txt_path.read_text())


@app.get("/jobs")
async def list_jobs(limit: int = 50):
    """List recent transcription jobs."""
    jobs = registry.list_jobs(limit=limit)
    return {"jobs": [j.to_dict() for j in jobs]}


# ---------------------------------------------------------------------------
# Background processing
# ---------------------------------------------------------------------------
async def _process_job(job: Job) -> None:
    """Run the transcription pipeline for a job."""
    job_dir = registry.get_job_dir(job.job_id)

    registry.update_status(job.job_id, JobStatus.PROCESSING)

    try:
        backend_kwargs = _get_backend_kwargs()

        if job.source_type == "zip":
            success = await _process_zip_job(job, job_dir, backend_kwargs)
        elif job.source_type == "wav":
            success = await _process_wav_job(job, job_dir, backend_kwargs)
        else:
            raise ValueError(f"Unknown source type: {job.source_type}")

        if success:
            registry.update_status(job.job_id, JobStatus.COMPLETED)
            logger.info("Job %s completed successfully", job.job_id)
        else:
            registry.update_status(
                job.job_id, JobStatus.FAILED, error="No speech detected"
            )
            logger.warning("Job %s: no speech detected", job.job_id)

    except Exception as e:
        logger.exception("Job %s failed", job.job_id)
        registry.update_status(job.job_id, JobStatus.FAILED, error=str(e))

    # Send webhook if configured
    if job.webhook_url:
        await _send_webhook(job)


async def _process_zip_job(job: Job, job_dir: Path, backend_kwargs: dict) -> bool:
    """Process a ZIP upload (multi-participant session)."""
    zip_path = job_dir / job.source_name

    # Extract ZIP to a working directory
    work_dir = job_dir / "session"
    work_dir.mkdir(exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(work_dir)

    # The ZIP from livekit-recorder wraps in a session_id/ folder.
    # Find the manifest — could be at work_dir/manifest.json or
    # work_dir/<session_id>/manifest.json
    session_dir = work_dir
    if not (session_dir / "manifest.json").exists():
        # Look one level deeper
        subdirs = [d for d in session_dir.iterdir() if d.is_dir()]
        for d in subdirs:
            if (d / "manifest.json").exists():
                session_dir = d
                break

    if not (session_dir / "manifest.json").exists():
        raise ValueError("ZIP does not contain a manifest.json")

    success = await transcribe_session(
        session_dir=session_dir,
        backend_type=TRANSCRIBE_BACKEND,
        language=job.language,
        denoise=TRANSCRIBE_DENOISE,
        vad_threshold=TRANSCRIBE_VAD_THRESHOLD,
        **backend_kwargs,
    )

    # Copy transcript files to job root for easy access
    if success:
        for fname in ("transcript.json", "transcript.txt"):
            src = session_dir / fname
            if src.exists():
                shutil.copy2(src, job_dir / fname)

    return success


async def _process_wav_job(job: Job, job_dir: Path, backend_kwargs: dict) -> bool:
    """Process a single WAV file upload."""
    wav_path = job_dir / job.source_name

    success = await transcribe_wav(
        wav_path=wav_path,
        output_dir=job_dir,
        backend_type=TRANSCRIBE_BACKEND,
        language=job.language,
        denoise=TRANSCRIBE_DENOISE,
        vad_threshold=TRANSCRIBE_VAD_THRESHOLD,
        speaker="speaker",
        **backend_kwargs,
    )

    return success


async def _send_webhook(job: Job) -> None:
    """Fire-and-forget webhook notification."""
    import httpx

    payload = {
        "job_id": job.job_id,
        "status": job.status.value,
        "transcript_url": f"/transcribe/{job.job_id}/transcript",
    }
    if job.error:
        payload["error"] = job.error

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            await client.post(job.webhook_url, json=payload)
        logger.info("Webhook sent for job %s to %s", job.job_id, job.webhook_url)
    except Exception:
        logger.warning("Failed to send webhook for job %s", job.job_id)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main():
    """Run the transcriber service."""
    import uvicorn

    uvicorn.run(
        "transcriber.main:app",
        host=HOST,
        port=PORT,
        log_level=LOG_LEVEL.lower(),
    )


if __name__ == "__main__":
    main()
