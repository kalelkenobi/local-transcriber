"""
Job registry — tracks transcription jobs and their state.

Jobs are stored as directories under JOBS_DIR with state persisted in job.json.
"""

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Job:
    job_id: str
    status: JobStatus
    created_at: float
    completed_at: float | None = None
    error: str | None = None
    language: str = "en"
    source_type: str = ""  # "zip" or "wav"
    source_name: str = ""
    webhook_url: str | None = None
    transcript: dict | None = None

    def to_dict(self) -> dict[str, Any]:
        d = {
            "job_id": self.job_id,
            "status": self.status.value,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "error": self.error,
            "language": self.language,
            "source_type": self.source_type,
            "source_name": self.source_name,
            "webhook_url": self.webhook_url,
        }
        if self.status == JobStatus.COMPLETED and self.transcript:
            d["transcript"] = self.transcript
        return d


class JobRegistry:
    """Manages transcription jobs on disk."""

    def __init__(self, jobs_dir: Path):
        self._jobs_dir = jobs_dir
        self._jobs_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, Job] = {}
        self._load_existing()

    def _load_existing(self) -> None:
        """Load existing jobs from disk on startup."""
        for job_dir in self._jobs_dir.iterdir():
            if not job_dir.is_dir():
                continue
            meta_path = job_dir / "job.json"
            if meta_path.exists():
                try:
                    with open(meta_path) as f:
                        data = json.load(f)
                    job = Job(
                        job_id=data["job_id"],
                        status=JobStatus(data["status"]),
                        created_at=data["created_at"],
                        completed_at=data.get("completed_at"),
                        error=data.get("error"),
                        language=data.get("language", "en"),
                        source_type=data.get("source_type", ""),
                        source_name=data.get("source_name", ""),
                        webhook_url=data.get("webhook_url"),
                    )
                    # Load transcript if completed
                    if job.status == JobStatus.COMPLETED:
                        transcript_path = job_dir / "transcript.json"
                        if transcript_path.exists():
                            with open(transcript_path) as f:
                                job.transcript = json.load(f)
                    self._cache[job.job_id] = job
                except (json.JSONDecodeError, KeyError, ValueError):
                    logger.warning("Skipping corrupt job dir: %s", job_dir.name)

    def create_job(
        self,
        source_type: str,
        source_name: str,
        language: str = "en",
        webhook_url: str | None = None,
    ) -> Job:
        """Create a new job and return it."""
        job_id = str(uuid.uuid4())[:12]
        job = Job(
            job_id=job_id,
            status=JobStatus.QUEUED,
            created_at=time.time(),
            language=language,
            source_type=source_type,
            source_name=source_name,
            webhook_url=webhook_url,
        )

        # Create job directory
        job_dir = self._jobs_dir / job_id
        job_dir.mkdir(parents=True)

        self._cache[job_id] = job
        self._persist(job)
        return job

    def get_job(self, job_id: str) -> Job | None:
        return self._cache.get(job_id)

    def get_job_dir(self, job_id: str) -> Path:
        return self._jobs_dir / job_id

    def update_status(
        self,
        job_id: str,
        status: JobStatus,
        error: str | None = None,
    ) -> None:
        job = self._cache.get(job_id)
        if not job:
            return
        job.status = status
        if error:
            job.error = error
        if status in (JobStatus.COMPLETED, JobStatus.FAILED):
            job.completed_at = time.time()
        # Load transcript if completed
        if status == JobStatus.COMPLETED:
            transcript_path = self.get_job_dir(job_id) / "transcript.json"
            if transcript_path.exists():
                with open(transcript_path) as f:
                    job.transcript = json.load(f)
        self._persist(job)

    def list_jobs(self, limit: int = 50) -> list[Job]:
        """List recent jobs sorted by creation time (newest first)."""
        jobs = sorted(self._cache.values(), key=lambda j: j.created_at, reverse=True)
        return jobs[:limit]

    def _persist(self, job: Job) -> None:
        """Write job metadata to disk."""
        job_dir = self._jobs_dir / job.job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "job_id": job.job_id,
            "status": job.status.value,
            "created_at": job.created_at,
            "completed_at": job.completed_at,
            "error": job.error,
            "language": job.language,
            "source_type": job.source_type,
            "source_name": job.source_name,
            "webhook_url": job.webhook_url,
        }
        with open(job_dir / "job.json", "w") as f:
            json.dump(meta, f, indent=2)
