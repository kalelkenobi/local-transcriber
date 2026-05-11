# API Reference

## Submit Transcription

```
POST /transcribe
Content-Type: multipart/form-data
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | file | yes | `.wav` or `.zip` file to transcribe |
| `language` | string | no | Language code (default `"en"`) |
| `webhook_url` | string | no | URL to POST notification on completion |

**Response** (201):
```json
{"job_id": "abc123", "status": "queued"}
```

**Errors**:
- `400` — missing filename or unsupported file type

---

## Get Job Status

```
GET /transcribe/{job_id}
```

**Response** (200):
```json
{
  "job_id": "abc123",
  "status": "completed",
  "source_type": "zip",
  "source_name": "my-room_20240101_120000.zip",
  "language": "en",
  "created_at": "2024-01-01T12:00:00",
  "transcript": [...]
}
```

**Status values**: `queued`, `processing`, `completed`, `failed`

**Errors**:
- `404` — job not found

---

## Download Transcript (JSON)

```
GET /transcribe/{job_id}/transcript
```

Returns the structured JSON transcript when the job is complete.

**Response codes**:
- `200` — transcript JSON body
- `202` — transcription still in progress
- `404` — job or file not found

---

## Download Transcript (Plain Text)

```
GET /transcribe/{job_id}/transcript.txt
```

Returns the plain-text formatted transcript.

**Response codes**:
- `200` — plain text body
- `202` — transcription still in progress
- `404` — job or file not found

---

## List Jobs

```
GET /jobs?limit=50
```

**Response** (200):
```json
{
  "jobs": [
    {"job_id": "...", "status": "completed", ...},
    {"job_id": "...", "status": "processing", ...}
  ]
}
```

---

## Webhook Notification

When `webhook_url` is provided during submission, a POST is sent on completion or failure:

```json
{
  "job_id": "abc123",
  "status": "completed",
  "transcript_url": "/transcribe/abc123/transcript"
}
```

On failure, an additional `"error"` field is included.

The webhook is fire-and-forget — failures are logged but not retried.
