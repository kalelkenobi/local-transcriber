#!/bin/sh
# Run local-transcriber via Apple container.
#
# Usage:
#   scripts/run.sh --url URL --model MODEL PATH
#   scripts/run.sh --url URL --model MODEL --recursive PATH
#
# Requires Apple container CLI (macOS). The image is pulled from
# docker.io/kalelkenobi/local-transcriber (override tag with --tag or $LT_TAG).

set -eu

IMAGE_REPO="docker.io/kalelkenobi/local-transcriber"
CONTAINER_MOUNT="/in"

# ── helpers ────────────────────────────────────────────────────────────

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS] PATH

Run local-transcriber via Apple container.

Required:
  --url URL         Base URL of an OpenAI-compatible ASR server
                    (or set TRANSCRIBE_URL in the environment).
  --model MODEL     Model name passed to the ASR server
                    (or set TRANSCRIBE_MODEL in the environment).

Options:
  --tag TAG            Image tag (default: latest, env: LT_TAG).
  --api-key KEY        Optional bearer token (env: TRANSCRIBE_API_KEY).
  --recursive, -r      Process every child dir containing manifest.json.
  --language LANG      Language code (default: en).
  --log-level LVL      Log level: DEBUG|INFO|WARNING|ERROR (default: INFO).
  --vad-threshold N    VAD threshold (default: 0.3).
  --timeout SEC        Per-segment HTTP timeout in seconds (default: 300).
  --concurrency N      Max parallel ASR requests (default: 4).
  --memory N           Container max memory in GB (default: 2).
  --help, -h           Show this help and exit.

Examples:
  $(basename "$0") --url http://host.docker.internal:8000 \\
      --model whisper-large-v3-mlx /Users/me/recordings/my-session

  $(basename "$0") --url https://api.example.com --api-key sk-... \\
      --model whisper-1 --recursive /Users/me/recordings
EOF
    exit 2
}

# ── defaults ───────────────────────────────────────────────────────────

TAG="${LT_TAG:-latest}"
IMAGE="${IMAGE_REPO}:${TAG}"

API_URL="${TRANSCRIBE_URL:-}"
MODEL="${TRANSCRIBE_MODEL:-}"
API_KEY="${TRANSCRIBE_API_KEY:-}"

RECURSIVE=""
LANGUAGE="it"
LOG_LEVEL="INFO"
VAD_THRESHOLD="0.2"
TIMEOUT=""
CONCURRENCY=""
MEMORY="2"

# ── parse flags ────────────────────────────────────────────────────────

while [ $# -gt 0 ]; do
    case "$1" in
        --url)       API_URL="$2"; shift 2 ;;
        --model)     MODEL="$2"; shift 2 ;;
        --tag)       TAG="$2"; IMAGE="${IMAGE_REPO}:${TAG}"; shift 2 ;;
        --api-key)   API_KEY="$2"; shift 2 ;;
        --language)  LANGUAGE="$2"; shift 2 ;;
        --log-level) LOG_LEVEL="$2"; shift 2 ;;
        --vad-threshold) VAD_THRESHOLD="$2"; shift 2 ;;
        --timeout)   TIMEOUT="$2"; shift 2 ;;
        --concurrency) CONCURRENCY="$2"; shift 2 ;;
        --memory) MEMORY="$2"; shift 2 ;;
        -r|--recursive) RECURSIVE="--recursive"; shift ;;
        -h|--help)   usage ;;
        --) shift; break ;;
        -*) echo "Unknown option: $1" >&2; usage ;;
        *) break ;;
    esac
done

PATH_ARG="${1:-}"
if [ -z "$PATH_ARG" ]; then
    echo "Error: PATH is required." >&2
    usage
fi

# ── validate PATH ──────────────────────────────────────────────────────

if [ ! -d "$PATH_ARG" ]; then
    echo "Error: '$PATH_ARG' is not a directory." >&2
    exit 1
fi

PATH_ABS="$(cd "$PATH_ARG" && pwd)"

# ── validate required env ──────────────────────────────────────────────

if [ -z "$API_URL" ]; then
    echo "Error: --url is required (or set TRANSCRIBE_URL)." >&2
    exit 1
fi
if [ -z "$MODEL" ]; then
    echo "Error: --model is required (or set TRANSCRIBE_MODEL)." >&2
    exit 1
fi

# ── check container CLI ────────────────────────────────────────────────

if ! command -v container >/dev/null 2>&1; then
    echo "Error: Apple 'container' CLI not found on PATH." >&2
    echo "Install: brew install container" >&2
    exit 1
fi

# ── build env-var args ─────────────────────────────────────────────────

set -- \
    -e "TRANSCRIBE_URL=$API_URL" \
    -e "TRANSCRIBE_MODEL=$MODEL"

if [ -n "$API_KEY" ]; then
    set -- "$@" -e "TRANSCRIBE_API_KEY=$API_KEY"
fi

set -- "$@" -e "LOG_LEVEL=$LOG_LEVEL"

# ── build CLI args ─────────────────────────────────────────────────────

CLI_ARGS="$CONTAINER_MOUNT"
if [ -n "$RECURSIVE" ]; then
    CLI_ARGS="$CLI_ARGS $RECURSIVE"
fi
if [ "$LANGUAGE" != "en" ]; then
    CLI_ARGS="$CLI_ARGS --language $LANGUAGE"
fi
if [ -n "$VAD_THRESHOLD" ]; then
    CLI_ARGS="$CLI_ARGS --vad-threshold $VAD_THRESHOLD"
fi
if [ -n "$TIMEOUT" ]; then
    CLI_ARGS="$CLI_ARGS --timeout $TIMEOUT"
fi
if [ -n "$CONCURRENCY" ]; then
    CLI_ARGS="$CLI_ARGS --concurrency $CONCURRENCY"
fi

# ── run ────────────────────────────────────────────────────────────────

RUNTIME_ARGS=""
if [ -n "$MEMORY" ]; then
    RUNTIME_ARGS="$RUNTIME_ARGS --memory $MEMORY"
fi

echo "Running: container run --rm $RUNTIME_ARGS -v $PATH_ABS:$CONTAINER_MOUNT [env] $IMAGE $CLI_ARGS"
container run --rm \
    $RUNTIME_ARGS \
    -v "$PATH_ABS:$CONTAINER_MOUNT" \
    "$@" \
    "$IMAGE" \
    $CLI_ARGS
