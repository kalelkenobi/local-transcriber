#!/bin/sh
# Sweep stranded end-to-end test artifacts.
#
# Detects available container runtime in this priority order:
#   1. $E2E_RUNTIME (one of: container, docker, podman)
#   2. Apple `container` CLI
#   3. `docker`
#   4. `podman`
# Then removes any containers and (optionally) images whose name starts
# with the `lt-e2e-` prefix.

set -eu

PREFIX="${E2E_PREFIX:-lt-e2e-}"

pick_runtime() {
    if [ -n "${E2E_RUNTIME:-}" ]; then
        case "$E2E_RUNTIME" in
            container|docker|podman) printf '%s' "$E2E_RUNTIME"; return 0 ;;
            *) echo "Unknown E2E_RUNTIME: $E2E_RUNTIME" >&2; exit 2 ;;
        esac
    fi
    for candidate in container docker podman; do
        if command -v "$candidate" >/dev/null 2>&1; then
            printf '%s' "$candidate"
            return 0
        fi
    done
    echo "No container runtime found (looked for container, docker, podman)" >&2
    exit 1
}

RUNTIME="$(pick_runtime)"
echo "Using runtime: $RUNTIME (prefix: $PREFIX)"

# List + remove matching containers (running or stopped).
case "$RUNTIME" in
    container)
        # Apple container CLI
        containers="$($RUNTIME list --all --format '{{.Name}}' 2>/dev/null \
            | grep "^$PREFIX" || true)"
        ;;
    docker|podman)
        containers="$($RUNTIME ps -a --format '{{.Names}}' 2>/dev/null \
            | grep "^$PREFIX" || true)"
        ;;
esac

if [ -n "$containers" ]; then
    echo "Removing containers:"
    echo "$containers"
    echo "$containers" | xargs -r "$RUNTIME" rm -f >/dev/null 2>&1 || true
else
    echo "No matching containers to remove."
fi

# List + remove matching images.
case "$RUNTIME" in
    container)
        images="$($RUNTIME images list --format '{{.Repository}}:{{.Tag}}' 2>/dev/null \
            | grep "^$PREFIX" || true)"
        ;;
    docker|podman)
        images="$($RUNTIME images --format '{{.Repository}}:{{.Tag}}' 2>/dev/null \
            | grep "^$PREFIX" || true)"
        ;;
esac

if [ -n "$images" ]; then
    echo "Removing images:"
    echo "$images"
    echo "$images" | xargs -r "$RUNTIME" rmi -f >/dev/null 2>&1 || true
else
    echo "No matching images to remove."
fi
