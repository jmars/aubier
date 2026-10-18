#!/bin/sh
# with_gpu_lock.sh — run a command under the project's exclusive GPU lock.
# The single global serial resource for every bench/gate run (lanes A-E depend
# on it). Passes the command's exit code through UNCHANGED: this script's exit
# status is the command's status, nothing else.
# Usage: scripts/with_gpu_lock.sh <command> [args...]
set -u

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
LOCK="$SCRIPT_DIR/gpu.lock"

if [ ! -f "$LOCK" ]; then
    echo "with_gpu_lock: FATAL: lock file $LOCK missing (scaffold broken)" >&2
    exit 69  # EX_UNAVAILABLE: the resource is not usable
fi

if ! command -v flock >/dev/null 2>&1; then
    echo "with_gpu_lock: FATAL: flock(1) not found on PATH; refusing to run WITHOUT" >&2
    echo "  the lock — the GPU is a global serial resource and an unlocked run can" >&2
    echo "  collide with a peer lane. Install/util-linux provides flock." >&2
    exit 69
fi

# exec: our pid BECOMES the locked command, so its exit status is ours,
# unchanged — signals and codes pass through with no wrapper layer.
exec flock -x "$LOCK" "$@"
