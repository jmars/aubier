#!/bin/sh
# env.sh — path contract for the mistral4-rx9070 project. POSIX sh, safe to source twice.
# ROOT resolution order (first hit wins):
#   1. $M4_ROOT if set and a directory (relocation escape hatch)
#   2. $0, if it names this file (executed directly: sh scripts/env.sh)
#   3. $0's dir, if it contains scripts/env.sh (caller script lives at project root)
#   4. the one canonical hardcode below (POSIX sh has no way to learn the path a
#      file was SOURCED from — ${BASH_SOURCE} is a bash parse error under dash —
#      so a sourced dash script lands here; verified to exist before use)
# Failure to resolve => loud nonzero, no silent defaults.
# Usage: . /var/data/workspace/mistral4-rx9070/scripts/env.sh

M4_CANONICAL="$(cd "$(dirname "$0")/.." && pwd)"

_m4_resolve_root() {
    if [ -n "${M4_ROOT:-}" ] && [ -d "$M4_ROOT" ]; then
        printf '%s\n' "$M4_ROOT"
        return 0
    fi
    _d=$(dirname -- "${0:-}")
    if [ -f "${0:-}" ] && [ "$(basename -- "${0:-}")" = "env.sh" ]; then
        printf '%s\n' "$_d"
        return 0
    fi
    if [ -f "$_d/scripts/env.sh" ]; then
        printf '%s\n' "$_d"
        return 0
    fi
    if [ -f "$M4_CANONICAL/scripts/env.sh" ]; then
        printf '%s\n' "$M4_CANONICAL"
        return 0
    fi
    return 1
}

if ! M4_ROOT=$( _m4_resolve_root ); then
    echo "env.sh: FATAL: cannot resolve project ROOT (set M4_ROOT or source by path)" >&2
    return 1 2>/dev/null || exit 1
fi

# -P: never trust a symlinked project path (realpath hazard for lane write scopes)
ROOT=$(CDPATH= cd -- "$M4_ROOT" && pwd -P) || {
    echo "env.sh: FATAL: ROOT '$M4_ROOT' is not enterable" >&2
    return 1 2>/dev/null || exit 1
}
if [ ! -f "$ROOT/scripts/env.sh" ]; then
    echo "env.sh: FATAL: '$ROOT' is not the project root (scripts/env.sh missing)" >&2
    return 1 2>/dev/null || exit 1
fi

MODELS="$ROOT/models"
BUILDS="$ROOT/builds"
VENVS="$ROOT/venvs"
PREFIXES="$ROOT/prefixes"
SRC_LLAMA="$ROOT/vendor/llama.cpp"
SRC_IK="$ROOT/src/ik_llama.cpp"
RESULTS="$ROOT/results"
LOGS="$ROOT/logs"
CONFIG="$ROOT/config"
LOCK_GPU="$ROOT/scripts/gpu.lock"
LOCK_DL="$ROOT/models/.dl.lock"

export ROOT MODELS BUILDS VENVS PREFIXES SRC_LLAMA SRC_IK RESULTS LOGS CONFIG LOCK_GPU LOCK_DL
