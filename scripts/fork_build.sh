#!/usr/bin/env bash
# P-FORK: build the llama-fork worktree (src/llama-fork -> builds/hip-fork) with the
# SAME configure line as the working HIP build (scripts/build_hip.sh). The fork is the
# Mistral-Small-4-specialized engine arm; it must build identically to stock so that
# M1's byte-identity gate is meaningful.
#
# Usage:  scripts/fork_build.sh [configure|build|all]   (default: all)
# Env:    JOBS (default 8), clean configure with RECONFIGURE=1
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$ROOT/venvs/rocm"
SRC="$ROOT/src/llama-fork"
BUILD="$ROOT/builds/hip-fork"
JOBS="${JOBS:-8}"
MODE="${1:-all}"

ROCM_PREFIX="$("$VENV/bin/rocm-sdk" path --root)"
ROCM_BIN="$("$VENV/bin/rocm-sdk" path --bin)"
[ -d "$ROCM_PREFIX" ] || { echo "FATAL: ROCm prefix not found: $ROCM_PREFIX" >&2; exit 1; }

HIP_CLANG="$ROCM_PREFIX/lib/llvm/bin/clang++"
[ -x "$HIP_CLANG" ] || { echo "FATAL: ROCm clang++ not found: $HIP_CLANG" >&2; exit 1; }

export ROCM_PATH="$ROCM_PREFIX"
export HIP_PATH="$ROCM_PREFIX"
export HIP_PLATFORM=amd
export PATH="$ROCM_BIN:$VENV/bin:$PATH"
export LD_LIBRARY_PATH="$ROCM_PREFIX/lib:${LD_LIBRARY_PATH:-}"

echo "== fork_build.sh =="
echo "SRC           = $SRC"
echo "BUILD         = $BUILD"
echo "ROCM_PREFIX   = $ROCM_PREFIX"
echo "HIP compiler  = $HIP_CLANG"
echo "JOBS          = $JOBS"

if [ "$MODE" = configure ] || [ "$MODE" = all ]; then
  [ "${RECONFIGURE:-0}" = 1 ] && rm -rf "$BUILD"
  cmake -S "$SRC" -B "$BUILD" \
    -DGGML_HIP=ON \
    -DCMAKE_HIP_ARCHITECTURES=gfx1201 \
    -DCMAKE_HIP_COMPILER="$HIP_CLANG" \
    -DCMAKE_HIP_FLAGS="--rocm-path=$ROCM_PREFIX --rocm-device-lib-path=$ROCM_PREFIX/lib/llvm/amdgcn/bitcode" \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLAMA_CURL=OFF \
    -DCMAKE_PREFIX_PATH="$ROCM_PREFIX" \
    -DHIP_PLATFORM=amd \
    -DCMAKE_BUILD_RPATH="$ROCM_PREFIX/lib" \
    -DCMAKE_INSTALL_RPATH="$ROCM_PREFIX/lib"
fi

if [ "$MODE" = build ] || [ "$MODE" = all ]; then
  cmake --build "$BUILD" -j "$JOBS"
fi

echo "OK: $BUILD/bin"
