#!/usr/bin/env bash
# LANE A: build llama.cpp for AMD gfx1201 (RX 9070, RDNA4) with HIP, using the
# no-sudo ROCm toolchain that lives entirely inside venvs/rocm (TheRock wheels).
#
# Usage:  scripts/build_hip.sh [configure|build|all]   (default: all)
# Env:    JOBS (default 8), clean configure with RECONFIGURE=1,
#         VENV/SRC/BUILD to build a tree other than this one
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${VENV:-$ROOT/venvs/rocm}"
SRC="${SRC:-$ROOT/vendor/llama.cpp}"
BUILD="${BUILD:-$ROOT/builds/hip}"
JOBS="${JOBS:-8}"
MODE="${1:-all}"

# --- ROCm prefix discovery: ask the toolchain itself, never hardcode ----------
# rocm-sdk path --root -> .../_rocm_sdk_devel (holds lib/cmake, bin/hipcc, lib/*.so)
ROCM_PREFIX="$("$VENV/bin/python" "$VENV/bin/rocm-sdk" path --root)"
ROCM_BIN="$("$VENV/bin/python" "$VENV/bin/rocm-sdk" path --bin)"
[ -d "$ROCM_PREFIX" ] || { echo "FATAL: ROCm prefix not found: $ROCM_PREFIX" >&2; exit 1; }

# CMake >= 4.x refuses the hipcc wrapper as CMAKE_HIP_COMPILER ("This is not
# supported. Use Clang directly") - hand it the ROCm clang that hipcc wraps.
HIP_CLANG="$ROCM_PREFIX/lib/llvm/bin/clang++"
[ -x "$HIP_CLANG" ] || { echo "FATAL: ROCm clang++ not found: $HIP_CLANG" >&2; exit 1; }

export ROCM_PATH="$ROCM_PREFIX"    # ggml-hip/CMakeLists.txt reads $ENV{ROCM_PATH}
export HIP_PATH="$ROCM_PREFIX"
# TheRock's hip-config.cmake cannot probe hipconfig (its gate expands to "OFF"),
# so HIP_PLATFORM must be supplied or find_package(hip) FATALs.
export HIP_PLATFORM=amd
export PATH="$ROCM_BIN:$VENV/bin:$PATH"
export LD_LIBRARY_PATH="$ROCM_PREFIX/lib:${LD_LIBRARY_PATH:-}"

echo "== build_hip.sh =="
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
    -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
    -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_SERVER=OFF \
    -DGGML_CCACHE=OFF \
    -DCMAKE_PREFIX_PATH="$ROCM_PREFIX" \
    -DHIP_PLATFORM=amd \
    -DCMAKE_BUILD_RPATH="$ROCM_PREFIX/lib" \
    -DCMAKE_INSTALL_RPATH="$ROCM_PREFIX/lib"
fi

if [ "$MODE" = build ] || [ "$MODE" = all ]; then
  # TARGETS, not "all": the vendored tree carries only what builds these four
  # libraries (the engine links exactly these). "all" would also want tests/,
  # examples/, tools/ and app/ sources, which are not vendored.
  cmake --build "$BUILD" --target ggml ggml-base ggml-cpu ggml-hip -j "$JOBS"
fi

echo "OK: $BUILD/bin"
