#!/bin/sh
# engine_build.sh — build the M0 split primitive (builds/engine/moe_split).
# Links ggml (base+cpu+hip) from builds/hip/bin for the LOADER, the ORACLE, and the
# compute kernels.  The engine's own loop/split/store-list/streams are in engine/.
#
# rpath: DT_RPATH (--disable-new-dtags) so the binary runs BARE — libggml-hip.so's
# own RUNPATH points into the DEAD pre-move tree, and only a transitive RPATH can
# reach the ROCm SDK libs through it.

set -e

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
BUILDS="$ROOT/builds"
SRC="$ROOT/vendor/llama.cpp"
ROCMLIB="$ROOT/venvs/rocm/lib/python3.13/site-packages/_rocm_sdk_devel/lib"
GGML_INC="$SRC/ggml/include"
GGML_LIB="$BUILDS/hip/bin"

for f in "$GGML_LIB/libggml-base.so" "$GGML_LIB/libggml-cpu.so" "$GGML_LIB/libggml-hip.so" \
         "$ROCMLIB/libamdhip64.so.7" "$GGML_INC/ggml.h" "$ROOT/engine/moe_split.cpp"; do
    [ -e "$f" ] || { echo "engine_build.sh: FATAL: missing $f" >&2; exit 1; }
done

mkdir -p "$BUILDS/engine"

/usr/bin/c++ -O2 -std=c++17 \
    -I "$GGML_INC" \
    "$ROOT/engine/moe_split.cpp" \
    -L "$GGML_LIB" -lggml-hip -lggml-cpu -lggml -lggml-base \
    -Wl,-rpath,"$GGML_LIB" -Wl,-rpath,"$ROCMLIB" \
    -Wl,--disable-new-dtags \
    -o "$BUILDS/engine/moe_split"

echo "engine_build.sh: built $BUILDS/engine/moe_split"
