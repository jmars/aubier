#!/bin/sh
# engine_build_m1.sh — build the M1 engine skeleton (builds/engine/m1).
#
# The M1 binary shares the GGUF reader and the store machinery with M0's moe_split via
# engine/gguf_model.h + engine/store.h; it links the SAME ggml (base+cpu+hip) from
# builds/hip/bin as scripts/engine_build.sh, with the SAME flags — engine_build.sh is left
# untouched so that M0's gated artifact stays reproducible from an unchanged build recipe.
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
         "$ROCMLIB/libamdhip64.so.7" "$GGML_INC/ggml.h" "$GGML_INC/gguf.h" \
         "$ROOT/engine/m1_engine.cpp" "$ROOT/engine/m1_layer.cpp" \
         "$ROOT/engine/m1_main.cpp" "$ROOT/engine/m1_session.h" \
         "$ROOT/engine/gguf_model.h" "$ROOT/engine/store.h" "$ROOT/engine/m1_layer.h"; do
    [ -e "$f" ] || { echo "engine_build_m1.sh: FATAL: missing $f" >&2; exit 1; }
done

mkdir -p "$BUILDS/engine"

/usr/bin/c++ -O2 -std=c++17 \
    -I "$GGML_INC" \
    "$ROOT/engine/m1_engine.cpp" "$ROOT/engine/m1_layer.cpp" "$ROOT/engine/m1_main.cpp" \
    -L "$GGML_LIB" -lggml-hip -lggml-cpu -lggml -lggml-base \
    -Wl,-rpath,"$GGML_LIB" -Wl,-rpath,"$ROCMLIB" \
    -Wl,--disable-new-dtags \
    -o "$BUILDS/engine/m1"

echo "engine_build_m1.sh: built $BUILDS/engine/m1"

# gg_fixture — the SEAM 3 fixture builder/exerciser (loader-groups unit).  Built CPU-ONLY
# (no -lggml-hip, no ROCm rpath): it must never touch the GPU or take gpu.lock, so its
# fixture runs stay runnable while a measurement unit holds the window.
if [ -e "$ROOT/tools/gg_fixture.cpp" ]; then
    /usr/bin/c++ -O2 -std=c++17 \
        -I "$GGML_INC" -I "$ROOT/engine" \
        "$ROOT/tools/gg_fixture.cpp" \
        -L "$GGML_LIB" -lggml-cpu -lggml-base \
        -Wl,-rpath,"$GGML_LIB" \
        -Wl,--disable-new-dtags \
        -o "$BUILDS/engine/gg_fixture"
    echo "engine_build_m1.sh: built $BUILDS/engine/gg_fixture"
fi

# m1_batch_test — B2's batch-of-N test rig.  The SAME engine TUs as the m1 binary plus its own
# driver, linked the same way (the rig drives M1Session through m1_session_step_batch; the m1
# compile line above is untouched, so that binary's reproducibility is unchanged).
if [ -e "$ROOT/tools/m1_batch_test.cpp" ]; then
    /usr/bin/c++ -O2 -std=c++17 \
        -I "$GGML_INC" -I "$ROOT/engine" \
        "$ROOT/engine/m1_engine.cpp" "$ROOT/engine/m1_layer.cpp" "$ROOT/tools/m1_batch_test.cpp" \
        -L "$GGML_LIB" -lggml-hip -lggml-cpu -lggml -lggml-base \
        -Wl,-rpath,"$GGML_LIB" -Wl,-rpath,"$ROCMLIB" \
        -Wl,--disable-new-dtags \
        -o "$BUILDS/engine/m1_batch_test"
    echo "engine_build_m1.sh: built $BUILDS/engine/m1_batch_test"
fi
