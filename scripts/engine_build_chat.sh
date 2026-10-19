#!/bin/sh
# engine_build_chat.sh — build the U2 chat-render unit (no GPU, no gpu.lock, no model weights).
#
#   builds/engine/m1_chat_ours   the renderer + its test driver (engine/m1_chat.cpp with
#                                -DM1_CHAT_DRIVER; plain C++17, NO ggml, NO llama.cpp, NO ROCm —
#                                it cannot touch the GPU even by accident)
#   builds/engine/m1_chat_ref    the TEST-ONLY reference renderer (tools/m1_chat_ref.cpp): it runs
#                                the GGUF's REAL chat template through llama.cpp's own Jinja from
#                                builds/hip-fork/bin/libllama-common.so.0.6.0, loading VOCAB ONLY
#                                (llama_model_params.vocab_only = true, llama.h:351) — ~0.5 s, no
#                                weights, no GPU.
#
# MUTANT=<id> builds a deliberately broken m1_chat_ours at builds/engine/m1_chat_ours-mutant-<id>
# instead of the real one, so scripts/engine_gate_chat.sh's FAIL path can be exercised (a gate
# whose failure path has never been seen is not a gate). The real binary is never overwritten by
# a mutant build: the mutant gets its own path and the gate must be pointed at it explicitly.
#
#   MUTANT=eos   drop the trailing '</s>' an assistant message renders
#   MUTANT=sha   shift the pinned template sha256 by one byte
#   MUTANT=dsm   corrupt one byte of the pinned default_system_message literal
#
# Usage: sh scripts/engine_build_chat.sh            (build both)
#        MUTANT=eos sh scripts/engine_build_chat.sh (build the mutant renderer only)

set -e

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
BUILDS="$ROOT/builds"
FORKBIN="$ROOT/builds/hip-fork/bin"
ROCMLIB="$ROOT/venvs/rocm/lib/python3.13/site-packages/_rocm_sdk_devel/lib"
SRC="$ROOT/vendor/llama.cpp"
MUTANT="${MUTANT:-}"

CXX=/usr/bin/c++

for f in "$ROOT/engine/m1_chat.cpp" "$ROOT/engine/m1_chat.h" "$ROOT/engine/m1_chat_dsm.inc" \
         "$ROOT/tools/m1_chat_ref.cpp"; do
    [ -e "$f" ] || { echo "engine_build_chat.sh: FATAL: missing $f" >&2; exit 1; }
done

mkdir -p "$BUILDS/engine"

DEFS=""
OUT="$BUILDS/engine/m1_chat_ours"
case "$MUTANT" in
    '')   ;;
    eos)  DEFS="-DM1_CHAT_MUTANT_EOS";     OUT="$OUT-mutant-eos"  ;;
    sha)  DEFS="-DM1_CHAT_MUTANT_SHA";     OUT="$OUT-mutant-sha"  ;;
    dsm)  DEFS="-DM1_CHAT_MUTANT_DSM";     OUT="$OUT-mutant-dsm"  ;;
    *)    echo "engine_build_chat.sh: FATAL: MUTANT='$MUTANT' (eos|sha|dsm)" >&2; exit 1 ;;
esac

# the renderer + driver: plain C++17, no third-party or GPU dependency at all
# shellcheck disable=SC2086
$CXX -O2 -std=c++17 -Wall -Wextra -DM1_CHAT_DRIVER $DEFS \
    -I "$ROOT/engine" \
    "$ROOT/engine/m1_chat.cpp" \
    -o "$OUT"
echo "engine_build_chat.sh: built $OUT${DEFS:+  [$DEFS]}"

[ -z "$MUTANT" ] || exit 0   # a mutant build does not touch the reference renderer

# the reference renderer: llama.cpp's own Jinja, VOCAB ONLY
for f in "$FORKBIN/libllama-common.so.0.6.0" "$FORKBIN/libllama.so" \
         "$SRC/common/chat.h" "$SRC/include/llama.h"; do
    [ -e "$f" ] || { echo "engine_build_chat.sh: FATAL: missing $f" >&2; exit 1; }
done

# -Wl,--allow-shlib-undefined + -rpath-link: libllama.so's DT_NEEDED chain reaches the ROCm SDK
# libs through libggml-hip.so, whose own RUNPATH points into a dead pre-move tree — the same
# workaround scripts/engine_build_m1.sh uses, and the reason llama-tokenize needs
# LD_LIBRARY_PATH=$FORKBIN:$ROCMLIB at RUN time (the gate exports it).
$CXX -O2 -std=c++17 -Wall -Wextra \
    -I "$SRC/common" -I "$SRC/include" -I "$SRC/ggml/include" -I "$ROOT/engine" \
    "$ROOT/tools/m1_chat_ref.cpp" \
    -L "$FORKBIN" -L "$ROCMLIB" -lllama-common -lllama \
    -Wl,-rpath,"$FORKBIN" -Wl,-rpath-link,"$ROCMLIB" \
    -Wl,--disable-new-dtags -Wl,--allow-shlib-undefined \
    -o "$BUILDS/engine/m1_chat_ref"
echo "engine_build_chat.sh: built $BUILDS/engine/m1_chat_ref"
