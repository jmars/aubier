#!/bin/sh
# engine_build_server.sh — build THE PRODUCT SERVER: builds/engine/m1_server (server plan §H row U5).
#
# WHAT THIS BUILDS: engine/m1_server.cpp (the OpenAI surface + routing) with -DM1_ENGINE_REAL, the
# real engine TU engine/m1_real.cpp, the session's own TU engine/m1_engine.cpp + engine/m1_layer.cpp,
# the tokenizer engine/m1_tok.cpp and the chat renderer engine/m1_chat.cpp, plus the two vendored
# unicode TUs (the pre-tokenizer's splitter and the byte<->unicode map, compiled in exactly as
# scripts/engine_build_tok.sh does). The engine TU that carries main() is engine/m1_main.cpp and it
# is NOT linked here (a second main would collide — the server's own main is in m1_server.cpp).
#
# LINKED LIKE scripts/engine_build_m1.sh, for the same reason: builds/hip/bin's libggml-hip.so has a
# RUNPATH into a dead pre-move tree, so only a transitive DT_RPATH (--disable-new-dtags) reaches the
# ROCm SDK libs and lets the binary run BARE.
#
# CPU-ONLY vs GPU: the BINARY always links libggml-hip (the engine is one binary, as the m1 binary
# is), but the CONFIGURATION decides what runs where — with no --gpu-store/--gpu-core the session
# builds every store on the CPU backend and the layer core on the CPU (the byte-identical correctness
# path), and nothing on the card is touched. The GPU configuration is the flag set
# (--arm-store --gpu-store --gpu-core --n-res 48 --rotate --rot-admit-k 4 --resident-set <hotset>).
#
# The GPU lock: the BINARY takes scripts/gpu.lock itself at boot, only when --engine real (see
# main() in engine/m1_server.cpp). This script takes no lock — it only compiles.
#
# -Wall -Wextra: a warning in m1_real.cpp / m1_server.cpp must not be silent (the U3 build had them;
# this script did not, and a reviewer flagged it). MEASURED: those two TUs are warning-free under
# these flags; the remaining warnings this recipe prints come from PRE-EXISTING files owned by other
# units (m1_layer.cpp:310's `&&` inside `||`, store.h:948's unused derive_store_specs, m1_tok.cpp:51,
# m1_chat.cpp:120, vendored unicode.cpp:22) and are left visible rather than suppressed.

# usage: sh scripts/engine_build_server.sh

set -e

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
BUILDS="$ROOT/builds"
SRC="$ROOT/vendor/llama.cpp"
ROCMLIB="$ROOT/venvs/rocm/lib/python3.13/site-packages/_rocm_sdk_devel/lib"
GGML_INC="$SRC/ggml/include"
GGML_LIB="$BUILDS/hip/bin"
OUT="$BUILDS/engine/m1_server"

for f in "$GGML_LIB/libggml-base.so" "$GGML_LIB/libggml-cpu.so" "$GGML_LIB/libggml-hip.so" \
         "$ROCMLIB/libamdhip64.so.7" "$GGML_INC/ggml.h" "$GGML_INC/gguf.h" \
         "$SRC/src/unicode.cpp" "$SRC/src/unicode-data.cpp" "$SRC/src/unicode.h" \
         "$ROOT/engine/m1_server.cpp" "$ROOT/engine/m1_real.cpp" "$ROOT/engine/m1_engine_api.h" \
         "$ROOT/engine/m1_engine.cpp" "$ROOT/engine/m1_layer.cpp" "$ROOT/engine/m1_layer.h" \
         "$ROOT/engine/m1_session.h" "$ROOT/engine/m1_tok.cpp" "$ROOT/engine/m1_tok.h" \
         "$ROOT/engine/m1_chat.cpp" "$ROOT/engine/m1_chat.h" "$ROOT/engine/m1_chat_dsm.inc" \
         "$ROOT/engine/m1_json.cpp" "$ROOT/engine/m1_json.h" \
         "$ROOT/engine/m1_http.cpp" "$ROOT/engine/m1_http.h" \
         "$ROOT/engine/gguf_model.h" "$ROOT/engine/store.h"; do
    [ -e "$f" ] || { echo "engine_build_server.sh: FATAL: missing $f" >&2; exit 1; }
done

mkdir -p "$BUILDS/engine"

/usr/bin/c++ -O2 -std=c++17 -Wall -Wextra -Wno-unused-parameter -DM1_ENGINE_REAL \
    -I "$GGML_INC" -I "$ROOT/engine" -I "$SRC/src" \
    "$ROOT/engine/m1_server.cpp" "$ROOT/engine/m1_real.cpp" \
    "$ROOT/engine/m1_json.cpp" "$ROOT/engine/m1_http.cpp" \
    "$ROOT/engine/m1_engine.cpp" "$ROOT/engine/m1_layer.cpp" \
    "$ROOT/engine/m1_tok.cpp" "$ROOT/engine/m1_chat.cpp" \
    "$SRC/src/unicode.cpp" "$SRC/src/unicode-data.cpp" \
    -L "$GGML_LIB" -lggml-hip -lggml-cpu -lggml -lggml-base \
    -Wl,-rpath,"$GGML_LIB" -Wl,-rpath,"$ROCMLIB" \
    -Wl,--disable-new-dtags \
    -pthread \
    -o "$OUT"

echo "engine_build_server.sh: built $OUT (REAL engine: m1_session + m1_tok + m1_chat)"
