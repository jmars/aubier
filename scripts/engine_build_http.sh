#!/bin/sh
# engine_build_http.sh — build the server's JSON + HTTP + OpenAI surface against the FAKE
# engine (server plan §F, unit U3).
#
# WHAT THIS BUILD IS: CPU-ONLY and ggml-FREE. It links libstdc++, libm, libpthread and nothing
# else — no libggml-*, no ROCm rpath, no CUDA/hip. That is a measured property, not an
# intention: scripts/engine_gate_http.sh asserts that `ldd` on the binary names no ggml object,
# so the whole wire-protocol gate runs on a loaded box, with no GPU and without taking
# scripts/gpu.lock.
#
# WHAT IT IS NOT: the product server. That one is scripts/engine_build_server.sh (U5): it links the
# real engine TU (engine/m1_engine.cpp + engine/m1_layer.cpp), engine/m1_real.cpp, the tokenizer and
# the chat render, which DO need ggml, with -DM1_ENGINE_REAL, and it writes builds/engine/m1_server.
# THIS script writes builds/engine/m1_server_fake: two different binaries with two different
# jobs, and the name says which is which — so running the wire-protocol gate can never leave a fake
# engine sitting at the path the product's own id-parity test boots.

set -e

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
BUILDS="$ROOT/builds"
OUT="$BUILDS/engine/m1_server_fake"

for f in "$ROOT/engine/m1_json.cpp" "$ROOT/engine/m1_http.cpp" "$ROOT/engine/m1_server.cpp"; do
    [ -e "$f" ] || { echo "engine_build_http.sh: FATAL: missing $f" >&2; exit 1; }
done

mkdir -p "$BUILDS/engine"

# -pthread: POSIX threads. No -lggml*, no ROCm rpath, no --disable-new-dtags (nothing to find
# at runtime). If a future edit reintroduces a ggml dependency, the gate's ldd arm fails.
/usr/bin/c++ -O2 -std=c++17 -Wall -Wextra -Wno-unused-parameter \
    -I "$ROOT/engine" \
    "$ROOT/engine/m1_json.cpp" "$ROOT/engine/m1_http.cpp" "$ROOT/engine/m1_server.cpp" \
    -pthread \
    -o "$OUT"

echo "engine_build_http.sh: built $OUT (FAKE engine; CPU-only, no ggml, no gpu.lock)"
