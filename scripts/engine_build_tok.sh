#!/bin/sh
# engine_build_tok.sh — build the tokenizer test driver (builds/engine/tok_test).
#
# CPU-ONLY, like gg_fixture (scripts/engine_build_m1.sh:43-54): ggml-base + ggml-cpu, no
# -lggml-hip, no ROCm rpath.  It must never touch the GPU or take gpu.lock, because the
# tokenizer unit (server plan §H U1) needs to stay runnable while a measurement row holds the
# window.
#
# The two vendored unicode translation units are COMPILED IN, not transcribed
# (vendor/llama.cpp/src/unicode.cpp + unicode-data.cpp — MIT, already in the tree, no new
# external dependency): they provide unicode_regex_split (the hand-rolled llama3 splitter),
# unicode_byte_to_utf8 and unicode_utf8_to_byte, i.e. the SAME code llama.cpp's tokenizer
# runs, so the pre-tokenization stage is parity by construction instead of by a 7000-line
# hand-copied unicode table.

set -e

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
BUILDS="$ROOT/builds"
SRC="$ROOT/vendor/llama.cpp"
GGML_INC="$SRC/ggml/include"
GGML_LIB="$BUILDS/hip/bin"

for f in "$GGML_LIB/libggml-base.so" "$GGML_LIB/libggml-cpu.so" \
         "$GGML_INC/ggml.h" "$GGML_INC/gguf.h" \
         "$SRC/src/unicode.cpp" "$SRC/src/unicode-data.cpp" "$SRC/src/unicode.h" \
         "$ROOT/engine/m1_tok.cpp" "$ROOT/engine/m1_tok.h" \
         "$ROOT/engine/gguf_model.h" "$ROOT/tools/tok_test.cpp"; do
    [ -e "$f" ] || { echo "engine_build_tok.sh: FATAL: missing $f" >&2; exit 1; }
done

mkdir -p "$BUILDS/engine"

/usr/bin/c++ -O2 -std=c++17 \
    -I "$GGML_INC" -I "$ROOT/engine" -I "$SRC/src" \
    "$ROOT/tools/tok_test.cpp" "$ROOT/engine/m1_tok.cpp" \
    "$SRC/src/unicode.cpp" "$SRC/src/unicode-data.cpp" \
    -L "$GGML_LIB" -lggml-cpu -lggml-base \
    -Wl,-rpath,"$GGML_LIB" \
    -Wl,--disable-new-dtags \
    -o "$BUILDS/engine/tok_test"

echo "engine_build_tok.sh: built $BUILDS/engine/tok_test"
