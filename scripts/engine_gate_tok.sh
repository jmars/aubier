#!/bin/sh
# engine_gate_tok.sh — THE TOKENIZER GATE (server plan §H U1): our byte-level BPE tokenizer and
# detokenizer vs the PINNED reference's own, over a corpus, with the exit code deciding.
#
#   THE VERDICT (exit code, never the tail of stdout):
#
#     TOK PARITY: for every corpus case, our id array is ELEMENT-FOR-ELEMENT identical to the
#     reference's, AND every token's detokenized BYTES are identical, AND the reference's own
#     detokenization is lossless on the case (concat(pieces) == the input bytes — the control
#     that makes the piece-parsing trustworthy), AND the round-trip and streaming arms hold.
#
#   THE REFERENCE is builds/hip-fork/bin/llama-tokenize, the same pinned tree gate1 pins
#   (scripts/engine_gate1.sh:170 TOK_BIN).  It loads the VOCAB ONLY (llama_model_params.
#   vocab_only) — no weights, no GPU, no gpu.lock — and needs two library paths, both
#   dead-pathed otherwise.  MEASURED: ~0.35-0.6 s per invocation.
#
#   HAZARD (MEASURED, cost a 3-minute hang): piping llama-tokenize's stdout into `tail`/`head`
#   can wedge the process at 100 % CPU.  Every invocation here redirects to a FILE.
#
#   EXIT CODES:
#     0 = PASS  — all arms verified.
#     1 = FAIL  — an id array, a detokenized piece, a round-trip, or the streaming decoder
#                 disagrees with the reference, or the piece-parse control failed.  A FAIL is
#                 never softened: quality discipline says OUR tokenizer is wrong, not the oracle.
#     2 = INSUFFICIENT / BLIND — the harness could not decide: the independent GGUF reader
#                 disagrees with the C++ reader, a case could not be built, or the reference
#                 produced unparsable output.  Reported per case, never a silent pass.
#     69 = environment missing (model, tok_test, llama-tokenize, python3, ROCm libdir).
#
#   THE GATE CAN FAIL — and this is how you show it (asymmetry proof, MEASURED this unit):
#       M1_TOK_MUTANT=merge-ranks-reversed  sh scripts/engine_gate_tok.sh  # -> exit 1 (id arm)
#       M1_TOK_MUTANT=detok-no-byte-decode  sh scripts/engine_gate_tok.sh  # -> exit 1 (PIECE
#                                                                          #     MISMATCH arm)
#   The mutants live in engine/m1_tok.cpp behind that ONE env var, are OFF by default, and print
#   a loud banner when active; the shipped build exits 0.  A third mutant,
#   M1_TOK_MUTANT=ignore-merges-off, is kept as a MEASURED-and-REFUTED hypothesis: on THIS vocab
#   forcing ignore_merges off changes nothing (see the note in m1_tok.cpp), so it does NOT make
#   the gate fail — evidence that the plan's "ignore_merges changes results" does not reproduce
#   here, and a reminder that a mutant which cannot fail proves nothing.
#
#   CORPUS (built fresh by scripts/tok_parity_check.py, which also reads the GGUF with an
#   INDEPENDENT python reader and must agree with the C++ one before anything is compared):
#     the known-good strings the plan measured; every printable ASCII byte 0x20-0x7e; 3-byte
#     (CJK/Hangul) and 4-byte (emoji) codepoints; combining accents; U+00A0/U+2003/U+2028/U+3000
#     whitespace; 40 spaces; 300 digits; CRLF; a lone CR; the empty file; all 1000 special-token
#     texts (in chunks of 50, and all 1000 in one file) BOTH with parse_special=true and as
#     literal text with it false; EVERY NORMAL TOKEN TEXT in the vocab, one per line (130072
#     texts / 1.39 MB — the BPE stage over effectively the whole vocab, and the perf sanity
#     arm); every scripts/*.sh; and a ~100 KB slice of LEDGER.md (an INTERNAL file: read at run
#     time, never written into the corpus of any tracked file).
#
#   HOST: CPU-only and load-insensitive (no GPU, deterministic string code).  Nothing here is a
#   speed claim; the one timing printed is a sanity number, marked.

set -u

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
ROCMLIB="$ROOT/venvs/rocm/lib/python3.13/site-packages/_rocm_sdk_devel/lib"
REF_DIR="$ROOT/builds/hip-fork/bin"
REF_BIN="${TOK_BIN:-$REF_DIR/llama-tokenize}"
TOK_TEST="${TOK_TEST:-$ROOT/builds/engine/tok_test}"
MODEL="${MODEL:-$ROOT/models/Mistral-Small-4-119B-2603-UD-IQ2_XXS.gguf}"
PY="$ROOT/scripts/tok_parity_check.py"
CASEDIR="${CASEDIR:-$ROOT/logs/engine/gate-tok-corpus}"
OUT="${OUT:-$ROOT/logs/engine/gate-tok-run}"
# ---------------------------------------------------------------------------
# reference invocation — ALWAYS to a file, ALWAYS with </dev/null and a timeout
# ---------------------------------------------------------------------------
# THE FLAKY REFERENCE (MEASURED this unit, on an idle-ish box): builds/hip-fork/bin/
# llama-tokenize fails intermittently on this host — over 15 consecutive runs of the SAME
# trivial prompt, 2 SEGFAULTED (rc 139) and 1 HUNG past 25 s (rc 124); with -t 1 it was 1/15.
# Both symptoms appear AFTER the `llama_context: n_ctx_seq ...` line, i.e. in context creation /
# tokenization, and are properties of the PINNED BINARY, not of our code (gate1 pins the same
# binary and only ever runs the vocab path once per prompt).
#
# A crashed run prints NO id array, so it can never be mistaken for a result: python's parser
# rejects it and the case goes BLIND.  The gate therefore RETRIES a failed invocation (bounded,
# counted, PRINTED — never silent), runs the reference with -t 1 (fewer ggml spin threads; the
# tokenizer path is single-threaded, and every known-good fixture in this gate was ALSO measured
# without -t 1 and matches), and goes BLIND only when every attempt failed.  If it fails out on 3
# invocations the whole gate stops BLIND rather than grinding through the corpus.
TO="${TO:-20}"                  # per-invocation timeout (seconds): a normal run is ~0.5 s
RETRIES="${RETRIES:-5}"         # attempts per reference invocation (see THE FLAKY REFERENCE)

ref_attempt_fail=0              # attempts that failed (any rc) across the whole gate
ref_invocations=0
ref_hard_fails=0                # invocations that exhausted every attempt
ref_abort=0

[ -x "$TOK_TEST" ] || { echo "engine_gate_tok.sh: FATAL: $TOK_TEST not built (run scripts/engine_build_tok.sh)" >&2; exit 69; }
[ -x "$REF_BIN" ]  || { echo "engine_gate_tok.sh: FATAL: reference $REF_BIN missing" >&2; exit 69; }
[ -f "$MODEL" ]    || { echo "engine_gate_tok.sh: FATAL: model $MODEL missing" >&2; exit 69; }
[ -f "$PY" ]       || { echo "engine_gate_tok.sh: FATAL: $PY missing" >&2; exit 69; }
[ -d "$ROCMLIB" ]  || { echo "engine_gate_tok.sh: FATAL: ROCm lib dir $ROCMLIB missing" >&2; exit 69; }
command -v python3 >/dev/null || { echo "engine_gate_tok.sh: FATAL: python3 missing" >&2; exit 69; }

mkdir -p "$OUT"

echo "# engine_gate_tok.sh"
echo "# reference: $REF_BIN (vocab-only, no GPU, no gpu.lock)"
echo "# ours     : $TOK_TEST"
echo "# model    : $MODEL"
echo "# corpus   : $CASEDIR"

# ---------------------------------------------------------------------------
# corpus
# ---------------------------------------------------------------------------
if ! python3 "$PY" build "$CASEDIR" "$ROOT" "$TOK_TEST" "$MODEL"; then
    echo "engine_gate_tok.sh: FAIL: the corpus/control stage did not pass — nothing was compared."
    echo "VERDICT: TOK PARITY BLIND (exit 2) — the independent GGUF reader and the C++ reader " \
         "disagree, or a case could not be built."
    exit 2
fi

MANIFEST="$CASEDIR/manifest.tsv"
[ -f "$MANIFEST" ] || { echo "engine_gate_tok.sh: FATAL: $MANIFEST vanished" >&2; exit 69; }

# ---------------------------------------------------------------------------
ref_run() {   # ref_run <outfile> [args...]  -> 0 on a completed run, else the last rc
    out=$1; shift
    ref_invocations=$((ref_invocations + 1))
    attempt=1
    while [ "$attempt" -le "$RETRIES" ]; do
        LD_LIBRARY_PATH="$REF_DIR:$ROCMLIB" timeout "$TO" \
            "$REF_BIN" -m "$MODEL" -t 1 "$@" </dev/null >"$out" 2>"$out.err"
        rc=$?
        if [ $rc -eq 0 ]; then
            if [ "$attempt" -gt 1 ]; then
                echo "# reference retry: '$out' succeeded on attempt $attempt (rc of the failed attempts is in $out.err)"
                ref_attempt_fail=$((ref_attempt_fail + attempt - 1))
            fi
            return 0
        fi
        attempt=$((attempt + 1))
    done
    ref_attempt_fail=$((ref_attempt_fail + RETRIES))
    ref_hard_fails=$((ref_hard_fails + 1))
    echo "# reference FAILED all $RETRIES attempts (last rc=$rc): $out"
    return $rc
}

n_pass=0; n_fail=0; n_blind=0
failed_cases=""
blind_cases=""

note() { echo "$@"; }

case_fail() {   # case_fail <name> <reason...>
    n_fail=$((n_fail + 1))
    failed_cases="$failed_cases $1"
    echo "case $1: FAIL"
    shift
    echo "  $*"
}

case_blind() {
    n_blind=$((n_blind + 1))
    blind_cases="$blind_cases $1"
    echo "case $1: BLIND (insufficient)"
    shift
    echo "  $*"
}

# ---------------------------------------------------------------------------
# the main loop: per case, the ids arm and the detokenizer arm
# ---------------------------------------------------------------------------
while IFS="$(printf '\t')" read -r name flags; do
    [ -n "$name" ] || continue
    case="$CASEDIR/$name.txt"
    [ -f "$case" ] || { case_blind "$name" "case file $case missing"; continue; }

    case "$flags" in
        "" )   our_flags=""                 ; ref_flags=""                     ; tag="" ;;
        sp0)   our_flags="--no-parse-special"; ref_flags="--no-parse-special"   ; tag=" [parse_special=false]" ;;
        *)     case_blind "$name" "unknown flags '$flags' in the manifest"; continue ;;
    esac

    # ---- arm 1: ids, BOS ON (the pixtral default), vs the reference's default ---------------
    ref_run "$OUT/$name.ref.ids" -f "$case" --no-escape --ids $ref_flags
    rc=$?
    if [ $rc -ne 0 ]; then
        case_blind "$name$tag" "reference --ids exited $rc after $RETRIES attempt(s) (see $OUT/$name.ref.ids.err)"
        [ "$ref_hard_fails" -ge 3 ] && { echo "# ABORT: the reference failed out on $ref_hard_fails invocations — the harness is not running on this host; stopping instead of grinding the corpus"; ref_abort=1; break; }
        continue
    fi
    # shellcheck disable=SC2086
    "$TOK_TEST" --encode "$case" $our_flags >"$OUT/$name.our.ids" 2>"$OUT/$name.our.ids.err"
    rc=$?
    if [ $rc -ne 0 ]; then
        case_fail "$name$tag" "tok_test --encode exited $rc: $(tail -2 "$OUT/$name.our.ids.err" | tr '\n' ' ')"
        continue
    fi
    cmp_out=$(python3 "$PY" ids "$OUT/$name.ref.ids" "$OUT/$name.our.ids" 2>&1)
    rc=$?
    if [ $rc -ne 0 ]; then
        case_fail "$name$tag" "id arrays differ: $cmp_out"
        continue
    fi
    ids_ev=$(echo "$cmp_out" | tr -d '\n' | sed 's/^ *//')

    # ---- arm 2: detokenized BYTES, per token, BOS OFF on both sides ------------------------
    if ! ref_run "$OUT/$name.ref.pieces" -f "$case" --no-escape --no-bos $ref_flags; then
        case_blind "$name$tag" "the reference's piece run failed all $RETRIES attempts"
        [ "$ref_hard_fails" -ge 3 ] && { echo "# ABORT: the reference failed out on $ref_hard_fails invocations — stopping"; ref_abort=1; break; }
        continue
    fi
    # shellcheck disable=SC2086
    "$TOK_TEST" --pieces "$case" --no-bos $our_flags >"$OUT/$name.our.pieces" 2>/dev/null || {
        case_fail "$name$tag" "tok_test --pieces failed"
        continue
    }
    python3 "$PY" nobos "$OUT/$name.ref.ids" "$OUT/$name.nobos.ids" || {
        case_blind "$name$tag" "could not derive the reference's no-BOS id list"
        continue
    }
    cmp_out=$(python3 "$PY" pieces "$OUT/$name.ref.pieces" "$OUT/$name.our.pieces" \
                                 "$case" "$OUT/$name.nobos.ids" 2>&1)
    rc=$?
    if [ $rc -ne 0 ]; then
        case_fail "$name$tag" "detokenizer arm: $cmp_out"
        continue
    fi
    pcs_ev=$(echo "$cmp_out" | tr -d '\n' | sed 's/^ *//')

    echo "case $name$tag: PASS  ($ids_ev; $pcs_ev)"
    n_pass=$((n_pass + 1))
done <"$MANIFEST"

# ---------------------------------------------------------------------------
# control: --no-bos must be exactly "the default array minus its leading BOS"
# ---------------------------------------------------------------------------
echo "# control: --no-bos vs the default id array"
CTRL="$CASEDIR/known-01-hello-world.txt"
if ref_run "$OUT/ctrl.nobos" -f "$CTRL" --no-escape --ids --no-bos; then
    out=$(python3 "$PY" nobos-check "$OUT/known-01-hello-world.ref.ids" "$OUT/ctrl.nobos" 2>&1)
    if [ $? -eq 0 ]; then
        echo "# control --no-bos: PASS —$out"
    else
        echo "# control --no-bos: FAIL — $out"
        echo "#   --no-bos is not 'drop the leading BOS' on this build, so the piece arm's id" \
             "anchor would be unsound; exit 1"
        n_fail=$((n_fail + 1))
        failed_cases="$failed_cases control-no-bos"
    fi
else
    echo "# control --no-bos: BLIND — the reference run failed"
    n_blind=$((n_blind + 1))
fi

# ---------------------------------------------------------------------------
# round-trip + streaming arms (round-trip is implied by the piece parse control above for
# EVERY case; these run it explicitly, plus the streaming decoder, on representative cases)
# ---------------------------------------------------------------------------
echo "# round-trip / streaming arms"
for c in known-01-hello-world known-02-nihao known-05-inst-token known-06-context \
         utf8-cjk3 utf8-emoji4 utf8-combining utf8-ws-mix utf8-mixed script-engine_gate1 \
         special-all vocab-all-normal ledger-100k empty; do
    f="$CASEDIR/$c.txt"
    [ -f "$f" ] || { echo "  $c: SKIP (not in the corpus)"; continue; }
    # round-trip with parse_special BOTH ways (a special text detokenizes to its literal text)
    ok=1
    for fl in "" "--no-parse-special"; do
        # shellcheck disable=SC2086
        if ! "$TOK_TEST" --roundtrip "$f" $fl >/dev/null 2>"$OUT/$c.rt.err"; then
            echo "  $c${fl:+ (--no-parse-special)}: ROUND-TRIP FAIL: $(tail -3 "$OUT/$c.rt.err" | tr '\n' ' ')"
            ok=0
        fi
        # shellcheck disable=SC2086
        if ! "$TOK_TEST" --stream "$f" $fl >/dev/null 2>"$OUT/$c.st.err"; then
            echo "  $c${fl:+ (--no-parse-special)}: STREAM FAIL: $(tail -3 "$OUT/$c.st.err" | tr '\n' ' ')"
            ok=0
        fi
    done
    if [ $ok -eq 1 ]; then
        echo "  $c: PASS (round-trip + streaming, both special arms)"
    else
        n_fail=$((n_fail + 1))
        failed_cases="$failed_cases rt-$c"
    fi
done

# ---------------------------------------------------------------------------
# verdict
# ---------------------------------------------------------------------------
echo "# cases: PASS=$n_pass FAIL=$n_fail BLIND=$n_blind"
echo "# reference invocations: $ref_invocations, failed attempts: $ref_attempt_fail (retried), " \
     "invocations that failed out: $ref_hard_fails, RETRIES=$RETRIES TO=${TO}s"
[ -n "$failed_cases" ] && echo "# failed:$failed_cases"
[ -n "$blind_cases" ]  && echo "# blind:$blind_cases"

if [ "$ref_abort" -eq 1 ]; then
    echo "VERDICT: TOK PARITY BLIND (exit 2) — ABORTED: the reference itself is not running " \
         "reliably on this host ($ref_hard_fails invocations failed out). Nothing about the " \
         "tokenizer was decided."
    exit 2
fi
if [ "$n_pass" -eq 0 ] && [ "$n_fail" -eq 0 ]; then
    echo "VERDICT: TOK PARITY BLIND (exit 2) — no case was compared."
    exit 2
fi
if [ "$n_fail" -gt 0 ]; then
    echo "VERDICT: TOK PARITY FAIL (exit 1) — $n_fail case(s) disagree with the reference. " \
         "OUR tokenizer is wrong: fix engine/m1_tok.cpp, never the expectation."
    exit 1
fi
if [ "$n_blind" -gt 0 ]; then
    echo "VERDICT: TOK PARITY BLIND (exit 2) — $n_blind case(s) could not be decided; " \
         "$n_pass passed. Not a pass."
    exit 2
fi
echo "VERDICT: TOK PARITY PASS (exit 0) — $n_pass cases: id arrays and per-token detokenized " \
     "bytes identical to the pinned reference, round-trip and streaming hold."
exit 0
