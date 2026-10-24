#!/bin/sh
# engine_gate_batchq.sh — BQ BATCHED-QUALITY GATE: batched-vs-solo on a SMALL position count.
#
#   THE VERDICT (exit code, never the tail of stdout):
#     0 = PASS   the paired comparison over the gate's positions is NOT DEGRADED at delta=0.005
#                (EQUIVALENT, IMPROVED, or INSUFFICIENT — undecided is not degraded);
#     1 = FAIL   DEGRADED (the batched arm is measurably worse beyond the margin), or any
#                plumbing failure (an engine/rig/instrument rc != 0, a pairing mismatch).
#
#   WHAT IT RUNS: chunks 0 and 1 of corpus Q3 (2 x 249 = 498 paired positions), the L88
#   certificate's own corpus files, scored TWICE with the stock baseline (the four m4_* hatches
#   OFF, rotation OFF) at --kv-max 512 on the GPU product path:
#     A = the m1 binary's single-stream --score (cached per m1 binary sha; the A side is a
#         property of the BINARY and the fixed chunks, not of the code under gate);
#     B = the batch rig at --batch 2 (one process, one group: chunks 0+1 lockstep).
#   Then the UNMODIFIED instrument's paired rule reads the widest of its three intervals.
#
#   ITS HONEST POWER (printed by the gate itself from its own run): at ~500 positions and the
#   chaotic sd this engine's perturbations carry (~0.4 nats), the MDD is ~0.035 nats — the gate
#   can only trip on GROSS degradation (>= ~3.6 % ppl).  It CANNOT certify equivalence at
#   delta=0.005; that is the full 39 840-position certificate's job (L88/L91).  A PASS here
#   means "not massively worse", nothing finer.
#
#   INSTRUMENT: the paired-NLL instrument is NOT part of this tree (it reads corpora and verdict
#   history kept out of the public repo).  Set NLL_EVAL=<path to nll_eval.py>; the gate REFUSES
#   loudly (exit 1) when it is unset or missing rather than passing on a comparison it never ran.
set -u
ROOT=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd -P); cd "$ROOT" || exit 1
export LD_LIBRARY_PATH="$ROOT/venvs/rocm/lib/python3.13/site-packages/_rocm_sdk_devel/lib:$ROOT/builds/hip-fork/bin"
export GGML_MOE_WR_DISABLE=1
NLL="${NLL_EVAL:-}"
[ -n "$NLL" ] && [ -e "$NLL" ] || {
    echo "FATAL: set NLL_EVAL to the path of the paired-NLL instrument (nll_eval.py)." >&2
    echo "  It is NOT part of this repository: it reads the corpora and verdict history that are" >&2
    echo "  deliberately kept out of it.  Without it this gate cannot measure anything, so it" >&2
    echo "  refuses rather than passing on a comparison it did not run." >&2
    exit 1
}
M1="$ROOT/builds/engine/m1"; RIG="$ROOT/builds/engine/m1_batch_test"
T="models/Mistral-Small-4-119B-2603-UD-IQ2_XXS.gguf"
H="logs/engine/m1-u6/hotset-A48.json"
CH="logs/engine/m1-h3-u11/q"
B="logs/engine/gate-batchq"; mkdir -p "$B"
NO_M4="--no-m4-kvwrite --no-m4-wvb --no-m4-kqkv --no-m4-wkb"
KV=512
N=2

fail() { echo "GATE BATCHQ: FAIL — $*"; exit 1; }

[ -x "$M1" ]  || fail "no engine binary $M1"
[ -x "$RIG" ] || fail "no rig binary $RIG (sh scripts/engine_build_m1.sh builds it)"
[ -f "$NLL" ] || fail "the quality instrument $NLL is missing — the gate needs it (NLL_EVAL=... overrides); refusing to run a silent no-op"
[ -f "$CH/q3c.chunk0.txt" ] && [ -f "$CH/q3c.chunk1.txt" ] || fail "corpus chunks $CH/q3c.chunk{0,1}.txt missing"

# ---- A side: single-stream, cached on the m1 binary's sha (the baseline is the binary's) ----
m1_sha=$(sha256sum "$M1" | cut -d' ' -f1)
CACHE="$B/cache/$m1_sha"; mkdir -p "$CACHE"
for c in 0 1; do
  [ -f "$CACHE/c$c.nll" ] && continue
  sh scripts/with_gpu_lock.sh "$M1" --loop --score "$CH/q3c.chunk$c.txt" --kv-max $KV --model "$T" \
      --out "$CACHE/c$c.logits" --arm-store --gpu-core --gpu-store --threads 16 \
      --resident-set "$H" $NO_M4 > "$B/solo-c$c.stdout" 2>&1 \
    || fail "solo --score chunk $c rc=$? (see $B/solo-c$c.stdout)"
  python3 "$NLL" score "$CACHE/c$c.logits" --nll-out "$CACHE/c$c.nll" > /dev/null || fail "instrument score on chunk $c"
  rm -f "$CACHE/c$c.logits" "$CACHE/c$c.logits.tokens"
done

# ---- B side: the batch, one process ---------------------------------------------------------
rm -f "$B/g0-s0.logits" "$B/g0-s0.logits.tokens" "$B/g0-s1.logits" "$B/g0-s1.logits.tokens"
sh scripts/with_gpu_lock.sh "$RIG" --score --batch $N \
    --tokens "$CH/q3c.chunk0.txt" --tokens "$CH/q3c.chunk1.txt" \
    --kv-max $KV --model "$T" --out "$B/b" --arm-store --gpu-core --gpu-store --threads 16 \
    --resident-set "$H" > "$B/rig.stdout" 2>&1 \
  || fail "the batched rig run rc=$? (see $B/rig.stdout)"
python3 "$NLL" score "$B/b-g0-s0.logits" --nll-out "$B/c0.nll" > /dev/null || fail "instrument score on the batch's slot 0"
python3 "$NLL" score "$B/b-g0-s1.logits" --nll-out "$B/c1.nll" > /dev/null || fail "instrument score on the batch's slot 1"
rm -f "$B"/b-g0-s*.logits "$B"/b-g0-s*.logits.tokens

# ---- the paired rule ------------------------------------------------------------------------
python3 scripts/splice_nll.py "$B/A.nll" "$CACHE/c0.nll" "$CACHE/c1.nll" > /dev/null
python3 scripts/splice_nll.py "$B/B.nll" "$B/c0.nll" "$B/c1.nll" > /dev/null
python3 "$NLL" compare-nll "$B/A.nll" "$B/B.nll" --delta 0.005 --label-B "batched N=2 (gate, 498 positions)" \
  > "$B/compare.txt"
rc=$?
cat "$B/compare.txt"
mdd=$(awk '/^MDD/{print $4}' "$B/compare.txt")
sd=$(awk '/^sd of dNLL/{print $5}' "$B/compare.txt")
echo "gate power: MDD $mdd nats at 498 positions (sd $sd) — this arm trips on GROSS degradation only;"
echo "  it CANNOT certify delta=0.005; the full 39 840-position certificate is the quality bar (L91)."
[ "$rc" = 1 ] && fail "DEGRADED: the batched arm is measurably worse beyond delta=0.005 on the gate's positions"
[ "$rc" = 0 ] || [ "$rc" = 2 ] || fail "instrument rc=$rc (input error — pairing or format)"
echo "GATE BATCHQ: PASS — not DEGRADED at delta=0.005 over 498 paired positions (verdict $(awk '/^VERDICT/{print $3}' "$B/compare.txt")); MDD $mdd nats."
exit 0
