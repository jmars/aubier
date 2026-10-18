#!/bin/sh
# loader_groups_verify.sh — the loader-groups unit's verification window (SEAM 3).
#
# Every command here was DEFERRED until engine-m1-u4c finished its headline measurement
# (the brief's priority rule): a contended run is not evidence.  Run this ONLY when
# (verified independently of any external record)
#
# The (c) fixture runs are CPU-only (builds/engine/gg_fixture links no HIP) but are deferred
# anyway per the brief; they pin 8 CPU threads and are nice'd.
#
# Evidence produced (see results/loader-groups.md for the numbers):
#   (a) fallback 1: engine_gate0.sh exit 0 AND gate1 (N=8 PROMPTS_N=1 REUSE=0) exit 0 with
#       the token line byte-identical (the gate asserts it internally vs the reference).
#   (b) fallback 2: the FULL one-group fixture built from the real model (tools/gg_onegroup_full.py),
#       then the U1 loader check on it (grouped surface, exit 0) and gate1 with MODEL= the
#       fixture and the FLAT file's reference artifacts reused as the oracle (REUSE=1):
#       the engine's greedy chain on the grouped file must be byte-identical to the flat
#       file's reference — the trivial-partition proof at full strength.
#   (c) G>1 mixed types: the derived store list + the MoE divergence as numbers
#       (gg_fixture moe mixed3 flat, and g128 flat for the no-special-case-G check),
#       plus flat vs onegroup BITWISE at fixture level.
#
# EVERY step runs (no early stop) — and the exit code is the verdict.  THE CONTRACT (0.2d):
#   0 = every step exit 0 (verified)
#   1 = at least one step FAILED (real failures are listed, and any deferred step is listed too)
#   4 = 0 failed, but at least one step DEFERRED — the gate's falsifier half did not decide
#       (engine_gate0.sh exit 4: the load precondition unmet, its rig control still run;
#       engine_gate1.sh exit 4: CORRECTNESS VERIFIED, falsifier deferred — the gate prints a
#       "CORRECTNESS: VERIFIED" line naming what held, and this runner surfaces it below).
#       NOT a pass and NOT a failure — re-run when the box is quiet.
#   5 = deferred before anything ran (the gate's DEFER_UPFRONT knob or a lock timeout): nothing
#       was verified.  Distinct from 4, and also not a failure.
#   2/3 from a gate = BLIND / INSUFFICIENT — real verdicts, counted as FAIL.  On a gate exit 2
#   the falsifier's per-run loadavg samples are surfaced next to the failure: gate1's precondition
#   is ONE pre-loop settled instant, so a peer burst arriving mid-loop makes the 2 contestable
#   (the gate prints the triplets itself; a quiet re-run decides).
# This is the project's own rule applied to the script itself: an earlier revision exited 0 while
# two steps failed, which made two real failures look green.  The last line names every failed
# step, and every deferred step is named WITH what it verified before deferring (a deferral must
# never be able to hide a correctness result).
#
# SELFTEST=1 exercises the CLASSIFICATION only (no gates run): stub steps at each code go
# through the same run()/run_note() below and the mapping is asserted; it exits 0/1 on the
# assertion and never verifies the loader groups.
#
# The two gg_fixture commands that MEASURE rather than build (moe, show) exit 0 whatever they
# print, so their claims are asserted separately by the check-* steps below (bitwise / identity
# / store count); a claim whose only instrument is a printed number is not verified.

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
cd "$ROOT"
FIX=fixtures/loader-groups
OUT=logs/engine/loader-groups
ROCMLIB="$ROOT/venvs/rocm/lib/python3.13/site-packages/_rocm_sdk_devel/lib"
MODEL=models/Mistral-Small-4-119B-2603-UD-IQ2_XXS.gguf
mkdir -p "$OUT" "$FIX"

for f in builds/engine/m1 builds/engine/gg_fixture "$MODEL"; do
    [ -e "$f" ] || { echo "loader_groups_verify.sh: FATAL: missing $f" >&2; exit 69; }
done

STEPS=0
FAILED=""
DEFERRED=""

# wait_quiet — the gate steps below carry the gate's OWN load precondition (QUIET_LOAD, default
# 4.0): it REFUSES to measure on a busy host and exits 4 (DEFERRED).  THIS RUNNER'S OWN earlier
# steps (fixture builds, model loads) raise the 1-minute load above that line, so the gate then
# defers on this script's own noise -- measured 2026-10-14: a run started at loadavg 2.96 and the
# gate steps saw >4.  Wait for the load to fall before each gate step, bounded (QUIET_WAIT, default 900s), and let the step
# itself still decide (a step that defers is reported as DEFERRED, not FAIL).
wait_quiet() {
    q="${QUIET_LOAD:-4.0}"; waited=0
    while :; do
        l=$(awk '{print $1}' /proc/loadavg)
        ok=$(awk -v a="$l" -v b="$q" 'BEGIN{print (a+0 < b+0) ? 1 : 0}')
        [ "$ok" = 1 ] && { echo "[wait_quiet] loadavg(1m)=$l < $q after ${waited}s"; return 0; }
        [ "$waited" -ge "${QUIET_WAIT:-900}" ] && { echo "[wait_quiet] loadavg(1m)=$l still >= $q after ${waited}s -- letting the gate decide (it will DEFER, exit 4)"; return 0; }
        sleep 15; waited=$((waited + 15))
    done
}
: > "$OUT/step-table.txt"

# run <logfile> <label> <command...> — captures, prints, records the REAL exit code and
# accumulates failures.  Never stops: the per-step table is the evidence.
run() {
    log=$1; label=$2; shift 2
    echo "===== $label"
    "$@" > "$OUT/$log" 2>&1
    rc=$?
    cat "$OUT/$log"
    echo "$label -> exit $rc"
    echo "exit=$rc" >> "$OUT/$log"
    echo
    STEPS=$((STEPS + 1))
    printf 'exit=%d\t%s\n' "$rc" "$label" >> "$OUT/step-table.txt"
    if [ "$rc" -eq 4 ] || [ "$rc" -eq 5 ]; then
        DEFERRED="$DEFERRED
  DEFERRED (exit $rc)  $label"
        # surface what the step DID hold before deferring: the gates print an explicit
        # CORRECTNESS / RIG CONTROL / DEFERRED UP FRONT line for exactly this reader
        surf=$(grep -E -A1 '^(CORRECTNESS|RIG CONTROL|DEFERRED UP FRONT)' "$OUT/$log" | sed 's/^/    /' || true)
        [ -n "$surf" ] && DEFERRED="$DEFERRED
$surf"
    elif [ "$rc" -eq 2 ]; then
        # 2 stays a FAIL (the gate's BLIND verdict) — but a gate1 exit 2 can be a mid-loop peer
        # burst judged by the pre-loop settled sample (the gates' STALENESS note), so the
        # falsifier's per-run loadavg lines are surfaced with the failure.
        FAILED="$FAILED
  FAIL (exit $rc, BLIND)  $label"
        surf=$(grep -E '^[[:space:]]*\[(real|control) [0-9]+/[0-9]+, loadavg |^GATE layer .*run [0-9]+/.*loadavg\(1m\)' "$OUT/$log" | sed 's/^/    /' || true)
        [ -n "$surf" ] && FAILED="$FAILED
  falsifier per-run loadavg (own rig included; a mid-loop burst makes the 2 contestable — quiet re-run decides):
$surf"
    elif [ "$rc" -ne 0 ]; then FAILED="$FAILED
  FAIL (exit $rc)  $label"; fi
    return 0
}

# run_note <logfile> <label> <command...> — runs, logs and prints, but is NOT a verdict: for a
# deliberately-failing CONTROL (whose exit 1 is the expected result) or an instrument whose
# statistic is not reproducible.  The content is asserted by the check-* step that follows.
run_note() {
    log=$1; label=$2; shift 2
    echo "===== [REPORT ONLY — not a verdict] $label"
    "$@" > "$OUT/$log" 2>&1
    rc=$?
    cat "$OUT/$log"
    echo "$label -> exit $rc (NOT counted; a check-* step asserts its content)"
    echo "exit=$rc" >> "$OUT/$log"
    echo
    return 0
}

# SELFTEST=1 — exercise the CLASSIFICATION, not the loader groups: stub steps at each verdict
# code go through the SAME run() above, and the accumulated FAILED/DEFERRED strings are
# ASSERTED against the 0.2d contract.  This is the instrument for "the runner's classification
# demonstrated at each new code"; it verifies nothing about the loader groups and runs no gate.
if [ "${SELFTEST:-0}" = 1 ]; then
    run st0.txt "selftest stub exit 0 (verified)"        sh -c 'echo "step body"; exit 0'
    run st1.txt "selftest stub exit 1 (fail)"            sh -c 'echo "FAIL: stub"; exit 1'
    run st4.txt "selftest stub exit 4 (deferred, correctness held)"   sh -c 'echo "CORRECTNESS: VERIFIED — token line byte-identical (stub)"; exit 4'
    run st5.txt "selftest stub exit 5 (deferred, nothing verified)"   sh -c 'exit 5'
    run st2.txt "selftest stub exit 2 (BLIND — counted FAIL, per-run loadavg surfaced)" \
        sh -c 'echo "  [real 1/3, loadavg 5.60 7.36 5.35] STORE-UNIT PAIRED FALSIFIER[REAL]: ratio=0.900 threshold=0.950 -> PASS"; exit 2'
    ok=1
    printf '%s\n' "$FAILED"   | grep -q  'stub exit 1'                    || ok=0
    printf '%s\n' "$FAILED"   | grep -q  'stub exit 0'                    && ok=0
    printf '%s\n' "$FAILED"   | grep -q  'stub exit 2'                    || ok=0
    printf '%s\n' "$FAILED"   | grep -q  'per-run loadavg'                || ok=0
    printf '%s\n' "$FAILED"   | grep -q  '5.60 7.36 5.35'                 || ok=0
    printf '%s\n' "$FAILED"   | grep -q  'stub exit 4'                    && ok=0
    printf '%s\n' "$FAILED"   | grep -q  'stub exit 5'                    && ok=0
    printf '%s\n' "$DEFERRED" | grep -q  'stub exit 4'                    || ok=0
    printf '%s\n' "$DEFERRED" | grep -q  'stub exit 5'                    || ok=0
    printf '%s\n' "$DEFERRED" | grep -q  'CORRECTNESS: VERIFIED'          || ok=0
    echo "===== selftest classification (FAILED/DEFERRED as accumulated by run())"
    printf 'FAILED:%s\n' "$FAILED"
    printf 'DEFERRED:%s\n' "$DEFERRED"
    if [ "$ok" -eq 1 ]; then
        echo "SELFTEST PASS: 0->verified, 1->FAIL, 2->FAIL with the falsifier loadavg surfaced, 4->DEFERRED with the correctness result surfaced, 5->DEFERRED with nothing verified."
        exit 0
    fi
    echo "SELFTEST FAIL: the classification does not match the 0.2d contract."
    exit 1
fi

# ---- fixtures: build whatever is missing (g128 alone costs ~20 min of quantization) --------
missing=""
for n in flat onegroup mixed3 g128; do
    [ -f "$FIX/$n.gguf" ] || missing="$missing $n"
done
if [ -n "$missing" ]; then
    # shellcheck disable=SC2086
    run build-fixtures.txt "build missing small fixtures:$missing" \
        nice -n 15 ./builds/engine/gg_fixture build "$FIX" $missing
else
    echo "===== small fixtures present (flat/onegroup/mixed3/g128) — not rebuilt"
    echo
fi

# ---- (c) fixture-level, CPU-only -------------------------------------------------
run show-mixed3.txt     "(c) show mixed3 (layout.md §2's 96/24/8 IQ2_XXS/IQ3_S/IQ4_XS)" \
    nice -n 15 ./builds/engine/gg_fixture show "$FIX/mixed3.gguf"
run show-g128.txt       "(c) show g128 (128 groups of one expert — no special-case G)" \
    nice -n 15 ./builds/engine/gg_fixture show "$FIX/g128.gguf"
run moe-flat-onegroup.txt "(b-fixture) flat vs onegroup — BITWISE expected" \
    nice -n 15 ./builds/engine/gg_fixture moe "$FIX/flat.gguf" "$FIX/onegroup.gguf"
run moe-mixed3-flat.txt "(c) mixed3 vs flat — store list grows, divergence as numbers" \
    nice -n 15 ./builds/engine/gg_fixture moe "$FIX/mixed3.gguf" "$FIX/flat.gguf"
run moe-g128-flat.txt   "(c) g128 vs flat — 128 stores, MoE computes, weights byte-identical" \
    nice -n 15 ./builds/engine/gg_fixture moe "$FIX/g128.gguf" "$FIX/flat.gguf"
# the same run twice more: the g128-vs-flat divergence must be REPRODUCIBLE to the digit
run moe-g128-flat-2.txt "(c) g128 vs flat — repeat 2 (reproducibility of the divergence)" \
    nice -n 15 ./builds/engine/gg_fixture moe "$FIX/g128.gguf" "$FIX/flat.gguf"
run moe-g128-flat-3.txt "(c) g128 vs flat — repeat 3 (reproducibility of the divergence)" \
    nice -n 15 ./builds/engine/gg_fixture moe "$FIX/g128.gguf" "$FIX/flat.gguf"

# the byte claim, tested in BYTES (the per-expert slice, via Store's own slice_src) — the exit
# code IS meaningful here: 0 = every expert of both tensors carries the same bytes.
run bytes-flat-onegroup.txt "(b) flat vs onegroup — per-expert bytes (the trivial partition)" \
    nice -n 15 ./builds/engine/gg_fixture bytes "$FIX/flat.gguf" "$FIX/onegroup.gguf"
run bytes-flat-g128.txt "(c) flat vs g128 — per-expert bytes ('same weights partitioned 128 ways')" \
    nice -n 15 ./builds/engine/gg_fixture bytes "$FIX/flat.gguf" "$FIX/g128.gguf"
# control: mixed3's types genuinely differ, so the byte instrument must FAIL — it is not a
# tautology that always prints 128/128
run_note bytes-flat-mixed3.txt "(c) CONTROL: flat vs mixed3 per-expert bytes (must report differences)" \
    nice -n 15 ./builds/engine/gg_fixture bytes "$FIX/flat.gguf" "$FIX/mixed3.gguf"
# REPORT ONLY: every expert routed ALONE, on both files.  MEASURED to be non-reproducible at
# cells whose y cancels to ~0 (three consecutive runs: {87}, {38,67,72,100}, {87}), so its exit
# code cannot be a verdict — but it is the instrument that shows the per-expert COMPUTE is not
# where the g128 divergence comes from.
run_note sweep-g128-flat.txt "(c) REPORT: every expert routed alone, g128 vs flat (per-expert compute)" \
    nice -n 15 ./builds/engine/gg_fixture sweep "$FIX/g128.gguf" "$FIX/flat.gguf"

# ---- the claims those runs PRINT, asserted (their exit codes cannot carry them) -------------
run check-bitwise-onegroup.txt \
    "check: (b) flat vs onegroup MoE BITWISE (0 cells differ, max|d| = 0) + loaded bytes IDENTICAL" \
    sh -c "grep -q '=== DIVERGENCE A vs B: cells 0/[0-9]* differ; max|d| = 0\.000000e+00;' '$OUT/moe-flat-onegroup.txt' &&
           grep -q 'LOADED BYTES: 1 gu + 1 dn segment(s) pairwise IDENTICAL' '$OUT/moe-flat-onegroup.txt'"
run check-bitwise-g128-weights.txt \
    "check: (c) g128 vs flat — 256/256 EXPERT SLICES byte-identical (the partition claim, in bytes)" \
    sh -c "grep -q '=== LOADED BYTES (per expert): 256/256 expert slices byte-identical' '$OUT/moe-g128-flat.txt' &&
           grep -q 'gu   128/128 expert slices byte-identical' '$OUT/moe-g128-flat.txt' &&
           grep -q 'dn   128/128 expert slices byte-identical' '$OUT/moe-g128-flat.txt'"
run check-g128-128stores.txt \
    "check: (c) g128 derives 128 stores (one per group — no special-case G)" \
    sh -c "grep -q 'derived store list (n_res=64): 128 store(s)' '$OUT/show-g128.txt' &&
           [ \$(grep -c '^    store \[' '$OUT/show-g128.txt') -eq 128 ] &&
           grep -q 'ne=\[4096,4096,128\]  128 segment(s)' '$OUT/show-g128.txt' &&
           grep -q 'ne=\[2048,4096,128\]  128 segment(s)' '$OUT/show-g128.txt'"
# g128 vs flat is NOT bitwise, and cannot be (the store PARTITION changes the order in which the
# routed experts' weighted outputs are summed; float addition is not associative).  The expected
# bitwise result is REFUTED here — see results/loader-groups.md §7 — so what is asserted instead
# is the bound that separates a rounding-order difference from a wrong-weight one: every
# difference <= 1e-6 relative to its own cell (MEASURED: 1.19e-07 worst per cell = ~1 float32
# ULP; a real weight error shows up ~5e-1, as mixed3's own divergence does).
run check-g128-divergence-shape.txt \
    "check: (c) g128 vs flat — divergence is 1-ULP PER CELL (rel<=1e-6), NOT bitwise: the finding" \
    sh -c "grep -q 'DIVERGENCE A vs B: cells [1-9]' '$OUT/moe-g128-flat.txt' &&
           r=\$(sed -n 's/^=== DIVERGENCE A vs B:.*rel = \([0-9.e+-]*\)$/\1/p' '$OUT/moe-g128-flat.txt') &&
           v=\$(sed -n 's/^=== DIVERGENCE SHAPE: worst per-cell relative difference = \([0-9.e+-]*\).*/\1/p' '$OUT/moe-g128-flat.txt') &&
           [ -n \"\$r\" ] && [ -n \"\$v\" ] &&
           awk -v r=\"\$r\" -v v=\"\$v\" 'BEGIN { exit !(r+0 <= 1e-6 && v+0 <= 1e-6) }'"
run check-g128-reproducible.txt \
    "check: (c) g128 vs flat — the divergence is REPRODUCIBLE to the digit over 3 runs" \
    sh -c "a=\$(grep '^=== DIVERGENCE A vs B' '$OUT/moe-g128-flat.txt') &&
           b=\$(grep '^=== DIVERGENCE A vs B' '$OUT/moe-g128-flat-2.txt') &&
           c=\$(grep '^=== DIVERGENCE A vs B' '$OUT/moe-g128-flat-3.txt') &&
           [ \"\$a\" = \"\$b\" ] && [ \"\$b\" = \"\$c\" ]"
run check-bytes-control.txt \
    "check: (c) CONTROL — the byte instrument DOES report mixed3's 64 type-differing experts" \
    sh -c "grep -q 'gu   96/128 expert slices byte-identical' '$OUT/bytes-flat-mixed3.txt' &&
           grep -q 'dn   96/128 expert slices byte-identical' '$OUT/bytes-flat-mixed3.txt' &&
           grep -q 'differing: 32 type' '$OUT/bytes-flat-mixed3.txt' &&
           grep -q 'DIFFERING EXPERT BYTES' '$OUT/bytes-flat-mixed3.txt'"
run check-mixed3-4stores.txt \
    "check: (c) mixed3 derives 4 stores over 3 segments at cuts {0,64,96,120,128}" \
    sh -c "grep -q 'derived store list (n_res=64): 4 store(s)' '$OUT/show-mixed3.txt' &&
           grep -q 'store \[ 96,120) RAM' '$OUT/show-mixed3.txt' &&
           grep -q 'store \[120,128) RAM' '$OUT/show-mixed3.txt'"

sha256sum "$FIX"/*.gguf | tee "$OUT/fixture-sha256.txt"

# ---- (b) full one-group fixture --------------------------------------------------
if [ ! -f "$FIX/onegroup-full.gguf" ]; then
    run build-onegroup.txt "(b) building the FULL one-group fixture (~35 GB stream copy)" \
        nice -n 10 python3 tools/gg_onegroup_full.py "$MODEL" "$FIX/onegroup-full.gguf"
else
    echo "===== (b) $FIX/onegroup-full.gguf present -- not rebuilt"
    echo
fi

run u1-onegroup-full.txt "(b) U1 loader check on the FULL one-group fixture (grouped surface)" \
    env LD_LIBRARY_PATH="$ROCMLIB" ./builds/engine/m1 --model "$FIX/onegroup-full.gguf"

# ---- (a) fallback 1: today's file, the two gates ---------------------------------
# Both gates carry the 0.2d deferral contract: on a host over their QUIET_LOAD the falsifier
# half defers (exit 4) with the correctness/rig half still run and surfaced; exit 5 = deferred
# before anything ran.  wait_quiet gives each step a quiet window first (bounded); the step
# itself still decides, and a step that defers is reported as DEFERRED — with what it verified
# — never as FAIL.
wait_quiet
run gate0.txt "(a) engine_gate0.sh (M0 gate, real flat file)" sh scripts/engine_gate0.sh

wait_quiet
run gate1-flat.txt "(a) engine_gate1.sh fresh (N=8 PROMPTS_N=1 REUSE=0, real flat file)" \
    env N=8 PROMPTS_N=1 REUSE=0 DIR="$PWD/$OUT/gate1-flat" sh scripts/engine_gate1.sh

# the flat-file LOADER control, same instrument as the grouped run below: the key-absent file
# must still be 579/579 with the checklist clean (the pre/post control for the grouped change)
run u1-flat.txt "(a) U1 loader check on the REAL flat model (key absent — the regression)" \
    env LD_LIBRARY_PATH="$ROCMLIB" ./builds/engine/m1 --model "$MODEL"
run check-u1-flat.txt \
    "check: (a) flat U1 says PASS at 579/579 FILE TENSORS, ungrouped message" \
    sh -c "grep -q '^CHECK: PASS.*579/579 FILE TENSORS classified \[ungrouped file:' '$OUT/u1-flat.txt'"

# ---- (b) the token line on the one-group fixture ---------------------------------
# The reference side REUSES the flat run's oracle artifacts (the proposition is that the
# grouped file behaves identically to the flat file, so the flat file's reference IS its
# reference); the tokenizer and the ENGINE run on the grouped fixture itself.
# The copy is a CHECKED step: if it were to fail, gate1 would silently REGENERATE the
# reference from the grouped file and the proof would collapse to grouped-vs-grouped.
run copy-oracle.txt "(b) copy the FLAT run's oracle artifacts into the onegroup DIR (REUSE=1)" \
    sh -c "mkdir -p '$OUT/gate1-onegroup' &&
           cp '$OUT/gate1-flat'/ref-p*.logits '$OUT/gate1-flat'/ref-p*.logits.tokens \
              '$OUT/gate1-flat'/ref-p*.stderr.txt '$OUT/gate1-flat'/canonical-p1.* \
              '$OUT/gate1-onegroup/'"
wait_quiet
run gate1-onegroup.txt "(b) engine_gate1.sh with MODEL=onegroup-full (REUSE=1, flat oracle)" \
    env MODEL="$PWD/$FIX/onegroup-full.gguf" N=8 PROMPTS_N=1 REUSE=1 DIR="$PWD/$OUT/gate1-onegroup" \
        sh scripts/engine_gate1.sh
run check-onegroup-reused.txt \
    "check: (b) gate1-onegroup REUSED the flat artifacts (the oracle was not regenerated)" \
    sh -c "grep -q 'reference: REUSING' '$OUT/gate1-onegroup.txt' &&
           grep -q 'oracle identity control: REUSING' '$OUT/gate1-onegroup.txt'"

# ---- the verdict -----------------------------------------------------------------
echo "===== per-step exit codes"
cat "$OUT/step-table.txt"
echo "===== $STEPS steps run"
if [ -n "$FAILED" ]; then
    echo "LOADER-GROUPS VERIFICATION: FAIL — $STEPS steps, failed:$FAILED"
    # A DEFERRED step is NOT a failure (the gate's own exit 4: the host could not support the
    # measurement) -- but the verification is incomplete, so it is reported here rather than hidden.
    [ -n "$DEFERRED" ] && echo "  (also deferred, not failed:$DEFERRED)"
    exit 1
fi
if [ -n "$DEFERRED" ]; then
    echo "LOADER-GROUPS VERIFICATION: DEFERRED — $STEPS steps ran, 0 failed, but these deferred:$DEFERRED"
    echo "  exit 4 = the gate's own DEFERRED code (loadavg above its quiet threshold): re-run when quiet."
    exit 4
fi
echo "LOADER-GROUPS VERIFICATION: PASS — all $STEPS steps exit 0"
echo "  (exit codes and the byte-identity lines above are the evidence;"
echo "   logs under $OUT/, the numbers in results/loader-groups.md)"
exit 0
