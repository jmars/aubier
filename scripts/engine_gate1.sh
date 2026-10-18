#!/bin/sh
# engine_gate1.sh — M1 GATE 1: THE TOKEN LINE (the whole-token engine vs the reference).
#
#   THE VERDICT (exit code, never the tail of stdout):
#
#     TOKEN LINE: for every prompt, the engine's greedy chain is BYTE-IDENTICAL to the
#     reference's own greedy chain over N decode positions, with 0 FAIL lines, and the
#     per-layer tier-A table (the LOCALISATION instrument) is clean: no layer above the
#     1e-6 trigger, no (layer, step) cell above the reference's 5e-10 print floor.
#
#   PASS conditions (all must hold):
#     1. every reference artifact was produced by the PINNED invocation below (or is a cached
#        artifact of it, hashed and named in this log), exits 0, and its OWN dumped
#        kq_soft_max shape says the reduction width is 256 for EVERY layer of EVERY eval;
#     2. the reference's greedy chain has N tokens (a shorter chain is INSUFFICIENT, exit 3 —
#        never a silent pass on fewer positions);
#     3. the engine exits 0 with 0 FAIL lines per prompt and its <out>.tokens is byte-identical
#        to the reference's <out>.tokens (the token line);
#     4. scripts/fork_logits_compare.py (the established tool, consuming our file unchanged)
#        reports 100.0000% top-token agreement with N positions per prompt — its full metric
#        set is printed as evidence;
#     5. the ORACLE IDENTITY control: the canonical unpatched harness and the dump-only patched
#        harness produce byte-identical .logits AND .tokens on prompt 1 (U2b/U2c verified this;
#        this gate re-derives it rather than trusting the record).
#
#   EXIT CODES (0.2d contract — CORRECTNESS IS NOT GATED BY TIMING; only the falsifier defers):
#               0 = gate PASS: ALL VERIFIED — the token line + the tier-A table + the oracle
#                   identity control held AND the falsifier decided (its settled PRE-LOOP
#                   precondition held; the verdict also quotes the loop's own load band).
#               1 = gate FAIL (token line / tier A / engine FAIL lines, or an invalid falsifier
#                   run: a falsifier exit code outside {0,70} or a ^FAIL line, or the falsifier's
#                   real side failed/mixed).  A FAIL is NEVER deferred away.
#               2 = BLIND: the gate cannot decide — the oracle identity control failed, the
#                   MISPAIR control went undetected, or the falsifier RAN (so the settled load
#                   precondition held at its start) and its no-overlap control was not rejected.
#                   The exit-2 verdict re-prints the falsifier's per-run loadavg triplets: a
#                   peer burst arriving mid-loop is visible there and makes the 2 contestable
#                   (the precondition is one PRE-LOOP instant — STALENESS below); a quiet
#                   re-run decides.  The deferral (4) can never report a blind the precondition
#                   refused to measure.
#               3 = INSUFFICIENT: the reference's chain is shorter than N (EOG), or a prompt
#                   would take the KV width past 256, so the line cannot be evaluated.
#               4 = CORRECTNESS VERIFIED, FALSIFIER DEFERRED (U4b's 4, redefined by 0.2d to mean
#                   exactly this): steps 1-3 RAN and held — 0 FAIL lines, byte-identical .tokens,
#                   tier A clean, oracle identity held — and ONLY the falsifier deferred (the
#                   settled loadavg(1m) >= QUIET_LOAD, or scripts/gpu.lock not free within
#                   LOCK_WAIT).  A "CORRECTNESS: VERIFIED" line names what held, so automation
#                   can tell this from exit 5 (nothing verified) and from 0 (all verified).
#               5 = DEFERRED UP FRONT: DEFER_UPFRONT=1 refused the run before ANY invocation —
#                   nothing was invoked, nothing is verified.  The gate NEVER defers on its own
#                   before the correctness half: that line is load-independent (U2 proved the rig
#                   byte-reproducible at 1/8/24 threads), so it always runs.
#               69 = environment/build missing (model, binaries, tokenizer, python3).
#
#   THE FALSIFIER — the SPLIT's paired statistic (U4a).  TWO units are printed; the gate
#   DECIDES on the second:
#
#     * LOOP-BODY ratio — min-of-N(t_conc)/min-of-N(t_ser) over one token through the layers
#       with the product store list (plan §Q2's statistic).  REPORTED, and NOT REACHABLE on this
#       rig: the loop body is ~70 % attention/MLA/router/shexp, which runs on the CPU in BOTH
#       arms and cannot overlap, so the overlap effect (M0 measured it as 23 % of the STORE
#       block) is diluted to ~4-5 % of the body.  MEASURED (U4a, 8 draws, n_res 64/96/104,
#       iters 15/60, loadavg 4.5-6.1): 0.925 0.950 0.956 0.962 0.967 0.970 0.979 0.979; and the
#       decomposition gives an optimum of ~0.953 over EVERY residency, i.e. AT the line.  No N
#       and no THRESH makes this unit decide (results/m1-u4a.md §3).
#     * STORE-UNIT ratio — the SAME paired drive timed on its own (the core graph still runs,
#       untimed, between the drives; same layer, stores, ids, weights and order).  This is M0's
#       unit and the unit the 0.950 line was calibrated on (results/m0-gate-fix.md §1: real
#       0.847-0.912, control 0.955-1.086).  THIS IS THE VERDICT.  It cannot pass vacuously: on
#       the CPU-only rig (SPLIT=0) the two arms are the SAME instruction sequence — the CPU
#       backend has no async iface (ggml/src/ggml-cpu/ggml-cpu.cpp NULL set_tensor_async/
#       synchronize) and ggml_backend_graph_compute_async == graph_compute (ggml-backend.cpp:601-604)
#       — and it measures ~1.00 (MEASURED, logs/engine/m1-u4a/cpuonly-storeunit.txt: 1.026).
#
#   THE SPLIT ITSELF IS REAL (U4a, MEASURED): --gpu-store builds each store on the backend its
#   spec's location axis names — VRAM ranges on ROCm0, RAM ranges on the CPU — with M0's
#   fill-all -> enqueue-all -> sync-all -> drain-all drive.  The two arms stop being the same
#   instruction sequence: the per-iteration paired ratio at n_res=64 is min 0.749 / med 0.879,
#   against 0.987-1.021 with no consistent direction on the pre-split rig (U3b §2), and at
#   n_res=64 the concurrent drive measures its ideal max(gpu_half,cpu_half) (results/m1-u4a.md §2).
#   SPLIT=1 (default) runs the falsifier on the split, so it NEEDS the GPU and scripts/gpu.lock;
#   SPLIT=0 runs the old CPU-only rig, for the comparison only (it cannot pass).
#
#   RULE (0.2d): the correctness half (steps 1-3) runs ALWAYS — it is load-independent and is
#   never gated by timing.  Only the falsifier defers: if the settled loadavg(1m) is >=
#   QUIET_LOAD, or scripts/gpu.lock cannot be taken within LOCK_WAIT seconds (a peer lane holds
#   the GPU), the falsifier section is DEFERRED and the gate exits 4 with a
#   "CORRECTNESS: VERIFIED" line — machine-distinguishable from 5 (nothing ran) and 0 (all
#   verified).  If the host IS quiet the falsifier is a PASS condition and a control that is not
#   rejected exits 2 — a BLIND is only ever reported for a falsifier that actually RAN.
#   THE SETTLED SAMPLE: the precondition compares a loadavg(1m) taken after a bounded
#   QUIET_WAIT-second settle (sampled every 5 s) against QUIET_LOAD.  The settle is not cosmetic:
#   the gate's OWN steps 1-3 raise the 1-minute average — MEASURED 5.53-5.99 right after an
#   N=8 engine loop (U3b §5) and 5.21 at U4b's fresh-run check ("the session's own tail") — so
#   a cold sample would self-defer the falsifier on an otherwise quiet host, which is exactly
#   why every fresh gate1-flat run exited 4 before this unit.  QUIET_LOAD=4.0 IS A JUDGMENT,
#   NOT A MEASUREMENT: it sits inside the gap between an idle host (~1) and ~10, a band this
#   unit's own rig never measured (it measured the noise band at load 17-51).  What IS measured,
#   on this host with a different rig (the M0 gate's falsifier, results/m0-gate-busy.md):
#   idle/load-6 -> a clean PASS, load ~10 -> exit 2 BLIND, load 25-39 -> exit 2 on 4/4 runs.
#   "Quiet" for a paired timing gate on this box therefore means load <~6 and 10 is already too
#   much, so 4.0 is a deliberately conservative point chosen inside the unmeasured gap — a
#   precondition, not a calibrated threshold.
#   STALENESS (stated, not hidden): the deferral decision compares the settled pre-loop loadavg
#   with QUIET_LOAD, so a host that goes busy DURING the falsifier loop is judged by the
#   pre-loop reading.  Mid-loop readings cannot referee this: the falsifier's own rig generates
#   load up to 4-18 on the 1-min average (MEASURED, U4b's decay-tail note), indistinguishable
#   from peer contention.  The falsifier prints its own loadavg at the start and end of every
#   run — the reading to trust for the run, reported for the reviewer, not consulted by the
#   verdict — and the PASS/BLIND verdicts quote the per-run 1-min band, and exit 2 re-prints
#   the gathered triplets, so a burst-blind is contestable without scrolling.  A control
#   rejected/not-rejected decision is therefore only ever issued for a falsifier that started
#   under the settled precondition.
#
#   CONTROL KNOBS (they exist so the gate's own failure branches are exercised on demand — a
#   gate whose failure path has never been seen is not a gate; none of them can fake a PASS):
#     WIDTH=pos       the engine runs with --kv-max 0 (U2's PRE-FIX width = the sequence length)
#                     instead of the reference's 256.  The engine asserts its width against the
#                     reference's own dumped kq_soft_max shape, so this must exit 1 with a FAIL.
#     MISPARE=1       pair prompt 1's ENGINE output against prompt 2's REFERENCE chain: a
#                     deliberate mismatch that must make the token line FAIL (exit 1).
#     THRESH=<x>      the falsifier's line (default 0.950), for its branch exercises.
#     SPLIT=1|0       the falsifier's configuration: 1 (default) = the GPU/CPU split
#                     (--gpu-store, needs scripts/gpu.lock); 0 = the CPU-only rig (it cannot pass).
#     LOCK_WAIT=<s>   seconds to wait for scripts/gpu.lock before reporting the falsifier DEFERRED
#                     (default 300).  A correctness gate must not hang behind a peer lane.
#     QUIET_LOAD=<x>  the falsifier's load precondition (default 4.0 — judgment, provenance in
#                     the RULE above).  QUIET_LOAD=-1 forces the deferral for the branch exercise
#                     (it cannot manufacture a PASS: no falsifier ratio is measured).
#     QUIET_WAIT=<s>  bounded settle before the falsifier's load sample (default 300 s; 0 =
#                     sample cold).  See THE SETTLED SAMPLE above.
#     DEFER_UPFRONT=1 exit 5 before ANY invocation — nothing verified (exercise/pre-check knob).
#     FALS_NRES=<n>   the falsifier's residency (default 64 — M0's default; at 4 layers it is
#                     1.67 GiB of VRAM and fits; the whole model at 64 does NOT fit, see §8 of
#                     results/m1-u4a.md).
#     REUSE=0         regenerate every artifact instead of reusing the cached ones.
#     DIR=<dir>       the artifact directory.  Default: logs/engine/gate1-run — gate-owned
#                     scratch with a params stamp (.gate1-params); a non-empty dir whose stamp
#                     does not match this run is REFUSED on the default, WARNED on explicit DIR=.
#
#   THE PINNED REFERENCE INVOCATION (verbatim; n_ctx 2048 is hardcoded in the harness,
#   moe_logits_dump.cpp:107, and repeated here to match the recorded U2 invocation):
#
#     LD_LIBRARY_PATH=<ROCMLIB>
#     LLAMA_MOE_SPLIT_DUMP=1 MOE_LOGITS_FILE=<out>.logits \
#     <REF_BIN> -m models/Mistral-Small-4-119B-2603-UD-IQ2_XXS.gguf \
#       -ngl 0 -fa off -ctk f16 -ctv f16 -c 2048 -b 256 -ub 1 -t <THREADS> -n <N> -p "<prompt>"
#
#     -ub 1 is what makes this gate tractable and it is NOT cosmetic: with -ub 256 the same
#     binary's greedy chain DIVERGES from its own -ub 1 chain at the third generated token
#     (U2's FINDING 1), so the per-token engine's oracle is the -ub 1 chain, and the position
#     mapping is ref step k <-> position (n_prompt-1)+k (the prompt is n_prompt single-token
#     graph evals landing in ONE get_logits_ith chunk; last-write-wins).  The engine re-derives
#     that mapping from the reference's own per-step logits checksum and prints it.
#
#   HOST: the correctness line is load-independent (the rig is thread-count independent and
#   byte-reproducible, U2), so this gate runs on a busy host; every timing/load number printed
#   here is marked, and NOTHING here is a speed claim.  The correctness path never touches the
#   GPU (-ngl 0 for the reference; the engine's token loop is CPU-only unless SPLIT=1 runs the
#   falsifier on the split, which then takes scripts/gpu.lock with a bounded wait).

set -u

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
ROCMLIB="$ROOT/venvs/rocm/lib/python3.13/site-packages/_rocm_sdk_devel/lib"
BIN="$ROOT/builds/engine/m1"
REF_BIN="${REF_BIN:-$ROOT/builds/engine/moe-logits-dump-taps}"
CANON_BIN="${CANON_BIN:-$ROOT/builds/hip-fork/bin/moe-logits-dump}"
REBUILD="$ROOT/logs/engine/m1-u2b-rebuild-ref.sh"
TOK_BIN="${TOK_BIN:-$ROOT/builds/hip-fork/bin/llama-tokenize}"
MODEL="${MODEL:-$ROOT/models/Mistral-Small-4-119B-2603-UD-IQ2_XXS.gguf}"
CMP_PY="$ROOT/scripts/fork_logits_compare.py"
# DIR's default is GATE-OWNED SCRATCH, never a recorded unit's evidence directory: the old
# default (logs/engine/m1-u3/gate1) was a fossil of the unit that created this gate, and an
# unpinned invocation silently overwrote that unit's recorded p1 artifacts — twice
# (results/gate-deferral.md §6).  Deliberately NOT timestamped: REUSE=1 must persist across
# invocations.  Every run stamps the dir (.gate1-params) after the guard below; a non-empty
# dir whose stamp does not match this run's parameters is REFUSED on the default, loudly
# warned on an explicit DIR=.
DIR_GIVEN="${DIR:-}"
DIR="${DIR:-$ROOT/logs/engine/gate1-run}"

N="${N:-128}"                       # decode positions per prompt (>=1)
THREADS="${THREADS:-24}"
KV="${KV:-256}"                     # the reference's pinned KV/mask width (asserted, not assumed)
THRESH="${THRESH:-0.950}"           # the falsifier's line
WIDTH="${WIDTH:-refpad}"            # refpad | pos  (pos = the U2 PRE-FIX width control)
MISPARE="${MISPARE:-0}"
REUSE="${REUSE:-1}"
FALSIFY_LAYERS="${FALSIFY_LAYERS:-4}"
FALSIFY_ITERS="${FALSIFY_ITERS:-15}"
FALSIFY_WARMUP="${FALSIFY_WARMUP:-3}"
FALSIFY_N="${FALSIFY_N:-3}"
SPLIT="${SPLIT:-1}"
LOCK_WAIT="${LOCK_WAIT:-300}"
FALS_NRES="${FALS_NRES:-64}"
QUIET_LOAD="${QUIET_LOAD:-4.0}"
QUIET_WAIT="${QUIET_WAIT:-300}"
DEFER_UPFRONT="${DEFER_UPFRONT:-0}"
PROMPTS_FILE="${PROMPTS_FILE:-}"
PROMPTS_N="${PROMPTS_N:-3}"

# ---------------------------------------------------------------------------
# guards: a gate must not decide without data
# ---------------------------------------------------------------------------
case "$N" in ''|*[!0-9]*) echo "engine_gate1.sh: FATAL: N='$N' is not a non-negative integer" >&2; exit 69 ;; esac
[ "$N" -ge 1 ] || { echo "engine_gate1.sh: FATAL: N=$N would compare no positions and PASS vacuously" >&2; exit 69; }
case "$PROMPTS_N" in ''|*[!0-9]*) echo "engine_gate1.sh: FATAL: PROMPTS_N='$PROMPTS_N' is not a non-negative integer" >&2; exit 69 ;; esac
[ "$PROMPTS_N" -ge 1 ] || { echo "engine_gate1.sh: FATAL: PROMPTS_N=$PROMPTS_N would compare no prompt" >&2; exit 69; }
case "$WIDTH" in refpad|pos) ;; *) echo "engine_gate1.sh: FATAL: WIDTH='$WIDTH' (refpad|pos)" >&2; exit 69 ;; esac
case "$SPLIT" in 0|1) ;; *) echo "engine_gate1.sh: FATAL: SPLIT='$SPLIT' (0|1)" >&2; exit 69 ;; esac
case "$LOCK_WAIT" in ''|*[!0-9]*) echo "engine_gate1.sh: FATAL: LOCK_WAIT='$LOCK_WAIT' is not a non-negative integer" >&2; exit 69 ;; esac
case "$FALS_NRES" in ''|*[!0-9]*) echo "engine_gate1.sh: FATAL: FALS_NRES='$FALS_NRES' is not a non-negative integer" >&2; exit 69 ;; esac
printf '%s\n' "$QUIET_LOAD" | grep -Eq '^-?[0-9]+(\.[0-9]+)?$' || { echo "engine_gate1.sh: FATAL: QUIET_LOAD='$QUIET_LOAD' is not a number" >&2; exit 69; }
case "$QUIET_WAIT" in ''|*[!0-9]*) echo "engine_gate1.sh: FATAL: QUIET_WAIT='$QUIET_WAIT' is not a non-negative integer" >&2; exit 69 ;; esac
case "$DEFER_UPFRONT" in 0|1) ;; *) echo "engine_gate1.sh: FATAL: DEFER_UPFRONT='$DEFER_UPFRONT' (0|1)" >&2; exit 69 ;; esac

[ -x "$BIN" ]         || { echo "engine_gate1.sh: FATAL: $BIN not built (run scripts/engine_build_m1.sh)" >&2; exit 69; }
[ -f "$MODEL" ]       || { echo "engine_gate1.sh: FATAL: model $MODEL missing" >&2; exit 69; }
[ -x "$TOK_BIN" ]     || { echo "engine_gate1.sh: FATAL: $TOK_BIN missing" >&2; exit 69; }
[ -x "$CANON_BIN" ]   || { echo "engine_gate1.sh: FATAL: $CANON_BIN missing" >&2; exit 69; }
[ -f "$CMP_PY" ]      || { echo "engine_gate1.sh: FATAL: $CMP_PY missing" >&2; exit 69; }
[ -d "$ROCMLIB" ]     || { echo "engine_gate1.sh: FATAL: ROCm lib dir $ROCMLIB missing" >&2; exit 69; }
command -v python3 >/dev/null || { echo "engine_gate1.sh: FATAL: python3 missing" >&2; exit 69; }

# DEFERRED UP FRONT (exit 5): a deliberate refusal BEFORE anything runs.  The gate never takes
# this path on its own — the correctness half is load-independent and always runs — so 5 means
# "nothing was invoked, nothing is verified", a different claim from 4 ("the correctness half
# ran and held; only the falsifier deferred").
if [ "$DEFER_UPFRONT" = 1 ]; then
    echo "DEFERRED UP FRONT (exit 5): DEFER_UPFRONT=1 — no invocation was run, nothing is verified."
    exit 5
fi

# the engine binary's own DT_RPATH resolves the ggml libs, but libggml-hip.so.0 carries a baked
# RUNPATH into the dead pre-move tree for ITS transitive ROCm deps, and the prebuilt harnesses in
# builds/hip-fork/bin have no rpath at all — both need this (the same workaround engine_gate0.sh
# and U2's recorded invocation use).
export LD_LIBRARY_PATH="$ROCMLIB:$ROOT/builds/hip-fork/bin"

# THE DIR GUARD (0.2d-fix): before ANY write, check the target dir against this run's
# parameters.  The stamp is written by every run that passes the guard, so a matching stamp
# means the last run here used exactly these parameters; a non-empty dir with a missing or
# mismatched stamp holds artifacts this run would mix with or overwrite.  On the DEFAULT dir
# that is refused (the briefs' exact bare command must stay safe); on an explicit DIR= it is
# a loud warning naming what is about to be overwritten.
GATE1_PARAMS="N=$N PROMPTS_N=$PROMPTS_N MODEL=$(basename "$MODEL") PROMPTS_FILE=${PROMPTS_FILE:-builtin}"
dir_stamp()     { cat "$DIR/.gate1-params" 2>/dev/null || true; }
dir_risk_list() { ls -Sl "$DIR" 2>/dev/null | awk 'NR>1 && NF>=9 { printf "    %12s B  %s\n", $5, $NF }' | head -n 8; }
mkdir -p "$DIR"
if [ -n "$(ls -A "$DIR")" ] && [ "$(dir_stamp)" != "$GATE1_PARAMS" ]; then
    if [ -z "$DIR_GIVEN" ]; then
        echo "engine_gate1.sh: FATAL: refusing to write into the default DIR: $DIR" >&2
        echo "  it is non-empty and its contents do not match this run's parameters" >&2
        echo "  (this run: $GATE1_PARAMS; the dir's stamp: $(dir_stamp)" >&2
        echo "  — a missing stamp means the dir was not created by this gate's current contract):" >&2
        dir_risk_list >&2
        echo "  writing here would mix or overwrite artifacts from a different specification, or" >&2
        echo "  destroy a recorded unit's evidence — the old default destroyed m1-u3's p1 exactly" >&2
        echo "  this way (results/gate-deferral.md §6).  Pass DIR=<dir> explicitly, or clear the scratch dir." >&2
        exit 69
    fi
    echo "WARNING: DIR=$DIR was set explicitly and its contents do not match this run's parameters"
    echo "  (this run: $GATE1_PARAMS; the dir's stamp: $(dir_stamp)"
    echo "  — a missing stamp means the dir predates this gate's params stamp).  Overwriting:"
    dir_risk_list
    echo "  Proceeding because DIR was explicit.  If this is a recorded unit's evidence directory,"
    echo "  this run clobbers it — that is how logs/engine/m1-u3/gate1 lost its p1 artifacts"
    echo "  (results/gate-deferral.md §6: recorded 67108876 B / 38f7d43b…, overwritten at 4194316 B)."
fi
printf '%s\n' "$GATE1_PARAMS" > "$DIR/.gate1-params"

# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
gate_fail=0
blind=0
insufficient=0
# the correctness half's evidence counters (for the "CORRECTNESS: VERIFIED" line on exit 4)
n_tokens_ok=0
n_engfail_total=0
oracle_ok=1
note()  { echo "$@"; }
lavg1() { cut -d' ' -f1 /proc/loadavg; }
lavg()  { cut -d' ' -f1-3 /proc/loadavg; }
mark()  { echo "---- $*  [elapsed ${SECONDS}s, loadavg $(lavg)]"; }
SECONDS=0

# the prompts: one per line, plain ASCII, no escape sequences (common_params_parse
# escape-processes -p, and both the harness and the tokenizer take the same string)
read_prompts() {
    if [ -n "$PROMPTS_FILE" ]; then cat "$PROMPTS_FILE"
    else cat <<'EOF'
The Roman Empire was one of the largest and most influential empires in world history. It began
1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
Once upon a time, in a small village at the edge of a dark forest, there lived
EOF
    fi
}
read_prompts | head -n "$PROMPTS_N" > "$DIR/prompts.txt"
n_prompts=$(grep -c . "$DIR/prompts.txt")
[ "$n_prompts" -ge 1 ] || { echo "engine_gate1.sh: FATAL: no prompts" >&2; exit 69; }

echo "================================================================"
echo "M1 GATE 1 — THE TOKEN LINE  (whole-token engine vs the reference)"
echo "================================================================"
echo "# start                    $(date -Is)  loadavg $(lavg)"
echo "# prompts                  $n_prompts  ($DIR/prompts.txt)"
sed 's/^/#   prompt: /' "$DIR/prompts.txt"
echo "# positions per prompt     $N"
echo "# threads                  $THREADS"
echo "# kv width pin             $KV  (WIDTH=$WIDTH; 'pos' is the PRE-FIX control that must FAIL)"
echo "# falsifier                layers=$FALSIFY_LAYERS iters=$FALSIFY_ITERS x $FALSIFY_N runs, thresh=$THRESH, quiet-load<=$QUIET_LOAD"
echo "# reuse cached artifacts   $REUSE"
echo "# engine                   $BIN   sha256 $(sha256sum "$BIN" | cut -c1-16)"
echo "# host                     $(uname -n), $(nproc) cpus; -ngl 0, no GPU lock, no GPU touch"

# ---------------------------------------------------------------------------
# step 1 — the reference binary (dump-only tap patch; staged from HEAD if needed)
# ---------------------------------------------------------------------------
if [ ! -x "$REF_BIN" ]; then
    echo "---- building the reference harness (in-tree tree already carries the tap patch):"
    sh "$REBUILD" "$ROOT/src/llama-fork/examples/moe-logits-dump/moe_logits_dump.cpp" "$REF_BIN" || {
        echo "engine_gate1.sh: FATAL: could not build $REF_BIN" >&2; exit 69; }
fi
[ -x "$REF_BIN" ] || { echo "engine_gate1.sh: FATAL: $REF_BIN not executable" >&2; exit 69; }

# reference provenance (a GUARD, not a print): the oracle-identity control catches a tap-vs-
# canonical DIVERGENCE but not a change that affects both binaries equally (e.g. a rebuild from a
# modified fork), so the two recorded hashes are asserted here.  These are the hashes of the
# committed-HEAD builds (src/llama-fork @ 07dccaf61).  A mismatch is a provenance failure, not a
# warning: update these two lines DELIBERATELY after re-verifying a rebuild, never to clear an
# assert.
REF_SHA="e3c22e2ea964beb292445c4729c0c6f6b0bcf779fe245a90f77f554209b36060"
CANON_SHA="e5d1e29357d6e61eb93e52b992508493105dea0d21be4875a9ffe0f95af6e5e7"
ref_sum_list() { printf '%s  %s\n%s  %s\n' "$REF_SHA" "$REF_BIN" "$CANON_SHA" "$CANON_BIN"; }
if ! ref_sum_list | sha256sum -c - >/dev/null 2>&1; then
    echo "engine_gate1.sh: FATAL: reference binary provenance — the recorded sha256 does not match" >&2
    ref_sum_list | sha256sum -c - 2>&1 | sed 's/^/  /' >&2
    echo "  (the recorded hashes are of the committed-HEAD builds; a rebuild from a modified fork" >&2
    echo "   changes both binaries together, which the oracle-identity control cannot see)" >&2
    exit 69
fi
echo "# reference (patched)      $REF_BIN  sha256 $(sha256sum "$REF_BIN" | cut -c1-16)  [PINNED, asserted]"
echo "# reference (canonical)    $CANON_BIN  sha256 $(sha256sum "$CANON_BIN" | cut -c1-16)  [PINNED, asserted]"
echo "# model                    $MODEL"

run_ref() {   # $1=bin $2=prompt $3=out-prefix
    LLAMA_MOE_SPLIT_DUMP=1 MOE_LOGITS_FILE="$3.logits" "$1" \
        -m "$MODEL" -ngl 0 -fa off -ctk f16 -ctv f16 -c 2048 -b 256 -ub 1 \
        -t "$THREADS" -n "$N" -p "$2" > "$3.stdout.txt" 2> "$3.stderr.txt"
}

# ---------------------------------------------------------------------------
# step 2 — per prompt: tokenize, run the reference, assert the width off its own artifact
# ---------------------------------------------------------------------------
i=0
while IFS= read -r PROMPT; do
    [ -n "$PROMPT" ] || continue
    i=$((i+1))
    tag="p$i"
    refpre="$DIR/ref-$tag"
    tokf="$DIR/tokens-$tag.txt"
    printf '%s' "$PROMPT" > "$DIR/prompt-$tag.txt"

    mark "prompt $i/$n_prompts: tokenize"
    ids=$(LC_ALL=C "$TOK_BIN" -m "$MODEL" -p "$PROMPT" --ids 2> "$DIR/tokenize-$tag.stderr.txt" | tr -d '[]' | tr ',' ' ')
    np=0
    for _ in $ids; do np=$((np+1)); done
    [ "$np" -ge 1 ] || { echo "FAIL: the tokenizer returned no ids for prompt $i"; gate_fail=1; continue; }
    { printf 'n_prompt %d\n' "$np"; printf '%s\n' $ids; } > "$tokf"
    total=$((np + N))
    echo "#   prompt $i: n_prompt=$np + n_decode=$N = $total positions"
    echo "#   ids: $(head -c 200 "$tokf" | tr '\n' ' ')"
    if [ "$total" -gt "$KV" ]; then
        echo "INSUFFICIENT: prompt $i needs $total cells but the reference's pinned width is $KV;"
        echo "  the width would move to 512 mid-run (llama-kv-cache.cpp:1285) and the pin would be"
        echo "  unsound — shorten the prompt or lower N.  NOT a pass."
        insufficient=1
    fi

    if [ "$REUSE" = 1 ] && [ -s "$refpre.logits" ] && [ -s "$refpre.stderr.txt" ] && [ -s "$refpre.logits.tokens" ]; then
        echo "#   reference: REUSING $refpre.logits (sha256 $(sha256sum "$refpre.logits" | cut -c1-16))"
    else
        mark "prompt $i/$n_prompts: the reference run (CPU-only, ~35 GB of weights)"
        echo "#   INVOCATION (pinned): LLAMA_MOE_SPLIT_DUMP=1 MOE_LOGITS_FILE=$refpre.logits"
        echo "#     $REF_BIN -m $MODEL -ngl 0 -fa off -ctk f16 -ctv f16 -c 2048 -b 256 -ub 1 -t $THREADS -n $N -p \"$PROMPT\""
        t0=$SECONDS
        run_ref "$REF_BIN" "$PROMPT" "$refpre"
        rc=$?
        echo "#   reference exit=$rc, $((SECONDS-t0)) s wall at loadavg $(lavg)  (LOAD-DEPENDENT; the"
        echo "#   correctness line is not: U2 proved the rig byte-reproducible at 1/8/24 threads)"
        [ "$rc" -eq 0 ] || { echo "FAIL: the reference run exited $rc (see $refpre.stderr.txt)"; gate_fail=1; }
    fi

    # the width, asserted off the reference's OWN dumped kq_soft_max shape (U2c's instrument)
    widths=$(grep -o 'kq_soft_max-[0-9]* shape=\[[0-9]*' "$refpre.stderr.txt" | sed 's/.*\[//' | sort -u | tr '\n' ' ')
    if [ -z "$widths" ]; then
        echo "FAIL: the reference dump carries no kq_soft_max shape — the reduction width cannot be"
        echo "  asserted against the artifact (is $REF_BIN the tap-patched harness?)"
        gate_fail=1
    else
        echo "#   reference KV width read off its own kq_soft_max dumps: [$widths] over $(grep -c 'kq_soft_max-' "$refpre.stderr.txt") lines"
        nw=$(printf '%s\n' $widths | wc -l)
        if [ "$nw" -ne 1 ] || [ "$widths" != "$KV " ]; then
            echo "FAIL: the reference's width is not a constant $KV ([$widths]) — the pin is unsound"
            gate_fail=1
        fi
    fi

    # the reference's own greedy chain length
    ntok=$(python3 - "$refpre.logits.tokens" <<'PY'
import struct,sys
d=open(sys.argv[1],'rb').read()
print(struct.unpack('<I',d[:4])[0] if len(d)>=4 else 0)
PY
)
    echo "#   reference chain: $ntok tokens for -n $N; get_logits_ith fingerprints: $(grep -c 'get_logits_ith' "$refpre.stderr.txt")"
    if [ "$ntok" -ne "$N" ]; then
        echo "INSUFFICIENT: the reference emitted $ntok tokens for -n $N (EOG?) — the chain cannot"
        echo "  support the $N-position line.  This is not a pass and not a failure of the engine."
        insufficient=1
    fi
done < "$DIR/prompts.txt"

# ---------------------------------------------------------------------------
# step 2b — the ORACLE IDENTITY control: canonical (unpatched) vs tap-patched harness
#   The patched harness is our oracle for the tier-A table AND the width assertion; the patch
#   is dump-only, so the two must be byte-identical.  U2b/U2c recorded that; this re-derives it.
# ---------------------------------------------------------------------------
canon="$DIR/canonical-p1"
if [ "$REUSE" = 1 ] && [ -s "$canon.logits" ] && [ -s "$canon.logits.tokens" ]; then
    echo "---- oracle identity control: REUSING $canon.logits"
else
    mark "the oracle identity control (canonical unpatched harness, prompt 1, -n $N)"
    PROMPT1=$(head -n 1 "$DIR/prompts.txt")
    run_ref "$CANON_BIN" "$PROMPT1" "$canon"
    rc=$?
    echo "#   canonical exit=$rc at loadavg $(lavg)"
fi
if [ -s "$canon.logits" ] && [ -s "$DIR/ref-p1.logits" ]; then
    if cmp -s "$canon.logits" "$DIR/ref-p1.logits" && cmp -s "$canon.logits.tokens" "$DIR/ref-p1.logits.tokens"; then
        echo "ORACLE IDENTITY: canonical vs tap-patched .logits AND .tokens byte-identical on prompt 1"
        echo "  (the dump-only patch does not move the oracle — the tier-A table and the width"
        echo "   assertion are read off the same values the canonical binary produces)"
    else
        echo "BLIND: the canonical and tap-patched harnesses disagree on prompt 1 — the oracle moved,"
        echo "  so neither the token line nor the tier A table can be interpreted.  exit 2."
        blind=1
        oracle_ok=0
    fi
else
    echo "FAIL: the oracle identity control produced no artifacts"
    gate_fail=1
    oracle_ok=0
fi

# ---------------------------------------------------------------------------
# step 3 — the engine's token loop, per prompt
# ---------------------------------------------------------------------------
kvx="$KV"
[ "$WIDTH" = pos ] && kvx=0
echo "---- engine: --kv-max $kvx  (WIDTH=$WIDTH, MISPARE=$MISPARE)"

i=0
while IFS= read -r PROMPT; do
    [ -n "$PROMPT" ] || continue
    i=$((i+1))
    tag="p$i"
    tokf="$DIR/tokens-$tag.txt"
    refpre="$DIR/ref-$tag"
    engpre="$DIR/eng-$tag"
    # MISPARE: pair prompt i's ENGINE output against prompt (i+1)'s REFERENCE chain
    if [ "$MISPARE" = 1 ]; then
        j=$((i % n_prompts + 1))
        pair_ref="$DIR/ref-p$j.logits.tokens"
        pair_ref_logits="$DIR/ref-p$j.logits"
        echo "#   prompt $i: ENGINE vs REFERENCE OF PROMPT $j  (the deliberate MISPAIR control)"
    else
        pair_ref="$refpre.logits.tokens"
        pair_ref_logits="$refpre.logits"
    fi
    [ -s "$tokf" ] || continue
    p_ok=1   # this prompt's token-line evidence, for the exit-4 CORRECTNESS line

    mark "prompt $i/$n_prompts: the engine's 36-layer token loop ($(( $(wc -l < "$tokf") - 1 + N )) positions)"
    t0=$SECONDS
    "$BIN" --loop --tokens "$tokf" --out "$engpre.logits" --n-decode "$N" --kv-max "$kvx" \
           --threads "$THREADS" --ref "$refpre.stderr.txt" --ref-tokens "$pair_ref" \
           --dump "$engpre.dump" > "$engpre.stdout.txt" 2>&1
    rc=$?
    echo "#   engine exit=$rc, $((SECONDS-t0)) s wall at loadavg $(lavg)  (LOAD-DEPENDENT, not a speed claim)"
    nfail=$(grep -c '^FAIL' "$engpre.stdout.txt")
    n_engfail_total=$((n_engfail_total + nfail))
    echo "#   engine FAIL lines: $nfail"
    grep -m5 '^FAIL' "$engpre.stdout.txt" | sed 's/^/    /'
    echo "#   --- engine summary (the localisation line + the tier-A verdict) ---"
    grep -E '^(# (kv width|arm |reference KV width|position mapping|reference: |head|built|loadavg after|wall clock)|TOKEN LINE|  first divergence|TIER A worst layer|TIER A logits fingerprint|VERDICT)' \
        "$engpre.stdout.txt" | sed 's/^/    /'
    [ "$rc" -eq 0 ] || { echo "FAIL: the engine exited $rc on prompt $i"; gate_fail=1; p_ok=0; }
    [ "$nfail" -eq 0 ] || { echo "FAIL: the engine printed $nfail FAIL line(s) on prompt $i"; gate_fail=1; p_ok=0; }

    # the token line, byte for byte
    mark "prompt $i/$n_prompts: the token line (ours vs the reference's own chain)"
    python3 - "$pair_ref" "$engpre.logits.tokens" "$N" <<'PY'
import struct,sys
def rt(p):
    d=open(p,'rb').read()
    n=struct.unpack('<I',d[:4])[0] if len(d)>=4 else 0
    return list(struct.unpack('<%di'%n, d[4:4+4*n]))
ref=rt(sys.argv[1]); eng=rt(sys.argv[2]); want=int(sys.argv[3])
n=min(len(ref),len(eng))
bad=[i for i in range(n) if ref[i]!=eng[i]]
print("  reference tokens=%d  engine tokens=%d  compared=%d  identical=%d  first_divergence=%s"
      % (len(ref),len(eng),n,n-len(bad), bad[0] if bad else "none"))
print("  ours      = %s" % (eng[:16],))
print("  reference = %s" % (ref[:16],))
ok = (not bad) and len(eng)>=want and len(ref)>=want
print("  TOKEN LINE: %s" % ("IDENTICAL over %d positions" % n if ok else "NOT identical / short"))
sys.exit(0 if ok else 1)
PY
    [ $? -eq 0 ] || { echo "FAIL: the token line is not identical on prompt $i"; gate_fail=1; p_ok=0; }
    if cmp -s "$pair_ref" "$engpre.logits.tokens"; then
        echo "  TOKEN LINE (bytes): the two .tokens files are BYTE-IDENTICAL (cmp -s), $(wc -c < "$engpre.logits.tokens") bytes"
    else
        echo "FAIL: the .tokens files differ byte for byte (cmp -s) on prompt $i"
        gate_fail=1
        p_ok=0
    fi
    if cmp -s "$pair_ref_logits" "$engpre.logits"; then
        echo "  logits (bytes): byte-identical to the reference's .logits as well"
    else
        echo "  logits (bytes): NOT byte-identical to the reference's .logits (the float last bits"
        echo "    differ; the token line and the tier A table above say by how much) — reported, not a FAIL"
    fi

    # the established comparison tool, consuming our file unchanged
    mark "prompt $i/$n_prompts: scripts/fork_logits_compare.py (reported metrics)"
    python3 "$CMP_PY" "$pair_ref_logits" "$engpre.logits" "$pair_ref" "$engpre.logits.tokens" \
        > "$engpre.compare.txt" 2>&1
    sed 's/^/    /' "$engpre.compare.txt"
    agr=$(sed -n 's/^top-token agreement  : .* = \([0-9.]*\)%.*/\1/p' "$engpre.compare.txt")
    pos=$(sed -n 's/^positions            : //p' "$engpre.compare.txt")
    echo "#   parsed: positions=$pos top-token=$agr%"
    if [ "$MISPARE" = 1 ]; then
        # the gate's FAIL path must fire on a deliberate mismatch
        if [ "$agr" = "100.0000" ] && [ "$n_prompts" -gt 1 ]; then
            echo "BLIND: the MISPAIR control produced 100.0000% agreement — the gate cannot detect a"
            echo "  mismatch.  exit 2."
            blind=1
        else
            echo "#   MISPAIR control: the mismatch is deliberate and the tool reports $agr% (not 100%)"
        fi
    else
        if [ "$pos" != "$N" ] || [ "$agr" != "100.0000" ]; then
            echo "FAIL: fork_logits_compare.py reports $pos positions / $agr% top-token agreement"
            echo "  (the gate's line is $N positions at 100.0000%; no tolerance is widened here)"
            gate_fail=1
            p_ok=0
        fi
    fi
    n_tokens_ok=$((n_tokens_ok + p_ok))
done < "$DIR/prompts.txt"

# ---------------------------------------------------------------------------
# step 4 — the falsifier: the split's paired statistic, DECIDED on the STORE unit
#   (the loop-body unit is reported alongside; it is not reachable at 0.950 on this rig —
#    results/m1-u4a.md §3 — and is therefore not the verdict.)
# ---------------------------------------------------------------------------
echo "---- the falsifier (the split's paired statistic; verdict = the STORE unit)"
echo "#   configuration: SPLIT=$SPLIT (1 = --gpu-store, ROCm0; 0 = the CPU-only rig), n_res=$FALS_NRES,"
echo "#   layers=$FALSIFY_LAYERS, iters=$FALSIFY_ITERS x $FALSIFY_N, thresh=$THRESH, quiet-load<=$QUIET_LOAD"
lf=$(lavg1)
fals_deferred=0
fals_1m=""        # the falsifier loop's own per-run loadavg(1m) readings (the band quoted at the verdict)
fals_triplets=""  # the same runs' full 1m/5m/15m triplets (re-printed gathered on exit 2)
# the load precondition is checked BEFORE any falsifier invocation (steps 1-3 above are the
# load-independent correctness runs and run regardless): a paired number taken at loadavg >=
# QUIET_LOAD is not a measurement, and with SPLIT=1 every skipped invocation would cost up to
# LOCK_WAIT seconds waiting on a peer's GPU hold.  (U3b ran them anyway and printed the bands;
# this unit skips them, because the bands it printed were noise by its own statement.)
# THE SAMPLE IS SETTLED, NOT COLD: the gate's own steps 1-3 raise the 1-minute average
# (MEASURED 5.53-5.99 after an N=8 engine loop, U3b §5; 5.21 at U4b), so the reading is taken
# after up to QUIET_WAIT seconds of settle (sampled every 5 s) — otherwise the gate would
# self-defer its falsifier on an otherwise quiet host.
echo "#   falsifier precondition: settled sample — QUIET_LOAD=$QUIET_LOAD, QUIET_WAIT=$QUIET_WAIT s"
settled=0
w=0
while :; do
    lf=$(lavg1)
    if awk -v a="$lf" -v b="$QUIET_LOAD" 'BEGIN { exit !(a+0 < b+0) }'; then
        settled=1
        [ "$w" -gt 0 ] && echo "#   falsifier precondition met after ${w}s of settle: loadavg(1m)=$lf < QUIET_LOAD=$QUIET_LOAD"
        break
    fi
    [ "$w" -ge "$QUIET_WAIT" ] && break
    sleep 5
    w=$((w + 5))
done
if [ "$settled" -eq 0 ]; then
    fals_deferred=1
    fals_reason="settled loadavg(1m)=$lf >= QUIET_LOAD=$QUIET_LOAD (waited ${w}s of QUIET_WAIT=$QUIET_WAIT) — the instrument's precondition is not met"
    echo "FALSIFIER: DEFERRED — $fals_reason."
    echo "  No falsifier invocation was run: a paired timing statistic measured at this load is not a"
    echo "  measurement (results/m0-gate-busy.md), and with SPLIT=1 each skipped invocation would"
    echo "  wait up to ${LOCK_WAIT}s for a peer lane's GPU hold.  NOT a PASS, NOT a FAIL: the exit"
    echo "  code (4) says the correctness half ran and held; only the falsifier deferred.  Re-run"
    echo "  on a quiet host."
fi

run_fals() {   # $1=real|control  $2=thresh  -> stdout; the exit code is the run's validity
    # the invocation, with --gpu-store when the split is selected.  When SPLIT=1 the run is
    # wrapped in the project's GPU lock with a bounded wait: flock exits 70 (our -E) if it could
    # not take the lock in LOCK_WAIT seconds, which is a DEFERRAL, not a gate failure.
    op="--loop --falsify --arm-store --tokens $DIR/tokens-p1.txt --n-res $FALS_NRES"
    op="$op --falsify-layers $FALSIFY_LAYERS --iters $FALSIFY_ITERS --warmup $FALSIFY_WARMUP"
    op="$op --threads $THREADS --thresh $2"
    [ "$SPLIT" = 1 ] && op="$op --gpu-store"
    [ "$1" = control ] && op="$op --serialize-both"
    if [ "$SPLIT" = 1 ]; then
        # shellcheck disable=SC2086
        flock -x -E 70 -w "$LOCK_WAIT" "$ROOT/scripts/gpu.lock" "$BIN" $op 2>&1
    else
        # shellcheck disable=SC2086
        "$BIN" $op 2>&1
    fi
}
# U4b (review Q7): a falsifier run is VALID only if it exited 0 and printed no ^FAIL line.
# The STORE-UNIT line is printed BEFORE the binary's own arm-divergence check (m1_engine.cpp:
# the line first, then the arm_diff check that returns rc=1), so a printed ratio from an
# invalid run — binary failure, crash, arm divergence — must not count toward PASS.
fals_valid() {   # $1=exit code, $2=output  -> 0 = valid
    [ "$1" -eq 0 ] || return 1
    ! printf '%s\n' "$2" | grep -q '^FAIL'
}
ratios_real=""
ratios_ctrl=""
body_real=""
body_ctrl=""
n_real_draws=0
n_ctrl_draws=0
r=1
while [ "$fals_deferred" = 0 ] && [ "$r" -le "$FALSIFY_N" ]; do
    out=$(run_fals real "$THRESH"); frc=$?
    if [ "$frc" -eq 70 ]; then
        echo "  [real $r/$FALSIFY_N] scripts/gpu.lock not free within ${LOCK_WAIT}s — DEFERRED"
        fals_deferred=1
        fals_reason="scripts/gpu.lock was not free within ${LOCK_WAIT}s (a peer lane holds the GPU)"
        break
    fi
    # the VERDICT line (the store unit) and the reported loop-body line
    line=$(printf '%s\n' "$out" | grep '^STORE-UNIT PAIRED FALSIFIER\[REAL\]:')
    bline=$(printf '%s\n' "$out" | grep '^PAIRED FALSIFIER\[REAL\]:')
    if [ -z "$line" ]; then
        echo "  [real $r/$FALSIFY_N] the falsifier produced no STORE-UNIT PAIRED line (binary exit $frc):"
        printf '%s\n' "$out" | tail -4 | sed 's/^/    /'
        gate_fail=1
    fi
    rla=$(lavg)
    echo "  [real $r/$FALSIFY_N, loadavg $rla] $line"
    fals_1m="$fals_1m${rla%% *}
"
    fals_triplets="$fals_triplets    real $r/$FALSIFY_N:  $rla
"
    printf '%s\n' "$out" | grep -E '^(store-unit c_|paired c_|# store-unit share|# paired arms|# loadavg falsifier)' | sed 's/^/    /'
    echo "    [reported, loop body — diluted by the non-overlappable core] $bline"
    ratio=$(printf '%s\n' "$line" | sed -n 's/.*ratio=\([0-9.]*\).*/\1/p')
    bratio=$(printf '%s\n' "$bline" | sed -n 's/.*ratio=\([0-9.]*\).*/\1/p')
    if fals_valid "$frc" "$out"; then
        ratios_real="$ratios_real$ratio
"
        body_real="$body_real$bratio
"
        n_real_draws=$((n_real_draws+1))
    else
        echo "  [real $r/$FALSIFY_N] FAIL: the falsifier run is INVALID (exit $frc, $(printf '%s\n' "$out" | grep -c '^FAIL') FAIL line(s)) — its ratio, if any, is NOT counted:"
        printf '%s\n' "$out" | grep '^FAIL' | head -3 | sed 's/^/    /'
        gate_fail=1
    fi
    out=$(run_fals control "$THRESH"); frc=$?
    if [ "$frc" -eq 70 ]; then
        echo "  [control $r/$FALSIFY_N] scripts/gpu.lock not free within ${LOCK_WAIT}s — DEFERRED"
        fals_deferred=1
        fals_reason="scripts/gpu.lock was not free within ${LOCK_WAIT}s (a peer lane holds the GPU)"
        break
    fi
    line=$(printf '%s\n' "$out" | grep '^STORE-UNIT PAIRED FALSIFIER\[CONTROL\]:')
    bline=$(printf '%s\n' "$out" | grep '^PAIRED FALSIFIER\[CONTROL\]:')
    if [ -z "$line" ]; then
        echo "  [control $r/$FALSIFY_N] the falsifier produced no STORE-UNIT PAIRED line (binary exit $frc):"
        printf '%s\n' "$out" | tail -4 | sed 's/^/    /'
        gate_fail=1
    fi
    cla=$(lavg)
    echo "  [control $r/$FALSIFY_N, loadavg $cla] $line"
    fals_1m="$fals_1m${cla%% *}
"
    fals_triplets="$fals_triplets    control $r/$FALSIFY_N:  $cla
"
    echo "    [reported, loop body] $bline"
    ratio=$(printf '%s\n' "$line" | sed -n 's/.*ratio=\([0-9.]*\).*/\1/p')
    bratio=$(printf '%s\n' "$bline" | sed -n 's/.*ratio=\([0-9.]*\).*/\1/p')
    if fals_valid "$frc" "$out"; then
        ratios_ctrl="$ratios_ctrl$ratio
"
        body_ctrl="$body_ctrl$bratio
"
        n_ctrl_draws=$((n_ctrl_draws+1))
    else
        echo "  [control $r/$FALSIFY_N] FAIL: the falsifier run is INVALID (exit $frc, $(printf '%s\n' "$out" | grep -c '^FAIL') FAIL line(s)) — its ratio, if any, is NOT counted:"
        printf '%s\n' "$out" | grep '^FAIL' | head -3 | sed 's/^/    /'
        gate_fail=1
    fi
    r=$((r+1))
done
# the falsifier loop's own load band; what each sample IS is stated where it is quoted
# (per-invocation readings, the falsifier's own rig included — STALENESS above)
if printf '%s' "$fals_1m" | grep -q .; then
    fals_band=$(printf '%s\n' "$fals_1m" | awk 'NF { if (lo=="" || $1+0<lo+0) lo=$1; if (hi=="" || $1+0>hi+0) hi=$1; n++ } END { printf "%s-%s of %d run(s)", lo, hi, n }')
else
    fals_band="no runs"
fi
if [ "$fals_deferred" = 0 ]; then
echo "  store-unit real    ratios (n=$n_real_draws of $FALSIFY_N drawn): $(printf '%s\n' "$ratios_real" | tr '\n' ' ')"
echo "  store-unit control ratios (n=$n_ctrl_draws of $FALSIFY_N drawn): $(printf '%s\n' "$ratios_ctrl" | tr '\n' ' ')"
echo "  loop-body  real    ratios: $(printf '%s\n' "$body_real" | tr '\n' ' ')  [reported]"
echo "  loop-body  control ratios: $(printf '%s\n' "$body_ctrl" | tr '\n' ' ')  [reported]"
n_ctrl_le=$(printf '%s\n' "$ratios_ctrl" | awk -v t="$THRESH" 'NF && $1+0 <= t+0 { n++ } END { print n+0 }')
n_real_le=$(printf '%s\n' "$ratios_real" | awk -v t="$THRESH" 'NF && $1+0 <= t+0 { n++ } END { print n+0 }')
echo "  store-unit real    <= $THRESH : $n_real_le of $n_real_draws drawn"
echo "  store-unit control <= $THRESH : $n_ctrl_le of $n_ctrl_draws drawn  (must be 0 for the control to be REJECTED)"
fi
if [ "$fals_deferred" = 1 ]; then
    echo "FALSIFIER: DEFERRED — $fals_reason."
    echo "  The split's paired statistic was NOT decided.  NOT a PASS and NOT a FAIL: the exit"
    echo "  code is 4 — the correctness half ran and held (see the CORRECTNESS: VERIFIED line"
    echo "  below); only the falsifier deferred.  Re-run when the GPU is free / the host is"
    echo "  quiet (results/m1-u4a.md §7)."
elif [ "$n_real_draws" -ne "$FALSIFY_N" ] || [ "$n_ctrl_draws" -ne "$FALSIFY_N" ]; then
    echo "FALSIFIER: INCOMPLETE — only $n_real_draws real / $n_ctrl_draws control draws of"
    echo "  $FALSIFY_N were valid (see the INVALID lines above); the split's statistic is not"
    echo "  decided on a partial set.  The gate's exit is the FAIL those runs already set."
else
    if [ "$n_ctrl_le" -eq 0 ] && [ "$n_real_le" -eq "$FALSIFY_N" ]; then
        echo "FALSIFIER: PASS — the split's STORE-unit ratio is under the line on all $FALSIFY_N runs and"
        echo "  the --serialize-both control is rejected on all of them.  The load, per sample:"
        echo "  the settled PRE-LOOP precondition sample was loadavg(1m) $lf (one instant, after up to"
        echo "  ${QUIET_WAIT}s of settle — the only quiet-host claim the verdict makes); the loop itself"
        echo "  ran at 1-min loadavg $fals_band (per-invocation readings, the falsifier's own rig"
        echo "  included — STALENESS above: printed, not consulted; each run's full triplet is inline)."
        echo "  (The loop-BODY ratios above are reported.  U4b MEASURED on a quiet host: at the"
        echo "   balance residency n_res=113 / 24 threads the body optimum sits just ABOVE the line"
        echo "   (0.959 0.980, controls 1.007/1.011), so the store unit stays the verdict; and at"
        echo "   32 threads the paired instrument itself is invalid (its controls fail at every"
        echo "   residency) — do not raise THREADS past the calibrated band to chase the body line.)"
    elif [ "$n_ctrl_le" -ne 0 ]; then
        echo "BLIND (exit 2): the falsifier RAN — the settled precondition held at its start (loadavg(1m)"
        echo "  $lf < QUIET_LOAD=$QUIET_LOAD, one PRE-LOOP instant) — and its --serialize-both control was"
        echo "  NOT rejected on $n_ctrl_le of $FALSIFY_N runs: the line cannot separate overlap from none."
        echo "  The loop itself ran at 1-min loadavg $fals_band (own rig included): if a peer burst arrived"
        echo "  mid-loop it is visible in the per-run triplets inline above, and this exit 2 is CONTESTABLE"
        echo "  by them — a quiet re-run decides."
        blind=1
    elif [ "$n_real_le" -eq 0 ]; then
        echo "FALSIFIER: FAIL (real side) — on a quiet host NO run of the concurrent split beat the"
        echo "  serialised split by the required margin ($n_real_le of $FALSIFY_N under $THRESH)."
        echo "  This is the split's own claim failing: check the SPLIT=1 / --gpu-store wiring first."
        gate_fail=1
    else
        echo "FALSIFIER: MIXED — $n_real_le of $FALSIFY_N real runs under the $THRESH line and"
        echo "  $((FALSIFY_N-n_real_le)) over it: a statistic that passes and fails on the same build"
        echo "  cannot decide (M0's rule).  exit 1."
        gate_fail=1
    fi
fi

# ---------------------------------------------------------------------------
# step 5 — the verdict
# ---------------------------------------------------------------------------
echo "================================================================"
echo "# end $(date -Is)  loadavg $(lavg)  elapsed ${SECONDS}s"
if [ "$insufficient" -eq 1 ]; then
    echo "GATE INSUFFICIENT (exit 3): a prompt's reference chain is shorter than N or past the pinned"
    echo "  KV width, so the token line could not be evaluated.  This is NOT a pass."
    exit 3
fi
if [ "$blind" -eq 1 ]; then
    echo "GATE FAILURE (exit 2): BLIND — the gate cannot decide (see the BLIND line above)."
    if [ -n "$fals_triplets" ]; then
        echo "  The falsifier's per-run loadavg triplets (1m 5m 15m, run order; own rig included):"
        printf '%s' "$fals_triplets"
        echo "  (the precondition was the ONE settled pre-loop instant quoted above — a mid-loop peer"
        echo "  burst would show in these triplets and make this exit 2 contestable; a quiet re-run decides)"
    else
        echo "  (no falsifier runs were made: this blind is the oracle-identity/MISPAIR control's, not the loop's)"
    fi
    exit 2
fi
if [ "$gate_fail" -ne 0 ]; then
    echo "GATE FAIL (exit 1): see the FAIL lines above."
    exit 1
fi
if [ "$fals_deferred" -eq 1 ]; then
    echo "GATE DEFERRED (exit 4): CORRECTNESS VERIFIED, FALSIFIER DEFERRED.  The token line held and"
    echo "  nothing failed (steps 1-3 above); the split's paired falsifier did NOT decide"
    echo "  ($fals_reason).  The falsifier is a timing claim and the precondition for it was not"
    echo "  met; the correctness line is load-independent (U2) and IS verified.  The exit code is"
    echo "  the verdict: NOT a verified split (0), NOT a refusal before anything ran (5).  Re-run"
    echo "  on a quiet host with the GPU free for the falsifier."
    echo "CORRECTNESS: VERIFIED — token line byte-identical on $n_tokens_ok/$n_prompts prompt(s) x"
    echo "  $N positions; engine FAIL lines: $n_engfail_total; oracle identity control: held;"
    echo "  reference KV width pin: $KV (WIDTH=$WIDTH)."
    exit 4
fi
echo "GATE PASS: THE TOKEN LINE holds — the engine's 36-layer greedy chain is byte-identical to"
echo "  the reference's own chain on all $n_prompts prompt(s) x $N positions, with 0 FAIL lines,"
echo "  the per-layer tier-A table clean at every layer (incl. ffn_inp/l_out, tier A's blind spot"
echo "  in U2), the reduction width read off the reference's own kq_soft_max dump and equal to"
echo "  $KV, the oracle identity control byte-identical, and fork_logits_compare.py reporting"
echo "  100.0000% top-token agreement per prompt."
echo "  SPEED: the split's paired falsifier above PASSED on this quiet run (STORE-unit verdict;"
echo "  loop-BODY reported — U4b: at 24 threads the body optimum sits just above the line)."
exit 0
