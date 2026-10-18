#!/bin/sh
# engine_gate0.sh — M0 gate: run the split primitive on BOTH down-type layers and check
# the falsifier.  The gate is the EXIT CODE, never the tail of stdout.
#
#   THE VERDICT (revision 3 — see "WITHDRAWN" below):
#
#     PAIRED FALSIFIER[REAL]: min-of-N(c_conc) / min-of-N(c_ser) <= 0.950
#
#   c_conc and c_ser are the SPLIT arm timed twice by the binary inside ONE timing loop —
#   one iteration each, alternating, over the same store list / ids / weights / activation /
#   iteration count / warmup.  Because both arms share the machine state, the pairing cancels
#   the between-invocation drift that killed the previous statistic, which divided c by a
#   SEPARATELY-MEASURED bound (see below).  gpu_half, cpu_half, c/max(a,b) and c/(a+b) are
#   still printed by the binary, as REPORTED decomposition — never verdicts.
#
#   WITHDRAWN (2026-10-11, results/m0-gate-idle.md, task 0.2b): the previous verdict
#   `c_min <= 0.95·(gpu_half_min + cpu_half_min)` was not reproducible on an idle host —
#   six invocations exited 0,0,1,1,0,1 because the denominator (a separately-measured bound)
#   drifts 13–41% between invocations against a 5% margin, while the effect it divides is
#   repeatable to ~3%.  That statistic is still PRINTED by the binary, labelled
#   `[reported, WITHDRAWN as a verdict]`, and is no longer read by this script.
#
#   PASS conditions (all must hold):
#     1. (b) CPU-only is BITWISE identical to the ggml CPU oracle   (harness control: (b)
#        runs the same kernel as the oracle, so any non-zero error means the rig is broken).
#     2. THE FALSIFIER: ALL N real runs of BOTH layers give a paired ratio <= 0.950.
#     3. (a)/(c) final y agree with the oracle within 0.1 abs        (gross sanity ceiling;
#        the per-(tensor,type) tolerance is DERIVED and printed by the binary).
#     4. THE ASYMMETRY: ALL N runs with `--serialize-both` give a paired ratio > 0.950.
#        `--serialize-both` forces the SERIALISED implementation into the CONCURRENT arm too,
#        so both arms do the same work and the ratio is ~1.00 BY CONSTRUCTION — the control is
#        produced by the binary, not by this script "knowing" the answer.  If a control run
#        is NOT rejected the gate exits 2 with "the falsifier is BLIND": a gate nobody has
#        seen fail is not a gate.
#
#   REPEATABILITY IS PART OF THE GATE.  N=5 (override with N=) real and N=5 control runs per
#   layer; every one of the N ratios is printed, and the decision is taken on all of them:
#     real side     ALL N <= 0.950 -> PASS;  ALL N  > 0.950 -> the premise is DEAD;
#                   otherwise (mixed)    -> FAIL, naming the runs that crossed.
#     control side  ALL N  > 0.950 -> the required rejection; ANY N <= 0.950 -> BLIND (exit 2).
#
#   EXIT CODES (0.2d deferral contract — a TIMING verdict is only issued when the host could
#   support the measurement; correctness/rig checks are NOT gated by load):
#     0 = gate PASS: all N real ratios <= THRESH on BOTH layers, ALL controls rejected, rig
#         checks clean, and the load precondition held for the whole run (QUIET_LOAD below).
#     1 = gate FAIL: the rig is broken (a structural check — mode lines, the (b) bitwise
#         control, arms-agree, the sanity ceiling), the real side is above the line, or MIXED.
#         A FAIL is NEVER deferred away — including one found by the deferred path's rig control.
#     2 = FALSIFIER IS BLIND: a --serialize-both control was not rejected — ON A QUIET HOST only
#         (the load precondition held at every reading of the run).  On a loaded host this exact
#         signature is what contention produces (0.2c MEASURED: exit 2 on 4/4 runs at load
#         25-39, controls 0.890-0.947), so a control at/under the line with the precondition
#         violated is DEFERRED (exit 4), never 2 — the deferral must not be able to hide a real
#         BLIND either: the numbers are printed and a quiet re-run decides.
#     4 = FALSIFIER DEFERRED: the load precondition was not met (at start, or violated mid-run),
#         so no timing verdict is issued.  The rig control STILL RUNS and is reported (the
#         "RIG CONTROL:" line), because that check is load-independent and is what distinguishes
#         "the rig is broken" (exit 1) from "the host could not support the measurement" (4).
#         NOT a PASS, NOT a FAIL: re-run on a quiet host.
#     5 = DEFERRED UP FRONT: nothing ran, nothing is verified — DEFER_UPFRONT=1 (exercise or
#         pre-check knob), or scripts/gpu.lock not free within LOCK_WAIT so even the deferred
#         path's rig control could not run.  Distinct from 4: "nothing was verified" is not the
#         same claim as "the falsifier deferred with the rig checked".
#    69 = environment/build missing.
#
#   The two layers cover both down-expert types (the file is MIXED within the model):
#     layer 0 -> ffn_down_exps = IQ2_S     (seed 1234 routes 2 hits / 2 misses at n-res 64)
#     layer 4 -> ffn_down_exps = IQ3_S     (seed 2    routes 2 hits / 2 misses at n-res 64)
#
# LD_LIBRARY_PATH: the engine binary's own DT_RPATH resolves the ggml libs, but
# libggml-hip.so.0 carries a baked RUNPATH into the DEAD pre-move tree for ITS transitive
# ROCm deps (libamdhip64/libhipblas/librocblas) — DT_RPATH does not propagate past a
# RUNPATH-bearing object, so the ROCm dir is supplied here (the only inherited workaround).

set -u

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
ROCMLIB="$ROOT/venvs/rocm/lib/python3.13/site-packages/_rocm_sdk_devel/lib"
BIN="$ROOT/builds/engine/moe_split"

# THRESH is overridable ONLY so the gate's own failure branches can be exercised on demand —
# a gate whose failure path is never seen is not a gate.  THRESH above the control's ~1.00
# leaves the control UN-rejected -> exit 2 (THE FALSIFIER IS BLIND); THRESH below the real
# build's ~0.85-0.93 fails the real side -> exit 1 (premise DEAD).  It cannot be used to fake a
# PASS: a PASS needs the real ratio <= THRESH AND the control ratio > THRESH in the SAME run,
# and the two measured populations are ~0.85-0.93 and ~0.97-1.06 — no single line satisfies
# both when the build is real.  Only the default 0.950 is a verdict.
THRESH="${THRESH:-0.950}"
N="${N:-5}"

#   THE LOAD PRECONDITION (0.2d).  A paired TIMING verdict is only issued when the host could
#   support the measurement.  The instrument is loadavg(1m) against QUIET_LOAD; readings are
#   taken at start and after every falsifier invocation, and the decision uses the MAX (a
#   post-invocation 1-min average mostly reflects that invocation's own window — the staleness
#   is stated here, not hidden).  QUIET_LOAD=4.0 IS A JUDGMENT, NOT A MEASUREMENT — the same
#   point engine_gate1.sh has used since U3b — chosen against the 0.2c bracket
#   (results/m0-gate-busy.md): idle and load ~6 -> the gate decided cleanly (exit 0; m1-u1's
#   ratios were taken at load 5.6-6.6), load ~10 -> exit 2 BLIND (a control draw at 0.949),
#   load 25-39 -> exit 2 on 4/4.  4.0 sits below the lowest load at which the gate ever refused
#   to decide, at the conservative end of the unmeasured 1..10 gap: a precondition, not a
#   calibrated threshold.  The gate's OWN falsifier runs are light (--nt 1; the m0-gate-fix
#   session read loadavg ~1.2 while its runs were going), but its own contribution is NOT
#   subtracted — the readings include it (INTERPRETATION: ~2.8 of headroom to 4.0).
#   CONSEQUENCE (INTERPRETATION, from that arithmetic): with sustained peer load between
#   ~QUIET_LOAD-1.2 and QUIET_LOAD the mid-run max is always above the line, so exit 2 is
#   effectively unreachable there and every genuine blind converts to 4 after a full run —
#   a stream of 4s on a moderately-loaded box is this property, not a rig fault.
QUIET_LOAD="${QUIET_LOAD:-4.0}"
QUIET_WAIT="${QUIET_WAIT:-0}"       # s to wait at start for the load to fall before deferring
LOCK_WAIT="${LOCK_WAIT:-300}"       # s the DEFERRED path's rig control waits for scripts/gpu.lock
DEFER_UPFRONT="${DEFER_UPFRONT:-0}" # 1 = refuse before ANY invocation (exit 5, nothing verified)

#   CONTROL KNOBS (they exist so the gate's own branches are exercised on demand — a gate whose
#   failure path has never been seen is not a gate; none of them can fake a PASS):
#     THRESH=<x>       the falsifier's line (default 0.950; only the default is a verdict).
#                      Under the 0.2d contract the THRESH branch exercises need a QUIET host:
#                      a loaded host defers (exit 4) instead of deciding.
#     N=<n>            runs per layer per mode (default 5; >= 1 enforced below).
#     QUIET_LOAD=<x>   the load precondition (default 4.0, provenance above).  QUIET_LOAD=-1
#                      forces the deferral for the branch exercise (it cannot manufacture a
#                      PASS: no falsifier ratio is measured).
#     QUIET_WAIT=<s>   bounded settle wait at start (default 0 = sample immediately; the loader
#                      runner runs its own wait_quiet before the gate).
#     LOCK_WAIT=<s>    the deferred path's rig-control lock wait (default 300).  The NORMAL path
#                      keeps with_gpu_lock.sh's block-until-free behaviour (unchanged since 0.2b).
#     DEFER_UPFRONT=1  exit 5 before anything runs (exercise/pre-check knob).

# A run of ZERO measurements satisfies "ALL N real ratios <= THRESH" vacuously and would print
# GATE PASS having measured nothing (independent review, 2026-10-11).  The knobs are for
# exercising the failure branches; neither may make the gate decide without data.
case "$N" in
    ''|*[!0-9]*) echo "engine_gate0.sh: FATAL: N='$N' is not a non-negative integer" >&2; exit 69 ;;
esac
[ "$N" -ge 1 ] || { echo "engine_gate0.sh: FATAL: N=$N would measure nothing and PASS vacuously" >&2; exit 69; }

# the new knobs are validated like N: a malformed float would make every awk comparison lie
printf '%s\n' "$QUIET_LOAD" | grep -Eq '^-?[0-9]+(\.[0-9]+)?$' || { echo "engine_gate0.sh: FATAL: QUIET_LOAD='$QUIET_LOAD' is not a number" >&2; exit 69; }
case "$QUIET_WAIT" in ''|*[!0-9]*) echo "engine_gate0.sh: FATAL: QUIET_WAIT='$QUIET_WAIT' is not a non-negative integer" >&2; exit 69 ;; esac
case "$LOCK_WAIT" in ''|*[!0-9]*) echo "engine_gate0.sh: FATAL: LOCK_WAIT='$LOCK_WAIT' is not a non-negative integer" >&2; exit 69 ;; esac
case "$DEFER_UPFRONT" in 0|1) ;; *) echo "engine_gate0.sh: FATAL: DEFER_UPFRONT='$DEFER_UPFRONT' (0|1)" >&2; exit 69 ;; esac

[ -x "$BIN" ] || { echo "engine_gate0.sh: FATAL: $BIN not built (run scripts/engine_build.sh)" >&2; exit 69; }
[ -d "$ROCMLIB" ] || { echo "engine_gate0.sh: FATAL: ROCm lib dir $ROCMLIB missing" >&2; exit 69; }

export LD_LIBRARY_PATH="$ROCMLIB"
export GGML_CUDA_GRAPH_OPT=0   # keep CUDA-graph capture deterministic (M0 hygiene)

lavg1() { cut -d' ' -f1 /proc/loadavg; }

# DEFERRED UP FRONT (exit 5): a deliberate refusal BEFORE anything runs.  Distinct from 4
# ("the falsifier deferred, the rig control ran and is reported"): nothing was invoked here,
# so nothing — not even the rig state — is verified.
if [ "$DEFER_UPFRONT" = 1 ]; then
    echo "DEFERRED UP FRONT (exit 5): DEFER_UPFRONT=1 — no invocation was run, nothing is verified."
    exit 5
fi

# load_ok_wait: 0 iff loadavg(1m) < QUIET_LOAD, after an optional bounded settle wait
# (QUIET_WAIT s, sampled every 5 s).  Sets _last_load to the final reading either way.
_last_load=$(lavg1)
load_ok_wait() {
    _waited=0
    while :; do
        _last_load=$(lavg1)
        if awk -v a="$_last_load" -v b="$QUIET_LOAD" 'BEGIN { exit !(a+0 < b+0) }'; then
            [ "$_waited" -gt 0 ] && echo "load precondition met after ${_waited}s: loadavg(1m)=$_last_load < QUIET_LOAD=$QUIET_LOAD"
            return 0
        fi
        [ "$_waited" -ge "$QUIET_WAIT" ] && return 1
        sleep 5
        _waited=$((_waited + 5))
    done
}


# run_bin: $1=layer $2=seed $3=mode(real|control)  ->  the binary's stdout+stderr.
run_bin() {
    if [ "$3" = control ]; then
        "$ROOT/scripts/with_gpu_lock.sh" "$BIN" --layer "$1" --seed "$2" --iters 20 --warmup 3 --nt 1 --serialize-both
    else
        "$ROOT/scripts/with_gpu_lock.sh" "$BIN" --layer "$1" --seed "$2" --iters 20 --warmup 3 --nt 1
    fi
}

# structural: $1=mode, run output on stdin.  0 iff every non-timing condition holds.
# These are the same conditions revision 2 checked: the mode lines, the (b) bitwise harness
# control, the per-arm agreement rig check and the gross sanity ceiling on (a)/(c).
structural() {
    _mode=$1
    _out=$(cat)
    _h=$(printf '%s\n' "$_out" | grep '^# halves mode:')
    case "$_h" in
        *CONCURRENT*) ;;
        *) echo "FAIL: the (c) split arm did not run CONCURRENT: $_h"; return 1 ;;
    esac
    _p=$(printf '%s\n' "$_out" | grep '^# paired mode:')
    if [ "$_mode" = control ]; then
        case "$_p" in
            *CONTROL*) ;;
            *) echo "FAIL: the --serialize-both control did not enter CONTROL mode: $_p"; return 1 ;;
        esac
    else
        case "$_p" in
            *REAL*) ;;
            *) echo "FAIL: the real run did not enter REAL paired mode (the flag must reach it): $_p"; return 1 ;;
        esac
    fi
    _b=$(printf '%s\n' "$_out" | grep '^\[(b) CPU-only\] vs CPU-oracle')
    case "$_b" in
        *"y max|d|=0.000e+00"*) ;;
        *) echo "FAIL: (b) is not bitwise-identical to the CPU oracle (harness broken): $_b"; return 1 ;;
    esac
    _r=$(printf '%s\n' "$_out" | grep '^# paired arms compute the same thing:')
    case "$_r" in
        *"= 0.000e+00"*) ;;
        *) echo "FAIL: the two paired arms do not compute the same thing (the ratio is meaningless): $_r"; return 1 ;;
    esac
    for arm in "(a) GPU-only" "(c) SPLIT"; do
        line=$(printf '%s\n' "$_out" | grep -F "[$arm] vs CPU-oracle")
        maxd=$(printf '%s\n' "$line" | sed -n 's/.*y max|d|=\([0-9.e+-]*\).*/\1/p')
        ok=$(printf '%s\n' "$maxd" | awk '{ exit !($1 < 0.1) }' && echo yes || echo no)
        if [ "$ok" != yes ]; then
            echo "FAIL: $arm final-y max|d|=$maxd >= 0.1 sanity ceiling"
            return 1
        fi
    done
    return 0
}

gate_fail=0
blind=0

count_le() { printf '%s\n' "$1" | awk -v t="$2" 'NF { if ($1+0 <= t+0) n++ } END { print n+0 }'; }
count_gt() { printf '%s\n' "$1" | awk -v t="$2" 'NF { if ($1+0 >  t+0) n++ } END { print n+0 }'; }
count_nonempty() { printf '%s\n' "$1" | awk 'NF { n++ } END { print n+0 }'; }
list_le() { printf '%s\n' "$1" | awk -v t="$2" 'NF && $1+0 <= t+0 { printf "%s ", $1 }'; }
list_gt() { printf '%s\n' "$1" | awk -v t="$2" 'NF && $1+0 >  t+0 { printf "%s ", $1 }'; }

# THE LOAD PRECONDITION, and THE DEFERRED PATH (exit 4): the falsifier's timing verdict is not
# issued, but the RIG CONTROL still runs — the (b)-bitwise / mode / arms-agree checks are
# load-independent, and they are what keeps "the rig is broken" (exit 1) distinguishable from
# "the host could not support the measurement" (4).  One real-mode invocation, iters 5 (a
# quarter of a normal run — only the numerics lines are read; any ratio this invocation prints
# is REPORTED, not a verdict).
deferred=0
defer_reason=""
load_readings=""
if ! load_ok_wait; then
    deferred=1
    defer_reason="loadavg(1m)=$_last_load >= QUIET_LOAD=$QUIET_LOAD at start (QUIET_WAIT=$QUIET_WAIT)"
fi
if [ "$deferred" -eq 1 ]; then
    echo "================================================================"
    echo "FALSIFIER: DEFERRED — $defer_reason."
    echo "  A paired timing ratio measured at this load is not a measurement (results/m0-gate-busy.md:"
    echo "  at load 25-39 the gate could not decide on 4/4 runs).  No timing verdict will be issued."
    echo "  The rig control still runs (layer 0, seed 1234, real mode, iters 5):"
    rig_out=$(flock -x -E 70 -w "$LOCK_WAIT" "$ROOT/scripts/gpu.lock" \
        "$BIN" --layer 0 --seed 1234 --iters 5 --warmup 1 --nt 1 2>&1)
    rrc=$?
    if [ "$rrc" -eq 70 ]; then
        echo "DEFERRED UP FRONT (exit 5): scripts/gpu.lock was not free within ${LOCK_WAIT}s — even the"
        echo "  rig control could not run.  NOTHING ran; nothing is verified."
        exit 5
    fi
    [ "$rrc" -eq 0 ] || { echo "FAIL: the rig-control invocation exited $rrc"; gate_fail=1; }
    printf '%s\n' "$rig_out" | structural real || gate_fail=1
    printf '%s\n' "$rig_out" | grep '^PAIRED FALSIFIER' | sed 's/^/  [reported, NOT a verdict — the falsifier is deferred] /'
    if [ "$gate_fail" -eq 0 ]; then
        echo "RIG CONTROL: OK — modes CONCURRENT/REAL, (b) bitwise-identical to the CPU oracle, the paired"
        echo "  arms compute the same thing, (a)/(c) within the sanity ceiling.  Scope: ONE layer-0,"
        echo "  seed-1234, real-mode, iters-5 invocation — a layer-4- or CONTROL-mode-specific break"
        echo "  would not be seen here (the full path exercises both layers and both modes); the"
        echo "  falsifier itself was not measured."
        echo "FALSIFIER: DEFERRED (exit 4) — $defer_reason.  Re-run on a quiet host."
        exit 4
    fi
    echo "RIG CONTROL: BROKEN — the structural checks above FAILED.  That is a real failure, not a"
    echo "  deferral: a deferral must never hide a broken rig."
    echo "GATE FAIL (exit 1): see the FAIL lines above."
    exit 1
fi
# the settled start reading counts as the run's first load reading
load_readings="$_last_load
"
for spec in "0 1234" "4 2"; do
    set -- $spec
    L=$1; S=$2
    for mode in real control; do
        if [ "$mode" = real ]; then tag=REAL; else tag=CONTROL; fi
        ratios=""
        i=1
        while [ "$i" -le "$N" ]; do
            echo "================================================================"
            echo "GATE layer $L (seed $S)  mode=$mode  run $i/$N  [loadavg(1m) $(lavg1)]"
            echo "================================================================"
            out=$(run_bin "$L" "$S" "$mode")
            rc=$?
            # mid-run load reading: the decision at the end uses the MAX over start+every run
            # (staleness stated in the header: a post-invocation 1-min average mostly reflects
            # that invocation's own window)
            lri=$(lavg1)
            load_readings="$load_readings$lri
"
            [ "$rc" -eq 0 ] || { echo "FAIL: the binary exited $rc"; gate_fail=1; }
            if [ "$i" -eq 1 ]; then
                # the full run (arms, error table, derived tolerances, decomposition) once
                printf '%s\n' "$out"
            else
                printf '%s\n' "$out" | grep -E '^(PAIRED FALSIFIER|VERDICT|\[reported, WITHDRAWN as a verdict\] SERIAL-BOUND CHECK \(min-of-N\))'
            fi
            printf '%s\n' "$out" | structural "$mode" || gate_fail=1
            pl=$(printf '%s\n' "$out" | grep "^PAIRED FALSIFIER\[$tag\]:")
            if [ -z "$pl" ]; then
                echo "FAIL: no PAIRED FALSIFIER[$tag] line in run $i (mode mismatch or old binary)"
                gate_fail=1
            else
                ratio=$(printf '%s\n' "$pl" | sed -n 's/.*ratio=\([0-9.]*\).*/\1/p')
                if [ -z "$ratio" ]; then
                    echo "FAIL: unparseable PAIRED FALSIFIER line: $pl"
                    gate_fail=1
                else
                    ratios="$ratios$ratio
"
                fi
                case "$pl" in
                    *"-> PASS"*)
                        if [ "$mode" != real ]; then
                            echo "FAIL: the control PASSED the falsifier on run $i — the line did not hold the control"
                        fi ;;
                    *"-> FAIL"*)
                        if [ "$mode" != control ]; then
                            echo "NOTE: the real build FAILED the falsifier on run $i"
                        fi ;;
                    *) echo "FAIL: no '-> PASS'/'-> FAIL' verdict in: $pl"; gate_fail=1 ;;
                esac
            fi
            i=$((i+1))
        done

        n_all=$(count_nonempty "$ratios")
        n_le=$(count_le "$ratios" "$THRESH")
        n_gt=$(count_gt "$ratios" "$THRESH")
        echo "----------------------------------------------------------------"
        echo "layer $L (seed $S)  mode=$mode  ratios (N=$n_all):"
        printf '%s\n' "$ratios" | awk 'NF { printf "   %s\n", $1 }'
        if [ "$mode" = real ]; then
            if [ "$n_le" -eq "$N" ] && [ "$n_all" -eq "$N" ]; then
                echo "   -> ALL $N real ratios <= $THRESH  (PASS)"
            elif [ "$n_gt" -eq "$N" ] && [ "$n_all" -eq "$N" ]; then
                echo "   -> ALL $N real ratios  > $THRESH  (the premise is DEAD: no overlap in the build)"
                gate_fail=1
            else
                echo "   -> MIXED: $n_le of $n_all <= $THRESH, $n_gt above it; the ones ABOVE the line (the failures for a real build): $(list_gt "$ratios" "$THRESH")"
                echo "      a falsifier that passes and fails on the same build cannot decide; exit non-zero."
                gate_fail=1
            fi
        else
            if [ "$n_le" -eq 0 ] && [ "$n_all" -eq "$N" ]; then
                echo "   -> ALL $N control ratios  > $THRESH  (the control is REJECTED, as it must be)"
            else
                echo "   -> THE CONTROL WAS NOT REJECTED: $n_le of $n_all <= $THRESH -> $(list_le "$ratios" "$THRESH")"
                echo "      a build with no overlap passed the falsifier: THE FALSIFIER IS BLIND."
                blind=1
            fi
        fi
    done
done

echo "================================================================"
# the load decision: MAX over the start reading and one reading after every falsifier run
max_load=$(printf '%s\n' "$load_readings" | awk 'NF { if ($1+0 > m+0) m=$1+0 } END { print m+0 }')
n_load_readings=$(printf '%s\n' "$load_readings" | awk 'NF { n++ } END { print n+0 }')
load_violated=0
if awk -v a="$max_load" -v b="$QUIET_LOAD" 'BEGIN { exit !(a+0 > b+0) }'; then load_violated=1; fi
echo "loadavg(1m) over the run: max $max_load of $n_load_readings readings (QUIET_LOAD=$QUIET_LOAD)"
if [ "$gate_fail" -eq 0 ] && [ "$load_violated" -eq 1 ]; then
    echo "  NOTE: the load precondition was VIOLATED during the run (max $max_load > $QUIET_LOAD);"
    echo "  the verdict below accounts for it — a BLIND under violation is DEFERRED (exit 4), never 2."
fi
if [ "$gate_fail" -eq 1 ]; then
    echo "GATE FAIL (exit 1): see the FAIL lines above."
    [ "$load_violated" -eq 1 ] && echo "  (measured under a violated load precondition — a FAIL is never deferred away; re-run quiet to confirm)"
    exit 1
fi
if [ "$blind" -eq 1 ]; then
    if [ "$load_violated" -eq 1 ]; then
        echo "FALSIFIER: DEFERRED (exit 4) — the control was NOT rejected, BUT the load precondition was"
        echo "  violated during the run (max loadavg(1m) $max_load > QUIET_LOAD=$QUIET_LOAD).  On a loaded"
        echo "  host a control at or under the line is the EXPECTED artifact of contention (results/"
        echo "  m0-gate-busy.md: 4/4 runs at load 25-39), so a BLIND is not established here.  The"
        echo "  numbers stand as printed; a QUIET re-run decides — if the control is still not rejected"
        echo "  with the precondition met, that is a genuine BLIND (exit 2)."
        exit 4
    fi
    echo "GATE FAILURE (exit 2): THE FALSIFIER IS BLIND — at least one --serialize-both control run was"
    echo "NOT rejected by the $THRESH line, so the gate cannot distinguish overlap from no overlap."
    echo "QUIET HOST (max loadavg(1m) $max_load <= QUIET_LOAD=$QUIET_LOAD), so this is a genuine blind:"
    echo "the measured separation is too weak to be called a falsifier.  exit 2."
    exit 2
fi
if [ "$load_violated" -eq 1 ]; then
    echo "  NOTE: the PASS stands under the mid-run load violation (max $max_load > $QUIET_LOAD): the"
    echo "  pass conditions held on the measured ratios, and load's measured effect (0.2c: control"
    echo "  ratios move DOWN, real UP) can only make this pass harder, not easier.  INTERPRETATION."
fi
if [ "$gate_fail" -eq 0 ]; then
    echo "GATE PASS: the paired falsifier holds on ALL $N real runs of BOTH down types"
    echo "(min-of-N(c_conc)/min-of-N(c_ser) <= $THRESH, both arms measured in the same host state);"
    echo "(b) is bitwise-identical to the CPU oracle on every run; (a)/(c) are within the sanity"
    echo "ceiling; and ALL $N no-overlap controls (--serialize-both) are REJECTED by the same line."
    exit 0
fi
echo "GATE FAIL (exit 1): see the FAIL lines above."
exit 1
