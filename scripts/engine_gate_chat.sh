#!/bin/sh
# engine_gate_chat.sh — U2 CHAT GATE: the Mistral Small 4 chat render, against llama.cpp's own Jinja.
#
#   THE VERDICT (exit code, never the tail of stdout):
#
#     THE RENDER: for every message shape, engine/m1_chat.cpp's render_chat() is BYTE-IDENTICAL
#     to the render the GGUF's own chat template produces through llama.cpp's Jinja runtime
#     (tools/m1_chat_ref.cpp -> common_chat_templates_apply), and the template the renderer was
#     written for is the template in the model file: its raw tokenizer.chat_template bytes hash
#     to the pinned sha256, and the default_system_message literal transcribed into
#     engine/m1_chat_dsm.inc equals the literal in those bytes.
#
#   PASS conditions (all must hold):
#     1. PIN      the GGUF's raw tokenizer.chat_template is 14121 bytes and hashes to the sha256
#                 written in engine/m1_chat.h — measured THREE ways: an independent python GGUF
#                 parse of the file, sha256sum of the bytes llama.cpp itself reports, and the
#                 renderer's own boot check (which FATALs on a mismatch);
#     2. LITERAL  the renderer's boot check #2 passes: the three default_system_message pieces
#                 extracted from those bytes equal the compiled-in transcription and render to
#                 2374 bytes;
#     3. RENDER   on every shape, `cmp` of our render against the oracle's is EMPTY — including
#                 the shapes that carry special-token TEXT as literal content, a tool message,
#                 a trailing assistant turn (its '</s>'), escapes and trailing whitespace, and
#                 the no-system-role default; and the oracle's llama-server path (which strips a
#                 leading BOS, common/chat.cpp:931-933) equals our render MINUS '<s>';
#     4. DATES    the template's rollback ladder agrees with ours on every boundary day, and — the
#                 sharp half — our REFUSALS coincide with the template's raise_exception() calls
#                 (1 March of a year outside {2024,2028,2032,1970}, 1 January outside
#                 {1970,2024..2032}): both sides render or both sides refuse, never one of each;
#     5. ROLES    a non-alternating conversation is REFUSED by both sides;
#     6. BOS      on the rendered text the tokenizer gives ids[0]==1 && ids[1]!=1 with
#                 --no-bos (the product policy, ONE bos) and ids[0]==1 && ids[1]==1 with BOS
#                 (llama-server's double-BOS, --chat-bos=tokenizer), with the first 4 ids logged;
#     7. INSTRUMENT the date injection used by arm 4 is faithful: rendering with --now <today>
#                 is byte-identical to rendering without it.
#
#   EXIT CODES:
#     0 = gate PASS: every arm above held.
#     1 = gate FAIL: at least one arm produced a FAIL line.  A FAIL is never tolerated away.
#     69 = BLIND/environment: a binary is missing, the model file is missing, the oracle link
#          failed, or a control knob is invalid — nothing was verified.
#
#   CONTROL KNOBS (a gate whose failure path has never been seen is not a gate):
#     OURS=<path>     the renderer binary under test (default builds/engine/m1_chat_ours).  Point
#                     it at a MUTANT build to see the FAIL path:
#                       MUTANT=eos sh scripts/engine_build_chat.sh
#                       OURS=builds/engine/m1_chat_ours-mutant-eos sh scripts/engine_gate_chat.sh
#                     (also MUTANT=sha, MUTANT=dsm).  A mutant can only make the gate FAIL —
#                     nothing here can manufacture a PASS.
#     DIR=<dir>       evidence directory (default logs/engine/gate-chat-run, gate-owned scratch).
#     DATES="..."     override the boundary-day list (only to exercise it; the default list is
#                     the pinned one and its length is asserted).
#
#   WHY THIS NEEDS NO GPU AND NO gpu.lock: both binaries load the model VOCAB ONLY
#   (llama_model_params.vocab_only = true) or not at all (the renderer is plain C++17), so
#   nothing here touches ROCm0 or the weights.  It is runnable on a loadavg-20 host.
#
#   THE ORACLE IS NOT OUR CODE: tools/m1_chat_ref.cpp does not include engine/m1_chat.h and
#   shares no parsing or rendering code with the unit under test — the case-file grammar is
#   parsed twice, independently, and any disagreement shows up as a byte diff.
#
#   ARM 5 AND THE TOKENIZER BINARY — BOUNDED, COUNTED, PRINTED RETRY.  This gate shells out to
#   builds/hip-fork/bin/llama-tokenize for the BOS arm, and that binary is INTERMITTENTLY BROKEN
#   on this host: U1 MEASURED 2/15 SEGV and 1/15 hangs past 25 s, and this unit's own mutant run
#   sat in arm 5 for 23 minutes under a peer's tokenizer storm before it was killed.  So every
#   invocation is (a) wrapped in `timeout $TOK_WAIT`, (b) retried up to $TOK_TRIES times, and
#   (c) PRINTED per attempt.  A tokenizer that never returns a clean id list makes arm 5 UNMEASURED,
#   which is a BLIND (exit 69) — never a FAIL (nothing about the render was disproved) and never
#   a PASS (the arm did not run).  TOK_WAIT=120 and TOK_TRIES=4 are knobs; every other invocation
#   in this gate is our own binary and completes in well under a second.

set -u

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
FORKBIN="$ROOT/builds/hip-fork/bin"
ROCMLIB="$ROOT/venvs/rocm/lib/python3.13/site-packages/_rocm_sdk_devel/lib"
OURS="${OURS:-$ROOT/builds/engine/m1_chat_ours}"
REF="$ROOT/builds/engine/m1_chat_ref"
TOK_BIN="${TOK_BIN:-$FORKBIN/llama-tokenize}"
MODEL="${MODEL:-$ROOT/models/Mistral-Small-4-119B-2603-UD-IQ2_XXS.gguf}"
HDR="$ROOT/engine/m1_chat.h"
DIR="${DIR:-$ROOT/logs/engine/gate-chat-run}"
TOK_WAIT="${TOK_WAIT:-120}"   # per-invocation timeout for llama-tokenize (see the header)
TOK_TRIES="${TOK_TRIES:-4}"   # bounded attempts before arm 5 reports BLIND

# ---------------------------------------------------------------------------
# guards: a gate must not decide without data
# ---------------------------------------------------------------------------
[ -x "$OURS" ]  || { echo "engine_gate_chat.sh: FATAL: $OURS not built (sh scripts/engine_build_chat.sh)" >&2; exit 69; }
[ -x "$REF" ]   || { echo "engine_gate_chat.sh: FATAL: $REF not built (sh scripts/engine_build_chat.sh)" >&2; exit 69; }
[ -x "$TOK_BIN" ] || { echo "engine_gate_chat.sh: FATAL: tokenizer oracle $TOK_BIN missing" >&2; exit 69; }
[ -f "$MODEL" ] || { echo "engine_gate_chat.sh: FATAL: model $MODEL missing" >&2; exit 69; }
[ -f "$HDR" ]   || { echo "engine_gate_chat.sh: FATAL: $HDR missing" >&2; exit 69; }
[ -d "$ROCMLIB" ] || { echo "engine_gate_chat.sh: FATAL: ROCm lib dir $ROCMLIB missing" >&2; exit 69; }
command -v python3 >/dev/null || { echo "engine_gate_chat.sh: FATAL: python3 missing" >&2; exit 69; }

# the oracle and llama-tokenize have no rpath of their own; libggml-hip's RUNPATH points into a
# dead pre-move tree (same workaround as gate1 and engine_build_m1.sh).
export LD_LIBRARY_PATH="$FORKBIN:$ROCMLIB"

mkdir -p "$DIR"

gate_fail=0
blind=0
SECONDS=0
lavg() { cut -d' ' -f1-3 /proc/loadavg; }
note() { echo "$@"; }
fail() { echo "FAIL: $*"; gate_fail=1; }
mark() { echo "---- $*  [elapsed ${SECONDS}s, loadavg $(lavg)]"; }

# the pinned template sha256, READ OUT OF THE HEADER (the constant the renderer compiles in) —
# never from a copy in this script: a gate that carries its own copy of the pin can agree with
# itself while both are wrong.
PIN=$(sed -n '/M1_CHAT_TEMPLATE_SHA256 =/,/;/p' "$HDR" | grep -o '[0-9a-f]\{64\}' | head -n 1)
PIN_LEN=$(sed -n 's/.*M1_CHAT_TEMPLATE_LEN *= *\([0-9]*\).*/\1/p' "$HDR" | head -n 1)
DSM_LEN=$(sed -n 's/.*M1_CHAT_DEFAULT_SYSTEM_MESSAGE_LEN *= *\([0-9]*\).*/\1/p' "$HDR" | head -n 1)
[ -n "$PIN" ] || { echo "engine_gate_chat.sh: FATAL: no pinned sha256 found in $HDR" >&2; exit 69; }
[ -n "$PIN_LEN" ] || { echo "engine_gate_chat.sh: FATAL: no M1_CHAT_TEMPLATE_LEN in $HDR" >&2; exit 69; }
[ -n "$DSM_LEN" ] || { echo "engine_gate_chat.sh: FATAL: no M1_CHAT_DEFAULT_SYSTEM_MESSAGE_LEN in $HDR" >&2; exit 69; }

echo "================================================================"
echo "U2 CHAT GATE — the Mistral Small 4 render vs llama.cpp's own Jinja"
echo "================================================================"
echo "# start              $(date -Is)  loadavg $(lavg)"
echo "# renderer           $OURS   sha256 $(sha256sum "$OURS" | cut -c1-16)"
echo "# oracle             $REF   sha256 $(sha256sum "$REF" | cut -c1-16)   (vocab-only, no GPU)"
echo "# tokenizer          $TOK_BIN"
echo "# model              $MODEL"
echo "# pinned template    sha256 $PIN  len $PIN_LEN  dsm $DSM_LEN bytes"
echo "# evidence           $DIR"
echo "# host               $(uname -n), $(nproc) cpus; no GPU, no gpu.lock"

# ---------------------------------------------------------------------------
# arm 0 — the SHA-256 and the ladder functions themselves (a wrong hash would make the pin a lie)
# ---------------------------------------------------------------------------
mark "arm 0: renderer self-test (SHA-256 FIPS vectors + the ladder's branches)"
if "$OURS" selftest > "$DIR/selftest.txt" 2>&1; then
    sed 's/^/#   /' "$DIR/selftest.txt"
else
    sed 's/^/#   /' "$DIR/selftest.txt"
    fail "the renderer's self-test failed (SHA-256 vectors and/or the date ladder)"
fi

# ---------------------------------------------------------------------------
# arm 1 — the PIN: three independent measurements of the same bytes
# ---------------------------------------------------------------------------
mark "arm 1: the template pin (python GGUF parse | llama.cpp's bytes | the renderer's boot check)"
"$REF" template "$MODEL" "$DIR/template.bin" > "$DIR/ref-template.log" 2>&1
if [ ! -s "$DIR/template.bin" ]; then
    echo "BLIND: the oracle produced no template bytes (see $DIR/ref-template.log)"
    sed 's/^/#   /' "$DIR/ref-template.log" | tail -5
    gate_fail=1
else
    got_len=$(wc -c < "$DIR/template.bin" | tr -d ' ')
    got_sha=$(sha256sum "$DIR/template.bin" | cut -d' ' -f1)
    echo "#   llama.cpp's llama_model_chat_template(): $got_len bytes, sha256 $got_sha"
    [ "$got_len" = "$PIN_LEN" ] || fail "the template is $got_len bytes, the header pins $PIN_LEN"
    [ "$got_sha" = "$PIN" ]     || fail "the template's sha256 ($got_sha) is not the pinned one ($PIN)"

    # an INDEPENDENT parse of the GGUF's KV section (no llama.cpp involved)
    py_sha=$(python3 - "$MODEL" <<'PY'
import hashlib, struct, sys

f = open(sys.argv[1], 'rb')
def rd(n):
    b = f.read(n)
    if len(b) != n: raise SystemExit("short read")
    return b
def u32(): return struct.unpack('<I', rd(4))[0]
def u64(): return struct.unpack('<Q', rd(8))[0]
SZ = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}

if rd(4) != b'GGUF': raise SystemExit("not a GGUF")
u32(); u64(); n_kv = u64()
want = None
for _ in range(n_kv):
    klen = u64(); key = rd(klen).decode('utf-8', 'replace')
    t = u32()
    if t == 8:
        vlen = u64(); val = rd(vlen)
        if key == 'tokenizer.chat_template': want = val
    elif t == 9:
        at = u32(); n = u64()
        if at == 8:
            for _ in range(n):
                ln = u64(); f.seek(ln, 1)
        else:
            f.seek(SZ[at] * n, 1)
    else:
        f.seek(SZ[t], 1)
    if want is not None: break
if want is None: raise SystemExit("tokenizer.chat_template absent")
print("%d %s" % (len(want), hashlib.sha256(want).hexdigest()))
PY
)
    echo "#   python GGUF KV parse of the file:        ${py_sha}"
    case "$py_sha" in
        "$PIN_LEN $PIN") echo "#   PIN: the two measurements agree, and both are the pinned value" ;;
        *) fail "the independent GGUF parse gives '${py_sha}', pin is '$PIN_LEN $PIN'" ;;
    esac

    # the renderer's own boot checks (both FATAL on a mismatch) — this is what the server runs
    if "$OURS" verify "$DIR/template.bin" > "$DIR/verify.txt" 2>&1; then
        sed 's/^/#   /' "$DIR/verify.txt"
    else
        sed 's/^/#   /' "$DIR/verify.txt"
        fail "the renderer's boot check REJECTED the model's template (pin or literal mismatch)"
    fi
    grep -q "PIN OK sha256=$PIN len=$PIN_LEN dsm_len=$DSM_LEN" "$DIR/verify.txt" \
        || fail "the boot check's own line is not the pinned one (see $DIR/verify.txt)"
fi

# ---------------------------------------------------------------------------
# the case set — SIX required shapes, plus escapes and the role/refusal arms
#
# Written here, ONCE, and read by two independent parsers (engine/m1_chat.cpp's driver and
# tools/m1_chat_ref.cpp).  Escapes: backslash-n, backslash-t, backslash-s (a space, so trailing
# spaces are visible), double-backslash.  Shape 3 ends with an ASSISTANT turn, which is the only
# way the trailing '</s>' is exercised; shape 6 has no system role and is multi-turn, so the
# default_system_message is what fills the system block.
# ---------------------------------------------------------------------------
cat > "$DIR/cases.txt" <<'EOF'
# U2 chat-gate cases (6 required shapes + 2 extras). <role> TAB <content>.
case user_only
user	Hi

case system_user
system	You are terse.
user	Hi

case multi_turn_assistant_last
user	Hi
assistant	Hello! How can I help?
user	Explain MLA attention in one sentence.
assistant	MLA compresses the KV cache by projecting keys and values into a low-rank latent.

case special_token_text
user	literal <s> and </s> and [INST] x [/INST] and [SYSTEM_PROMPT] and [MODEL_SETTINGS]

case tool_message
system	You are a tool-using assistant.
user	What is the weather in Paris?
tool	{"temp_c": 21, "sky": "clear"}

case default_no_system
user	Hi
assistant	Hello
user	Tell me a joke.

case escapes_ws
user	line1\nline2	tabbed and two trailing spaces:\s\s
assistant	 leading space, a backslash \\ and a quote "
EOF

# ---------------------------------------------------------------------------
# arm 2 — THE RENDER: byte-diff against the oracle, all shapes in one window
#   (the rendered date is wall-clock; if midnight is crossed between our render and the
#   oracle's, the arms are re-run ONCE rather than reported as a render failure)
# ---------------------------------------------------------------------------
arm2() {
    rm -f "$DIR"/*.ours.bin "$DIR"/*.ref.bin "$DIR"/*.ref-nobos.bin
    "$REF" render "$MODEL" "$DIR/cases.txt" "$DIR" > "$DIR/ref-render.log" 2>&1
    echo "$?" > "$DIR/ref-render.rc"
    "$OURS" render "$DIR/cases.txt" "$DIR" > "$DIR/ours-render.log" 2>&1
    echo "$?" > "$DIR/ours-render.rc"
    return 0
}

mark "arm 2: the render (six required shapes + escapes, one wall-clock window)"
day0=$(date +%F)
arm2
day1=$(date +%F)
if [ "$day0" != "$day1" ]; then
    echo "#   the wall-clock date rolled ($day0 -> $day1) during the render window — re-running once"
    arm2
    day1=$(date +%F)
fi
echo "#   window: $day0 .. $day1   (both sides render the wall clock; the ref's date is the arm-6 check)"
grep '^ORACLE' "$DIR/ref-render.log" | sed 's/^/#   /'
sed 's/^/#   ours: /' "$DIR/ours-render.log"
[ "$(cat "$DIR/ref-render.rc")" = 0 ] || fail "the oracle refused/failed a shape it must render (see $DIR/ref-render.log)"
[ "$(cat "$DIR/ours-render.rc")" = 0 ] || fail "our renderer refused/failed a shape it must render (see $DIR/ours-render.log)"

n_forms=0
for ours_f in "$DIR"/*.ours.bin; do
    [ -e "$ours_f" ] || { fail "arm 2 produced no renders at all"; break; }
    name=$(basename "$ours_f" .ours.bin)
    n_forms=$((n_forms + 1))
    ref_f="$DIR/$name.ref-nobos.bin"     # add_bos=false: the '<s>' survives -> THE comparison
    srv_f="$DIR/$name.ref.bin"           # model-driven: llama-server's text, leading '<s>' stripped
    if [ ! -s "$ref_f" ]; then
        fail "$name: the oracle produced no ref-nobos render"
        continue
    fi
    if cmp -s "$ref_f" "$ours_f"; then
        echo "  RENDER $name: BYTE-IDENTICAL to the template's own render ($(wc -c < "$ours_f") bytes)"
    else
        fail "$name: our render differs from the template's own render"
        cmp "$ref_f" "$ours_f" 2>&1 | sed 's/^/    /'
        # the first differing byte, for the operator (never a tolerance: the render is exact)
        python3 - "$ref_f" "$ours_f" <<'PY'
import sys
a = open(sys.argv[1], 'rb').read(); b = open(sys.argv[2], 'rb').read()
n = min(len(a), len(b))
i = next((k for k in range(n) if a[k] != b[k]), None)
print("    ref %d bytes, ours %d bytes, first diff at %s" % (len(a), len(b), i))
if i is not None:
    print("    ref : %r" % (a[max(0, i-40):i+40],))
    print("    ours: %r" % (b[max(0, i-40):i+40],))
PY
    fi
    # the llama-server path: our render minus its leading '<s>' (the oracle strips a leading BOS
    # when add_bos, common/chat.cpp:931-933 — that IS the difference between the two policies)
    ours_nobos="$DIR/$name.ours-nobos.bin"
    tail -c +4 "$ours_f" > "$ours_nobos"
    head -c 3 "$ours_f" > "$DIR/$name.ours-prefix-3"
    if [ "$(cat "$DIR/$name.ours-prefix-3")" != "<s>" ]; then
        fail "$name: our render does not start with '<s>' (it must: the template emits it)"
    fi
    if cmp -s "$srv_f" "$ours_nobos"; then
        echo "  SERVER-PATH $name: the oracle's BOS-stripped render == our render minus '<s>'"
    else
        fail "$name: the oracle's model-driven render is not our render minus '<s>'"
    fi
done
[ "$n_forms" -ge 7 ] || fail "arm 2 compared only $n_forms shape(s); the case set has 7"

# ---------------------------------------------------------------------------
# arm 3 — the date ladder on the boundary days, including the days the template REFUSES
#   24 boundary days = the 1st of each month in 2026 and in the leap 2028, plus the edge cases
#   (Feb 29 of a leap year, Mar 1 after one, and the years the template's list does not cover).
# ---------------------------------------------------------------------------
DATES="${DATES:-2026-01-01 2026-02-01 2026-03-01 2026-04-01 2026-05-01 2026-06-01 2026-07-01 2026-08-01 2026-09-01 2026-10-01 2026-11-01 2026-12-01 2028-01-01 2028-02-01 2028-03-01 2028-04-01 2028-05-01 2028-06-01 2028-07-01 2028-08-01 2028-09-01 2028-10-01 2028-11-01 2028-12-01 2028-02-29 2027-03-01 2033-01-01 1970-01-01 2024-03-01 2032-01-01}"
n_dates=0
for d in $DATES; do n_dates=$((n_dates + 1)); done
mark "arm 3: the date ladder on $n_dates dates (the 24 boundary days + the template's refusals)"
[ "$n_dates" -ge 24 ] || fail "the boundary-day list has only $n_dates dates (the contract is 24+)"

# the instrument's own faithfulness, BEFORE its readings are used:
#   rendering with --now <today> must be byte-identical to rendering with no --now.
today=$(date +%F)
mkdir -p "$DIR/dates" "$DIR/dates/probe"
printf 'case user_only\nuser\tHi\n' > "$DIR/dates/one.txt"
"$REF" render "$MODEL" "$DIR/dates/one.txt" "$DIR/dates/probe" > /dev/null 2>&1
cp "$DIR/dates/probe/user_only.ref-nobos.bin" "$DIR/dates/probe/plain.bin"
"$REF" render "$MODEL" "$DIR/dates/one.txt" "$DIR/dates/probe" --now "$today" > /dev/null 2>&1
if cmp -s "$DIR/dates/probe/plain.bin" "$DIR/dates/probe/user_only.ref-nobos.bin"; then
    echo "  INSTRUMENT: the --now injection is faithful (--now $today == no --now, byte-identical)"
else
    fail "the --now date injection is NOT faithful: --now $today differs from the wall-clock render"
fi

n_render=0; n_refuse=0
for d in $DATES; do
    sub="$DIR/dates/$d"
    mkdir -p "$sub"
    "$OURS" render "$DIR/dates/one.txt" "$sub" --now "$d" > "$sub/ours.log" 2>&1
    orc=$?
    "$REF" render "$MODEL" "$DIR/dates/one.txt" "$sub" --now "$d" > "$sub/ref.log" 2>&1
    rrc=$?
    orc_v=$( [ "$orc" -eq 0 ] && echo RENDER || echo REFUSE )
    if grep -q 'REFUSED' "$sub/ref.log"; then rrc_v=REFUSE; else rrc_v=RENDER; fi
    if [ "$orc_v" != "$rrc_v" ]; then
        fail "$d: ours=$orc_v but the template says $rrc_v  (this is the ladder disagreeing, not a tolerance)"
        sed 's/^/    ours: /' "$sub/ours.log" | tail -2
        sed 's/^/    ref : /' "$sub/ref.log" | tail -2
        continue
    fi
    if [ "$orc_v" = RENDER ]; then
        n_render=$((n_render + 1))
        if cmp -s "$sub/user_only.ref-nobos.bin" "$sub/user_only.ours.bin"; then
            yday=$(grep -o 'yesterday" is [0-9][0-9-]*' "$sub/user_only.ours.bin" | head -1 | sed 's/.* is //')
            echo "  DATE $d: both render, BYTE-IDENTICAL (the template's yesterday_date = $yday)"
        else
            fail "$d: both render but the bytes differ"
            cmp "$sub/user_only.ref-nobos.bin" "$sub/user_only.ours.bin" 2>&1 | sed 's/^/    /'
        fi
    else
        n_refuse=$((n_refuse + 1))
        echo "  DATE $d: BOTH REFUSE (the template's raise_exception() and our ladder agree)"
    fi
done
echo "  dates: $n_render render byte-identically, $n_refuse refused by both sides, of $n_dates"
[ $((n_render + n_refuse)) -eq "$n_dates" ] || fail "not every boundary date was decided"

# ---------------------------------------------------------------------------
# arm 4 — roles: a non-alternating conversation is refused by BOTH sides, and an assistant turn
#          with EMPTY content is RENDered by both (the raw template raises there, but llama.cpp
#          patches exactly that condition out — the oracle is llama.cpp, see engine/m1_chat.h)
# ---------------------------------------------------------------------------
mark "arm 4: role handling (non-alternating must be refused; empty assistant content must render)"
printf 'case non_alternating\nuser\ta\nuser\tb\n' > "$DIR/dates/nonalternating.txt"
mkdir -p "$DIR/roles"
"$OURS" refuse "$DIR/dates/nonalternating.txt" > "$DIR/roles/ours-refuse.log" 2>&1
orc=$?
"$REF" render "$MODEL" "$DIR/dates/nonalternating.txt" "$DIR/roles" --refuse > "$DIR/roles/ref-refuse.log" 2>&1
rrc=$?
sed 's/^/#   /' "$DIR/roles/ours-refuse.log"
grep '^ORACLE' "$DIR/roles/ref-refuse.log" | sed 's/^/#   /'
[ "$orc" -eq 0 ] || fail "our renderer did NOT refuse a non-alternating conversation"
[ "$rrc" -eq 0 ] || fail "the oracle's template did NOT raise on a non-alternating conversation"
grep -q 'refused (ok)' "$DIR/roles/ours-refuse.log" || fail "our refuse arm did not report a refusal"

printf 'case empty_assistant\nuser\ta\nassistant\t\n' > "$DIR/dates/empty_assistant.txt"
mkdir -p "$DIR/roles/ea-ours" "$DIR/roles/ea-ref"
"$OURS" render "$DIR/dates/empty_assistant.txt" "$DIR/roles/ea-ours" > "$DIR/roles/ea-ours.log" 2>&1
orc=$?
"$REF" render "$MODEL" "$DIR/dates/empty_assistant.txt" "$DIR/roles/ea-ref" > "$DIR/roles/ea-ref.log" 2>&1
rrc=$?
[ "$orc" -eq 0 ] && [ "$rrc" -eq 0 ] || fail "an empty assistant turn must RENDER on both sides (ours rc=$orc ref rc=$rrc)"
if cmp -s "$DIR/roles/ea-ref/empty_assistant.ref-nobos.bin" "$DIR/roles/ea-ours/empty_assistant.ours.bin"; then
    echo "  EMPTY-ASSISTANT: byte-identical (both emit just '</s>' for the turn)"
else
    fail "an empty assistant turn renders differently from the oracle"
fi

# ---------------------------------------------------------------------------
# arm 5 — BOS: the rendered text through the tokenizer, BOTH policies, first 4 ids logged
#   <s>=1 [SYSTEM_PROMPT]=17 [INST]=3 [/INST]=4 </s>=2 (MEASURED: ids 0..17 are the control set)
# ---------------------------------------------------------------------------
mark "arm 5: the BOS trap on the rendered text (template policy: one BOS; tokenizer policy: two)"
# BOUNDED, COUNTED, PRINTED RETRY — llama-tokenize is intermittently broken on this host (header).
# $1 = the BOS flag ("" or "--no-bos"), $2 = log tag. Sets ids_out, returns 0 only on a clean list.
ids_out=""
tok_ids() {
    tok_attempt=1
    while [ "$tok_attempt" -le "$TOK_TRIES" ]; do
        timeout "$TOK_WAIT" "$TOK_BIN" -m "$MODEL" -f "$DIR/bos-prompt.txt" --no-escape --ids $1 \
            </dev/null > "$DIR/tok-$2.txt" 2> "$DIR/tok-$2.err"
        tok_rc=$?
        if [ "$tok_rc" -eq 0 ] && grep -q '^\[[0-9]' "$DIR/tok-$2.txt"; then
            [ "$tok_attempt" -gt 1 ] && echo "#   tokenizer ($2): clean id list on attempt $tok_attempt/$TOK_TRIES"
            ids_out=$(tr -d '[] ' < "$DIR/tok-$2.txt" | tr ',' ' ')
            return 0
        fi
        tok_why="exit $tok_rc"
        [ "$tok_rc" -eq 124 ] && tok_why="exit 124 (TIMEOUT after ${TOK_WAIT}s)"
        echo "#   tokenizer ($2): attempt $tok_attempt/$TOK_TRIES failed — $tok_why — retrying"
        tok_attempt=$((tok_attempt + 1))
    done
    ids_out=""
    return 124
}
bos_case="$DIR/system_user.ours.bin"
if [ -s "$bos_case" ]; then
    cp "$bos_case" "$DIR/bos-prompt.txt"
    tok_ids "--no-bos" nobos; tok_rc_a=$?; ids_nobos="$ids_out"
    tok_ids "" bos;           tok_rc_b=$?; ids_bos="$ids_out"
  if [ "$tok_rc_a" -ne 0 ] || [ "$tok_rc_b" -ne 0 ]; then
    blind=1
    echo "BLIND: arm 5 could not be measured — llama-tokenize gave no clean id list in $TOK_TRIES"
    echo "  bounded attempt(s) of ${TOK_WAIT}s each ($TOK_BIN is intermittently broken on this host;"
    echo "  see the header).  Nothing about the render is disproved, and nothing is verified either:"
    echo "  this is exit 69, not a PASS and not a FAIL.  Re-run."
  else
    first4() { printf '%s\n' $1 | head -4 | tr '\n' ' '; }
    echo "#   --chat-bos=template  (--no-bos)   first 4 ids: $(first4 "$ids_nobos")   <- product default"
    echo "#   --chat-bos=tokenizer (BOS added)  first 4 ids: $(first4 "$ids_bos")   <- llama-server's double BOS"
    i0=$(printf '%s\n' $ids_nobos | sed -n '1p')
    i1=$(printf '%s\n' $ids_nobos | sed -n '2p')
    j0=$(printf '%s\n' $ids_bos   | sed -n '1p')
    j1=$(printf '%s\n' $ids_bos   | sed -n '2p')
    j2=$(printf '%s\n' $ids_bos   | sed -n '3p')
    [ "$i0" = 1 ] && [ "$i1" != 1 ] && [ "$i1" = 17 ] \
        || fail "--no-bos must give ids[0]==1 (the template's <s>) and ids[1]==17 ([SYSTEM_PROMPT]), got [$i0,$i1]"
    [ "$j0" = 1 ] && [ "$j1" = 1 ] && [ "$j2" = 17 ] \
        || fail "with BOS the prompt must be DOUBLE-BOS ([1,1,17]) — the trap — got [$j0,$j1,$j2]"
    n_nobos=$(printf '%s\n' $ids_nobos | wc -l)
    n_bos=$(printf '%s\n' $ids_bos | wc -l)
    echo "  BOS: $n_bos ids with BOS vs $n_nobos with --no-bos (the difference is exactly the added id 1)"
    [ $((n_bos - n_nobos)) -eq 1 ] || fail "the two policies differ by $((n_bos - n_nobos)) ids, not 1"
  fi
else
    fail "no system_user render to feed the tokenizer"
fi

# ---------------------------------------------------------------------------
# the verdict
# ---------------------------------------------------------------------------
echo "================================================================"
echo "# end $(date -Is)  loadavg $(lavg)  elapsed ${SECONDS}s"
if [ "$gate_fail" -ne 0 ]; then
    echo "VERDICT: FAIL (exit 1) — see the FAIL lines above. The render is NOT the template's render,"
    echo "  or the template is not the pinned one; either way nothing here may be shipped."
    exit 1
fi
if [ "$blind" -ne 0 ]; then
    echo "VERDICT: BLIND (exit 69) — at least one arm could not be measured (see the BLIND line"
    echo "  above), so this run verifies nothing.  Nothing failed; re-run."
    exit 69
fi
echo "VERDICT: PASS (exit 0) — the render is byte-identical to the pinned template's own render"
echo "  through llama.cpp's Jinja on all $n_forms shape(s); the template pin holds (three independent"
echo "  measurements: python GGUF parse, llama.cpp's own bytes, the renderer's boot check) and the"
echo "  default_system_message literal matches the template byte for byte; the date ladder agrees on"
echo "  all $n_dates boundary dates INCLUDING the $n_refuse the template itself refuses; both BOS"
echo "  policies produce the pinned id prefixes ([1,17,...] and [1,1,17,...])."
exit 0
