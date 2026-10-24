#!/bin/sh
# engine_gate_server.sh — U5 GATE: THE FIRST FUNCTIONAL PROOF. The REAL engine behind the OpenAI
# server must produce the SAME token ids the single-shot binary produces (server plan §H row U5).
#
# EXIT CODE IS THE VERDICT: 0 = every arm passed, 1 = at least one arm failed (each FAIL line names
# the arm and what it saw), 69 = BLIND (the reference run produced nothing, so nothing was verified
# and nothing disproved — never reported as a pass).
#
# THE CLAIM UNDER TEST, in one sentence: greedy generation through HTTP over the SAME ids the
# binary was given is BYTE-IDENTICAL to the binary's `--n-decode` chain, it stays identical across
# requests (the per-request reset is real), and the server holds scripts/gpu.lock for its lifetime.
#
# THE ARMS
#   U5-arm-01 link-ggml              the product binary links the engine (libggml-hip, ROCm rpath)
#   U5-arm-02 reference-run          builds/engine/m1 under scripts/with_gpu_lock.sh, product flags
#   U5-arm-03 config-identity        the server's M1Session boot block == the m1 run's (same prints,
#                                    same code): configuration identity BEFORE numerics identity
#   U5-arm-04 id-parity              `cmp` of the server's req-0.tokens vs the m1 run's .tokens EMPTY
#   U5-arm-05 repeat-after-request   a second request over the same ids gives the same first G ids
#   U5-arm-06 chat-nostream          text == tok_test --decode(the ids the engine recorded), the
#                                    budget respected, finish_reason "length" IFF the budget bound,
#                                    no EOS id in an EOS stop, completion_tokens == the ids recorded,
#                                    prompt_tokens == the prompt the renderer produces TODAY
#   U5-arm-07 chat-stream-eq         the SSE deltas concatenate to the non-stream text, exactly one
#                                    finish_reason in the last frame before [DONE], [DONE] last, the
#                                    chain identical to arm 06's (same prompt, same boot)
#   U5-arm-08 eos-stop               EOS(2) never appears in the chain and an EOS stop is "stop"
#                                    (the finish/length correspondence), and the chain is arm 06's
#   U5-arm-09 stop-string            a stop string truncates the text (no stop string in it, and it
#                                    stays a PREFIX of decode(ids)) and reports "stop"
#   U5-arm-10 clamp                  max_tokens is clamped against kv_max -> finish_reason "length"
#   U5-arm-11 prompt-too-long         n_prompt >= kv_max -> 400 BEFORE the engine, server still alive
#   U5-arm-12 drift                  a long request (so the rotation drifts) completes
#   U5-arm-13 reset-parity           B after A == a FRESH BOOT's B, byte for byte
#   U5-arm-14 reset-control          --reset=none: B after A DIFFERS from a fresh boot's B (the
#                                    asymmetry: without the reset the state leaks)
#   U5-arm-15 entry-invariant        M1_SERVER_MUTANT=skip_reset: the NEXT request is refused 500
#                                    engine_state_dirty (the entry check can fire)
#   U5-arm-16 lock-held              `flock -n scripts/gpu.lock true` FAILS while the server is up,
#                                    and /proc/<pid>/fd holds the lock file
#   U5-arm-17 lock-refuses-69        with the lock held by someone else the server exits 69
#   U5-arm-18 lock-released          `flock -n` SUCCEEDS again after a clean shutdown
#   U5-arm-19 reset-cost             the --reset=exact cost is printed per request (MEASURED here,
#                                    the plan's ~0.75 s was a PROJECTION)
#   U5-arm-20 clean-shutdown         SIGTERM -> exit 0 and the clean-shutdown line
#   U5-arm-21 chat-refusals          non-alternating turns / a late system message -> 400 (the
#                                    pinned template RAISES on both: a client must not reach that)
#   U5-arm-22 cancel-rst             a client that RSTs mid-stream: the NEXT request is byte-identical
#                                    to a fresh boot and the server still serves (plan §G2, real engine)
#   U5-arm-23 reset-bytes-verified   --rotate-verify: the reset's replayed VRAM rows are read back and
#                                    memcmp'd against the source experts (the bytes, not an argument)
#   U5-arm-24 nres-conflict-refused  a conflicting --n-res / --resident-set is a STARTUP FATAL with
#                                    the binary's own message (the store split is a numeric parameter)
#
# THE CHAT ARMS ARE CHAIN-LENGTH-AGNOSTIC ON PURPOSE — DO NOT "RESTORE" A LITERAL CHAIN.
# The pinned chat template embeds the WALL-CLOCK DATE (today_date/yesterday_date, engine/m1_chat.cpp
# via strftime_now), so the continuation the model gives — how many ids it produces, and whether it
# stops at EOS or runs to max_tokens — is a function of the CALENDAR, not of the server. MEASURED
# 2026-10-24: the product binary answers "Say nothing at all." in 3 ids + EOS ("Alright.",
# finish "stop") on the Oct-19 binary (builds/engine/m1_server.pre-u5fix1), the pre-B3 binary and
# the current one ALIKE, while these arms had been written on a day it produced 16 ids (finish
# "length"). Both are CORRECT SERVER BEHAVIOUR. So arms 06-09 assert the CONTRACT that must hold on
# EVERY day (text == decode(ids), the finish/length correspondence, EOS never in the chain, the SSE
# framing, the usage counts) and never a stored chain. The ONE value they still pin is the prompt
# count — and that one is DERIVED, not literal: the pinned renderer renders the same messages with
# today's date and tok_test encodes the result with the product BOS policy (add_special=false),
# exactly as scripts/engine_gate_chat.sh obtains its ground truth. (The prompt count is also
# MEASURED date-invariant in LENGTH on this template: 562 ids on 8 sampled days spanning
# 2026-10-19..2030-12-31 including the leap day, the render 2501 bytes on all — the date strings
# are fixed-width. It is still derived rather than pinned, so a template/message change cannot rot
# it silently.)
#
# WHY THE TWO SIDES RUN SEQUENTIALLY: two engine instances do not fit the 15.92 GiB card
# (LEDGER.md:2078), so the m1 reference runs first, under scripts/with_gpu_lock.sh, and the server
# boots afterwards and takes the lock ITSELF. The gate never holds the lock across a server run and
# NEVER runs gate1 inside it (an outer flock makes gate1 exit 4 falsely).
#
# usage: sh scripts/engine_gate_server.sh          (needs the GPU; takes ~15 min on a loaded box)
#        No arguments. Artefacts: logs/engine/gate-server-run/ (one directory per server instance).
#        Requires the model, the m1 binary and scripts/with_gpu_lock.sh; TOKDIR and HOTSET are
#        overridable because the prompt fixture and the resident set are not part of this repo.

set -u

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
BIN="$ROOT/builds/engine/m1_server"
M1="$ROOT/builds/engine/m1"

echo "engine_gate_server.sh: building the REAL server (scripts/engine_build_server.sh)"
if ! sh "$ROOT/scripts/engine_build_server.sh"; then
    echo "GATE FAIL: the build did not succeed" >&2
    exit 1
fi
for f in "$BIN" "$M1" "$ROOT/builds/engine/tok_test"; do
    if [ ! -x "$f" ]; then
        echo "GATE FAIL: $f is not executable (run scripts/engine_build_m1.sh and scripts/engine_build_tok.sh)" >&2
        exit 1
    fi
done

exec python3 - "$ROOT" "$BIN" "$M1" <<'PY'
import atexit, glob, json, os, re, shutil, socket, struct, subprocess, sys, time, traceback, urllib.request, urllib.error

def _excepthook(t, v, tb):
    traceback.print_exception(t, v, tb)
    print("GATE FAIL: an arm raised an exception (see the traceback above)")
    sys.exit(1)

sys.excepthook = _excepthook

ROOT, BIN, M1 = sys.argv[1], sys.argv[2], sys.argv[3]
TOK = os.path.join(ROOT, "builds/engine/tok_test")
MODEL = os.path.join(ROOT, "models/Mistral-Small-4-119B-2603-UD-IQ2_XXS.gguf")
# The prompt file and the resident set are inputs this repository does NOT ship (the weights are not
# here either): point TOKDIR/HOTSET at your own, or the gate cannot run. TOKDIR holds the
# tokens-B1.txt prompt ids; HOTSET is the per-layer expert set that seeds --rotate.
TOKDIR = os.environ.get("TOKDIR") or os.path.join(ROOT, "logs/engine/m1-u6/tokens")
HOTSET = os.environ.get("HOTSET") or os.path.join(ROOT, "logs/engine/m1-u6/hotset-A48.json")
LOCK = os.path.join(ROOT, "scripts/gpu.lock")
OUT = os.path.join(ROOT, "logs/engine/gate-server-run")
KV_MAX = 1024          # NUMERIC (plan §0 C4): BOTH sides must use this value
N_PARITY = 32          # the parity chain length (barged against the m1 run's --n-decode)
N_RESET = 128          # B's chain length for the reset arms (long enough to be sensitive)
CHAT_MAX = 16          # the cheap chat arms

# the product configuration, flag for flag on both sides
PRODUCT = ["--arm-store", "--gpu-core", "--gpu-store", "--n-res", "48", "--threads", "16",
           "--rotate", "--rot-admit-k", "4", "--resident-set", HOTSET, "--kv-max", str(KV_MAX)]

FAILS = []
BLIND = []

def arm(name, ok, detail=""):
    print(("PASS  " if ok else "FAIL  ") + name + (("  — " + detail) if detail else ""))
    if not ok:
        FAILS.append(name + (("  — " + detail) if detail else ""))
    return ok

def blind(name, detail):
    print("BLIND " + name + " — " + detail)
    BLIND.append(name + " — " + detail)

def read_ids(path):
    with open(path, "rb") as f:
        raw = f.read()
    n = struct.unpack("<I", raw[:4])[0]
    return list(struct.unpack("<%di" % n, raw[4:4 + 4 * n]))

def read_tokfile(path):
    """the text token format: 'n_prompt N' then the ids (whitespace separated)"""
    ids = []
    with open(path) as f:
        for line in f:
            if line.startswith("n_prompt"):
                continue
            ids += [int(x) for x in line.split()]
    return ids

def first_diff(a, b):
    for i in range(min(len(a), len(b))):
        if a[i] != b[i]:
            return "first divergence at index %d: server %d, reference %d (len %d vs %d)" % (
                i, a[i], b[i], len(a), len(b))
    if len(a) != len(b):
        return "same prefix, different length: %d vs %d" % (len(a), len(b))
    return ""

def chat_prompt_ids(messages, tag):
    """THE PROMPT COUNT'S GROUND TRUTH, for TODAY, derived the way scripts/engine_gate_chat.sh
    derives its own: the pinned renderer renders the conversation with the wall-clock date and
    tok_test encodes the result with the PRODUCT BOS POLICY (add_special=false — the template
    already emits '<s>', engine/m1_real.cpp's encode_chat). The renderer is compiled HERE, from the
    tree under gate, so a stale builds/ artefact cannot become the oracle. Returns (ids, why) with
    ids=None when it could not be measured (the caller then reports the count as unmeasured rather
    than pinning a literal)."""
    d = os.path.join(OUT, "chat-prompt-" + tag)
    os.makedirs(d, exist_ok=True)
    renderer = os.path.join(d, "m1_chat_ours")
    r = subprocess.run(["/usr/bin/c++", "-O2", "-std=c++17", "-DM1_CHAT_DRIVER",
                        "-I", os.path.join(ROOT, "engine"),
                        os.path.join(ROOT, "engine/m1_chat.cpp"), "-o", renderer],
                       cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        return None, "the renderer did not compile: " + (r.stderr or "")[-300:]
    cases = os.path.join(d, "cases.txt")
    with open(cases, "w") as f:
        f.write("case prompt\n")
        for m in messages:
            content = (m["content"].replace("\\", "\\\\").replace("\t", "\\t")
                       .replace("\n", "\\n"))
            f.write(m["role"] + "\t" + content + "\n")
    today = time.strftime("%Y-%m-%d")
    outdir = os.path.join(d, "render")
    os.makedirs(outdir, exist_ok=True)
    r = subprocess.run([renderer, "render", cases, outdir, "--now", today],
                       cwd=ROOT, capture_output=True, text=True)
    text_path = os.path.join(outdir, "prompt.ours.bin")
    if r.returncode != 0 or not os.path.exists(text_path):
        return None, "the renderer refused for %s (the template's own date ladder): %s" % (
            today, (r.stderr or r.stdout)[-300:])
    r = subprocess.run([TOK, "-m", MODEL, "--encode", text_path, "--no-bos"],
                       cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        return None, "tok_test --encode failed: " + (r.stderr or "")[-300:]
    arrays = [ln for ln in r.stdout.splitlines() if ln.startswith("[")]
    if not arrays:
        return None, "tok_test --encode printed no id array"
    try:
        ids = json.loads(arrays[-1])
    except json.JSONDecodeError as ex:
        return None, "tok_test --encode printed an unparsable id array (%s)" % ex
    return ids, "rendered for %s: %d ids, add_special=0 (the product BOS policy)" % (today, len(ids))

B1 = read_tokfile(os.path.join(TOKDIR, "tokens-B1.txt"))
A3 = read_tokfile(os.path.join(TOKDIR, "tokens-A3.txt"))
LONG = (A3 * 9)[:1000]          # the clamp arm: a 1000-id prompt (kv_max 1024)

os.makedirs(OUT, exist_ok=True)


SERVERS = []

def stop_all_servers():
    """atexit: a traceback (or any early exit) must not leave an engine holding scripts/gpu.lock"""
    for s in SERVERS:
        try:
            s.stop()
        except Exception:
            pass

import atexit
atexit.register(stop_all_servers)

def next_ordinal(srv):
    """the engine request ordinal the NEXT request on this server will get"""
    return len(glob.glob(os.path.join(srv.dir, "req-*.tokens")))

class Srv:
    def __init__(self, tag, extra, env=None):
        self.tag = tag
        self.dir = os.path.join(OUT, tag)
        shutil.rmtree(self.dir, ignore_errors=True)
        os.makedirs(self.dir, exist_ok=True)
        self.portfile = os.path.join(self.dir, "port")
        self.log = os.path.join(self.dir, "server.log")
        self.cmd = [BIN, "--engine", "real", "--model", MODEL, "--port", "0",
                    "--port-file", self.portfile, "--request-log", self.dir] + PRODUCT + list(extra)
        e = dict(os.environ)
        if env:
            e.update(env)
        self.p = subprocess.Popen(self.cmd, cwd=ROOT, stdout=open(self.log, "wb"),
                                  stderr=subprocess.STDOUT, env=e)
        self.port = None
        SERVERS.append(self)

    def wait(self, seconds=900):
        t0 = time.time()
        while time.time() - t0 < seconds:
            if os.path.exists(self.portfile):
                try:
                    self.port = int(open(self.portfile).read().strip())
                    return True
                except ValueError:
                    pass
            if self.p.poll() is not None:
                return False
            time.sleep(0.5)
        return False

    def post(self, path, body, timeout=3600):
        url = "http://127.0.0.1:%d%s" % (self.port, path)
        data = json.dumps(body).encode()
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.status, r.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as ex:
            return ex.code, ex.read().decode("utf-8", "replace")

    def get(self, path, timeout=120):
        url = "http://127.0.0.1:%d%s" % (self.port, path)
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return r.status, r.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as ex:
            return ex.code, ex.read().decode("utf-8", "replace")

    def stop(self, sig=15, timeout=300):
        if self.p.poll() is None:
            self.p.send_signal(sig)
        try:
            rc = self.p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.p.kill()
            rc = -9
        return rc

    def log_text(self):
        try:
            with open(self.log, "rb") as f:
                return f.read().decode("utf-8", "replace")
        except FileNotFoundError:
            return ""

    def post_ord(self, path, body, timeout=3600):
        """POST and return (status, body, the engine request ordinal it consumed or None).
        The ordinal is READ OFF THE LOG DIRECTORY rather than counted by hand: a request that the
        engine never entered (a 400 from the router) consumes none, and a mis-counted ordinal would
        silently compare the wrong artifact."""
        before = next_ordinal(self)
        st, txt = self.post(path, body, timeout)
        after = next_ordinal(self)
        # after-1, NOT before: another request may have been IN FLIGHT when this one was posted (the
        # RST arm has exactly that — the aborted request writes its artifact while this request waits
        # on the single-flight lock), so `before` can belong to that request. The artifact THIS post
        # produced is the highest-numbered one.
        return st, txt, ((after - 1) if after > before else None)

    def reqfile(self, n):
        return os.path.join(self.dir, "req-%d.tokens" % n)


def free_lock():
    return subprocess.run(["flock", "-n", LOCK, "true"], cwd=ROOT).returncode == 0


print("engine_gate_server.sh: gate-server-run under %s (kv_max %d)" % (OUT, KV_MAX))
print("loadavg: " + open("/proc/loadavg").read().strip())

# ---------------------------------------------------------------------------
# arm 01 — the product binary really links the engine
# ---------------------------------------------------------------------------
ldd = subprocess.run(["ldd", BIN], capture_output=True, text=True).stdout
arm("U5-arm-01 link-ggml",
    "libggml-hip" in ldd and "libggml-cpu" in ldd,
    "ldd names libggml-hip and libggml-cpu" if "libggml-hip" in ldd else "ldd: " + ldd.replace("\n", " ")[:200])

if not free_lock():
    print("GATE FAIL: scripts/gpu.lock is held by someone else before the gate starts; refusing")
    sys.exit(1)

# ---------------------------------------------------------------------------
# arm 02 — the reference: the single-shot binary, product flags, same kv_max
# ---------------------------------------------------------------------------
refdir = os.path.join(OUT, "ref")
shutil.rmtree(refdir, ignore_errors=True)
os.makedirs(refdir, exist_ok=True)
ref_log = os.path.join(refdir, "stdout.txt")
ref_cmd = ["sh", os.path.join(ROOT, "scripts/with_gpu_lock.sh"), M1, "--loop",
           "--tokens", os.path.join(TOKDIR, "tokens-B1.txt"), "--model", MODEL,
           "--out", os.path.join(refdir, "ref.logits"), "--n-decode", str(N_PARITY)] + PRODUCT
t0 = time.time()
with open(ref_log, "wb") as f:
    rc = subprocess.run(ref_cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode
ref_tokens = os.path.join(refdir, "ref.logits.tokens")
if rc != 0 or not os.path.exists(ref_tokens):
    blind("U5-arm-02 reference-run", "m1 exited %d without a token file (%s)" % (rc, ref_log))
    print("GATE BLIND: the reference produced nothing; nothing was verified")
    sys.exit(69)
REF = read_ids(ref_tokens)
arm("U5-arm-02 reference-run", len(REF) == N_PARITY,
    "m1 produced %d ids in %.0f s (%s)" % (len(REF), time.time() - t0, " ".join(str(x) for x in REF[:8])))

# ---------------------------------------------------------------------------
# arm 24 — a CONFLICTING --n-res / --resident-set pair is refused at STARTUP, exactly as the binary
# refuses it (U5FIX 1). The store split is a numeric parameter, so the server must not silently
# prefer one of the two; the refusal happens before the session is built, so nothing is loaded.
# ---------------------------------------------------------------------------
p24 = subprocess.run([BIN, "--engine", "real", "--model", MODEL, "--kv-max", "64", "--arm-store",
                      "--n-res", "32", "--resident-set", HOTSET, "--port", "0"],
                     cwd=ROOT, capture_output=True, text=True, timeout=900)
arm("U5-arm-24 nres-conflict-refused",
    p24.returncode == 1 and "--n-res 32 conflicts with --resident-set" in p24.stderr,
    "exit %d: %s" % (p24.returncode,
                     [l for l in p24.stderr.split("\n") if "conflicts" in l][:1]))

# ---------------------------------------------------------------------------
# S1 — the product server: arms 03-14, 16, 19, 20, 22
# ---------------------------------------------------------------------------
S1 = Srv("s1-product", ["--verbose"])
booted = S1.wait()
if not booted:
    arm("U5-arm-03 config-identity", False, "the server did not come up: " + S1.log_text()[-800:])
    print("GATE FAIL: the product server failed to boot")
    S1.stop()
    sys.exit(1)

# arm 16 (part 1): the lock really is held while the server runs
held = subprocess.run(["flock", "-n", LOCK, "true"], cwd=ROOT).returncode != 0
fd_link = ""
try:
    fd_dir = "/proc/%d/fd" % S1.p.pid
    for fd in os.listdir(fd_dir):
        try:
            tgt = os.readlink(os.path.join(fd_dir, fd))
        except OSError:
            continue
        if tgt.endswith("gpu.lock"):
            fd_link = tgt
except FileNotFoundError:
    pass
arm("U5-arm-16 lock-held", held and bool(fd_link),
    "flock -n refused while the server runs (%s), /proc/%d/fd holds %s"
    % ("rc!=0" if held else "rc==0", S1.p.pid, fd_link or "NOTHING"))

h, health = S1.get("/health")
ok_health = h == 200 and '"engine":"real"' in health and '"kv_max":%d' % KV_MAX in health
arm("U5-arm-03a health", ok_health, health[:160])

# arm 03 — the configuration block: the SAME prints from the SAME code
KEYLINES = ["# model      =", "# layers     =", "# kv width   =", "# arm        =", "# n_res      =",
            "# kv type    =", "# attention  =", "# backend    =", "# store split =",
            "# gpu core   =", "# resident-set =", "# rotate     =", "# rotate grid =", "# rotate verify =",
            "# head       =", "# built      =", "# store list ="]
def block(path):
    lines = {}
    for line in open(path, errors="replace"):
        for k in KEYLINES:
            if line.startswith(k):
                lines.setdefault(k, line.rstrip("\n"))
    return lines
srv_block = block(S1.log)
ref_block = block(ref_log)
missing = [k for k in KEYLINES if k not in srv_block or k not in ref_block]
mismatch = [k for k in KEYLINES if k in srv_block and k in ref_block and srv_block[k] != ref_block[k]]
arm("U5-arm-03 config-identity", not missing and not mismatch,
    "%d/%d configuration lines identical" % (len([k for k in KEYLINES if k in srv_block]), len(KEYLINES))
    if not missing and not mismatch else
    "missing %s; differing %s" % (missing, [(k, srv_block.get(k), ref_block.get(k)) for k in mismatch]))

# arm 04 — THE ID PARITY
st, body, o4 = S1.post_ord("/debug/tokens", {"ids": B1, "max_tokens": N_PARITY})
got = read_ids(S1.reqfile(o4))
same_bytes = open(S1.reqfile(o4), "rb").read() == open(ref_tokens, "rb").read()
arm("U5-arm-04 id-parity", st == 200 and same_bytes,
    "req-0.tokens == the m1 run's .tokens byte for byte (%d ids)" % len(got) if same_bytes else
    ("status %d body %s" % (st, body[:300]) if st != 200 else first_diff(got, REF)))

# arm 05 — a SECOND request over the same ids (the reset did not disturb the chain)
st, body, o5 = S1.post_ord("/debug/tokens", {"ids": B1, "max_tokens": 8})
got5 = read_ids(S1.reqfile(o5))
arm("U5-arm-05 repeat-after-request", st == 200 and got5 == REF[:8],
    "8 ids %s" % got5 if got5 == REF[:8] else first_diff(got5, REF[:8]))

# ---------------------------------------------------------------------------
# arm 06 — the chat route: the NON-STREAM CONTRACT, as properties (see the header: the prompt
# embeds the wall-clock date, so no chain length may be pinned here).
#   text == decode(ids) · the budget respected · finish_reason "length" IFF the budget was what
#   stopped it · an EOS stop leaves no eos id in the chain · completion_tokens counts exactly the
#   ids recorded · prompt_tokens is the prompt the RENDERER produces for these messages today
# ---------------------------------------------------------------------------
CHAT = {"messages": [{"role": "user", "content": "Say nothing at all."}], "max_tokens": CHAT_MAX}
day6 = time.strftime("%Y-%m-%d")
st6, body6, o6 = S1.post_ord("/v1/chat/completions", CHAT)
try:
    j6 = json.loads(body6)
except json.JSONDecodeError:
    j6 = {}
text6 = (j6.get("choices") or [{}])[0].get("message", {}).get("content") if st6 == 200 else None
ids6 = read_ids(S1.reqfile(o6))
dec6 = subprocess.run([TOK, "-m", MODEL, "--decode", "[" + ",".join(map(str, ids6)) + "]"],
                      cwd=ROOT, capture_output=True).stdout.decode("utf-8", "replace")
u6 = j6.get("usage") or {}
fr6 = (j6.get("choices") or [{}])[0].get("finish_reason")
prompt6, prompt6_why = chat_prompt_ids(CHAT["messages"], "06")
arm("U5-arm-06 chat-nostream",
    st6 == 200 and text6 == dec6 and len(ids6) <= CHAT_MAX
    and fr6 in ("stop", "length")
    and ((fr6 == "length") == (len(ids6) == CHAT_MAX))
    and (fr6 != "stop" or 2 not in ids6)
    and u6.get("completion_tokens") == len(ids6)
    and isinstance(u6.get("prompt_tokens"), int) and isinstance(u6.get("total_tokens"), int)
    and u6["total_tokens"] == u6["prompt_tokens"] + len(ids6)
    and (prompt6 is None or u6.get("prompt_tokens") == len(prompt6)),
    "status %d prompt %s (expected %s) ids %d/%d text %r == decode(ids) %r finish %s "
    "completion_tokens %s eos-in-ids %s · %s" %
    (st6, u6.get("prompt_tokens"), len(prompt6) if prompt6 is not None else "UNMEASURED",
     len(ids6), CHAT_MAX, (text6 or "")[:60], dec6[:60], fr6,
     u6.get("completion_tokens"), 2 in ids6, prompt6_why))

# ---------------------------------------------------------------------------
# arm 07 — the SSE contract: the deltas CONCATENATE to the non-stream text, the finish_reason
# appears EXACTLY ONCE and in the LAST frame before [DONE], [DONE] is last, and the chain is arm
# 06's (same prompt, same boot, greedy: repeatable — that is a same-minute comparison, not a chain
# pinned against a stored expectation).
# ---------------------------------------------------------------------------
st7, body7, o7 = S1.post_ord("/v1/chat/completions", dict(CHAT, stream=True))
raw_frames = [ln[6:] for ln in body7.split("\n") if ln.startswith("data: ")]
payloads, bad7 = [], ""
for fr in raw_frames:
    if fr == "[DONE]":
        continue
    try:
        payloads.append(json.loads(fr))
    except json.JSONDecodeError as ex:
        bad7 = "a frame is not JSON (%s): %r" % (ex, fr[:80])
deltas = [d["choices"][0]["delta"]["content"] for d in payloads
          if (d.get("choices") or [{}])[0].get("delta", {}).get("content")]
finish_at = [i for i, d in enumerate(payloads)
             if (d.get("choices") or [{}])[0].get("finish_reason") is not None]
streamed = "".join(deltas)
ids7 = read_ids(S1.reqfile(o7))
fr7 = payloads[finish_at[0]]["choices"][0]["finish_reason"] if finish_at else None
done_last = bool(raw_frames) and raw_frames[-1] == "[DONE]"
finish_is_last = len(finish_at) == 1 and finish_at[0] == len(payloads) - 1
# a midnight crossing between arms 06 and 07 changes the prompt (the date is IN it): the chain
# comparison is still the right assertion, so this is a NOTE, not a failure — and the note is the
# only place a reader would see why a one-in-a-million FAIL is not the server's fault.
rolled = time.strftime("%Y-%m-%d") != day6
arm("U5-arm-07 chat-stream-eq",
    st7 == 200 and not bad7 and streamed == text6 and done_last
    and raw_frames.count("[DONE]") == 1 and finish_is_last
    and fr7 == fr6 and ids7 == ids6,
    "stream concat == non-stream text (%d bytes, %d content frames), %d frames, exactly one "
    "finish_reason %s in the last frame before [DONE]=%s, ids identical to arm 06's=%s%s%s" %
    (len(streamed), len(deltas), len(raw_frames), fr7, done_last, ids7 == ids6,
     ("; " + bad7) if bad7 else "",
     "; NOTE the DATE ROLLED between the arms" if rolled else ""))

# ---------------------------------------------------------------------------
# arm 08 — EOS ends generation. WHICH branch (an EOS stop, or a chain that reaches the budget) a
# given day takes is the calendar's business, so the arm asserts the CORRESPONDENCE and prints the
# branch it measured; the chain itself must also be arm 06's (greedy, same prompt).
# ---------------------------------------------------------------------------
st8, body8, o8 = S1.post_ord("/v1/chat/completions", dict(CHAT, max_tokens=200))
j8 = json.loads(body8) if st8 == 200 else {}
ids8 = read_ids(S1.reqfile(o8))
fr8 = (j8.get("choices") or [{}])[0].get("finish_reason")
u8 = j8.get("usage") or {}
n8 = min(len(ids8), len(ids6))
arm("U5-arm-08 eos-stop",
    st8 == 200 and fr8 in ("stop", "length") and 2 not in ids8 and len(ids8) <= 200
    and ((fr8 == "length") == (len(ids8) == 200))
    and u8.get("completion_tokens") == len(ids8)
    and ids8[:n8] == ids6[:n8],
    "finish %s after %d ids (budget 200), EOS id 2 absent=%s, the first %d ids are arm 06's=%s" %
    (fr8, len(ids8), 2 not in ids8, n8, ids8[:n8] == ids6[:n8]))

# ---------------------------------------------------------------------------
# arm 09 — a stop string truncates the text and reports "stop". The chain is the model's (calendar-
# dependent), so the properties are: the stop string is absent from the text, the text stays a
# PREFIX of decode(ids) (the tail after the match was dropped), completion_tokens still counts the
# ids the engine produced, and the finish/length correspondence holds. Whether the stop actually
# FIRED today is printed, not assumed.
# ---------------------------------------------------------------------------
st9, body9, o9 = S1.post_ord("/v1/chat/completions", dict(CHAT, max_tokens=200, stop=" "))
j9 = json.loads(body9) if st9 == 200 else {}
t9 = (j9.get("choices") or [{}])[0].get("message", {}).get("content", "")
fr9 = (j9.get("choices") or [{}])[0].get("finish_reason")
u9 = j9.get("usage") or {}
ids9 = read_ids(S1.reqfile(o9))
dec9 = subprocess.run([TOK, "-m", MODEL, "--decode", "[" + ",".join(map(str, ids9)) + "]"],
                      cwd=ROOT, capture_output=True).stdout.decode("utf-8", "replace")
arm("U5-arm-09 stop-string",
    st9 == 200 and " " not in t9 and dec9.startswith(t9) and len(ids9) <= 200
    and ((fr9 == "length") == (len(ids9) == 200))
    and u9.get("completion_tokens") == len(ids9)
    # an EMPTY text is only honest when the chain's very first decoded byte IS the stop string
    and (len(t9) > 0 or dec9.startswith(" ")),
    "finish %s, text %r has no space, %d ids, text is a prefix of decode(ids)=%s, stop FIRED=%s" %
    (fr9, t9, len(ids9), dec9.startswith(t9), fr9 == "stop" and t9 != dec9))

# arm 10 — the max_tokens clamp against kv_max
st10, body10, o10 = S1.post_ord("/debug/tokens", {"ids": LONG, "max_tokens": 100000})
j10 = json.loads(body10) if st10 == 200 else {}
ids10 = read_ids(S1.reqfile(o10))
room = KV_MAX - len(LONG)
arm("U5-arm-10 clamp",
    st10 == 200 and len(ids10) == room and j10.get("finish_reason") == "length"
    and j10.get("usage", {}).get("completion_tokens") == room
    and "CLAMPED" in S1.log_text(),
    "clamped to %d ids (kv_max %d - prompt %d), finish %s, the server logged the clamp"
    % (len(ids10), KV_MAX, len(LONG), j10.get("finish_reason")))

# arm 11 — n_prompt >= kv_max is a 400 BEFORE the engine, and the server survives
st11, body11 = S1.post("/debug/tokens", {"ids": LONG + [1046] * 24, "max_tokens": 4})
ok11 = st11 == 400 and "prompt_too_long" in body11 and '"kv_max":%d' % KV_MAX in body11
st11b, _ = S1.get("/health")
arm("U5-arm-11 prompt-too-long", ok11 and st11b == 200,
    "status %d %s, then /health %d" % (st11, body11[:200], st11b))

# arm 21 — a CLIENT-caused conversation the pinned template would RAISE on is refused 400 before the
# renderer (render_chat() is a FATAL there; U2's note 5). Both shapes: non-alternating turns, and a
# system message that is not messages[0].
ord_before = next_ordinal(S1)
st21a, body21a = S1.post("/v1/chat/completions",
                         {"messages": [{"role": "user", "content": "a"},
                                       {"role": "user", "content": "b"}], "max_tokens": 4})
st21b, body21b = S1.post("/v1/chat/completions",
                         {"messages": [{"role": "user", "content": "a"},
                                       {"role": "assistant", "content": "b"},
                                       {"role": "system", "content": "late system"}], "max_tokens": 4})
st21c, _ = S1.get("/health")
arm("U5-arm-21 chat-refusals",
    st21a == 400 and "alternate" in body21a and st21b == 400 and "messages[0]" in body21b
    and st21c == 200 and next_ordinal(S1) == ord_before,
    "non-alternating -> %d (%s), late system -> %d (%s), server alive %d, engine NOT entered"
    % (st21a, body21a[26:130], st21b, body21b[26:130], st21c))

# arm 12 — the drift: a long request
st12, body12, long_req = S1.post_ord("/debug/tokens", {"ids": A3, "max_tokens": N_RESET})
n12 = len(read_ids(S1.reqfile(long_req)))
arm("U5-arm-12 drift", st12 == 200 and n12 == N_RESET,
    "%d ids after a %d-position request (status %d)" % (n12, len(A3) + N_RESET, st12))

# arm 13 (part 1) — B after the drift, on the SAME server
stB, bodyB, oB = S1.post_ord("/debug/tokens", {"ids": B1, "max_tokens": N_RESET})
after = read_ids(S1.reqfile(oB))

# arm 22 (part 1) — A CLIENT THAT VANISHES MID-REQUEST, on the REAL engine (plan §G2, U5FIX 2).
# The socket sends a streaming chat request and then disappears with an RST (SO_LINGER {on,0}) while
# the engine is still prefilling, so the abort can only be noticed at a POSITION BOUNDARY. The next
# request must be byte-identical to a fresh boot's (checked in the S2 block below against `fresh`),
# and the server must still serve. This is the one exit path the fake-engine gate covered and the
# real engine did not: it decides whether a cancelled request can leave half-applied store/rotation
# state behind.
body22 = json.dumps(dict(CHAT, stream=True, max_tokens=200)).encode()
sk = socket.create_connection(("127.0.0.1", S1.port), timeout=120)
sk.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: t\r\nContent-Type: application/json\r\n"
           b"Content-Length: " + str(len(body22)).encode() + b"\r\n\r\n" + body22)
time.sleep(2.0)                      # inside the prefill: the engine is mid-request
sk.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
sk.close()                           # RST, not a clean FIN
st22, body22r, o22 = S1.post_ord("/debug/tokens", {"ids": B1, "max_tokens": N_RESET})
after_rst = read_ids(S1.reqfile(o22))
h22, _ = S1.get("/health")
arm("U5-arm-22 cancel-rst", st22 == 200 and h22 == 200,
    "the client RST mid-stream; the next /debug/tokens -> %d (%d ids) and /health -> %d "
    "(identical-to-fresh is checked in the S2 block)" % (st22, len(after_rst), h22))

reset_lines = re.findall(r"reset\(exact\)=([0-9.]+)s", S1.log_text())
s1_cost = max((float(x) for x in reset_lines), default=-1.0)

rc1 = S1.stop()
clean = "shut down cleanly" in S1.log_text()
arm("U5-arm-20 clean-shutdown", rc1 == 0 and clean,
    "SIGTERM -> rc %d, clean-shutdown line %s" % (rc1, clean))
arm("U5-arm-19 reset-cost", len(reset_lines) > 0 and 0.0 <= s1_cost < 60.0,
    "%d request(s) paid it, max %.3f s, min %.3f s (MEASURED, this box's load; the plan's ~0.75 s "
    "was a projection)" % (len(reset_lines), s1_cost,
                           min((float(x) for x in reset_lines), default=-1.0)))

# arm 18 — the lock is released by a clean shutdown
arm("U5-arm-18 lock-released", free_lock(), "flock -n succeeds after the server exited")

# ---------------------------------------------------------------------------
# S2 — a FRESH BOOT, B only: the reset-parity comparison
# ---------------------------------------------------------------------------
S2 = Srv("s2-fresh", [])
if not S2.wait():
    arm("U5-arm-13 reset-parity", False, "the fresh server did not come up: " + S2.log_text()[-600:])
else:
    stF, bodyF, oF = S2.post_ord("/debug/tokens", {"ids": B1, "max_tokens": N_RESET})
    fresh = read_ids(S2.reqfile(oF))
    same = open(S2.reqfile(0), "rb").read() == open(S1.reqfile(oB), "rb").read()
    arm("U5-arm-13 reset-parity", same and len(fresh) == N_RESET,
        "B after A (%d ids) == a fresh boot's B, byte for byte" % len(after) if same
        else first_diff(after, fresh))
    # arm 22 (part 2) — the ids AFTER the aborted request equal a fresh boot's, byte for byte
    same22 = open(S2.reqfile(oF), "rb").read() == open(S1.reqfile(o22), "rb").read()
    arm("U5-arm-22b cancel-rst-parity", same22 and len(after_rst) == N_RESET,
        "the request after the RST is byte-identical to a fresh boot (%d ids)" % len(after_rst)
        if same22 else first_diff(after_rst, fresh))
    S2.stop()

# ---------------------------------------------------------------------------
# S3 — the CONTROL: --reset=none must NOT reproduce a fresh boot
# ---------------------------------------------------------------------------
S3 = Srv("s3-control-noreset", ["--reset", "none"])
if not S3.wait():
    arm("U5-arm-14 reset-control", False, "the control server did not come up: " + S3.log_text()[-600:])
else:
    S3.post_ord("/debug/tokens", {"ids": A3, "max_tokens": N_RESET})
    stC, _, oC = S3.post_ord("/debug/tokens", {"ids": B1, "max_tokens": N_RESET})
    ctrl = read_ids(S3.reqfile(oC))
    differs = ctrl != fresh
    at = next((i for i in range(min(len(ctrl), len(fresh))) if ctrl[i] != fresh[i]), -1)
    if differs:
        detail = ("WITHOUT the reset, B after A differs from a fresh boot at index %d (drifted %d vs "
                  "boot %d) — the reset in arm 13 is what makes the parity hold"
                  % (at, ctrl[at] if at >= 0 else -1, fresh[at] if at >= 0 else -1))
    else:
        detail = ("the control EQUALS the fresh boot (%d ids): the drift is invisible in this "
                  "configuration, so arm 13 is NOT evidence that the reset works" % len(ctrl))
    arm("U5-arm-14 reset-control", stC == 200 and differs, detail)
    S3.stop()

# ---------------------------------------------------------------------------
# S5 — arm 23: --rotate-verify makes the reset's REPLAYED BYTES measured, not argued (U5FIX 3a).
# The store's own verify_slot() reads each rewrote row back and memcmps it against the source
# expert's bytes AND checks the source pointer; a mismatch is a FATAL inside the reset. So this
# server booting, completing A then B, and matching the fresh boot proves the replayed rows are the
# right bytes.
# ---------------------------------------------------------------------------
S5 = Srv("s5-rotate-verify", ["--rotate-verify"])
if not S5.wait():
    arm("U5-arm-23 reset-bytes-verified", False,
        "the --rotate-verify server did not come up: " + S5.log_text()[-600:])
else:
    rv_on = "rotate verify = ON" in S5.log_text()
    S5.post_ord("/debug/tokens", {"ids": A3, "max_tokens": N_RESET})
    st23, _, o23 = S5.post_ord("/debug/tokens", {"ids": B1, "max_tokens": N_RESET})
    v23 = read_ids(S5.reqfile(o23))
    same23 = open(S5.reqfile(o23), "rb").read() == open(S2.reqfile(oF), "rb").read()
    arm("U5-arm-23 reset-bytes-verified", rv_on and st23 == 200 and same23,
        "readback ON, A then B, B == the fresh boot (%d ids) with every replayed slot memcmp'd"
        % len(v23) if same23 else
        ("readback ON=%s, status %d, " % (rv_on, st23)) + first_diff(v23, fresh))
    S5.stop()

# ---------------------------------------------------------------------------
# S4 — the entry invariant must be able to FIRE (mutant: the reset is skipped)
# ---------------------------------------------------------------------------
S4 = Srv("s4-mutant-skip-reset", [], env={"M1_SERVER_MUTANT": "skip_reset"})
if not S4.wait():
    arm("U5-arm-15 entry-invariant", False, "the mutant server did not come up: " + S4.log_text()[-600:])
else:
    s_ok, _ = S4.post("/debug/tokens", {"ids": B1, "max_tokens": 4})
    s_bad, bad_body = S4.post("/debug/tokens", {"ids": B1, "max_tokens": 4})
    arm("U5-arm-15 entry-invariant",
        s_ok == 200 and s_bad == 500 and "engine_state_dirty" in bad_body,
        "request 1 -> %d, request 2 (after the SKIPPED reset) -> %d %s" %
        (s_ok, s_bad, bad_body[:200]))
    S4.stop()

# ---------------------------------------------------------------------------
# arm 17 — the server exits 69 when the lock is held by someone else
# ---------------------------------------------------------------------------
holder = subprocess.Popen(["flock", "-x", LOCK, "-c", "sleep 60"], cwd=ROOT)
time.sleep(1.0)
try:
    p = subprocess.run([BIN, "--engine", "real", "--model", MODEL, "--port", "0",
                        "--lock-wait-ms", "500"] + PRODUCT,
                       cwd=ROOT, capture_output=True, text=True, timeout=600)
    rc17, out17 = p.returncode, (p.stdout + p.stderr)
finally:
    holder.terminate()
    holder.wait(timeout=60)
arm("U5-arm-17 lock-refuses-69", rc17 == 69 and "cannot take the exclusive GPU lock" in out17,
    "exit %d: %s" % (rc17, [l for l in out17.split("\n") if "lock" in l][:1]))

# ---------------------------------------------------------------------------
print()
if BLIND:
    print("GATE BLIND: %d arm(s) could not be measured: %s" % (len(BLIND), "; ".join(BLIND)))
if FAILS:
    print("GATE FAIL: %d arm(s) failed" % len(FAILS))
    for f in FAILS:
        print("  FAIL: " + f)
    sys.exit(1)
print("GATE PASS: the server's ids are the binary's ids, request after request, and the lock is "
      "held and released as designed")
sys.exit(0)
PY
