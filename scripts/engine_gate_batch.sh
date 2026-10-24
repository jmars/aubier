#!/bin/sh
# engine_gate_batch.sh — B3 GATE: the batch assembler's scheduling, on the FAKE engine.
#
#   THE VERDICT (exit code, never stdout's tail):
#     0 = every arm passed; 1 = at least one arm failed (each FAIL line names the arm and
#         what it saw); 2 = the build failed.
#
#   WHAT IT RUNS: builds/engine/m1_server_fake (CPU-only, no ggml, no gpu.lock) over the
#   batch_* seam — the fake's minimal batch mode (plan §H B3) whose greedy chain is the SOLO
#   chain by construction, so every difference these arms can see is an ASSEMBLER bug
#   (seating, phases, attribution, pads, departure, streaming, refusal shapes), which is
#   exactly this unit's surface.  The ENGINE-side batch numerics (regions, masks, kidx) are
#   B2's bars plus the real-server arms of B3 — not this gate's.
#
# THE ARMS
#   B3-arm-01 n1-identity       the whole fixed request set, default vs --batch-slots 1:
#                               status lines and (created-normalised) bodies byte-identical
#   B3-arm-02 batch-eq-solo     /debug/tokens at --batch-slots 2: the batched greedy IDS ==
#                               the single-flight IDS for the same request (fake-only promise)
#   B3-arm-03 fixed-width       Bar B: X alone (other slot an unjoined pad) vs X concurrent
#                               with Y — X's ids identical; Y too; matched per REQUEST
#   B3-arm-04 middecode-join    X runs, Y arrives 300 ms later (Y's prefill pauses X's decode
#                               mid-chain): X's ids still identical — the freeze/replay arm
#   B3-arm-05 departure         Bar C: Y finishes mid-batch (shorter max_tokens): X's ids
#                               unchanged; the pad is invisible
#   B3-arm-06 client-rst        Y's client vanishes mid-stream: X's ids unchanged, the server
#                               still serves a fresh request byte-identically
#   B3-arm-07 stream-framing    SSE at N=2: role chunk first, [DONE] LAST, the deltas
#                               concatenate to the non-stream text
#   B3-arm-08 slot-reuse        after X+Y depart, X' arrives into a freed slot: ids == X
#   B3-arm-09 stop-eos-clamp    stop string / EOS / clamp finish_reason shapes at N=2
#   B3-arm-10 budget-400        a prompt >= the per-slot region: 400 with {prompt_tokens,
#                               kv_max_slot} BEFORE the engine; the clamp lands inside
#   B3-arm-11 window-503        both slots held past --batch-window-ms: the third request
#                               gets 503 engine_busy + Retry-After: 1 (the ONE 503 shape)
#   B3-arm-12 rotate-fatal      --batch-slots 4 --rotate: startup FATAL naming BOTH flags
#   B3-arm-13 split-fatals      --kv-max not divisible by N, and a region not 16-aligned:
#                               both startup FATALs (rc 1)
#   B3-arm-14 occupancy-print   the shutdown summary prints steps/mean-live-slots (the
#                               observation instrument exists and is nonzero after work)
#
# usage: sh scripts/engine_gate_batch.sh        (CPU-only; no GPU, no gpu.lock)
set -u
ROOT=$(cd -- "$(dirname -- "$0")/.." && pwd)
BIN="$ROOT/builds/engine/m1_server_fake"

echo "engine_gate_batch.sh: building the FAKE server (scripts/engine_build_http.sh)"
if ! sh "$ROOT/scripts/engine_build_http.sh" >/dev/null; then
    echo "GATE FAIL: the build did not succeed" >&2
    exit 2
fi
[ -x "$BIN" ] || { echo "GATE FAIL: $BIN missing" >&2; exit 2; }

exec python3 - "$BIN" <<'PY'
import http.client, json, os, re, socket, subprocess, sys, threading, time

BIN = sys.argv[1]
FAILS = []

def arm(name, ok, detail=""):
    print(("PASS  " if ok else "FAIL  ") + name + (("  — " + detail) if detail else ""))
    if not ok:
        FAILS.append(name)
    return ok

class Srv:
    """One fake server instance on an ephemeral port."""
    def __init__(self, *flags, delay=0):
        self.dir = None
        self.port_file = "/tmp/m1_gate_batch_port.%d" % os.getpid()
        if os.path.exists(self.port_file):
            os.unlink(self.port_file)
        cmd = [BIN, "--port", "0", "--port-file", self.port_file] + list(flags)
        if delay:
            cmd += ["--fake-delay-ms", str(delay)]
        self.out = open("/tmp/m1_gate_batch_srv.%d.log" % os.getpid(), "ab")
        self.p = subprocess.Popen(cmd, stdout=self.out, stderr=self.out)
        for _ in range(120):
            if os.path.exists(self.port_file) and os.path.getsize(self.port_file) > 0:
                break
            if self.p.poll() is not None:
                raise RuntimeError("server died at boot rc=%s" % self.p.returncode)
            time.sleep(0.25)
        self.port = int(open(self.port_file).read().strip())

    def post(self, path, body, timeout=60):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=timeout)
        c.request("POST", path, json.dumps(body), {"Content-Type": "application/json"})
        r = c.getresponse()
        data = r.read().decode("utf-8", "replace")
        hdrs = dict(r.getheaders())
        c.close()
        return r.status, data, hdrs

    def stream(self, path, body, timeout=60, cut_after=None):
        """POST and read SSE frames; if cut_after, close the socket after that many bytes."""
        s = socket.create_connection(("127.0.0.1", self.port), timeout=timeout)
        req = ("POST %s HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
               "Content-Length: %d\r\nConnection: close\r\n\r\n%s"
               % (path, len(json.dumps(body)), json.dumps(body))).encode()
        s.sendall(req)
        if cut_after is not None:
            got = b""
            while len(got) < cut_after:
                chunk = s.recv(4096)
                if not chunk:
                    break
                got += chunk
            s.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, b"\x01\x00\x00\x00\x00\x00\x00\x00")
            s.close()
            return None  # RST'd deliberately
        buf = b""
        s.settimeout(120)
        try:
            while True:
                chunk = s.recv(4096)
                if not chunk:
                    break
                buf += chunk
        except socket.timeout:
            pass
        s.close()
        return buf.decode("utf-8", "replace")

    def stop(self):
        self.p.terminate()
        try:
            self.p.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self.p.kill()
        return open("/tmp/m1_gate_batch_srv.%d.log" % os.getpid(), "rb").read().decode(
            "utf-8", "replace")

DT = "/debug/tokens"
X_IDS = [1, 17, 64, 65, 66, 17, 24, 25, 64, 18]
Y_IDS = [3, 16, 17, 16, 17, 19, 20, 21]
def dt(ids, n, **kw):
    b = {"ids": ids, "max_tokens": n}
    b.update(kw)
    return b
def ids_of(body):
    return json.loads(body)["ids"]

# ---------------- arm 01: N=1 default vs --batch-slots 1 ----------------
FIXED = [
    (DT, dt(X_IDS, 24)),
    (DT, dt(Y_IDS, 8, stop=["abc"])),
    (DT, dt([1, 17], 100000)),
    (DT, dt([1, 2, 3], 4, temperature=0.5, seed=7)),
    ("/v1/completions", {"prompt": "Hello world", "max_tokens": 16}),
    ("/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 12}),
    ("/v1/chat/completions", {"messages": [{"role": "user", "content": "u"}], "max_tokens": 8,
                              "stream": True, "stream_options": {"include_usage": True}}),
]
CREATED = re.compile(r'"created":\d+')
def run_fixed(srv):
    out = []
    for path, body in FIXED:
        if body.get("stream"):
            raw = srv.stream(path, body)
            st = 200 if raw and raw.startswith("HTTP/1.1 200") else 0
            out.append((st, CREATED.sub('"created":N', raw or "")))
        else:
            st, data, _ = srv.post(path, body)
            out.append((st, CREATED.sub('"created":N', data)))
    return out

s1 = Srv();  a1 = run_fixed(s1); log1 = s1.stop()
s2 = Srv("--batch-slots", "1"); a2 = run_fixed(s2); log2 = s2.stop()
arm("B3-arm-01 n1-identity", a1 == a2,
    "the fixed request set differs between default and --batch-slots 1" if a1 != a2
    else "%d request(s) byte-identical (created-normalised)" % len(a1))

# ---------------- arm 02: batched == single-flight ids (the fake's promise) ----------------
solo = Srv()
st, body, _ = solo.post(DT, dt(X_IDS, 24)); ids_solo = ids_of(body)
st, body_y, _ = solo.post(DT, dt(Y_IDS, 16)); ids_y_solo = ids_of(body_y)
solo.stop()
sb = Srv("--batch-slots", "2")
st, body, _ = sb.post(DT, dt(X_IDS, 24)); ids_x_batch = ids_of(body)
arm("B3-arm-02 batch-eq-solo", ids_solo == ids_x_batch,
    "first divergence at %d" % next((i for i, (a, b) in enumerate(zip(ids_solo, ids_x_batch))
                                     if a != b), -1))

# ---------------- arm 03: Bar B — X alone vs X concurrent with Y ----------------
st, body, _ = sb.post(DT, dt(X_IDS, 24)); x_alone = ids_of(body)          # slot 0, other pad
st, body, _ = sb.post(DT, dt(Y_IDS, 16)); y_alone = ids_of(body)          # reuse of slot 0
res = {}
def go(tag, ids, n):
    st, body, _ = sb.post(DT, dt(ids, n))
    res[tag] = (st, ids_of(body))
tX = threading.Thread(target=go, args=("X", X_IDS, 24))
tY = threading.Thread(target=go, args=("Y", Y_IDS, 16))
tX.start(); tY.start(); tX.join(); tY.join()
ok3 = (res["X"][1] == x_alone and res["Y"][1] == y_alone and res["X"][0] == res["Y"][0] == 200)
arm("B3-arm-03 fixed-width", ok3,
    "X concurrent differs from X alone" if res["X"][1] != x_alone
    else "Y concurrent differs from Y alone" if res["Y"][1] != y_alone else "both identical")

# ---------------- arm 04: mid-decode join (freeze/replay) ----------------
# On a DELAYED engine (20 ms/step): X's 24 positions take ~0.6 s, so Y arriving 0.3 s in
# joins while X is MID-DECODE — X is frozen (idempotent replay) through Y's prefill.
s4 = Srv("--batch-slots", "2", delay=20)
x_alone_d = None
st, body, _ = s4.post(DT, dt(X_IDS, 24)); x_alone_d = ids_of(body)
res4 = {}
def late_join():
    time.sleep(0.3)
    st, body, _ = s4.post(DT, dt(Y_IDS, 16))
    res4["Y"] = (st, ids_of(body))
def go4(tag, ids, n):
    st, body, _ = s4.post(DT, dt(ids, n), timeout=120)
    res4[tag] = (st, ids_of(body))
tX = threading.Thread(target=go4, args=("X4", X_IDS, 24)); tX.start()
tY = threading.Thread(target=late_join); tY.start()
tX.join(); tY.join()
arm("B3-arm-04 middecode-join",
    res4["X4"][1] == x_alone_d and res4["Y"][1] == y_alone,
    "X after Y's mid-decode join differs at %d" % next(
        (i for i, (a, b) in enumerate(zip(res4["X4"][1], x_alone_d)) if a != b), -1))

# ---------------- arm 05: departure mid-batch ----------------
res5 = {}
def short_y():
    time.sleep(0.2)
    st, body, _ = s4.post(DT, dt(Y_IDS, 3))     # departs long before X
    res5["Y"] = (st, ids_of(body))
def go5(tag, ids, n):
    st, body, _ = s4.post(DT, dt(ids, n), timeout=120)
    res5[tag] = (st, ids_of(body))
tX = threading.Thread(target=go5, args=("X5", X_IDS, 24)); tX.start()
tY = threading.Thread(target=short_y); tY.start()
tX.join(); tY.join()
arm("B3-arm-05 departure", res5["X5"][1] == x_alone_d and res5["Y"][0] == 200,
    "X differs after Y departed mid-batch")

# ---------------- arm 06: client RST mid-stream ----------------
s4.stream("/v1/chat/completions",
          {"messages": [{"role": "user", "content": "u"}], "max_tokens": 40, "stream": True},
          cut_after=200)
st6, body6, _ = s4.post(DT, dt(X_IDS, 24), timeout=120)
arm("B3-arm-06 client-rst", st6 == 200 and ids_of(body6) == x_alone_d,
    "after an RST the next request differs or failed (st=%d)" % st6)

# ---------------- arm 07: SSE framing at N=2 ----------------
raw = sb.stream("/v1/chat/completions",
                {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 16,
                 "stream": True, "stream_options": {"include_usage": True}})
body7raw = raw.split("\r\n\r\n", 1)[1] if raw and "\r\n\r\n" in raw else (raw or "")
ok7 = bool(body7raw) and body7raw.endswith("data: [DONE]\n\n")
frames = [f[6:] for f in body7raw.split("\n\n") if f.startswith("data: ")]
ok7 = ok7 and frames and json.loads(frames[0])["choices"][0]["delta"].get("role") == "assistant"
text7 = "".join(json.loads(f)["choices"][0]["delta"].get("content", "")
                for f in frames[:-1] if f != "[DONE]" and json.loads(f).get("choices"))
st7, body7, _ = sb.post("/v1/chat/completions",
                        {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 16})
arm("B3-arm-07 stream-framing", ok7 and text7 == json.loads(body7)["choices"][0]["message"]["content"],
    "[DONE] last = %s" % (raw or "").endswith("data: [DONE]\n\n"))

# ---------------- arm 08: slot reuse ----------------
st, body, _ = sb.post(DT, dt(X_IDS, 24))
arm("B3-arm-08 slot-reuse", st == 200 and ids_of(body) == x_alone,
    "a request into a reused slot differs")
logb = sb.stop(); s4.stop()

# ---------------- arm 09: stop / EOS / clamp shapes at N=2 ----------------
s9 = Srv("--batch-slots", "2", delay=5)
st, body, _ = s9.post(DT, dt(Y_IDS, 16, stop=["stop"]))
j = json.loads(body)
ok9 = j["finish_reason"] == "stop" and "stop" not in j["text"] and j["usage"]["completion_tokens"] > 0
arm("B3-arm-09 stop-eos-clamp", ok9, "stop arm: finish=%s text=%r" % (j["finish_reason"], j["text"]))
s9b = Srv("--batch-slots", "2", "--kv-max", "64", delay=5)
st, body, _ = s9b.post(DT, dt(X_IDS, 100000), timeout=120)   # clamp to the region 64/2=32
j = json.loads(body)
arm("B3-arm-09b clamp", j["finish_reason"] == "length" and len(j["ids"]) == 32 - len(X_IDS),
    "clamped to %d, finish=%s" % (len(j["ids"]), j["finish_reason"]))
s9b.stop()
s9.stop()

# ---------------- arm 10: the per-slot budget 400 ----------------
s10 = Srv("--batch-slots", "2", "--kv-max", "64")
st, body, _ = s10.post(DT, dt(list(range(1, 33)), 4))     # 32 ids, region 32: n_prompt >= cap
j = json.loads(body)
ok10 = (st == 400 and j["error"]["code"] == "prompt_too_long"
        and j["error"]["prompt_tokens"] == 32 and j["error"]["kv_max_slot"] == 32)
st2, body2, _ = s10.post(DT, dt(list(range(1, 31)), 4))   # 30 ids fit: served
arm("B3-arm-10 budget-400", ok10 and st2 == 200,
    "status %d body %s" % (st, j["error"]))
s10.stop()

# ---------------- arm 11: the window 503 ----------------
s11 = Srv("--batch-slots", "2", "--batch-window-ms", "150", delay=40)
r11 = {}
def slow(tag):
    r11[tag] = s11.post(DT, dt(X_IDS, 30), timeout=120)
tA = threading.Thread(target=slow, args=("A",)); tB = threading.Thread(target=slow, args=("B",))
tA.start(); tB.start(); time.sleep(0.8)
st11, body11, h11 = s11.post(DT, dt(Y_IDS, 4))
tA.join(); tB.join()
j11 = json.loads(body11)
arm("B3-arm-11 window-503",
    st11 == 503 and j11["error"]["code"] == "engine_busy"
    and h11.get("Retry-After") == "1"
    and r11["A"][0] == 200 and r11["B"][0] == 200,
    "third request st=%d code=%s retry=%s" % (st11, j11["error"]["code"], h11.get("Retry-After")))
s11.stop()

# ---------------- arms 12/13: startup FATALs ----------------
def dies(*flags, needle=""):
    p = subprocess.run([BIN] + list(flags), capture_output=True, text=True, timeout=30)
    txt = (p.stdout + p.stderr)
    return p.returncode, txt
rc, txt = dies("--batch-slots", "4", "--rotate", needle="--rotate")
arm("B3-arm-12 rotate-fatal",
    rc == 1 and "--batch-slots" in txt and "--rotate" in txt,
    "rc=%d, names both flags: %s" % (rc, ("--batch-slots" in txt and "--rotate" in txt)))
rc, txt = dies("--batch-slots", "3")
rc2, txt2 = dies("--batch-slots", "2", "--kv-max", "2050")
arm("B3-arm-13 split-fatals",
    rc == 1 and "divisible" in txt and rc2 == 1 and "multiple of 16" in txt2,
    "divisible rc=%d, align rc=%d" % (rc, rc2))

# ---------------- arm 14: the occupancy instrument ----------------
arm("B3-arm-14 occupancy-print",
    "mean-live-slots" in logb and "steps=0," not in logb,
    "the shutdown summary prints steps/mean-live-slots")

print()
if FAILS:
    print("GATE FAIL: %d arm(s): %s" % (len(FAILS), ", ".join(FAILS)))
    sys.exit(1)
print("GATE PASS: the batch assembler's scheduling is consistent on the fake engine "
      "(%d arms)" % 14)
sys.exit(0)
PY
