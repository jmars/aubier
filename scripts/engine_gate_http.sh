#!/bin/sh
# engine_gate_http.sh — U3 GATE: the JSON + HTTP layer and the whole OpenAI surface, proven
# against the FAKE engine (server plan §H row U3). EXIT CODE IS THE VERDICT: 0 = every arm
# passed, 1 = at least one arm failed (each FAIL line names the arm and what it saw).
#
# WHAT IS UNDER TEST is the WIRE PROTOCOL, not the model: JSON strictness, HTTP framing, route
# and parameter validation, the budget rules, stop strings across token boundaries, SSE framing
# and UTF-8 chunk boundaries, usage accounting, and the /debug/tokens seam that later makes the
# engine id-parity test exact. The engine is the FAKE one in engine/m1_server.cpp, so this gate
# touches NO ggml, NO GPU and does NOT take scripts/gpu.lock — and it PROVES the first two (ldd
# on the binary names no ggml object; /proc/<pid>/fd of the running server holds no gpu.lock).
#
# THE ARMS (each prints PASS/FAIL by name):
#   U3-arm-01 link-cpu-only            no ggml/hip/rocm in ldd
#   U3-arm-02 no-gpu-lock-fd           the running server holds no fd on scripts/gpu.lock
#   U3-arm-03 models-route             GET /v1/models
#   U3-arm-04 health-route             GET /health
#   U3-arm-05 chat-nostream            exact expected text + shape
#   U3-arm-06 chat-prompt-tokens       prompt_tokens == the rendered prompt's code points
#   U3-arm-07 completions-nostream     the legacy shape, exact text
#   U3-arm-08 completions-stream       the legacy SSE chunk shape
#   U3-arm-09 sse-framing              every line is 'data: ', role chunk first, [DONE] last
#   U3-arm-10 sse-eq-nostream          concatenated stream text == non-stream text
#   U3-arm-11 sse-usage                the usage chunk only with include_usage, and it is right
#   U3-arm-12 sse-utf8-boundaries      no chunk splits a UTF-8 codepoint (strict per-chunk parse)
#   U3-arm-13 stop-single-token        stop == one token's text
#   U3-arm-14 stop-spanning-tokens     stop == 'o w', which spans "Hello" and " world"
#   U3-arm-15 stop-spanning-codepoint  stop == '你你', which spans three tokens
#   U3-arm-16 stop-streaming           the same truncation over SSE
#   U3-arm-17 max-tokens-clamp         n_prompt+max_tokens > kv_max clamps, finish_reason length
#   U3-arm-18 usage-accounting         prompt+completion == total, both routes
#   U3-arm-19 budget-prompt-too-long   400 {prompt_tokens,kv_max} BEFORE the engine, server alive
#   U3-arm-20 err-400-malformed        byte-offset diagnostic
#   U3-arm-21 err-400-bad-type
#   U3-arm-22 err-404-route
#   U3-arm-23 err-404-model            the one model is validated, never routed
#   U3-arm-24 err-405-method           + Allow
#   U3-arm-25 err-413-body
#   U3-arm-26..35 err-422-*            n, tools, image_url, logprobs, logit_bias,
#                                      response_format, penalties, echo, stream_options, stop>4
#   U3-arm-36 unsupported-defaults     a DEFAULT value of the same params is ACCEPTED
#   U3-arm-37 debug-tokens             /debug/tokens drives the engine from ids, ids echoed
#   U3-arm-38 prefix-fuzz              EVERY truncation of a valid body is 400, never 200/crash
#   U3-arm-39 concurrent-busy-503      --max-conns 1: the second connection gets 503+Retry-After
#   U3-arm-40 concurrent-served        two connections: both served, serialised (not concurrent)
#   U3-arm-41 mid-stream-disconnect    a killed client does not disturb the NEXT request
#   U3-arm-42 gate-fails-on-mutant     the framing check MUST fail on M1_SERVER_MUTANT=drop_done
#   U3-arm-43 gate-fails-on-utf8-mutant the UTF-8 boundary check MUST fail on
#                                      M1_SERVER_MUTANT=split_utf8
#   U3-arm-44 sampling-applied         temperature/top_p/top_k/seed reach the engine: same seed
#                                      reproduces, another seed differs, greedy stays greedy
#   U3-arm-45 json-escapes             \" \\ \n \t \uXXXX and surrogate pairs are DECODED (a
#                                      parser that passed escapes through literally shipped once)
#   U3-arm-46 json-strictness          control byte / invalid UTF-8 / lone surrogate / leading
#                                      zero / depth > 64 are each refused by the right check
#   U3-arm-47 graceful-shutdown        SIGTERM mid-request: exits 0, drains at a position
#                                      boundary, serves nothing afterwards
#
# THE ASYMMETRY PROOF: arm 42 runs arm 09's checker against a server with
# M1_SERVER_MUTANT=drop_done and REQUIRES it to fail; arm 43 does the same for arm 12 with
# M1_SERVER_MUTANT=split_utf8. The whole gate can also be run that way:
#     M1_SERVER_MUTANT=drop_done  sh scripts/engine_gate_http.sh  # exits 1
#     M1_SERVER_MUTANT=split_utf8 sh scripts/engine_gate_http.sh  # exits 1
#     sh scripts/engine_gate_http.sh                              # exits 0
#
# usage: sh scripts/engine_gate_http.sh
# No arguments. It builds the server itself (scripts/engine_build_http.sh), binds 127.0.0.1 on
# an EPHEMERAL port per instance (--port 0, the chosen port is reported through --port-file), so
# parallel runs cannot collide and no fixed port is ever held.

set -u

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
BIN="$ROOT/builds/engine/m1_server_fake"

echo "engine_gate_http.sh: building (scripts/engine_build_http.sh)"
if ! sh "$ROOT/scripts/engine_build_http.sh"; then
    echo "GATE FAIL: the build did not succeed" >&2
    exit 1
fi
if [ ! -x "$BIN" ]; then
    echo "GATE FAIL: $BIN is not executable" >&2
    exit 1
fi

exec python3 - "$ROOT" "$BIN" <<'PY'
import json, os, re, signal, socket, subprocess, sys, tempfile, threading, time

ROOT, BIN = sys.argv[1], sys.argv[2]

# ---------------------------------------------------------------------------
# the fake engine's documented output (engine/m1_server.cpp file header)
# ---------------------------------------------------------------------------
PIECES = {16: "Hello", 17: " world", 18: "!", 19: "\n",
          20: "\u4f60", 21: "\u00e9", 22: b"\xe4", 23: b"\xbd\xa0",
          24: "stop", 25: "abc"}
CYCLE = [16, 17, 18, 19, 20, 22, 23, 24, 25, 21]
EXPECT10 = "Hello world!\n\u4f60\u4f60stopabc\u00e9"

# the cycle's text, assembled from the same table the server uses: an independent recomputation
def cycle_text(n):
    out = b""
    for i in range(n):
        p = PIECES[CYCLE[i % len(CYCLE)]]
        out += p.encode("utf-8") if isinstance(p, str) else p
    return out.decode("utf-8")

assert cycle_text(10) == EXPECT10, (cycle_text(10), EXPECT10)

KV_MAX = 128
model_name = "mistral-small-4"

# ---------------------------------------------------------------------------
# tiny harness
# ---------------------------------------------------------------------------
class ArmFail(Exception):
    pass

RESULTS = []

def arm(name):
    def deco(fn):
        try:
            fn()
            print(f"PASS {name}")
            RESULTS.append((name, None))
        except ArmFail as e:
            print(f"FAIL {name}: {e}")
            RESULTS.append((name, str(e)))
        except Exception as e:  # a harness error is a FAIL, not a pass
            print(f"FAIL {name}: harness error: {type(e).__name__}: {e}")
            RESULTS.append((name, f"{type(e).__name__}: {e}"))
        return fn
    return deco

def eq(what, got, want):
    if got != want:
        raise ArmFail(f"{what}: got {got!r}, want {want!r}")

def ok(what, cond, detail=""):
    if not cond:
        raise ArmFail(f"{what}{(': ' + str(detail)) if detail else ''}")

# ---------------------------------------------------------------------------
# a server instance
# ---------------------------------------------------------------------------
SERVERS = []

class Srv:
    def __init__(self, extra=(), env=None):
        self.portfile = tempfile.mktemp(prefix="m1srv-port-")
        self.logfile = tempfile.mktemp(prefix="m1srv-log-")
        self.logf = open(self.logfile, "wb")
        e = dict(os.environ)
        if env:
            e.update(env)
        cmd = [BIN, "--port", "0", "--port-file", self.portfile,
               "--fake-kv-max", str(KV_MAX)] + list(extra)
        self.cmd = cmd
        self.p = subprocess.Popen(cmd, stdout=self.logf, stderr=subprocess.STDOUT, env=e)
        SERVERS.append(self)
        for _ in range(500):
            if os.path.exists(self.portfile):
                txt = open(self.portfile).read().strip()
                if txt:
                    self.port = int(txt)
                    return
            if self.p.poll() is not None:
                raise ArmFail("the server exited during boot:\n" + self.log())
            time.sleep(0.01)
        raise ArmFail("the server never reported its port")

    def log(self):
        try:
            self.logf.flush()
            return open(self.logfile, errors="replace").read()
        except Exception:
            return "<no log>"

    def stop(self):
        if self.p.poll() is None:
            self.p.send_signal(signal.SIGTERM)
            try:
                self.p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.p.kill()
                self.p.wait()
        try:
            self.logf.close()
        except Exception:
            pass

def stop_all():
    for s in SERVERS:
        s.stop()

# ---------------------------------------------------------------------------
# HTTP over a raw socket (full control: truncations, SSE, mid-stream disconnects)
# ---------------------------------------------------------------------------
def raw_request(port, method, path, body=None, headers=None, timeout=20, read_limit=None):
    s = socket.create_connection(("127.0.0.1", port), timeout=timeout)
    try:
        hdrs = {"Host": "127.0.0.1:1"}
        if headers:
            hdrs.update(headers)
        if body is not None:
            if isinstance(body, str):
                body = body.encode()
            hdrs["Content-Length"] = str(len(body))
        req = f"{method} {path} HTTP/1.1\r\n"
        req += "".join(f"{k}: {v}\r\n" for k, v in hdrs.items())
        req += "Connection: close\r\n\r\n"
        try:
            s.sendall(req.encode())
            if body is not None:
                s.sendall(body)
        except (BrokenPipeError, ConnectionResetError):
            pass  # the server may answer 413 and close before the body is done
        buf = b""
        while True:
            if read_limit is not None and len(buf) >= read_limit:
                break
            try:
                chunk = s.recv(65536)
            except (ConnectionResetError, socket.timeout):
                break
            if not chunk:
                break
            buf += chunk
            if read_limit is None:
                continue
    finally:
        s.close()
    return buf

def parse_response(buf):
    if not buf:
        raise ArmFail("empty response (the connection was closed with nothing written)")
    head, sep, body = buf.partition(b"\r\n\r\n")
    if not sep:
        raise ArmFail(f"no header terminator in the response: {buf[:200]!r}")
    lines = head.decode("latin-1").split("\r\n")
    m = re.match(r"HTTP/1\.[01] (\d{3}) ", lines[0])
    if not m:
        raise ArmFail(f"malformed status line: {lines[0]!r}")
    status = int(m.group(1))
    headers = {}
    for ln in lines[1:]:
        k, _, v = ln.partition(":")
        headers[k.strip().lower()] = v.strip()
    return status, headers, body

def call(port, method, path, body=None, headers=None, timeout=20):
    st, hdrs, raw = parse_response(raw_request(port, method, path, body, headers, timeout))
    return st, hdrs, raw

def call_json(port, method, path, body=None, headers=None):
    st, hdrs, raw = call(port, method, path, body, headers)
    try:
        doc = json.loads(raw)
    except Exception as e:
        raise ArmFail(f"{method} {path}: the body is not JSON ({e}): {raw[:300]!r}")
    return st, hdrs, doc

def post_json(port, path, obj, headers=None):
    return call_json(port, "POST", path, json.dumps(obj),
                     dict({"Content-Type": "application/json"}, **(headers or {})))

# ---------------------------------------------------------------------------
# SSE parsing: the framing contract, checked in one place so the mutant arm can reuse it
# ---------------------------------------------------------------------------
def sse_lines(port, body):
    st, hdrs, raw = call(port, "POST", "/v1/chat/completions", json.dumps(body))
    eq("status", st, 200)
    eq("content-type", hdrs.get("content-type"), "text/event-stream")
    payloads = []
    for part in raw.split(b"\n\n"):
        if not part:
            continue
        # STRICT per-line decode: if the server split a codepoint across chunks, this line
        # carries a bare continuation byte and decoding raises.
        try:
            line = part.decode("utf-8")
        except UnicodeDecodeError as e:
            raise ArmFail(f"an SSE line is not valid UTF-8 ({e}); a codepoint was split "
                          f"across chunks: {part!r}")
        ok("every SSE line starts with the exact 'data: ' prefix", line.startswith("data: "),
           repr(line[:80]))
        ok("no SSE line carries an embedded newline", "\n" not in line, repr(line[:120]))
        payloads.append(line[len("data: "):])
    ok("the stream has more than one frame", len(payloads) >= 2, payloads)
    return payloads

def check_framing(payloads, expect_text, include_usage=False):
    """The framing contract of plan §F. Raises ArmFail; arm 42 requires it to raise."""
    eq("the LAST frame is exactly [DONE]", payloads[-1], "[DONE]")
    docs = []
    for p in payloads[:-1]:
        try:
            docs.append(json.loads(p))
        except Exception as e:
            raise ArmFail(f"an SSE payload is not JSON ({e}): {p[:200]!r}")
    role_chunks = [d for d in docs if d["choices"] and d["choices"][0].get("delta", {}).get("role")]
    eq("exactly one chunk carries delta.role", len(role_chunks), 1)
    ok("the FIRST frame is the role chunk", docs[0] is role_chunks[0], docs[0])
    eq("the role chunk has finish_reason null", docs[0]["choices"][0]["finish_reason"], None)
    finish_chunks = [d for d in docs if d["choices"] and d["choices"][0]["finish_reason"] is not None]
    eq("exactly one chunk carries a finish_reason", len(finish_chunks), 1)
    text = "".join(d["choices"][0]["delta"].get("content", "")
                   for d in docs if d["choices"] and d["choices"][0].get("delta"))
    eq("the concatenated stream text", text, expect_text)
    usage_chunks = [d for d in docs if "usage" in d and d["usage"] is not None]
    if include_usage:
        eq("exactly one usage chunk", len(usage_chunks), 1)
        ok("the usage chunk has empty choices (the OpenAI shape)", usage_chunks[0]["choices"] == [],
           usage_chunks[0])
        eq("the usage chunk is the LAST data frame before [DONE]", docs[-1], usage_chunks[0])
        ok("the finish chunk precedes the usage chunk", docs[-2] is finish_chunks[0], docs[-2])
    else:
        eq("no usage chunk without stream_options.include_usage", len(usage_chunks), 0)
        eq("the finish chunk is the LAST data frame before [DONE]", docs[-1], finish_chunks[0])
    return text, finish_chunks[0]["choices"][0]["finish_reason"]

# ---------------------------------------------------------------------------
# the shared request bodies
# ---------------------------------------------------------------------------
CHAT = {"model": model_name, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 10}
STREAM = dict(CHAT, stream=True)

main = Srv(["--verbose", "--max-conns", "8"])

# ---------------------------------------------------------------------------
# 01-02: the CPU-only / no-lock policy, MEASURED on the artifact
# ---------------------------------------------------------------------------
@arm("U3-arm-01 link-cpu-only")
def _a01():
    out = subprocess.run(["ldd", BIN], capture_output=True, text=True).stdout
    ok("the binary links no ggml object", "ggml" not in out, out.replace("\n", " | "))
    low = out.lower()
    ok("the binary links no hip/rocm library", not any(x in low for x in ("amdhip", "rocm", "libhipblas")),
       out.replace("\n", " | "))

@arm("U3-arm-02 no-gpu-lock-fd")
def _a02():
    fds = []
    for fd in os.listdir(f"/proc/{main.p.pid}/fd"):
        try:
            fds.append(os.readlink(f"/proc/{main.p.pid}/fd/{fd}"))
        except OSError:
            pass
    hits = [f for f in fds if "gpu.lock" in f]
    ok("no fd points at scripts/gpu.lock (U3 takes no GPU lock)", hits == [], hits)
    st, _, doc = call_json(main.port, "GET", "/health")
    eq("status", st, 200)

# ---------------------------------------------------------------------------
# 03-04: the metadata routes
# ---------------------------------------------------------------------------
@arm("U3-arm-03 models-route")
def _a03():
    st, _, doc = call_json(main.port, "GET", "/v1/models")
    eq("status", st, 200)
    eq("object", doc["object"], "list")
    eq("exactly one model", len(doc["data"]), 1)
    eq("the model id", doc["data"][0]["id"], model_name)
    eq("the model object", doc["data"][0]["object"], "model")

@arm("U3-arm-04 health-route")
def _a04():
    st, _, doc = call_json(main.port, "GET", "/health")
    eq("status", st, 200)
    eq("status field", doc["status"], "ok")
    eq("the engine is the fake one", doc["engine"], "fake")
    eq("kv_max", doc["kv_max"], KV_MAX)
    ok("/health reports the request counter", isinstance(doc["requests"], int), doc)

# ---------------------------------------------------------------------------
# 05-08: the two completion routes, exact output
# ---------------------------------------------------------------------------
@arm("U3-arm-05 chat-nostream")
def _a05():
    st, _, doc = post_json(main.port, "/v1/chat/completions", CHAT)
    eq("status", st, 200)
    eq("object", doc["object"], "chat.completion")
    eq("model", doc["model"], model_name)
    ok("the id is a chatcmpl id", doc["id"].startswith("chatcmpl-"), doc["id"])
    eq("choices length", len(doc["choices"]), 1)
    eq("role", doc["choices"][0]["message"]["role"], "assistant")
    eq("the exact detokenized text", doc["choices"][0]["message"]["content"], EXPECT10)
    eq("finish_reason", doc["choices"][0]["finish_reason"], "length")

@arm("U3-arm-06 chat-prompt-tokens")
def _a06():
    msgs = [{"role": "user", "content": "hi"},
            {"role": "assistant", "content": "there"},
            {"role": "system", "content": "Be terse."}]
    st, _, doc = post_json(main.port, "/v1/chat/completions",
                           {"messages": msgs, "max_tokens": 4})
    eq("status", st, 200)
    # the U3 placeholder render, stated in engine/m1_server.cpp's render_chat_prompt
    rendered = "".join(m["role"] + ": " + m["content"] + "\n" for m in msgs)
    eq("prompt_tokens == the rendered prompt's code points", doc["usage"]["prompt_tokens"],
       len(rendered))
    eq("completion_tokens", doc["usage"]["completion_tokens"], 4)

@arm("U3-arm-07 completions-nostream")
def _a07():
    st, _, doc = post_json(main.port, "/v1/completions",
                           {"model": model_name, "prompt": "abc", "max_tokens": 10})
    eq("status", st, 200)
    eq("object", doc["object"], "text_completion")
    ok("the id is a cmpl id", doc["id"].startswith("cmpl-"), doc["id"])
    eq("the exact text", doc["choices"][0]["text"], EXPECT10)
    eq("finish_reason", doc["choices"][0]["finish_reason"], "length")
    eq("prompt_tokens", doc["usage"]["prompt_tokens"], 3)

@arm("U3-arm-08 completions-stream")
def _a08():
    st, hdrs, raw = call(main.port, "POST", "/v1/completions",
                         json.dumps({"prompt": "abc", "max_tokens": 4, "stream": True}))
    eq("status", st, 200)
    eq("content-type", hdrs.get("content-type"), "text/event-stream")
    payloads = [p for p in raw.split(b"\n\n") if p]
    eq("the last frame", payloads[-1], b"data: [DONE]")
    docs = [json.loads(p.decode()[len("data: "):]) for p in payloads[:-1]]
    eq("every chunk is a text_completion", {d["object"] for d in docs}, {"text_completion"})
    text = "".join(d["choices"][0].get("text", "") for d in docs if d["choices"])
    eq("the concatenated text", text, cycle_text(4))

# ---------------------------------------------------------------------------
# 09-12: SSE framing, stream == non-stream, usage, UTF-8 boundaries
# ---------------------------------------------------------------------------
@arm("U3-arm-09 sse-framing")
def _a09():
    check_framing(sse_lines(main.port, STREAM), EXPECT10, include_usage=False)

@arm("U3-arm-10 sse-eq-nostream")
def _a10():
    _, _, doc = post_json(main.port, "/v1/chat/completions", CHAT)
    streamed, _ = check_framing(sse_lines(main.port, STREAM), EXPECT10)
    eq("the streamed text equals the non-stream text", streamed, doc["choices"][0]["message"]["content"])

@arm("U3-arm-11 sse-usage")
def _a11():
    body = dict(STREAM, stream_options={"include_usage": True})
    payloads = sse_lines(main.port, body)
    text, finish = check_framing(payloads, EXPECT10, include_usage=True)
    eq("finish_reason", finish, "length")
    usage = json.loads(payloads[-2])["usage"]
    eq("prompt_tokens", usage["prompt_tokens"], 9)
    eq("completion_tokens", usage["completion_tokens"], 10)
    eq("total_tokens", usage["total_tokens"], 19)

@arm("U3-arm-12 sse-utf8-boundaries")
def _a12():
    # The fake cycle SPLITS a codepoint across tokens 22/23 (0xE4 then 0xBD 0xA0). sse_lines
    # decodes every frame strictly, so a chunk boundary inside that codepoint fails there.
    payloads = sse_lines(main.port, STREAM)
    contents = [json.loads(p)["choices"][0]["delta"].get("content", "")
                for p in payloads[:-1] if json.loads(p)["choices"]]
    joined = "".join(contents)
    eq("the text survives the token-split codepoint", joined, EXPECT10)
    ok("the codepoint was emitted in one chunk, not two",
       any("\u4f60" in c for c in contents), contents)

# ---------------------------------------------------------------------------
# 13-16: stop strings (single token, spanning tokens, spanning a codepoint, streaming)
# ---------------------------------------------------------------------------
@arm("U3-arm-13 stop-single-token")
def _a13():
    st, _, doc = post_json(main.port, "/v1/chat/completions", dict(CHAT, stop=" world"))
    eq("status", st, 200)
    eq("the text is truncated at the stop string", doc["choices"][0]["message"]["content"], "Hello")
    eq("finish_reason", doc["choices"][0]["finish_reason"], "stop")

@arm("U3-arm-14 stop-spanning-tokens")
def _a14():
    # "o w" starts INSIDE token 16 ("Hello") and ends inside token 17 (" world")
    st, _, doc = post_json(main.port, "/v1/chat/completions", dict(CHAT, stop=["zzz", "o w"]))
    eq("status", st, 200)
    eq("the text is truncated mid-token at the first match", doc["choices"][0]["message"]["content"], "Hell")
    eq("finish_reason", doc["choices"][0]["finish_reason"], "stop")

@arm("U3-arm-15 stop-spanning-codepoint")
def _a15():
    # "你你" spans tokens 20, 22 and 23 (a codepoint split across two tokens)
    st, _, doc = post_json(main.port, "/v1/chat/completions", dict(CHAT, stop="\u4f60\u4f60"))
    eq("status", st, 200)
    eq("the text before the three-token match", doc["choices"][0]["message"]["content"],
       "Hello world!\n")
    eq("finish_reason", doc["choices"][0]["finish_reason"], "stop")

@arm("U3-arm-16 stop-streaming")
def _a16():
    body = dict(STREAM, stop="o w")
    payloads = sse_lines(main.port, body)
    text, finish = check_framing(payloads, "Hell")
    eq("finish_reason over SSE", finish, "stop")

# ---------------------------------------------------------------------------
# 17-18: the budget clamp and usage accounting
# ---------------------------------------------------------------------------
@arm("U3-arm-17 max-tokens-clamp")
def _a17():
    st, _, doc = post_json(main.port, "/v1/chat/completions", dict(CHAT, max_tokens=100000))
    eq("status", st, 200)
    room = KV_MAX - 9  # prompt tokens of "user: hi\n"
    eq("completion_tokens is clamped to kv_max - prompt_tokens",
       doc["usage"]["completion_tokens"], room)
    eq("finish_reason is length", doc["choices"][0]["finish_reason"], "length")
    eq("the clamped text is exactly that many tokens", doc["choices"][0]["message"]["content"],
       cycle_text(room))

@arm("U3-arm-18 usage-accounting")
def _a18():
    _, _, doc = post_json(main.port, "/v1/chat/completions", CHAT)
    u = doc["usage"]
    eq("prompt_tokens", u["prompt_tokens"], 9)
    eq("completion_tokens", u["completion_tokens"], 10)
    eq("total == prompt + completion", u["total_tokens"], u["prompt_tokens"] + u["completion_tokens"])
    _, _, cdoc = post_json(main.port, "/v1/completions", {"prompt": "abc", "max_tokens": 3})
    cu = cdoc["usage"]
    eq("completions prompt_tokens", cu["prompt_tokens"], 3)
    eq("completions completion_tokens", cu["completion_tokens"], 3)
    eq("completions total", cu["total_tokens"], 6)

# ---------------------------------------------------------------------------
# 19: the prompt budget, checked BEFORE the engine, server still alive after
# ---------------------------------------------------------------------------
@arm("U3-arm-19 budget-prompt-too-long")
def _a19():
    huge = "x" * (KV_MAX + 50)
    st, _, doc = post_json(main.port, "/v1/chat/completions",
                           {"messages": [{"role": "user", "content": huge}], "max_tokens": 4})
    eq("status", st, 400)
    ok("prompt_tokens is reported", isinstance(doc["error"].get("prompt_tokens"), int), doc["error"])
    eq("kv_max is reported", doc["error"]["kv_max"], KV_MAX)
    ok("prompt_tokens >= kv_max", doc["error"]["prompt_tokens"] >= KV_MAX, doc["error"])
    eq("the code", doc["error"]["code"], "prompt_too_long")
    # the engine was never entered, so every later request still works
    st, _, doc = call_json(main.port, "GET", "/health")
    eq("the server is still alive after an over-long prompt", st, 200)

# ---------------------------------------------------------------------------
# 20-25: the error table, one arm per code
# ---------------------------------------------------------------------------
@arm("U3-arm-20 err-400-malformed")
def _a20():
    st, _, doc = call_json(main.port, "POST", "/v1/chat/completions", '{"messages": [')
    eq("status", st, 400)
    ok("the message names a byte offset", "byte" in doc["error"]["message"], doc["error"])
    ok("the offset is reported as a number", isinstance(doc["error"].get("offset"), int), doc["error"])
    ok("type", doc["error"]["type"] == "invalid_request_error", doc["error"])

@arm("U3-arm-21 err-400-bad-type")
def _a21():
    for body, param in [(dict(CHAT, max_tokens="ten"), "max_tokens"),
                        (dict(CHAT, stream="yes"), "stream"),
                        ({"messages": "hi"}, "messages"),
                        (dict(CHAT, temperature=[1]), "temperature")]:
        st, _, doc = post_json(main.port, "/v1/chat/completions", body)
        eq(f"status for {param}", st, 400)
        eq(f"the failing param for {param}", doc["error"]["param"], param)

@arm("U3-arm-22 err-404-route")
def _a22():
    st, _, doc = call_json(main.port, "GET", "/v1/nope")
    eq("status", st, 404)
    eq("the code", doc["error"]["code"], "unknown_route")

@arm("U3-arm-23 err-404-model")
def _a23():
    st, _, doc = post_json(main.port, "/v1/chat/completions", dict(CHAT, model="gpt-4o"))
    eq("status", st, 404)
    eq("the failing param", doc["error"]["param"], "model")
    eq("the code", doc["error"]["code"], "model_not_found")
    # the served name is accepted (no routing: exactly one)
    st, _, _ = post_json(main.port, "/v1/chat/completions", CHAT)
    eq("the served model is accepted", st, 200)

@arm("U3-arm-24 err-405-method")
def _a24():
    for method, path in [("GET", "/v1/chat/completions"), ("POST", "/v1/models"),
                         ("POST", "/health"), ("GET", "/debug/tokens")]:
        st, hdrs, doc = call_json(main.port, method, path, "{}" if method == "POST" else None)
        eq(f"status for {method} {path}", st, 405)
        ok(f"Allow is sent for {method} {path}", "allow" in hdrs, hdrs)
        eq(f"the code for {method} {path}", doc["error"]["code"], "method_not_allowed")

@arm("U3-arm-25 err-413-body")
def _a25():
    body = b'{"messages":[{"role":"user","content":"' + b"a" * (5 * 1024 * 1024) + b'"}]}'
    st, _, doc = call_json(main.port, "POST", "/v1/chat/completions", body)
    eq("status", st, 413)
    ok("it is the refusal shape", "error" in doc, doc)
    st, _, _ = call_json(main.port, "GET", "/health")
    eq("the server is still alive after an oversize body", st, 200)

# ---------------------------------------------------------------------------
# 26-35: recognised but unsupported => 422 BY NAME (never silently ignored)
# ---------------------------------------------------------------------------
def expect_422(body, param, path="/v1/chat/completions"):
    st, _, doc = post_json(main.port, path, body)
    eq(f"status for {param}", st, 422)
    eq(f"the 422 names the parameter ({param})", doc["error"]["param"], param)
    eq(f"the code for {param}", doc["error"]["code"], "unsupported_parameter")

@arm("U3-arm-26 err-422-n")
def _a26():
    expect_422(dict(CHAT, n=2), "n")
    expect_422(dict(CHAT, n=0), "n")

@arm("U3-arm-27 err-422-tools")
def _a27():
    expect_422(dict(CHAT, tools=[{"type": "function", "function": {"name": "f"}}]), "tools")
    expect_422(dict(CHAT, tool_choice="auto"), "tool_choice")
    expect_422({"messages": [{"role": "tool", "content": "x"}]}, "tools")

@arm("U3-arm-28 err-422-image_url")
def _a28():
    body = {"messages": [{"role": "user", "content": [
        {"type": "text", "text": "what is this"},
        {"type": "image_url", "image_url": {"url": "http://example/1.png"}}]}],
        "max_tokens": 4}
    expect_422(body, "image_url")

@arm("U3-arm-29 err-422-logprobs")
def _a29():
    expect_422(dict(CHAT, logprobs=True), "logprobs")
    expect_422(dict(CHAT, top_logprobs=5), "top_logprobs")

@arm("U3-arm-30 err-422-logit_bias")
def _a30():
    expect_422(dict(CHAT, logit_bias={"50256": -100}), "logit_bias")

@arm("U3-arm-31 err-422-response_format")
def _a31():
    expect_422(dict(CHAT, response_format={"type": "json_object"}), "response_format")

@arm("U3-arm-32 err-422-penalties")
def _a32():
    expect_422(dict(CHAT, frequency_penalty=0.5), "frequency_penalty")
    expect_422(dict(CHAT, presence_penalty=0.5), "presence_penalty")

@arm("U3-arm-33 err-422-echo")
def _a33():
    expect_422({"prompt": "abc", "echo": True}, "echo", path="/v1/completions")

@arm("U3-arm-34 err-422-stream_options")
def _a34():
    expect_422(dict(STREAM, stream_options={"include_usage": True, "chunk_size": 4}),
               "stream_options.chunk_size")

@arm("U3-arm-35 err-422-stop-too-many")
def _a35():
    expect_422(dict(CHAT, stop=["a", "b", "c", "d", "e"]), "stop")

@arm("U3-arm-36 unsupported-defaults-accepted")
def _a36():
    # The OTHER half of the 422 rule: a DEFAULT value must be accepted, not refused. This is
    # what keeps us from over-refusing ordinary OpenAI clients.
    body = dict(CHAT, n=1, logprobs=False, top_logprobs=0, logit_bias={},
                response_format={"type": "text"}, frequency_penalty=0, presence_penalty=0,
                repetition_penalty=1.0, echo=False, best_of=1, suffix="",
                stream_options={"include_usage": False})
    st, _, doc = post_json(main.port, "/v1/chat/completions", body)
    eq("status", st, 200)
    eq("the text is unaffected", doc["choices"][0]["message"]["content"], EXPECT10)

# ---------------------------------------------------------------------------
# 37: the /debug/tokens seam
# ---------------------------------------------------------------------------
@arm("U3-arm-37 debug-tokens")
def _a37():
    ids = [1, 3, 4, 15, 7]  # inside the fake vocab (1024 ids)
    st, _, doc = post_json(main.port, "/debug/tokens", {"ids": ids, "max_tokens": 10})
    eq("status", st, 200)
    eq("the generated ids", doc["ids"], CYCLE)
    eq("the detokenized text", doc["text"], EXPECT10)
    eq("finish_reason", doc["finish_reason"], "length")
    eq("prompt_tokens", doc["usage"]["prompt_tokens"], len(ids))
    eq("completion_tokens", doc["usage"]["completion_tokens"], 10)
    # the same ids drive the SAME engine path the chat route uses: identical text, both ways
    _, _, cdoc = post_json(main.port, "/v1/chat/completions", CHAT)
    eq("/debug/tokens and /v1/chat/completions agree on the text",
       doc["text"], cdoc["choices"][0]["message"]["content"])
    # an out-of-vocabulary id is a 400 here, never an engine FATAL
    st, _, edoc = post_json(main.port, "/debug/tokens", {"ids": [1, 99999], "max_tokens": 2})  # out of vocab
    eq("status for a bad id", st, 400)
    eq("the failing param", edoc["error"]["param"], "ids[]")

# ---------------------------------------------------------------------------
# 38: prefix fuzz — every truncation of a valid body must be refused, never crash
# ---------------------------------------------------------------------------
@arm("U3-arm-38 prefix-fuzz")
def _a38():
    body = json.dumps({"model": model_name, "messages": [{"role": "user", "content": "hi"}],
                       "max_tokens": 10, "temperature": 0.0, "stop": ["x"]})
    bad = []
    for cut in range(len(body)):
        piece = body[:cut]
        st, _, raw = call(main.port, "POST", "/v1/chat/completions", piece)
        if st != 400:
            bad.append((cut, st, piece[:40], raw[:120]))
            continue
        try:
            doc = json.loads(raw)
        except Exception:
            bad.append((cut, st, piece[:40], b"non-JSON body: " + raw[:80]))
            continue
        if "error" not in doc or not doc["error"].get("message"):
            bad.append((cut, st, piece[:40], raw[:120]))
    ok(f"every one of {len(body)} truncations is a 400 with an error message", not bad,
       f"{len(bad)} bad, first: {bad[:3]}")
    st, _, doc = call_json(main.port, "GET", "/health")
    eq("the server survived the fuzz", st, 200)
    rc = main.p.poll()
    ok("the server process is still running", rc is None, f"exit code {rc}")

# ---------------------------------------------------------------------------
# 39-41: concurrency and the disappearing client
# ---------------------------------------------------------------------------
@arm("U3-arm-39 concurrent-busy-503")
def _a39():
    s = Srv(["--max-conns", "1", "--fake-delay-ms", "400"])
    got = {}

    def slow():
        body = dict(CHAT, max_tokens=6, stream=True)
        got["slow"] = call(s.port, "POST", "/v1/chat/completions", json.dumps(body))

    t = threading.Thread(target=slow)
    t.start()
    time.sleep(0.25)  # the first connection is in flight and holding the only slot
    st, hdrs, doc = call_json(s.port, "POST", "/v1/chat/completions", json.dumps(dict(CHAT, max_tokens=2)))
    eq("the second connection is refused with 503", st, 503)
    eq("Retry-After is set", hdrs.get("retry-after"), "1")
    eq("the code", doc["error"]["code"], "server_busy")
    t.join(30)
    ok("the first connection was served", got["slow"][0] == 200, got["slow"][0])
    st, _, _ = call_json(s.port, "GET", "/health")
    eq("the server is still healthy", st, 200)

@arm("U3-arm-40 concurrent-served")
def _a40():
    s = Srv(["--max-conns", "4", "--fake-delay-ms", "250"])
    out = {}

    def one(k):
        out[k] = call(s.port, "POST", "/v1/chat/completions", json.dumps(dict(CHAT, max_tokens=4)))

    t0 = time.time()
    ts = [threading.Thread(target=one, args=(k,)) for k in (0, 1)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(60)
    elapsed = time.time() - t0
    for k in (0, 1):
        eq(f"connection {k} status", out[k][0], 200)
    ok("the two requests were serialised on the single-flight engine "
       f"(elapsed {elapsed:.2f}s for 2 x 250ms x 4 tokens)", elapsed >= 1.6, f"{elapsed:.2f}s")
    d0 = json.loads(out[0][2])
    d1 = json.loads(out[1][2])
    eq("both requests produce the identical text (the fake is deterministic)",
       d0["choices"][0]["message"]["content"], d1["choices"][0]["message"]["content"])
    eq("both report 4 completion tokens", d0["usage"]["completion_tokens"], 4)

@arm("U3-arm-41 mid-stream-disconnect")
def _a41():
    s = Srv(["--fake-delay-ms", "40"])
    _, _, base = post_json(s.port, "/v1/chat/completions", CHAT)
    expect = base["choices"][0]["message"]["content"]
    # start a long stream and kill the client half way
    sock = socket.create_connection(("127.0.0.1", s.port), timeout=20)
    body = json.dumps(dict(STREAM, max_tokens=100)).encode()
    req = (b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
           b"Content-Length: " + str(len(body)).encode() + b"\r\nConnection: close\r\n\r\n" + body)
    sock.sendall(req)
    time.sleep(0.3)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, b"\x01\x00\x00\x00\x00\x00\x00\x00")
    sock.close()  # RST: the next server write fails with EPIPE/ECONNRESET
    time.sleep(0.4)
    st, _, doc = call_json(s.port, "GET", "/health")
    eq("the server survived the disconnect", st, 200)
    st, _, doc = post_json(s.port, "/v1/chat/completions", CHAT)
    eq("the next request is served", st, 200)
    eq("and it is byte-identical to the pre-disconnect baseline",
       doc["choices"][0]["message"]["content"], expect)
    rc = s.p.poll()
    ok("the server process is still running", rc is None, f"exit code {rc}")

# ---------------------------------------------------------------------------
# 42: THE ASYMMETRY PROOF — the framing check must fail on a mutant
# ---------------------------------------------------------------------------
@arm("U3-arm-42 gate-fails-on-mutant")
def _a42():
    m = Srv(env={"M1_SERVER_MUTANT": "drop_done"})
    try:
        ok("the mutant announces itself at boot", "TEST MUTANT ACTIVE" in m.log(), m.log()[:400])
        detail = None
        try:
            check_framing(sse_lines(m.port, STREAM), EXPECT10)
        except ArmFail as e:
            detail = str(e)
        ok("the framing check FAILS on the mutant (it drops data: [DONE])", detail is not None,
           "the mutant PASSED the framing check, so that check cannot fail")
        ok("the failure names the missing [DONE] terminator", "[DONE]" in detail, detail)
        # and the pass case really passes on the shipped binary
        check_framing(sse_lines(main.port, STREAM), EXPECT10)
    finally:
        m.stop()

@arm("U3-arm-43 gate-fails-on-utf8-mutant")
def _a43():
    m = Srv(env={"M1_SERVER_MUTANT": "split_utf8"})
    try:
        ok("the mutant announces itself at boot", "TEST MUTANT ACTIVE" in m.log(), m.log()[:400])
        detail = None
        try:
            check_framing(sse_lines(m.port, STREAM), EXPECT10)
        except ArmFail as e:
            detail = str(e)
        ok("the framing+text check FAILS on the mutant (it flushes mid-codepoint bytes, so the "
           "split codepoint becomes U+FFFD and the text no longer matches)",
           detail is not None, "the split_utf8 mutant PASSED the check")
        ok("the failure is the text one, not a framing accident",
           "text" in detail.lower(), detail)
        # and the pass case really passes on the shipped binary
        check_framing(sse_lines(main.port, STREAM), EXPECT10)
    finally:
        m.stop()

@arm("U3-arm-44 sampling-applied")
def _a44():
    base = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 10}
    texts = []
    for seed in (7, 8, 9):
        st, _, doc = post_json(main.port, "/v1/chat/completions",
                               dict(base, temperature=0.8, top_p=0.9, top_k=40, seed=seed))
        eq(f"status with seed {seed}", st, 200)
        eq(f"completion_tokens with seed {seed}", doc["usage"]["completion_tokens"], 10)
        texts.append(doc["choices"][0]["message"]["content"])
    # same seed twice => the same chain (the per-request RNG stream is seeded, never shared)
    st, _, again = post_json(main.port, "/v1/chat/completions",
                             dict(base, temperature=0.8, top_p=0.9, top_k=40, seed=7))
    eq("the same seed reproduces the same text", again["choices"][0]["message"]["content"], texts[0])
    ok("different seeds give different text (sampling is APPLIED, not ignored)",
       len(set(texts)) >= 2, texts)
    ok("sampling does not silently reduce to the greedy chain",
       any(t != EXPECT10 for t in texts), texts)

@arm("U3-arm-45 json-escapes")
def _a45():
    # observability: the 404 message for an unknown model echoes the model string verbatim
    tricky = 'A"B\\C\nD\tE\u00e9\u4f60\U0001F600'
    st, _, doc = post_json(main.port, "/v1/chat/completions",
                           {"model": tricky, "messages": [{"role": "user", "content": "hi"}]})
    eq("status", st, 404)
    msg = doc["error"]["message"]
    ok("the escapes were DECODED, not passed through as text", "\\u" not in msg, msg)
    ok("the decoded string round-trips byte for byte", tricky in msg, msg)

@arm("U3-arm-46 json-strictness")
def _a46():
    cases = [
        (b'{"messages":[{"role":"user","content":"a\x01b"}]}', "control byte"),
        (b'{"messages":[{"role":"user","content":"a\xffb"}]}', "UTF-8"),
        (b'{"messages":[{"role":"user","content":"\\ud800"}]}', "surrogate"),
        (b'{"max_tokens":01,"messages":[{"role":"user","content":"hi"}]}', "leading zero"),
        (b"[" * 80 + b"]" * 80, "depth"),
    ]
    for raw, needle in cases:
        st, _, doc = call_json(main.port, "POST", "/v1/chat/completions", raw)
        eq(f"status for {needle}", st, 400)
        ok(f"the refusal names the {needle} problem", needle in doc["error"]["message"],
           doc["error"]["message"])

@arm("U3-arm-47 graceful-shutdown")
def _a47():
    s = Srv(["--fake-delay-ms", "100", "--drain-ms", "5000"])
    done = {}

    def slow():
        try:
            done["r"] = call(s.port, "POST", "/v1/chat/completions",
                             json.dumps(dict(STREAM, max_tokens=200)), timeout=30)
        except Exception as e:
            done["e"] = str(e)

    t = threading.Thread(target=slow)
    t.start()
    time.sleep(0.5)  # the stream is in flight (200 x 100 ms of fake generation)
    t0 = time.time()
    s.p.send_signal(signal.SIGTERM)
    rc = s.p.wait(timeout=30)
    dt = time.time() - t0
    eq("the exit code is 0 (a handled shutdown, not a signal death)", rc, 0)
    ok(f"it stopped accepting and drained at a position boundary in {dt:.2f}s", dt < 5.0, f"{dt:.2f}s")
    ok("the log reports a clean shutdown", "shut down cleanly" in s.log(), s.log()[-400:])
    t.join(30)
    try:
        st, _, _ = call(s.port, "GET", "/health", timeout=3)
        raise ArmFail(f"a connection was still served after shutdown (status {st})")
    except (ConnectionRefusedError, OSError):
        pass

# ---------------------------------------------------------------------------
# verdict
# ---------------------------------------------------------------------------
stop_all()
failed = [r for r in RESULTS if r[1] is not None]
print()
print(f"engine_gate_http.sh: {len(RESULTS) - len(failed)}/{len(RESULTS)} arms passed")
for name, why in failed:
    print(f"  FAILED {name}: {why}")
if failed:
    print("GATE FAIL")
    sys.exit(1)
print("GATE PASS: the wire protocol (JSON, HTTP, SSE, errors, budget, stop strings, usage, the "
      "/debug/tokens seam, concurrency, disconnect) is proven against the fake engine; the "
      "binary links no ggml and the server holds no gpu.lock")
sys.exit(0)
PY
