# aubier

A **single-stream inference engine for Mistral Small 4** (119B, 36 all-MoE layers, 128 routed experts, top-4).

The model does not fit in the VRAM of a consumer card, so the engine treats *where an expert's weights live* as a
first-class, policy-driven decision and keeps the rest of the model resident:

* **A split expert store.** Each layer's experts are partitioned across a VRAM store and a host store. The host side
  is a **zero-copy alias** over the mmap'd weights — no packed copy — so the memory a consumer card needs is the
  resident set and nothing else.
* **A resident set that moves.** A 2Q admission-controlled **rotation** follows the routing, because the hot set
  rotates within a generation and a static set cannot follow it. On the worst held-out prompt it serves **0.68** of
  routed slots at 48 slots/layer; the admission threshold and the VRAM budget are the two knobs.
* **Fused HIP kernels behind one custom-op hatch.** Three ops are appended to ggml (`GGML_OP_M4_WKB`, `_KQKV`,
  `_PHASE`), each **capture-safe** (a plain launch, no host sync), each with a **FATAL CPU handler** so the CPU-only
  configuration — the numerics oracle — can never take a fused path, and each behind a tri-state flag whose OFF path
  is byte-identical to the pre-kernel binary. Attention is correct from a 256-key decode window to a **262 144-key**
  context (tiled online softmax + split-K).
* **A quality instrument, because fused kernels cannot preserve the token line.** The greedy chain is chaotic — a
  one-ulp change in one kernel becomes O(1) in the logits by position 100 — so every numerics-changing default ships
  against a **paired per-position NLL** comparison with a confidence interval and an explicit *INSUFFICIENT* verdict,
  not against output equality.

Measured on a single 16 GB RX 9070 at 2-bit: **37 decode tok/s** on the worst held-out prompt (llama.cpp's
`tg128` on the same box: 31.7-31.8). On 1-4 × 32 GB cards the model is resident and batching becomes the lever.

## Layout

```
engine/        the engine: the split store, the rotation, the layer graph, the GGUF loader; the
               tokenizer, the chat renderer, and the server (m1_session.h, m1_real.cpp, m1_server.cpp)
patches/       our delta to the vendored ggml (the three M4 custom ops), as a unified diff + the sources
scripts/       build and gate: engine_build_m1.sh, engine_build_server.sh, engine_build_http.sh,
               build_hip.sh, engine_gate{0,1,tok,chat,http,server}.sh, env.sh
tools/         fixture builders (grouped-expert GGUF fixtures for the loader)
```

## Building

The ggml tree the engine builds against is **vendored and already patched** — `vendor/llama.cpp/` is llama.cpp at
`50e3e3e4` with our three M4 ops applied. It is pruned to the four libraries the engine links, so `vendor/README.md`
is where the prune, the reason the build names library targets instead of `all`, and the recipe for restoring a full
upstream tree are recorded.

Two prerequisites are machine-local, so they are not in this repository:

* a **ROCm toolchain as a Python venv** at `venvs/rocm` (AMD's ROCm pip wheels — `rocm`, `rocm-sdk-devel`,
  `rocm-sdk-libraries-<arch>-all`), used through `venvs/rocm/bin/python`; `build_hip.sh` asks `rocm-sdk path`
  for the prefix rather than hardcoding one;
* the model weights (a 2-bit GGUF of Mistral Small 4).

```sh
sh scripts/build_hip.sh          # the four ggml libraries -> builds/hip   (VENV/SRC/BUILD override)
sh scripts/engine_build_m1.sh    # the engine -> builds/engine/m1
```

`builds/hip` is the engine's `GGML_LIB`; `builds/hip-fork` is a second build used only for the reference binaries the
gates compare against.

## The server

`builds/engine/m1_server` is **a separate binary**: an OpenAI-compatible HTTP front end over the same engine the
single-shot `builds/engine/m1` drives. It serves Mistral Small 4 and nothing else — the model id is checked against
the one served name, and any other name is a 404 rather than being routed — and it runs **one request at a time**. A
second request waits on the single-flight engine and, if it waits longer than `--queue-wait-ms` (30 s), gets 503
with `Retry-After: 1`. Beyond `--max-conns` (32) concurrent connections the answer is 503 too.

The routes are `POST /v1/chat/completions` (SSE when `stream` is true), `POST /v1/completions` (a string prompt; an
array prompt is refused), `GET /v1/models` and `GET /health`, plus one route that is deliberately not OpenAI's:
`POST /debug/tokens` takes `{ids, max_tokens}` and drives the engine from token ids, returning the ids it produced.
That seam is what makes the engine's id-parity check independent of the tokenizer — a tokenizer bug cannot mask an
engine bug.

`model`, `messages` (system/user/assistant, text content), `prompt`, `max_tokens` or `max_completion_tokens`,
`temperature`, `top_p`, `top_k` (an extension), `stop` (up to four strings), `stream`,
`stream_options.include_usage` and `seed` are accepted. Sixteen more are **recognised and refused by name with 422**
whenever they carry a non-default value — `n`, `tools`/`functions`, `logprobs`, `logit_bias`, `response_format`, the
penalty family, `echo`, `best_of`, `suffix` and so on — because a parameter silently ignored yields an answer that
looks like the one the client asked for. Sent at their defaults, those same parameters are accepted, which is what
lets ordinary clients talk to it.

**The product path is the default configuration, not a mode.** `--arm-store`, `--gpu-core`, `--gpu-store` and the 2Q
rotation run exactly as they do in the single-shot binary, because the server drives the same session object
(`M1Session::open`/`step`) and fills the same argument struct: there is one decode loop, not two. `--kv-max` is a
**numeric** engine parameter, not only a capacity — the attention softmax reduces over the padded KV width — so a
client's usable context is exactly that width, and prompt + `max_tokens` are clamped to it (with
`finish_reason: "length"`). The pinned template's default system message renders to 2374 bytes, so even a one-line
chat request is ~562 prompt ids and `--kv-max` has to clear roughly 570 before the chat route is usable at all; the
completions and `/debug/tokens` routes are unaffected.

The server **takes `scripts/gpu.lock` itself at boot and holds it for its lifetime**, exiting 69 if it cannot get it
within `--lock-wait-ms`. The mutual exclusion is structural rather than a convention: while a server is up, no
measurement row can take the lock. That is the trade a long-lived GPU tenant makes.

Each request starts from the boot state. The rotation's membership and 2Q policy are replayed back to boot between
requests (`--reset=exact`, the default); the replay is timed and printed per request, and costs up to ~0.7 s under
load. The KV needs no reset: the attention mask is built for each request's own positions, so a cell a previous
request wrote is masked rather than read. `--reset=none` is the control that shows the replay is load-bearing —
with it, a second request no longer matches a fresh boot.

```sh
sh scripts/engine_build_server.sh    # the server -> builds/engine/m1_server
builds/engine/m1_server --engine real --port 8000 --kv-max 1024 \
    --arm-store --gpu-core --gpu-store --n-res 48 --threads 16

curl -s 127.0.0.1:8000/v1/chat/completions \
    -d '{"messages":[{"role":"user","content":"Hi"}]}'
```

Rotation is **not** in that invocation on purpose: `--rotate` needs a resident set to rotate over, so it is refused
(`FATAL: --rotate is FLAT-FILE and SET-backed only`) unless `--resident-set <hotset.json>` names the per-layer expert
set that seeds it. Without one the server runs the identity set — the store split and the GPU core still apply, and
it is a valid configuration, just not the adaptive one.

`--engine real` is what selects the real engine; the default is the fake test engine, so a binary built without the
engine can never quietly stand in for one that has it. `scripts/engine_build_http.sh` builds that fake variant as
`builds/engine/m1_server_fake`, a CPU-only binary that links no ggml and takes no lock.

## Gates

The gates exit on a decision rather than on the tail of stdout:

* `scripts/engine_gate1.sh` — the CPU-only configuration must be **byte-identical** to a pinned reference on a
  36-layer greedy chain, and the split's paired falsifier must still decide. It **takes the GPU lock itself**: run
  it bare, never inside an outer `flock`.
* `scripts/engine_gate0.sh` — the split primitive's own rig.

Four more cover the server work. Three need no GPU and take no lock: `engine_gate_tok.sh` (the tokenizer against the
pinned reference's own, over a corpus), `engine_gate_chat.sh` (the chat render against llama.cpp's Jinja on the
GGUF's template) and `engine_gate_http.sh` (the whole wire protocol against the fake engine — status shapes, SSE
framing and UTF-8 boundaries, stop strings across tokens, the budget, and the truncation fuzz). The fourth,
`engine_gate_server.sh`, needs the card and **takes the lock itself**: the product server must produce token ids
**byte-identical** to the single-shot binary's chain at the same `--kv-max` and flags (`cmp` on its `req-*.tokens`),
the identity must survive a later request (the reset) and a client that disappears mid-stream, and the boot lock
must be held, refuse with 69, and be released on shutdown.

## Status

The engine is a research vehicle for one model on one class of hardware; it is not a general GGUF runner. The
records that produced it are kept outside this repository.
