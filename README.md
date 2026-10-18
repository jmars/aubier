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
engine/        the engine: the split store, the rotation, the layer graph, the GGUF loader
patches/       our delta to the vendored ggml (the three M4 custom ops), as a unified diff + the sources
scripts/       build and gate: engine_build_m1.sh, engine_build.sh, build_hip.sh, engine_gate1.sh, env.sh
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

## Gates

Two gates, both exiting on a decision rather than on the tail of stdout:

* `scripts/engine_gate1.sh` — the CPU-only configuration must be **byte-identical** to a pinned reference on a
  36-layer greedy chain, and the split's paired falsifier must still decide. It **takes the GPU lock itself**: run
  it bare, never inside an outer `flock`.
* `scripts/engine_gate0.sh` — the split primitive's own rig.

## Status

The engine is a research vehicle for one model on one class of hardware; it is not a general GGUF runner. The
records that produced it are kept outside this repository.
