# Aubier

**A Mistral Small 4 specific inference engine** — a llama.cpp fork whose Mixture-of-Experts execution path is
replaced with one written for this model and this hardware. Generality is not a goal: the loader, MLA attention,
KV cache, sampler and server are inherited from llama.cpp; the expert path is ours.

The name: *aubier* is sapwood — the living layer between bark and heartwood. Apt for an engine whose entire
problem is deciding what lives in which layer.

## Why this exists

Measured on the reference box (RX 9070 16 GB, Ryzen 9 3950X), decoding Mistral Small 4 (119B MoE, ~6B active,
128 experts of 4 per layer, 36 all-MoE blocks, MLA attention) with the stock engine:

| | ms per token |
|---|---|
| GPU kernels | 11.16 |
| copies | 1.10 |
| **GPU idle** | **~19.5** |
| wall | ~31.7 |

Decode is **CPU-bound, not GPU-bound**: the CPU's expert share is serialised against the GPU at ~74 backend
switches per token (`ggml_backend_sched_compute_splits` synchronises at every backend switch), so the card idles
while the CPU grinds. Overlapping the two halves of each layer collapses the step toward
`max(11.16, ~13.5)` ≈ **60-75 tok/s** on a single 16 GB card — which is what this fork is for.

## Design

Per layer: split the MoE by where each expert's weights live, compute the parts **concurrently** on the devices
that hold them, and **sum** the partials — one code path covering 1 GPU (GPU residents / CPU remainder), 2 GPUs
(all experts resident), and N devices (a placement policy over capacity × speed × hotness). The split is
expressed as **one custom op** so the graph and scheduler see a single node per layer; that is what avoids both
the per-switch synchronisation and the fixed-shape padding. See `AUBIER-ARCHITECTURE.md`.

## Status

| milestone | state |
|---|---|
| **M1** — fork builds; flag-OFF path | **green**: byte-identical to stock (2/2 and 3/3 across two harnesses) |
| **M2** — specialised MoE split path | **green**: computes the correct sum to `max|Δ| = 1e-8` against stride-aware ground truth at every layer; byte-identical to stock (`8/8`, KL 0.000000) when the weighted-reduction fusion is disabled on both arms |
| **M3** — concurrency (the point of the fork) | **in progress** |

M2 is gated by `GGML_MOE_WR_DISABLE=1` on both arms: stock's `moe_weighted_reduction` CUDA-graph fusion fires for
stock's contiguous graph but cannot fire for an interleaved split graph, and on a 2-bit model that ordinary FP
difference is amplified by near-tie greedy chaos into full trajectory divergence. **Any fork-vs-stock byte
comparison must disable that fusion on both arms**, or it measures the fusion rather than the change.

## Building

The fork is built with the same configure line as the parent project's HIP build:

```sh
cmake -S . -B build -DGGML_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1201 -DCMAKE_BUILD_TYPE=Release
cmake --build build -j8
```

The full build/toolchain recipes (no-root ROCm via TheRock wheels; a dual CUDA+HIP build for a second NVIDIA
card) live in the parent project's `scripts/`, which is not part of this repository.

## Measuring anything here

Five execution-geometry axes change greedy output on this model *without any code change*: `-ub`, `-ncmoe`,
`-t`, the op-offload flag, and the `moe_weighted_reduction` fusion. Hold all five fixed across any byte
comparison, and prefer reporting top-token agreement / KL / perplexity over byte-identity for any change that
alters accumulation order.
