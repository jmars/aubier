# Aubier — target architecture: a resource-agnostic expert pool

Project name: **Aubier**. Status: **adopted direction, not yet implemented.** Written 2026-10-10. Claims are marked
**MEASURED** (with the artefact), **INTERPRETATION**, or **PROJECTION**. Anything unmarked is a
statement of intent.

---

## 1. The product constraint

The shipping target is a machine with **one 16 GB GPU and system RAM**. The reference hardware is the
one in this folder (RX 9070 16 GB, gfx1201, Ryzen 9 3950X, 125 GB RAM) — but note that the box also
carries a **Tesla P40 (24 GB, Pascal)**, which is *not* part of the product. Anything that only works
because both cards are present is a **local tier**, not the answer. The design below is required to
work on 16 GB alone; the second card is an extension of the same mechanism.

## 2. The design

**Per layer, split the MoE by where each expert's weights live, compute the parts concurrently on the
devices that hold them, and sum the partial outputs.**

One code path serves every configuration:

| configuration | resident (hits) | remainder (misses) | notes |
|---|---|---|---|
| **1 GPU, 16 GB — the product** | GPU | **CPU** | Strata's design; the CPU and GPU work *at the same time* |
| **2 GPUs, 16 + 24 GB** | GPU A | **GPU B** | on this box the ~30 GB expert set fits across both, so the CPU is idle and the slow x1 link only carries hidden state (~16 KB/layer × 36 ≈ 0.6 MB/token) |
| **2 GPUs + CPU (general)** | hottest → fastest card | next tier → second card | remainder → CPU; for models larger than the combined VRAM |

The placement decision is a **policy over resources** (capacity × speed × hotness), not a special case
per configuration. That is what makes the runner scale instead of being tuned for one machine.

## 3. What Strata does that we do not (code-evidenced)

Read from the clone at `/tmp/strata` (github.com/Niko1221/Strata):

- **Per-EXPERT VRAM residency.** `include/strata/core/expert_cache.hpp`: `(layer, expert) -> slot or -1`,
  an LRU with a static plan, ~4,105 slots, measured hit rate **0.6447** (ten-fold leave-one-out). The
  same file states that `h_layer` held-out is only **0.0456**, so the split *cannot* be per-layer
  grouping — it must be by expert.
- **Per-layer hit/miss split into two buffers.** `src/core/session.cpp:526-536,637-640`: `y_miss` is
  allocated with `cudaHostAlloc(cudaHostAllocMapped | cudaHostAllocPortable)` (pinned, device-mapped);
  the comment notes the first layer has no previous experts so "hits are empty" and `y_miss` starts at
  zero. `moe_combine` (`src/core/layer.cpp:479`) **sums** the two.
- **The two halves run concurrently.** `docs/HOW_IT_WORKS.md:17,27`: *"The CPU computes the experts that
  are not on the GPU **in place**, **at the same time as the graphics card, so neither waits for the
  other**."*
- **A DRAM-tuned CPU expert pool.** `include/strata/kernels/cpu/pool.hpp` — "keep every physical core
  reading expert bytes for the whole layer", with a specified claim/done protocol. It measures the CPU
  expert work at **663.6 MB of expert bytes per token at ~40 GB/s = 16.2 ms of a ~53 ms token**, and
  states honestly that the CPU work is **96% exposed** because the residual chain is strictly serial.
- **They measure the two-resource claim themselves.** `src/core/concurrent_main.cpp`: CPU expert kernel
  **42.55 GB/s of DRAM** + PCIe stream **28.9 GB/s**, run together, targeting **552.7 MB of misses in
  7.74 ms** ("100 tps inside reach") — with their own caveat that a pinned H2D copy also reads host
  DRAM, so the two resources may contend rather than being independent.

**The gap in llama.cpp (our current base):** it places **one `MUL_MAT_ID` per MoE op on one backend**.
Its in-tree cache (`src/llama.cpp/src/llama-moe-cache.cpp`, flag `--moe-cache-mib`) is a per-layer LRU
that moves the layer's **entire** compute onto the cache's backend and **uploads** misses on that same
stream — so a miss costs upload *then* GPU compute, serially, with the CPU idle for that layer. There is
no mechanism anywhere in it for a per-layer hit/miss split with the halves computed on two devices.

**This explains our measured negatives** (all MEASURED, `results/p-{pi,pm,po,pn,pq}.md`): the cache is
non-additive with `-ncmoe` (they are alternative placements of the same work), the cache is decode-only
(zero `ubatch > 8` lines in 10/10 rows), `-ncmoe` alternates CPU/GPU with a synchronisation at every
backend switch (**44 splits/token**), and the GPU either idles (~50%) or saturates (~90%) depending on
which side owns the layer. An earlier "CPU/GPU overlap is dead" verdict was an **argument**, not a
measurement (`handoff-mistral4-rx9070-pn`), and was withdrawn as such.

## 4. Dependency order

1. **Per-expert views + N partial `MUL_MAT_ID`s summed per layer.** Today: one op, one backend, one
   tensor. This is the primitive; nothing else works without it.
2. **Concurrent execution across backends.** `ggml_backend_sched_compute_splits`
   (`ggml-backend.cpp:1859-1888`) walks splits in order and synchronises at every backend switch. This
   is the measured cause of the 44 splits/token and of the alternating idle. INTERPRETATION: it will
   need per-backend streams with event-gated joins, or an async CPU path.
3. **Per-device expert stores** — VRAM A, VRAM B, RAM — with an eviction/hotness rule.
4. **The placement policy** over the resource set (sized from hotness; Strata's equivalent is 4,105
   slots at a 64% hit rate).

Build 1 → 2 → 3 → 4. Do not build the policy before the split works.

## 5. Assets in hand

- **Dual CUDA+HIP build**: `builds/hip-p40` enumerates `CUDA0: Tesla P40 (24438 MiB)` and
  `ROCm0: AMD Radeon RX 9070 (16304 MiB)` in one binary (required `GGML_BACKEND_DL=ON`; with DL off both
  backends export the same `ggml_backend_cuda_reg` symbol and static registration keeps one).
  `--spec-draft-device CUDA0` is proven to route a draft model onto the P40 with byte-identical output.
  Recipe: `scripts/p40_build_dual.sh`, `scripts/p40_setup_cuda_toolkit.sh`. MEASURED.
- **Two proven engine patches**: P-D's F32-src1 `MUL_MAT_ID` implementation (the expert cache depends on
  it) and U2.4's `t_layer_inp` publication (the DFlash speculator depends on it); composed without
  conflict in `builds/hip-combo` (`scripts/patch_combo.sh`).
- **Measured costs to size the policy**: CPU expert path ~42 GB/s DRAM equivalent (Strata's figure);
  PCIe **26.6 GiB/s** H2D post-hardware-fix (was 0.746 GiB/s on the old x1 link); **976 MiB** of expert
  bytes per token for this model (36 layers × 4 of 128 experts × ~6.5 MB); the GPU runs ~90% busy with
  the in-tree cache at `-ncmoe 30`.
- **llama.cpp's CPU MoE kernels are 548 GFLOP/s against ik_llama.cpp's 988** (P-C, op-level) — relevant
  if the CPU half of the split proves to be the bottleneck on the 16 GB path.

## 6. Baselines to beat (all MEASURED on this box)

| configuration | prefill | decode |
|---|---|---|
| pin: `-ngl 99 -ncmoe 12 -t 16 -ctk q8_0 -ctv q8_0 -ot ffn_down_exps=CPU --no-op-offload`, UD-IQ2_XXS | **179.6** pp512 / 86.7 pp48 | **31.50** tg128 |
| pin + `--spec-type ngram-simple` (llama-server) | — | **42.05** copy-class / 31.7 baseline |
| Strata, published, comparable hardware (RX 9070 XT 16 GB + 3900X + 47 GB) | ~1,160 pp (32K prompts) | **44–60** |

Target: **~55 tok/s** on agent-shaped work (long prompts, repeated/edited code), which is the user's
stated bar for auditable agent experiments.

## 7. Staging and gates

1. **2-way split** (GPU hits / CPU misses, concurrent, summed) — the product path *and* the shared core.
2. **N stores** (VRAM A / VRAM B / RAM) — the two-GPU case falls out of the same code.
3. **Placement policy** — hotness × capacity × speed.

Every stage is gated the way this project now runs: **bitwise-inert when the flag is OFF**, output
equality when ON, an **in-session control** per comparison, and a **noise floor measured for the harness
in use** — the CLI harness measured 1.7–1.8% p-p adjacent after the hardware fix, while the llama-server
harness measured **24–32%**, and greedy output on this model is **not invariant to execution geometry**
(the no-spec control itself changes text between `-ub 16` and `-ub 32`), so `-ub` and `-ncmoe` must be
held fixed across any byte comparison.

## 8. Open questions (units in flight)

- **P-S** — does `-ot` accept another backend's device (`CUDA0`), and what does the P40 actually deliver
  on these quantized expert kernels? This sizes its share. PROJECTION: Pascal, no tensor cores, 347 GB/s
  against the 9070's ~600 GB/s, so expect roughly a third to a half of the per-byte throughput — a
  plausible 1.5–1.8× over the 9070 alone, not 2.5×.
- **rev-stream** — is the per-layer hit/miss split expressible in llama.cpp with a bounded patch, or does
  the per-switch synchronisation make it a design change?

## 9. Scope, stated honestly

This is the project's **largest** piece of engine work: a new MoE execution layer, not a flag or a small
patch. It is worth attempting only with the two open questions settled. But it is the same mechanism the
16 GB product path needs and the one the two-GPU tier needs — build it once, ship it twice — and the
prerequisites (dual-backend build, measured costs, Strata's design as a blueprint, a measured baseline)
are already in place.

---

## 10. Post-llama.cpp goal: a Mistral-Small-4-specific engine

Status: **FUTURE — do not start until the llama.cpp work in §4 and §7 has run its course.** Recorded
now (user direction, 2026-10-10) so the things the llama.cpp phase must *establish* are collected as we
go, and so nobody re-derives this from scratch later.

### Why an engine at all

Three properties of llama.cpp are structural, not tuning gaps:

1. **The scheduler forbids the split.** One backend per op, with a synchronisation at every backend
   switch (`ggml-backend.cpp:1859-1888`; `:1843` for the copy path; both verified in source). The
   per-layer hit/miss concurrency of §2/§3 is therefore not expressible — the in-tree cache is
   per-layer-eligible (`llama-moe-cache.cpp:236`) and **uploads** misses rather than CPU-computing them
   (`:515-583`).
2. **Placement is static and per-tensor.** `-ot`/`-ncmoe` move whole tensors or whole layers; there is no
   per-expert residency with a hotness policy, and no way to consume part of an uploaded tensor early.
3. **The weight layout is generic.** It is not chosen for this model's shapes (H=4096, expert FF=2048,
   4-of-128 routing) nor for the two ISAs in play (RDNA4 and sm_61/dp4a).

### What it would implement, and the measured evidence that each item is real

| mechanism | evidence already in hand |
|---|---|
| Concurrent per-layer hit/miss split (GPU computes residents, CPU computes the rest, summed) | Strata's design, code-evidenced; **inexpressible in llama.cpp**; on 16 GB this is the main lever, bounded below by `max(CPU, GPU)` instead of their sum |
| Per-expert residency across N stores, with a hotness/eviction policy | Residency removes the **11.2 ms of a 32.4 ms step** that P-Q measured as pure weight movement; `-ot` can already target `CUDA0`; the ~32.5 GiB expert set fits across 24+16 GiB |
| A weight layout matched to the kernels and both ISAs | llama.cpp's CPU MoE path measures **548 GFLOP/s against ik_llama.cpp's 988** at the op level (P-C) — a 1.8x gap in exactly this choice |
| A fused speculative verify cycle | open-jet's own `-d 0` control shows the *kernel* bought **+6%**; their 1.53-1.93x came from making the **verify** nearly free. In llama.cpp the verify is a separate graph pass with its own overhead |
| Prefill that streams experts while attention runs | We measure **235 tok/s** at pp512; Strata publishes **~1,160** on 32K prompts with up to 8,192-token chunks — ~5x, and architectural |

### Entry criteria — what the llama.cpp phase must first establish

Only start the engine once these are answered, because each one changes its design:

1. **The split's real gain on 16 GB** — the concurrency win, measured, not projected (the split primitive
   in §4 item 1 is the probe).
2. **Whether llama.cpp's scheduler can be made to overlap two backends at all** (rev-stream: `HOLDS` that
   it cannot today; the open question is whether a bounded patch changes that).
3. **The residency numbers** — P-U's two-GPU measurement, which says how much expert work leaves the CPU
   and whether cross-vendor crossing latency is ~100 µs (80-120 tok/s) or ~1 ms (dead).
4. **The CPU-kernel gap** — reproducible 548 vs 988 GFLOP/s on our workload, so the layout work has a
   target.
5. **The prefill gap** — a real measurement of what our batched path does at 8K-32K prompts versus
   Strata's published figure.
6. **The long-context decode curve** — never measured on this project (MLA + KV quant at 32K/128K/256K),
   and a bespoke engine's KV path should be designed against it.

### Expected payoff, and what is explicitly NOT worth building

Payoff is "the last bit" **on this box but not on the product path**: with two GPUs, residency alone
recovers most of the 11.2 ms upload term inside llama.cpp, so an engine's *irreplaceable* gains there are
the split and the layout. On a **single 16 GB card** — the shipping target — the concurrency split and the
layout are precisely what llama.cpp cannot give, so that is where the engine earns its cost.

**Do not build, in an engine or anywhere else:** (a) MoE kernel fusion — measured single-digit % (P-T:
the fusable fraction is ~0.2-0.5 ms of ~17.1 ms of kernel time, and decode is already CUDA-graph
captured); (b) a whole-model persistent megakernel — open-jet's own control shows it buys +6% on decode;
adopt its *verify-cycle* idea instead.
