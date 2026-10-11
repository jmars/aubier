# Aubier — resume after reboot

State as of 2026-10-10 ~20:20, immediately before a reboot that was required to clear a wedged
NVIDIA driver call (the P40's fan temperature probe had been displaced during a card swap, the card
**overheated** and fell off the PCIe bus, and a `test-backend-ops` process is stuck in state R at
~170% CPU, immune to SIGKILL, holding the AMD render node and therefore `scripts/gpu.lock`).

Everything below is on disk. The authoritative record is the agent memory graph
(`fx-agent-memory read handoff-mistral4-rx9070-result` and the per-unit nodes `-p*`, `-rev*`); this
file is the short version of what to check and what to run first.

## 0. Verify the hardware is back before anything else

```sh
nvidia-smi                      # expect: Tesla P40, 24 GB, temps sane; if "no devices" -> probe/thermal again
for d in 0000:29:00.0 0000:00:03.1; do
  p=/sys/bus/pci/devices/$d
  echo "$d cur=$(cat $p/current_link_speed 2>/dev/null) x$(cat $p/current_link_width 2>/dev/null)"
done                            # 9070 must be x16 on the CPU-attached root port (00:03.1)
scripts/ggml_h2d_bench.sh postreboot   # expect ~26 GiB/s; if it is ~0.75 GiB/s the 9070 moved back
```

The 9070 must stay on the **CPU-attached x16** slot. The P40 stays on the chipset x1 chain (fine for
a resident draft or resident experts; only ~1 GB/s of host bandwidth).

## 1. Immediate queue (in order)

| # | what | why | how |
|---|---|---|---|
| 1 | **`-nopo` A/B at the new link** | The reviewer's #1 unowned item: `--no-op-offload` was pinned when H2D was 0.746 GiB/s. Op-offload *is* misses-only expert streaming to the GPU; at 26.6 GiB/s it may now beat the CPU compute it replaces. One A/B settles whether our pin is now **wrong**. | `llama-bench -m models/Mistral-Small-4-119B-2603-UD-IQ2_XXS.gguf -ngl 99 -ncmoe 12 -t 16 -ctk q8_0 -ctv q8_0 -ot ffn_down_exps=CPU -p 512 -n 128 -r 1` **with and without `-nopo 1`**, under `scripts/with_gpu_lock.sh`, load recorded. |
| 2 | **Two-GPU expert placement** (P-S's remainder) | P-S proved `CUDA0` is a legal `-ot` buffer type (`Available buffer types: CPU, CUDA0, ROCm0`) and that per-tensor `-ot` moves weights off the P40 — but its rows were never taken. Full expert residency fits: gate_up 528 MiB×36 = 18.6 GiB + down 328 MiB×36 = 11.97 GiB + 2.0 GiB ≈ 32.5 GiB against 24+16 GiB. | `bash scripts/ps_run.sh c12c c12p c12r c18p c24p c36p c36pr`, after `source scripts/ps_env.sh` (the `builds/hip-ps` binary needs the CUDA runtime on `LD_LIBRARY_PATH` or it silently exposes no P40). |
| 3 | **Speculation at the pin** (P-R's remainder) | P-R validated its equality checker but died before the arms. Open: draft-depth sweep, a generic small draft on the P40, and the **clean** DFlash-on-P40 test (target `--device ROCm0`, draft `--spec-draft-device CUDA0`, verified from the loader log *and* `nvidia-smi` — the previous attempt was load-contended and had 931 MiB of the target's output layer on the P40). | `results/p-pr.md` + `scripts/pr_*.sh`; remember `--spec-type` is honoured only by `llama-server`. |
| 4 | **Specialized kernel** (P-T's remainder) | P-T established that llama.cpp's decode is **already CUDA-graph captured**, so the launch-overhead premise is dead; the remaining kernel inefficiency is intermediate HBM traffic and small-M occupancy. Its Phase-1 design is in `results/p-pt.md`. | read `results/p-pt.md` before writing any kernel. |

## 2. What is settled — do not re-litigate

- **Pin**: `-ngl 99 -ncmoe 12 -t 16 -ctk q8_0 -ctv q8_0 -ot ffn_down_exps=CPU --no-op-offload`, model
  UD-IQ2_XXS, on `builds/hip`. Measured **pp512 179.6 / tg128 31.50**; with `--spec-type ngram-simple`
  (llama-server only) **42.05 copy-class / 31.7 baseline**, output byte-identical.
- **The design to build** is in `docs/TARGET-ARCHITECTURE.md`: per layer, split the MoE by where each
  expert's weights live, compute the parts concurrently, sum. One code path for 1 GPU (GPU hits / CPU
  misses), 2 GPUs (all resident), N devices (a placement policy).
- **Measured negatives that stand**: CPU/GPU *split* overlap is dead (verified in source:
  `ggml-backend.cpp:1843`, `ggml-cpu.cpp:396-399`); the in-tree `--moe-cache-mib` is per-layer-eligible
  and uploads misses rather than CPU-computing them, so **Strata's core trick is inexpressible in it**;
  it is decode-only and non-additive with speculation (an implementation cliff at `max_batch=32`).
- **Measured ceiling that bounds expectations**: the amortizable term is ~11.2 ms of a ~32.4 ms step
  (the expert-weight upload, 306–317 MiB/token), so any drafter caps near **1.5×**, and `ngram-simple`
  is already at **1.32×**.
- **Gates to hold**: bitwise-inert with the flag OFF, output equality with it ON, an in-session control
  per comparison, `-ub` and `-ncmoe` **held fixed** for any byte comparison (greedy output here shifts
  with geometry alone), and a noise floor measured **for the harness in use** (~1.7–1.8% adjacent for
  the CLI harness, ~24–32% for the llama-server harness).

## 3. Traps that cost real time — do not repeat

- `flock` **forks**: killing a task wrapper does not kill the process holding `scripts/gpu.lock`; kill
  by PID and verify `flock -n scripts/gpu.lock true` exits 0 **and** VRAM is idle before starting.
- **Never `pkill -f`** a pattern that appears in your own command line (it killed an orchestrator shell).
- `-nopo 1` is llama-bench only; llama-cli/llama-server want the long boolean `--no-op-offload`, no value.
- `--spec-type` is honoured only by **llama-server**; llama-cli accepts and silently ignores it.
- `--moe-cache-mib` changes greedy output (three independent pairs) and exists only on cli/server.
- `builds/hip-p40` / `hip-ps` silently expose **no** P40 without the CUDA runtime on `LD_LIBRARY_PATH`.
- Units on this host have been **SIGTERM-killed mid-run** repeatedly: write findings to memory as they
  land, not at the end.
