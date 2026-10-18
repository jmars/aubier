// moe_split.cpp — M0: the split primitive, standalone.
//
// ONE layer, ONE token batch, ONE MoE block, computed THREE ways, plus the ggml oracle.
// Mistral Small 4 ONLY, hardcoded, no generality (user directive 2026-10-11).
//
//   (a) GPU-only : every routed expert computed on the VRAM store
//   (b) CPU-only : every routed expert computed on the RAM store
//   (c) SPLIT    : resident experts on VRAM, misses on RAM, CONCURRENTLY, on our own
//                  streams, no cross-store sync until the sum
//
// The falsifier (engine-plan.md §2): (c) ~= max(a, b), NOT a + b.  A sum-like (c) kills
// the engine premise and is a complete answer.
//
// EXECUTION MODEL (stated, because the brief forbids a ggml graph *as the engine*):
//   The engine owns the loop, the store list, the split and the combine.  Compute kernels
//   are ggml's — invoked through a minimal per-store cgraph because that is ggml's only
//   public compute entry (there is no raw-kernel API).  No ggml_backend_sched, no model
//   graph, no per-backend-switch sync.  Concurrency in (c): enqueue the GPU store's groups
//   async on the HIP stream, then run the CPU store's groups blocking on the host thread
//   (overlapping the GPU), then one ggml_backend_synchronize, then sum on the host.
//
// SEAM 1 (store list): a Store has fill()/enqueue()/drain(); fill() is a REAL virtual step
//   with a no-op implementation — a streamed store (M4) drops in as a policy variant.
//   Kernel selection is per store (per (expert range x type x location)).
// SEAM 2 (drafter): nt is a parameter of compute_layer; the layer input activations are
//   published via layer_input() (the DFlash speculator hooks here).
// SEAM 3 (loader/type axis): weights are ordered lists of (expert_lo, expert_hi, type,
//   bytes) SEGMENTS, populated today from ggml's loader (one segment per tensor).  Exact
//   fallbacks (taillis/layout.md §2 rule 2): key absent => today's file.
//
// CORRECTNESS: the oracle is ggml's own MUL_MAT_ID — the exact llama.cpp MoE op sequence
//   (mul_mat_id -> swiglu_split -> mul_mat_id with GGML_PREC_F32 src1 for MISTRAL4 ->
//   weighted sum), run on the CPU backend (reference dequant) and on the HIP backend
//   (llama.cpp's own GPU path).  Arms and oracle run different dequant kernels over
//   different types, so none need be bitwise; errors are reported per (tensor, type).

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"   // HIP backend (ggml-cuda API under ROCm)
#include "gguf.h"

// M1/U1 extraction: the GGUF reader and the FULL per-layer tensor surface live in
// engine/gguf_model.h; the store machinery, the store list and the segment/range derivation
// live in engine/store.h (which includes gguf_model.h).  M0's own code below is unchanged.
#include "store.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <chrono>
#include <map>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

// ---------------------------------------------------------------------------
// Shared with the M1 loop (M1/U1 extraction): the constants H_DIM / FF_DIM / N_EXPERT /
// N_USED / N_LAYERS and the FATAL/CHECK macros now live in engine/gguf_model.h, the GGUF
// reader and the store machinery in engine/gguf_model.h + engine/store.h.  M0's behaviour
// is unchanged: it just no longer owns those definitions.
// ---------------------------------------------------------------------------
static double now_s() {
    return std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count();
}

// ---------------------------------------------------------------------------
// Oracle — ggml's own MUL_MAT_ID, the exact llama.cpp MoE op sequence
// ---------------------------------------------------------------------------
struct Oracle {
    ggml_backend_t backend = nullptr;
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_tensor * as_gu = nullptr, * as_d = nullptr;
    ggml_tensor * ids = nullptr, * w = nullptr, * x = nullptr;
    ggml_tensor * gu = nullptr, * d = nullptr, * h = nullptr, * y = nullptr;
    ggml_cgraph * gf = nullptr;

    ~Oracle() {
        // NOTE: the backend is owned by main and SHARED — not freed here.
        if (buf) ggml_backend_buffer_free(buf);
        if (ctx) ggml_free(ctx);
    }

    void build(const LogicalTensor & gu_lt, const LogicalTensor & d_lt,
               ggml_backend_t backend_, int nt) {
        backend = backend_;
        ggml_init_params p = { 8*1024*1024, nullptr, true };
        ctx = ggml_init(p);
        CHECK(ctx, "ggml_init failed");

        ggml_type gu_type = gu_lt.segs[0].type;
        ggml_type d_type  = d_lt.segs[0].type;
        size_t gu_row = ggml_row_size(gu_type, H_DIM);
        size_t d_row  = ggml_row_size(d_type,  FF_DIM);
        as_gu = ggml_new_tensor_3d(ctx, gu_type, H_DIM, 2*FF_DIM, N_EXPERT);
        as_d  = ggml_new_tensor_3d(ctx, d_type,  FF_DIM, H_DIM,   N_EXPERT);

        ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, N_USED, nt);
        w   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, N_USED, nt);
        x   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H_DIM, nt);

        ggml_tensor * xb  = ggml_reshape_3d(ctx, x, H_DIM, 1, nt);        // b: [H, 1, nt]
        gu = ggml_mul_mat_id(ctx, as_gu, xb, ids);                        // [2*FF, N_USED, nt]
        ggml_tensor * gate = ggml_view_3d(ctx, gu, FF_DIM, N_USED, nt,
                                          gu->nb[1], gu->nb[2], 0);
        ggml_tensor * up   = ggml_view_3d(ctx, gu, FF_DIM, N_USED, nt,
                                          gu->nb[1], gu->nb[2], FF_DIM * gu->nb[0]);
        h = ggml_swiglu_split(ctx, gate, up);                             // [FF, N_USED, nt]
        d = ggml_mul_mat_id(ctx, as_d, h, ids);                           // [H, N_USED, nt]
        CHECK(ggml_prec_set_src(d, GGML_PREC_F32, 1), "prec_set_src failed");
        ggml_tensor * dm = ggml_mul(ctx, d, w);                           // [H, N_USED, nt]

        ggml_tensor * acc = ggml_view_2d(ctx, dm, H_DIM, nt, dm->nb[2], 0);
        for (int e = 1; e < N_USED; ++e) {
            ggml_tensor * col = ggml_view_2d(ctx, dm, H_DIM, nt, dm->nb[2], e * dm->nb[1]);
            acc = ggml_add(ctx, acc, col);
        }
        y = acc;
        gf = ggml_new_graph_custom(ctx, 64, false);

        buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        CHECK(buf, "oracle backend alloc failed");
        // weights must be set AFTER the backend buffers are allocated
        ggml_backend_tensor_set(as_gu, gu_lt.segs[0].bytes, 0,
                                gu_row * (size_t)(2*FF_DIM) * N_EXPERT);
        ggml_backend_tensor_set(as_d, d_lt.segs[0].bytes, 0,
                                d_row * (size_t) H_DIM * N_EXPERT);
        ggml_build_forward_expand(gf, y);
    }

    void run(const int32_t * ids4, const float * w4, const float * x_h, int nt,
             float * y_out, float * gu_out, float * d_out, float * h_out) {
        ggml_backend_tensor_set(ids, ids4, 0, (size_t) N_USED*nt*sizeof(int32_t));
        ggml_backend_tensor_set(w,   w4,   0, (size_t) N_USED*nt*sizeof(float));
        ggml_backend_tensor_set(x,   x_h,  0, (size_t) H_DIM*nt*sizeof(float));
        ggml_backend_graph_compute(backend, gf);
        ggml_backend_tensor_get(y,  y_out,  0, (size_t) H_DIM*nt*sizeof(float));
        if (gu_out) ggml_backend_tensor_get(gu, gu_out, 0, (size_t)(2*FF_DIM)*N_USED*nt*sizeof(float));
        if (d_out)  ggml_backend_tensor_get(d,  d_out,  0, (size_t) H_DIM*N_USED*nt*sizeof(float));
        if (h_out)  ggml_backend_tensor_get(h,  h_out,  0, (size_t) FF_DIM*N_USED*nt*sizeof(float));
    }
};

// ---------------------------------------------------------------------------
// Router (input generation, NOT under test): logits = gate_inp^T x, softmax, top-4
// ---------------------------------------------------------------------------
static void gen_ids_and_weights(const float * gate_inp, const float * x, int nt,
                                std::vector<int32_t> & ids, std::vector<float> & w) {
    ids.assign((size_t) N_USED * nt, 0);
    w.assign((size_t) N_USED * nt, 0.0f);
    for (int t = 0; t < nt; ++t) {
        float logits[N_EXPERT];
        for (int k = 0; k < N_EXPERT; ++k) {
            double s = 0.0;
            const float * g = gate_inp + k * H_DIM;   // gate_inp is [H, N_EXPERT]
            for (int64_t j = 0; j < H_DIM; ++j) s += (double) g[j] * x[j + H_DIM*t];
            logits[k] = (float) s;
        }
        float mx = *std::max_element(logits, logits + N_EXPERT);
        double z = 0.0; float p[N_EXPERT];
        for (int k = 0; k < N_EXPERT; ++k) { p[k] = std::exp(logits[k] - mx); z += p[k]; }
        for (int k = 0; k < N_EXPERT; ++k) p[k] /= (float) z;
        int idx[N_EXPERT];
        std::iota(idx, idx + N_EXPERT, 0);
        std::partial_sort(idx, idx + N_USED, idx + N_EXPERT,
                          [&](int a, int b) { return p[a] > p[b]; });
        for (int e = 0; e < N_USED; ++e) {
            ids[e + N_USED*t] = idx[e];
            w[e + N_USED*t]   = p[idx[e]];
        }
    }
}

// ---------------------------------------------------------------------------
// small numerical helpers
// ---------------------------------------------------------------------------
static float max_abs(const float * a, const float * b, size_t n) {
    float m = 0.0f;
    for (size_t i = 0; i < n; ++i) m = std::max(m, std::fabs(a[i] - b[i]));
    return m;
}
static double rms(const float * a, const float * b, size_t n) {
    double s = 0.0;
    for (size_t i = 0; i < n; ++i) { double d = a[i] - b[i]; s += d*d; }
    return std::sqrt(s / n);
}
static float max_abs_val(const float * a, size_t n) {
    float m = 0.0f;
    for (size_t i = 0; i < n; ++i) m = std::max(m, std::fabs(a[i]));
    return m;
}

// The derived tolerance bound (Correction 1).  Both arms and the oracle read IDENTICAL
// weight bytes, so the per-weight dequant can differ only by scale/partial-sum precision.
// With F32 accumulation everywhere (MISTRAL4's down matmul forces GGML_PREC_F32 src1; the
// IQ vec-dot kernels accumulate across blocks in F32), the cross-kernel difference between
// two F32 reorderings of a K-term dot is bounded by
//      |Δy_i| <= 2 * γ_K * S_i ,   S_i = Σ_j |w_ij * act_j| ,   γ_K = K * 2^-24
// (each of the two paths carries <= γ_K·S_i vs the exact sum, so their distance <= 2γ_K·S_i).
// If the measured max|d| exceeds this, the difference is NOT reorder — it is activation
// quantization divergence (the q8_1 block scale, the ds4-class surface): report that.
// S_i is computed exactly by materialising the routed expert's weights to F32 via ggml_cpy.
//   wbytes  : full quantized tensor bytes [ne0, ne1, 128] (gguf layout, expert axis ne[2])
//   act     : [ne0] activation for this expert (x for gate_up, swiglu output h for down)
//   returns max_i S_i over output rows.
static double abs_dot_max(const uint8_t * wbytes, ggml_type type, int64_t ne0, int64_t ne1,
                          int expert, const float * act, ggml_backend_t cpu) {
    ggml_init_params p = { 1u << 20, nullptr, true };
    ggml_context * ctx = ggml_init(p);
    ggml_tensor * wq = ggml_new_tensor_2d(ctx, type, ne0, ne1);
    ggml_tensor * wf = ggml_cpy(ctx, wq, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne0, ne1));
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 4, false);
    ggml_build_forward_expand(gf, wf);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, cpu);
    if (!buf) { ggml_free(ctx); return -1.0; }  // cpy unsupported -> caller reports formula only
    size_t row = ggml_row_size(type, ne0);
    size_t stride = row * (size_t) ne1;
    ggml_backend_tensor_set(wq, wbytes + (size_t) expert * stride, 0, stride);
    ggml_backend_graph_compute(cpu, gf);
    std::vector<float> wf_h((size_t) ne0 * ne1);
    ggml_backend_tensor_get(wf, wf_h.data(), 0, (size_t) ne0 * ne1 * sizeof(float));
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    double s_max = 0.0;
    for (int64_t i = 0; i < ne1; ++i) {
        double s = 0.0;
        for (int64_t j = 0; j < ne0; ++j) s += std::fabs((double) wf_h[j + ne0*i] * act[j]);
        s_max = std::max(s_max, s);
    }
    return s_max;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char ** argv) {
    std::string model = "models/Mistral-Small-4-119B-2603-UD-IQ2_XXS.gguf";
    int layer = 0, nt = 1, n_res = 64, iters = 30, warmup = 3, cpu_threads = 16;
    uint32_t seed = 1234;
    bool list_layers = false;
    bool serialize_halves = std::getenv("MOE_SPLIT_SERIALIZE") != nullptr;
    std::vector<int> extra_cuts;      // SEAM 1/3: extra store boundaries (see --store-cuts)
    bool empty_store = false;         // SEAM 1: append a zero-expert entry to the store list
    // THE FALSIFIER'S OWN FALSIFIER: force the SERIALISED implementation into the paired
    // loop's CONCURRENT arm too, so the two arms do the same work and their ratio is ~1.00
    // by construction — it MUST trip the 0.95 line.  The control run is produced HERE, by
    // the binary; the gate script only asks for it and reads back what it got.
    bool serialize_both = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto need = [&](const char * opt) -> std::string {
            if (i + 1 >= argc) FATAL("%s needs a value", opt);
            return argv[++i];
        };
        if (a == "--model")       model = need("--model");
        else if (a == "--layer")  layer = std::atoi(need("--layer").c_str());
        else if (a == "--nt")     nt = std::atoi(need("--nt").c_str());
        else if (a == "--n-res")  n_res = std::atoi(need("--n-res").c_str());
        else if (a == "--iters")  iters = std::atoi(need("--iters").c_str());
        else if (a == "--warmup") warmup = std::atoi(need("--warmup").c_str());
        else if (a == "--cpu-threads") cpu_threads = std::atoi(need("--cpu-threads").c_str());
        else if (a == "--seed")   seed = (uint32_t) std::strtoul(need("--seed").c_str(), nullptr, 10);
        else if (a == "--list-layers") list_layers = true;
        else if (a == "--serialize-halves") serialize_halves = true;
        else if (a == "--serialize-both" || a == "--no-overlap-control") serialize_both = true;
        else if (a == "--empty-store") empty_store = true;
        else if (a == "--store-cuts") {
            // comma-separated extra expert-index cut points; each adds a STORE to the split
            // arm's list (SEAM 1) and a boundary derived from the geometry (SEAM 3).
            std::string v = need("--store-cuts");
            size_t pos = 0;
            while (pos <= v.size()) {
                size_t nxt = v.find(',', pos);
                std::string tok = v.substr(pos, nxt == std::string::npos ? std::string::npos : nxt - pos);
                if (!tok.empty()) extra_cuts.push_back(std::atoi(tok.c_str()));
                if (nxt == std::string::npos) break;
                pos = nxt + 1;
            }
        }
        else FATAL("unknown arg '%s'", a.c_str());
    }
    CHECK(layer >= 0 && layer < N_LAYERS, "layer %d out of range", layer);
    CHECK(nt >= 1, "nt must be >= 1");
    CHECK(n_res >= 0 && n_res <= N_EXPERT, "n-res %d out of range", n_res);

    GGUFModel * m = GGUFModel::load(model.c_str());
    if (list_layers) { m->list_layers(); return 0; }

    char gu_name[256], dn_name[256], gi_name[256];
    std::snprintf(gu_name, sizeof gu_name, "blk.%d.ffn_gate_up_exps.weight", layer);
    std::snprintf(dn_name, sizeof dn_name, "blk.%d.ffn_down_exps.weight", layer);
    std::snprintf(gi_name, sizeof gi_name, "blk.%d.ffn_gate_inp.weight", layer);
    LogicalTensor gu_lt = m->logical(gu_name);
    LogicalTensor dn_lt = m->logical(dn_name);
    size_t gi_n = 0;
    const float * gate_inp = m->f32_tensor(gi_name, &gi_n);
    CHECK(gi_n == (size_t) H_DIM * N_EXPERT, "gate_inp size mismatch");

    std::printf("# M0 moe_split  model=%s\n", m->path.c_str());
    std::printf("# layer=%d  nt=%d  n_res=%d  iters=%d  warmup=%d  cpu_threads=%d  seed=%u\n",
                layer, nt, n_res, iters, warmup, cpu_threads, seed);
    std::printf("# types: gate_up=%s  down=%s\n",
                ggml_type_name(gu_lt.segs[0].type), ggml_type_name(dn_lt.segs[0].type));

    // ---------- deterministic input ----------
    std::mt19937 rng(seed);
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    std::vector<float> x((size_t) H_DIM * nt);
    for (auto & v : x) v = gauss(rng);
    std::vector<int32_t> ids; std::vector<float> wgt;
    gen_ids_and_weights(gate_inp, x.data(), nt, ids, wgt);
    {
        int hits = 0;
        std::printf("# routed experts (top-4):");
        for (int t = 0; t < nt; ++t) {
            for (int e = 0; e < N_USED; ++e) {
                int ex = ids[e + N_USED*t];
                if (ex < n_res) ++hits;
                std::printf(" %d", ex);
            }
        }
        std::printf("  | hits in resident [0,%d): %d of %d\n", n_res, hits, (int) N_USED*nt);
    }

    // ---------- backends ----------
    ggml_backend_t cpu_backend = ggml_backend_cpu_init();
    CHECK(cpu_backend, "cpu backend init failed");
    ggml_backend_cpu_set_n_threads(cpu_backend, cpu_threads);
    ggml_backend_t gpu_backend = ggml_backend_cuda_init(0);
    CHECK(gpu_backend, "gpu (HIP) backend init failed");
    std::printf("# backends: CPU=%s  GPU=%s\n",
                ggml_backend_name(cpu_backend), ggml_backend_name(gpu_backend));

    // ---------- oracle (CPU reference + GPU path) ----------
    std::vector<float> y_orc((size_t) H_DIM * nt), gu_orc((size_t)(2*FF_DIM)*N_USED*nt), d_orc((size_t) H_DIM*N_USED*nt), h_orc((size_t) FF_DIM*N_USED*nt);
    std::vector<float> y_gpu_orc((size_t) H_DIM * nt), gu_gpu_orc((size_t)(2*FF_DIM)*N_USED*nt), d_gpu_orc((size_t) H_DIM*N_USED*nt);
    {
        Oracle oc;
        oc.build(gu_lt, dn_lt, cpu_backend, nt);
        oc.run(ids.data(), wgt.data(), x.data(), nt, y_orc.data(), gu_orc.data(), d_orc.data(), h_orc.data());
    }
    {
        Oracle og;
        og.build(gu_lt, dn_lt, gpu_backend, nt);
        og.run(ids.data(), wgt.data(), x.data(), nt, y_gpu_orc.data(), gu_gpu_orc.data(), d_gpu_orc.data(), nullptr);
    }
    std::printf("# oracle cpu-vs-gpu (cross-kernel reference): y max|d|=%.3e  gu max|d|=%.3e  d max|d|=%.3e\n",
                max_abs(y_orc.data(), y_gpu_orc.data(), y_orc.size()),
                max_abs(gu_orc.data(), gu_gpu_orc.data(), gu_orc.size()),
                max_abs(d_orc.data(), d_gpu_orc.data(), d_orc.size()));

    // ---------- derived tolerance bound (Correction 1) ----------
    {
        auto gamma = [](int64_t K) { return (double) K * 0x1p-24; };   // K * 2^-24
        // gate_up: act = x (same for every expert of a token); K = H = 4096
        double s_gu = 0.0;
        for (int t = 0; t < nt; ++t)
            for (int e = 0; e < N_USED; ++e) {
                int ex = ids[e + N_USED*t];
                double s = abs_dot_max(gu_lt.segs[0].bytes, gu_lt.segs[0].type, H_DIM, 2*FF_DIM,
                                       ex, x.data() + (size_t) H_DIM*t, cpu_backend);
                if (s >= 0) s_gu = std::max(s_gu, s);
            }
        // down: act = h (per-expert swiglu output); K = FF = 2048
        double s_dn = 0.0;
        for (int t = 0; t < nt; ++t)
            for (int e = 0; e < N_USED; ++e) {
                int ex = ids[e + N_USED*t];
                const float * h_act = h_orc.data() + (size_t) FF_DIM*e + (size_t) FF_DIM*N_USED*t;
                double s = abs_dot_max(dn_lt.segs[0].bytes, dn_lt.segs[0].type, FF_DIM, H_DIM,
                                       ex, h_act, cpu_backend);
                if (s >= 0) s_dn = std::max(s_dn, s);
            }
        const double eps_scale = 0x1p-11;   // F16 scale: block d (weights) and q8_1 d (activations)
        std::printf("# derived tolerance bound (S=Σ|w·act| over the routed experts, materialised):\n");
        std::printf("#   (ffn_gate_up_exps,%s) K=%d S_max=%.4e | F32-reorder 2*K*2^-24*S = %.3e | F16-scale 2*2^-11*S = %.3e | measured gu max|d| = %.3e\n",
                    ggml_type_name(gu_lt.segs[0].type), (int) H_DIM, s_gu,
                    2.0*gamma(H_DIM)*s_gu, 2.0*eps_scale*s_gu,
                    max_abs(gu_orc.data(), gu_gpu_orc.data(), gu_orc.size()));
        std::printf("#   (ffn_down_exps,%s)    K=%d S_max=%.4e | F32-reorder 2*K*2^-24*S = %.3e | F16-scale 2*2^-11*S = %.3e | measured d  max|d| = %.3e\n",
                    ggml_type_name(dn_lt.segs[0].type), (int) FF_DIM, s_dn,
                    2.0*gamma(FF_DIM)*s_dn, 2.0*eps_scale*s_dn,
                    max_abs(d_orc.data(), d_gpu_orc.data(), d_orc.size()));
        std::printf("#   READING: measured >> F32-reorder => the divergence is the quant-block scale surface, not reorder.\n");
        std::printf("#   SOURCE-VERIFIED (CORRECTED 2026-10-11, independent-review FIX 2 — the CPU side was\n"
                    "#   mislabelled q8_1; it is Q8_K, and M3d's root cause 1 is the right one):\n"
                    "#   • CPU: quantises src1 to type_traits_cpu[IQ2_XXS/IQ2_S/IQ3_S].vec_dot_type =\n"
                    "#     GGML_TYPE_Q8_K (256-wide, ggml_vec_dot_iq*_q8_K) at ggml-cpu.c:337/356/362,\n"
                    "#     applied UNCONDITIONALLY in ggml_compute_forward_mul_mat_id (ggml-cpu.c:1570-1644).\n"
                    "#     There is no 'prec' read ANYWHERE in ggml-cpu.c, and ops.cpp's only GGML_PREC_F32\n"
                    "#     is flash-attention (ops.cpp:9580).\n"
                    "#   • GPU: ggml_cuda_mul_mat_id dispatches to ggml_cuda_mul_mat_vec_q without reading\n"
                    "#     prec (ggml-cuda.cu:1992-2031), whose src1 quantiser is quantize_row_q8_1_cuda\n"
                    "#     (32-wide, mmvq.cu:1536).\n"
                    "#   => GPU = Q8_1, CPU = Q8_K on the decode vec path (exactly results/p-m3d.md:67-72),\n"
                    "#      and NEITHER honours the GGML_PREC_F32 src1 the model's graph requests.\n");
    }

    // SEAM 2(b): the layer input activations are published — the DFlash speculator hooks
    // this accessor.  (M0 exposes it; it does not implement the speculator.)
    const std::vector<float> & layer_input = x;

    // ---------- arm timing + check ----------
    // An arm is a LIST of stores (SEAM 1).  The canonical order is the spec's
    // (engine-plan.md §3 seam 1): every fill() first, then every enqueue(), then one
    // synchronize per store, then the drains.  fill() is a virtual no-op for a RAM/VRAM
    // store, so the order is immaterial to M0's numbers — but it is the order an M4
    // streamed store needs (its fill() issues the async H2D before any compute starts),
    // and hardcoding two stores would have made that a rewrite.
    struct ArmResult {
        double t_min, t_med, t_mean;
        std::vector<float> y, gu, d;
    };

    // The halves mode is an ARGUMENT, not a global: the paired falsifier below must be able to
    // run BOTH implementations inside one timing loop over the same store list and inputs.
    auto one_pass = [&](const std::vector<Store *> & stores, bool export_intermediates,
                        std::vector<float> & y, std::vector<float> & gu, std::vector<float> & d,
                        bool serial) {
        for (Store * s : stores) s->fill(ids.data(), nt);
        if (serial) {
            // NO-OVERLAP CONTROL (the falsifier's own falsifier): drain each store FULLY
            // before the next is enqueued.  This is the pre-fix serial implementation of the
            // split arm — c ~= gpu_half + cpu_half — and it must make the gate exit non-zero.
            for (Store * s : stores) {
                s->enqueue(ids.data(), wgt.data(), x.data(), nt);
                ggml_backend_synchronize(s->backend);
                s->drain(y.data(), nt, export_intermediates,
                         gu.empty() ? nullptr : gu.data(), d.empty() ? nullptr : d.data());
            }
        } else {
            for (Store * s : stores) s->enqueue(ids.data(), wgt.data(), x.data(), nt);
            for (Store * s : stores) ggml_backend_synchronize(s->backend);
            for (Store * s : stores)
                s->drain(y.data(), nt, export_intermediates,
                         gu.empty() ? nullptr : gu.data(), d.empty() ? nullptr : d.data());
        }
    };

    auto run_arm = [&](const std::vector<Store *> & stores) -> ArmResult {
        ArmResult r;
        std::vector<double> times;
        for (int it = -warmup; it < iters; ++it) {
            std::vector<float> y((size_t) H_DIM * nt, 0.0f);
            std::vector<float> none;
            // the y-drain is INSIDE the timed window: the sum needs the result on the host.
            double t0 = now_s();
            one_pass(stores, false, y, none, none, serialize_halves);
            double t1 = now_s();
            if (it >= 0) times.push_back((t1 - t0) * 1000.0);
        }
        std::sort(times.begin(), times.end());
        r.t_min = times.front();
        r.t_mean = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
        r.t_med = times[times.size() / 2];
        // export pass (untimed)
        r.y.assign((size_t) H_DIM * nt, 0.0f);
        r.gu.assign((size_t)(2*FF_DIM)*N_USED*nt, 0.0f);
        r.d.assign((size_t) H_DIM*N_USED*nt, 0.0f);
        one_pass(stores, true, r.y, r.gu, r.d, serialize_halves);
        return r;
    };

    // timing is measured here (the y-drain above is inside the timed window — the sum
    // needs the result on the host, and the CPU store's result is already host-side).

    auto report = [&](const char * name, const ArmResult & r,
                      const std::vector<float> & y_ref) {
        std::printf("[%s] ms/layer: min=%.4f med=%.4f mean=%.4f   (tok/s x36 = %.1f)\n",
                    name, r.t_min, r.t_med, r.t_mean, 1000.0 / (36.0 * r.t_med));
        std::printf("[%s] vs CPU-oracle: y max|d|=%.3e rms=%.3e  gu max|d|=%.3e  d max|d|=%.3e\n",
                    name, max_abs(r.y.data(), y_ref.data(), r.y.size()),
                    rms(r.y.data(), y_ref.data(), r.y.size()),
                    max_abs(r.gu.data(), gu_orc.data(), r.gu.size()),
                    max_abs(r.d.data(), d_orc.data(), r.d.size()));
    };

    // ---- the split arm's STORE LIST (SEAM 1) -------------------------------------------
    // The ranges are DERIVED from the geometry, never hardcoded (SEAM 3): every segment
    // boundary of either logical tensor becomes a store boundary, plus the requested
    // resident-expert cut and any --store-cuts.  A flat file yields exactly the two stores
    // M0 has always measured ([0,n_res) VRAM / [n_res,128) RAM); a grouped file yields one
    // boundary per group, and a store whose range spans groups is representable (Store cuts
    // it into one SLICE per (range x type)).
    // The derivation itself lives in engine/store.h (M1/U1 extraction) — same code, one owner.
    std::vector<StoreSpec> split_specs =
        derive_store_specs(gu_lt, dn_lt, n_res, extra_cuts, empty_store);
    std::printf("# store list (SEAM 1): %zu store(s); %d segment(s) seen per tensor, ranges derived from them\n",
                split_specs.size(), (int) dn_lt.segs.size());
    for (size_t i = 0; i < split_specs.size(); ++i)
        std::printf("#   [%zu] %-4s experts [%d,%d)  n_e=%d  resident=%s\n", i,
                    split_specs[i].loc == Store::VRAM ? "VRAM" : "RAM",
                    split_specs[i].lo, split_specs[i].hi, split_specs[i].hi - split_specs[i].lo,
                    split_specs[i].lo < n_res ? "yes" : "no");
    std::printf("# halves mode: %s\n",
                serialize_halves ? "SERIALIZED (no-overlap control — the pre-fix serial split)" : "CONCURRENT");

    // build the store list; owned here so the stores outlive the timing loops.
    auto build_stores = [&](std::vector<std::unique_ptr<Store>> & own) {
        std::vector<Store *> ptrs;
        for (const StoreSpec & sp : split_specs) {
            auto st = std::unique_ptr<Store>(new Store());
            st->build(sp.loc, gu_lt, dn_lt, sp.lo, sp.hi,
                      sp.loc == Store::VRAM ? gpu_backend : cpu_backend, nt);
            ptrs.push_back(st.get());
            own.push_back(std::move(st));
        }
        return ptrs;
    };

    // ---- (a) GPU-only ----
    ArmResult ra;
    {
        Store s;
        s.build(Store::VRAM, gu_lt, dn_lt, 0, N_EXPERT, gpu_backend, nt);
        ra = run_arm({ &s });
        report("(a) GPU-only", ra, y_orc);
    }
    // ---- (b) CPU-only ----
    ArmResult rb;
    {
        Store s;
        s.build(Store::RAM, gu_lt, dn_lt, 0, N_EXPERT, cpu_backend, nt);
        rb = run_arm({ &s });
        report("(b) CPU-only", rb, y_orc);
    }
    // ---- (c) split ----
    ArmResult rc;
    double gpu_half = 0.0, cpu_half = 0.0, gpu_half_min = 0.0, cpu_half_min = 0.0;
    {
        std::vector<std::unique_ptr<Store>> own;
        std::vector<Store *> split_ptrs = build_stores(own);
        rc = run_arm(split_ptrs);

        // which half bounds (c)?  measure each store entry STANDALONE and sum by location.
        // These two sums ARE the serial split bound the falsifier tests against.
        for (const StoreSpec & sp : split_specs) {
            Store st;
            st.build(sp.loc, gu_lt, dn_lt, sp.lo, sp.hi,
                     sp.loc == Store::VRAM ? gpu_backend : cpu_backend, nt);
            std::vector<double> ts;
            for (int it = -warmup; it < iters; ++it) {
                std::vector<float> y((size_t) H_DIM * nt, 0.0f);
                std::vector<float> none;
                double t0 = now_s();
                // single store: the serial and concurrent paths issue the same
                // enqueue/sync/drain sequence, so the mode is immaterial here.
                one_pass({ &st }, false, y, none, none, false);
                if (it >= 0) ts.push_back((now_s() - t0) * 1000.0);
            }
            std::sort(ts.begin(), ts.end());
            // median AND min: the min-of-N is the robust statistic for a falsifier that asks
            // "can this beat the no-overlap bound?" — a contended CPU store inflates the
            // median, and the bound must not move with the contention.
            if (sp.loc == Store::VRAM) { gpu_half += ts[ts.size()/2]; gpu_half_min += ts.front(); }
            else                       { cpu_half += ts[ts.size()/2]; cpu_half_min += ts.front(); }
        }
        report("(c) SPLIT", rc, y_orc);

        // decomposition of the split arm's serial residue: where does c - max(halves) go?
        {
            std::vector<std::unique_ptr<Store>> own2;
            std::vector<Store *> ps = build_stores(own2);
            double g_enq = 0, c_enq = 0, sync = 0, dr = 0, tot = 0;
            for (int it = -warmup; it < iters; ++it) {
                double t0 = now_s();
                for (Store * s : ps) s->fill(ids.data(), nt);
                double t1 = now_s();
                for (size_t i = 0; i < ps.size(); ++i) {
                    double u0 = now_s();
                    ps[i]->enqueue(ids.data(), wgt.data(), x.data(), nt);
                    double u1 = now_s();
                    if (split_specs[i].loc == Store::VRAM) g_enq += (u1-u0)*1000;
                    else                                   c_enq += (u1-u0)*1000;
                }
                double t2 = now_s();
                for (Store * s : ps) ggml_backend_synchronize(s->backend);
                double t3 = now_s();
                std::vector<float> y((size_t) H_DIM * nt, 0.0f);
                for (Store * s : ps) s->drain(y.data(), nt, false, nullptr, nullptr);
                double t4 = now_s();
                if (it >= 0) {
                    sync += ((t1-t0)+(t3-t2))*1000; dr += (t4-t3)*1000; tot += (t4-t0)*1000;
                }
            }
            int n = iters;
            std::printf("[c-decomp] gpu.enqueue=%.4f  cpu.enqueue=%.4f  sync=%.4f  drain=%.4f  total=%.4f ms\n",
                        g_enq/n, c_enq/n, sync/n, dr/n, tot/n);
            std::printf("[c-decomp] floor=max(gpu_half,cpu_half)=%.4f  => residual c-floor=%.4f ms is gpu.enqueue+drain (input staging on the sync path)\n",
                        std::max(gpu_half, cpu_half), (tot/n) - std::max(gpu_half, cpu_half));
        }
    }

    // ---- THE PAIRED (interleaved) MEASUREMENT — the falsifier's measurement ------------
    // WHY THIS EXISTS (results/m0-gate-idle.md): the withdrawn statistic c/(gpu_half+cpu_half)
    // divided a measured effect by a SEPARATELY-MEASURED bound, so it inherited the bound's
    // between-invocation drift — 13-41% on an idle host, against a 5% margin.  The effect
    // itself was repeatable to 3% interleaved.  Here BOTH arms are timed in the SAME host
    // state, one iteration each, alternating inside ONE timing loop, over the same store
    // list, ids, weights, activations, iteration count and warmup — the pairing cancels the
    // machine drift instead of dividing by it.
    std::vector<double> pc, pser;     // per-iteration c_conc and c_ser, in ms
    std::printf("# paired mode: %s\n",
                serialize_both ? "CONTROL (BOTH arms serialised — the no-overlap control: the ratio is ~1.00 by construction and MUST trip the line)"
                               : "REAL (paired concurrent arm vs serialised arm, interleaved)");
    std::printf("# paired loop: %d timed iterations (+%d warmup), one iteration per arm, alternating CONCURRENT then SERIALIZED\n",
                iters, warmup);
    {
        std::vector<std::unique_ptr<Store>> own_p;
        std::vector<Store *> ps = build_stores(own_p);
        std::vector<float> yc((size_t) H_DIM * nt, 0.0f), ys((size_t) H_DIM * nt, 0.0f);
        std::vector<float> none;
        for (int it = -warmup; it < iters; ++it) {
            // ARM 1 (the "concurrent" arm — under --serialize-both it runs the SERIALISED
            // implementation as well, which is the whole control).
            double t0 = now_s();
            one_pass(ps, false, yc, none, none, serialize_both);
            // ARM 2 (always the serialised split: each store enqueued AND drained before the
            // next starts).  Same iteration, same host state, immediately adjacent.
            double t1 = now_s();
            one_pass(ps, false, ys, none, none, true);
            double t2 = now_s();
            if (it >= 0) { pc.push_back((t1 - t0) * 1000.0); pser.push_back((t2 - t1) * 1000.0); }
        }
        // rig check (reported, not a verdict): the two arms run the SAME store list, ids,
        // weights and combine order and differ only in the schedule, so their results must
        // agree.  A non-zero value here means the ratio compares two different computations.
        std::printf("# paired arms compute the same thing: max|y_conc - y_ser| = %.3e\n",
                    max_abs(yc.data(), ys.data(), yc.size()));
    }

    // ---- the gate ---------------------------------------------------------------------
    // THE VERDICT IS THE PAIRED STATISTIC.  A gate must be able to fail AND to pass.
    // WITHDRAWN as a verdict (2026-10-11, results/m0-gate-idle.md, task 0.2b): the previous
    // condition c <= 0.95*(gpu_half+cpu_half) divided c by a SEPARATELY-MEASURED bound whose
    // between-invocation drift (13-41% on an idle host) was far larger than the 5% margin,
    // so on six idle invocations the gate exited 0,0,1,1,0,1 — it could not reliably pass.
    // The verdict is now min-of-N(c_conc) / min-of-N(c_ser): BOTH arms are measured by the
    // interleaved loop above, in the same host state, so the pairing removes the drift
    // instead of dividing by it.  The old statistic's numbers (and gpu_half, cpu_half,
    // c/max(a,b), c/(a+b)) are still printed as REPORTED decomposition — never verdicts.
    const double a = ra.t_med, b = rb.t_med, c_ = rc.t_med;
    const double mx = std::max(a, b), sm = a + b;
    const double serial_split = gpu_half + cpu_half;   // the NO-OVERLAP bound of the split
    const double c_over_max = c_ / mx, c_over_sum = c_ / sm;
    const double c_over_serial = serial_split > 0 ? c_ / serial_split : 0.0;
    const double serial_thresh = 0.95;                 // the paired falsifier's threshold
    const bool overlap_ok = c_ <= serial_thresh * serial_split;
    std::printf("\n=== GATE ===\n");
    std::printf("(a)=%.4f ms  (b)=%.4f ms  (c)=%.4f ms\n", a, b, c_);
    std::printf("REPORTED (not verdicts): c/max(a,b) = %.3f   c/(a+b) = %.3f\n", c_over_max, c_over_sum);
    std::printf("serial-vs-overlap: serial=a+b=%.4f  overlapped=max(a,b)=%.4f  measured(c)=%.4f\n",
                sm, mx, c_);
    std::printf("split halves: GPU(resident %d ex) med=%.4f ms  CPU(miss) med=%.4f ms  floor=max=%.4f ms\n",
                n_res, gpu_half, cpu_half, std::max(gpu_half, cpu_half));
    std::printf("[reported] split halves (min-of-N): GPU=%.4f  CPU=%.4f  serial_bound_min=gpu_min+cpu_min=%.4f\n",
                gpu_half_min, cpu_half_min, gpu_half_min + cpu_half_min);
    // The withdrawn statistic, kept as REPORTED decomposition for continuity with
    // results/m0-engine.md §2: it is not a verdict — on an idle host its denominator drifts
    // more between invocations than the margin it is tested against (results/m0-gate-idle.md).
    std::printf("[reported, WITHDRAWN as a verdict] SERIAL-BOUND CHECK: c=%.4f  gpu_half=%.4f  cpu_half=%.4f  serial_bound=gpu_half+cpu_half=%.4f  c/serial_bound=%.3f  threshold=%.3f  -> %s\n",
                c_, gpu_half, cpu_half, serial_split, c_over_serial, serial_thresh,
                overlap_ok ? "PASS" : "FAIL");
    const double serial_split_min = gpu_half_min + cpu_half_min;
    const double c_over_serial_min = serial_split_min > 0 ? rc.t_min / serial_split_min : 0.0;
    const bool overlap_ok_min = rc.t_min <= serial_thresh * serial_split_min;
    std::printf("[reported, WITHDRAWN as a verdict] SERIAL-BOUND CHECK (min-of-N): c_min=%.4f  serial_bound_min=%.4f  c_min/serial_bound_min=%.3f  threshold=%.3f  -> %s\n",
                rc.t_min, serial_split_min, c_over_serial_min, serial_thresh,
                overlap_ok_min ? "PASS" : "FAIL");
    std::printf("which half bounds (c): %s (%.4f ms)\n",
                (gpu_half >= cpu_half) ? "GPU half" : "CPU half", std::max(gpu_half, cpu_half));
    if (!serialize_halves) {
        std::printf("c = floor(%.4f) + loop-overhead(%.4f) = %.4f ms;  the GPU half (%.4f) runs fully underneath the CPU half\n",
                    std::max(gpu_half, cpu_half), c_ - std::max(gpu_half, cpu_half), c_, std::min(gpu_half, cpu_half));
        std::printf("loop-overhead = input staging + stream sync + result D2H (see [c-decomp] above)\n");
    } else {
        std::printf("c = gpu_half + cpu_half + per-store loop overhead = %.4f ms; NO overlap\n", c_);
        std::printf("this is the NO-OVERLAP CONTROL: the pre-fix serial split, and it must FAIL the check above\n");
    }
    // ---- THE PAIRED FALSIFIER — the verdict --------------------------------------------
    auto stat3 = [](std::vector<double> v, double & mn, double & md, double & mean) {
        std::sort(v.begin(), v.end());
        mn = v.front();
        md = v[v.size() / 2];
        mean = std::accumulate(v.begin(), v.end(), 0.0) / (double) v.size();
    };
    double c_min_, c_med_, c_mean_, s_min_, s_med_, s_mean_;
    stat3(pc, c_min_, c_med_, c_mean_);
    stat3(pser, s_min_, s_med_, s_mean_);
    std::vector<double> pr(pc.size());
    for (size_t i = 0; i < pc.size(); ++i) pr[i] = pc[i] / pser[i];
    double r_min_, r_med_, r_mean_;
    stat3(pr, r_min_, r_med_, r_mean_);
    const double r_max_ = *std::max_element(pr.begin(), pr.end());
    const double paired_ratio = s_min_ > 0 ? c_min_ / s_min_ : 0.0;
    const bool paired_ok = paired_ratio <= serial_thresh;
    const char * ptag = serialize_both ? "CONTROL" : "REAL";
    std::printf("\n=== PAIRED FALSIFIER (%s) — THE VERDICT ===\n", ptag);
    std::printf("paired c_CONC ms (concurrent arm): min=%.4f  med=%.4f  mean=%.4f\n", c_min_, c_med_, c_mean_);
    std::printf("paired c_SER  ms (serialised arm): min=%.4f  med=%.4f  mean=%.4f\n", s_min_, s_med_, s_mean_);
    std::printf("paired per-iteration ratio c_conc/c_ser: min=%.3f  med=%.3f  mean=%.3f  max=%.3f  (spread of the pairing itself)\n",
                r_min_, r_med_, r_mean_, r_max_);
    std::printf("PAIRED FALSIFIER[%s]: conc_min=%.4f  ser_min=%.4f  ratio=%.3f  threshold=%.3f  -> %s\n",
                ptag, c_min_, s_min_, paired_ratio, serial_thresh, paired_ok ? "PASS" : "FAIL");
    std::printf("VERDICT: %s\n",
                paired_ok ? "PASS — PAIRED: the concurrent split beats the serialised split measured in the SAME host state (min-of-N ratio below the 0.95 line); engine premise stands"
                          : "FAIL — PAIRED: NO OVERLAP: the concurrent split does not beat the serialised split by the required margin; premise DEAD (a complete, valuable result)");
    if (serialize_both) {
        std::printf("[reported] this run is the NO-OVERLAP CONTROL (both arms serialised): a FAIL above is the required outcome;\n"
                    "           a PASS above would mean the falsifier is BLIND (the gate script exits 2).\n");
    }
    std::printf("[reported] overlap characterization from the WITHDRAWN statistic (not a verdict): %s\n",
                overlap_ok ? "MAX-LIKE" : "SERIAL-LIKE");
    std::printf("\n# timing convention: median of %d iterations after %d warmup (min/median/mean reported).\n",
                iters, warmup);
    std::printf("# NOTE: y-drain (D2H of the result to host) is INSIDE the timed window for every arm;\n"
                "#       the split arm's combine (host sum) is included; intermediate gu/d export is NOT\n"
                "#       (it runs in an untimed second pass).\n");
    (void) layer_input;
    ggml_backend_free(cpu_backend);
    ggml_backend_free(gpu_backend);
    delete m;
    return 0;
}
