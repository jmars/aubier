// engine/m1_layer.h — M1/U2: ONE layer's forward for ONE token batch.
//
// attention (= q path + kv path + rope + MLA absorption) + KV cache + router + MoE split +
// shared expert + residual, built from ggml ops in the engine's OWN cgraphs (the M0 execution
// model: no ggml_backend_sched for the model, no llama_decode, no re-entering llama.cpp's
// graph — engine-plan.md §"M0 execution model").
//
// The op sequence is transcribed from the reference source and every step is named with the
// line it came from:
//   * src/llama-fork/src/models/deepseek2.cpp (deepseek2 graph, MLA branch)
//   * src/llama-fork/src/llama-graph.cpp      (build_attn_mha with v_mla, build_moe_ffn,
//                                              build_ffn, build_norm, build_inp_embd)
// Claims in this file are MEASURED (read from that source this session) unless marked.
//
// TWO ARMS, and they are the point of the file:
//   ARM_MOE_ORACLE — one cgraph for the WHOLE layer: the router stays in-graph and the expert
//                    block is a single ggml_mul_mat_id over all 128 experts.  This is the
//                    reference's own op sequence (tier B / M0's Oracle pattern).
//   ARM_MOE_STORE  — the same core cgraph, but the expert block runs through the STORE LIST
//                    (engine/store.h, M0's machinery): the router's ids/weights are read back
//                    to the host and each store computes mul_mat_id over its own expert range.
//                    This is the product path (the GPU/CPU residency split).
// The two arms must be BITWISE identical on the CPU backend (M0's rig control, generalised
// from the MoE block to the whole layer).  A non-zero difference means the rig is broken, not
// the model.
//
// Tier B is NOT an independent oracle: if the op ORDER is wrong here, tier B is wrong the same
// way.  Only the reference checksums (tier A, --ref <moedump>) can certify the attention path.

#pragma once

#include "store.h"

#include <cstdint>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Geometry — MEASURED from the GGUF metadata (mistral4.*) and the reference source
//   attention.head_count=32  key_length_mla=128  kv_lora_rank=256  q_lora_rank=1024
//   value_length_mla=128     key_length=320      rope.dimension_count=64
//   layer_norm_rms_epsilon=9.999999974752427e-07  expert_used_count=4  expert_count=128
//   rope.freq_base=10000  rope.scaling.{type=yarn, factor=128, original_context_length=8192,
//   yarn_beta_fast=32, yarn_beta_slow=1, yarn_log_multiplier=0.1 -> 1.0 after the
//   [TAG_DEEPSEEK2_YARN_LOG_MUL_FIX] cancel}
// ---------------------------------------------------------------------------
static constexpr int64_t L_N_HEAD      = 32;
static constexpr int64_t L_HEAD_K_MLA  = 128;                 // n_embd_head_k_mla
static constexpr int64_t L_ROT         = 64;                  // n_rot
static constexpr int64_t L_NOPE        = L_HEAD_K_MLA - L_ROT;// 64
static constexpr int64_t L_KV_RANK     = 256;                 // n_lora_kv
static constexpr int64_t L_Q_RANK      = 1024;                // n_lora_q
static constexpr int64_t L_K_ROW       = L_KV_RANK + L_ROT;   // 320 = the KV cache row
static constexpr int64_t L_V_HEAD      = 128;                 // n_embd_head_v_mla
static constexpr float   L_RMS_EPS     = 9.999999974752427e-07f;

static constexpr int64_t L_ROPE_N_CTX_ORIG  = 8192;
static constexpr float   L_ROPE_FREQ_BASE   = 10000.0f;
// freq_scale = hparams.rope_freq_scale_train = 1/rope.scaling.factor (llama-model.cpp:1460),
// and cparams.rope_freq_scale = that (llama-context.cpp:138).  NOT 1.0 — see m1_attn_params().
static constexpr float   L_ROPE_FREQ_SCALE  = 1.0f / 128.0f;
static constexpr float   L_ROPE_EXT_FACTOR  = 1.0f;   // yarn (llama-context.cpp:177)
static constexpr float   L_ROPE_BETA_FAST   = 32.0f;
static constexpr float   L_ROPE_BETA_SLOW   = 1.0f;
static constexpr float   L_ROPE_ATTN_FACTOR = 1.0f;   // hparams.rope_attn_factor default
static constexpr float   L_YARN_LOG_MUL     = 1.0f;   // GGUF 0.1, cancelled by the loader
static constexpr int64_t L_ROPE_MODE        = 0;      // LLAMA_ROPE_TYPE_NORM == GGML_ROPE_TYPE_NORMAL

static constexpr float   L_W_SUM_MIN = 6.103515625e-5f;  // the F16 minimum (build_moe_ffn clamp)

// The per-layer tensor order == LAYER_TENSOR_SPECS order in engine/gguf_model.h (U1's checklist).
enum LayerTensorIdx {
    LW_ATTN_NORM = 0, LW_Q_A, LW_Q_A_NORM, LW_Q_B, LW_KV_A_MQA, LW_KV_A_NORM,
    LW_K_B, LW_V_B, LW_ATTN_OUT, LW_FFN_NORM, LW_GATE_INP, LW_GATE_UP_EXPS, LW_DOWN_EXPS,
    LW_GATE_SHEXP, LW_UP_SHEXP, LW_DOWN_SHEXP, LW_N
};

// ---------------------------------------------------------------------------
// The attention scalars — derived the way llama_context derives them, with the values the
// reference prints in its own log (MEASURED: "setting new yarn_attn_factor = 1.0000",
// "freq_scale = 0.0078125", "freq_base = 10000.0", "n_ctx_orig_yarn = 8192").
// ---------------------------------------------------------------------------
struct M1AttnParams {
    float freq_base   = L_ROPE_FREQ_BASE;
    float freq_scale  = L_ROPE_FREQ_SCALE;
    float ext_factor  = L_ROPE_EXT_FACTOR;
    float attn_factor = 0;   // cparams.yarn_attn_factor, the `mscale` arg of ggml_rope_ext
    float beta_fast   = L_ROPE_BETA_FAST;
    float beta_slow   = L_ROPE_BETA_SLOW;
    float attn_factor_org = 0;                 // deepseek2.cpp:468
    float mscale          = 0;                 // deepseek2.cpp:470
    float kq_scale        = 0;                 // deepseek2.cpp:471
    float rope_mscale_kernel = 0;              // attn_factor * (1 + 0.1*ln(1/freq_scale))
    float corr_dims[2]    = {0, 0};            // ggml_rope_yarn_corr_dims
};
M1AttnParams m1_attn_params();
const char * m1_attn_params_str(char * buf, size_t n);

// ---------------------------------------------------------------------------
// One layer: weights + KV cache + the graphs.
// ---------------------------------------------------------------------------
struct M1Layer {
    int             layer     = -1;
    ggml_context *  ctx       = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_tensor *   w[LW_N]   = {};
    ggml_tensor *   kcache    = nullptr;   // F16 [L_K_ROW, n_kv_max] — the compressed K cache
    int64_t         n_kv_max  = 0;
    int             nt        = 1;
    int             n_res     = 64;        // resident-expert cut (the store list's one knob)
    int             fold_slots_ = -1;      // U9: the fold's slot count (-1 = N_USED, 0 = none)
    bool            m4_wkb      = false;   // H3/U4: the wk_b custom-op hatch (GPU-core only)
    bool            m4_kqkv     = false;   // H3/U5: the fused attention-core hatch (GPU-core only)
    bool            m4_kvwrite  = false;   // H3/U11: the fused KV-write phase (GPU-core only)
    bool            m4_wvb      = false;   // H3/U11: the fused wv_b / kqv_mla phase (GPU-core only)
    LogicalTensor   gu_lt, dn_lt;          // the expert tensors, for the store arm
    size_t          wbytes    = 0;

    // graph inputs
    ggml_tensor * g_x    = nullptr;        // F32 [H, nt]      layer input activations
    ggml_tensor * g_pos  = nullptr;        // I32 [nt]         positions
    ggml_tensor * g_kidx = nullptr;        // I64 [nt]         KV cache cells (= positions)
    ggml_tensor * g_mask = nullptr;        // F32 [n_kv_max, nt]  0 / -INF

    // graph taps (bisection): every one of these is a real node of the reference sequence
    ggml_tensor * g_q_a     = nullptr;     // [Q_RANK, nt]     q after attn_q_a
    ggml_tensor * g_q_norm  = nullptr;     // [Q_RANK, nt]     q after attn_q_a_norm (H3/U3 tap)
    ggml_tensor * g_q       = nullptr;     // [H, nt]          q after attn_q_b
    ggml_tensor * g_kv_cmpr = nullptr;     // [KV_RANK, nt]    normed compressed kv
    ggml_tensor * g_k_pe    = nullptr;     // [ROT, 1, nt]     roped k_pe
    ggml_tensor * g_q_nope_abs = nullptr;  // [KV_RANK, N_HEAD, nt]  absorbed q
    ggml_tensor * g_qcur    = nullptr;     // [K_ROW, N_HEAD, nt]  Qcur (concat)
    ggml_tensor * g_kcur    = nullptr;     // [K_ROW, 1, nt]       Kcur (concat) == the cache row
    ggml_tensor * g_kv_write = nullptr;    // the SET_ROWS node
    ggml_tensor * g_kq      = nullptr;     // [n_kv_max, nt, N_HEAD] softmaxed scores
    ggml_tensor * g_kqv     = nullptr;     // [KV_RANK, nt, N_HEAD] pre-v_b attention (ref: "kqv")
    ggml_tensor * g_kqv_mla = nullptr;     // [L_V_HEAD, nt, N_HEAD] after v_b (ref: "kqv_mla")
    ggml_tensor * g_kqv_out = nullptr;     // [H, nt]     head-recombined, feeds wo (ref: "kqv_out")
    ggml_tensor * g_attn_out= nullptr;     // [H, nt]          attn_output projection
    ggml_tensor * g_kv_cmpr_pe = nullptr;  // [K_ROW, nt]      raw wkv_a_mqa out, pre-split/norm/rope
    ggml_tensor * g_ffn_inp = nullptr;     // [H, nt]          the layer residual input
    ggml_tensor * g_cur     = nullptr;     // [H, nt]          normed MoE input
    ggml_tensor * g_logits  = nullptr;     // [N_EXPERT, nt]
    ggml_tensor * g_probs   = nullptr;     // [N_EXPERT, nt]
    ggml_tensor * g_sel     = nullptr;     // [N_USED, nt]     I32
    ggml_tensor * g_wts     = nullptr;     // [1, N_USED, nt]  F32 normed routing weights
    ggml_tensor * g_shexp   = nullptr;     // [H, nt]          shared expert
    ggml_tensor * g_moe     = nullptr;     // [H, nt]          MoE block (oracle arm only)
    ggml_tensor * g_moe_res = nullptr;     // [H, nt]          resident experts' partial (U8 fold arm)
    ggml_tensor * g_remap   = nullptr;     // [1, N_EXPERT, 1] I32 id-clamp table (U8 fold arm)
    ggml_tensor * g_remap_mask = nullptr;  // [1, N_EXPERT, 1] F32 weight-mask table (U8 fold arm)

    ggml_cgraph * gf_core   = nullptr;     // core: attention+router+shexp  (both arms)
    ggml_cgraph * gf_oracle = nullptr;     // core + the monolithic MoE      (ARM_MOE_ORACLE)

    // gpu_core (U4c, default false = the unchanged CPU-only path): build this layer's NON-expert
    // surface — the 14 attention/router/shexp weights + KV cache + inputs + intermediates — on
    // `backend` (ROCm0 when the flag is on).  The two expert tensors are NOT created in the
    // layer buffer in that mode: the store arm reads them from the stores' own slices
    // (engine/store.h), and the oracle arm (which mul_mat_ids the layer's own copies) is
    // therefore unavailable — m1_engine.cpp rejects the combination loudly.
    //
    // want_oracle (U4e, default true = the unchanged behaviour): when false, the two expert
    // tensors and gf_oracle are skipped exactly as in the gpu_core case — the pure store arm
    // (--arm-store) never reads them.  This is what makes a MIXED-type grouped file loadable:
    // make_layer_tensor refuses mixed member types (no single type exists for the oracle's
    // monolithic mul_mat_id), but the store arm slices per segment with per-segment types and
    // needs no monolithic tensor.  gpu_core implies !want_oracle (CHECKed).
    // H3/U1 — WHERE THE CORE GRAPH STOPS.  CORE_STOP_FULL (the default) is the landed graph,
    // node for node: the probe never changes what the product runs.  The two shorter stops exist
    // only for the --core-probe phase split (a TIMING-ONLY decomposition of the core bracket into
    // attention / +router / +shared-expert), and each of them is built as its own LAYER so that
    // the backend's CUDA-graph capture (keyed on the graph's FIRST node pointer,
    // ggml-cuda.cu:2691) stays stable per graph.
    enum CoreStop { CORE_STOP_FULL = 0, CORE_STOP_ROUTER = 1, CORE_STOP_ATTN = 2,
        // H3/U3 — the attention-chain bisection frontiers (prefixes of the attention expand
        // list, in build order).  Each is a SUPERSET of the previous, so the incremental
        // us/layer of a kernel group is the difference of two adjacent frontiers.
        CORE_STOP_KVWRITE = 3,   // attn_norm + kv path + set_rows (the first expand)
        CORE_STOP_Q_A     = 4,   // + wq_a
        CORE_STOP_Q_NOPE  = 5,   // + q_a_norm + wq_b + wk_b absorption (+ q_pe rope)
        CORE_STOP_QCUR    = 6,   // + qcur concat
        CORE_STOP_KQ      = 7,   // + kq matmul + softmax
        CORE_STOP_KQV     = 8,   // + vT copy + kqv matmul
        CORE_STOP_KQV_MLA = 9,   // + wv_b matmul
        CORE_STOP_KQV_OUT = 10,  // + head recombination
        CORE_STOP_Q_NORM  = 11,  // H3/U3: + q_a_norm (the normed q, before wq_b)
        CORE_STOP_Q_B     = 12,  // H3/U3: + wq_b (the 4.46 MiB up-projection)
    };
    // true for the attention-chain-only stops (nothing of the router/shexp is expanded)
    static bool stop_is_attn_only(int s) { return s >= CORE_STOP_ATTN; }
    // H3/U2 — WHICH EXPRESSION THE ATTENTION CHAIN IS BUILT WITH.  Both variants compute the
    // same values node for node; they differ only in whether an intermediate copy is
    // materialised.  attn_expr = 0 is the reference expansion (llama-graph.cpp build_attn_mha)
    // and stays the default so the landed path is unchanged until a variant is measured.
    //   0 = reference: (a) vT = cont(transpose(v)) for the kqv matmul, (b) kqv_out = cont_2d(
    //       permute(kqv_mla, 0,2,1,3)) for the head recombination.
    //   1 = (b) replaced by reshape_2d (valid at nt == 1: the swapped dim has extent 1, so the
    //       two layouts are the same bytes and the CONT copies contiguous data onto itself).
    // There is no option 2: (a) is NOT removable through ggml — ggml_mul_mat refuses a
    // transposed src0 outright (ggml.c:3356 GGML_ASSERT(!ggml_is_transposed(a))), measured by
    // building exactly that graph (H3/U2).  See results/m1-h3-u2.md.
    void build(GGUFModel & m, int layer, ggml_backend_t backend, int64_t n_kv_max, int nt, int n_res,
               bool gpu_core = false, bool want_oracle = true, bool fold_resident = false,
               int fold_slots = -1,   // U9: -1 = N_USED; 0 = no resident block (timing A/B)
               int core_stop = CORE_STOP_FULL,   // H3/U1: the probe's phase stops
               int attn_expr = 0,     // H3/U2: the attention chain's expression (0/1, see above)
               bool m4_wkb = false,   // H3/U4: the wk_b custom-op hatch (GPU-core only)
               bool m4_kqkv = false,  // H3/U5: the fused attention-core hatch (GPU-core only)
               bool m4_kvwrite = false, // H3/U11: the fused KV-write phase (GPU-core only)
               bool m4_wvb = false);  // H3/U11: the fused wv_b / kqv_mla phase (GPU-core only)
    void set_input(const float * x, const int32_t * pos, const int64_t * kidx,
                    const int64_t * n_kv_per_token);
    void read(const ggml_tensor * t, void * dst) const;

    // ---- U5: the GPU-core staging path (gpu_core builds only; timing/transfer structure ONLY
    // — the same bytes move, so the numerics are identical wherever it runs; the CPU-only
    // correctness path never calls these and its code is unchanged) ----
    // The blocking set_input/read pair costs one full stream synchronize PER TRANSFER
    // (ggml_backend_cuda_buffer_set/get_tensor = cudaMemcpyAsync + cudaStreamSynchronize): 9
    // round-trips per layer, 324 per token.  Here the four inputs and the five store-arm taps
    // stage through ONE pinned host allocation per layer, queued with *_tensor_*_async on the
    // backend's own stream, and the caller synchronizes ONCE (read_taps does it after the
    // queueing) — one stream wait per layer instead of nine.
    bool hstage = false;                    // set by build() when gpu_core allocated the staging
    ggml_backend_buffer_t host_buf = nullptr;   // pinned (backend's host buffer type)
    void * hraw = nullptr;                  // malloc fallback when no pinned buffer type exists
    // U5: marks the core graph's end on the stream, so the loop's decomposition can still
    // separate "core execution" from "result staging" once everything is queued (the blocking
    // compute used to measure that boundary implicitly).  Null => the split degrades silently.
    ggml_backend_event_t ev_core = nullptr;
    float    * hx = nullptr,    * hmask = nullptr,    * hcur = nullptr,
             * hffn_inp = nullptr, * hshexp = nullptr, * hmoe_res = nullptr, * hwts = nullptr;
    int32_t  * hpos = nullptr,  * hsel = nullptr;
    int64_t  * hkidx = nullptr;

    // pinned-staging input enqueue (queues 4 async H2D; no sync — ordered before the graph by
    // the stream).  x/pos/kidx/mask are read immediately into the pinned buffers, so the
    // caller's memory is free as soon as this returns.
    void set_input_async(const float * x, const int32_t * pos, const int64_t * kidx,
                         const int64_t * n_kv_per_token, ggml_backend_t be);
    // queue the five store-arm taps D2H (async; the caller then event-synchronizes inside its
    // core-timing bracket, so the graph's execution is attributed to the core term as before)
    void queue_taps(ggml_backend_t be);
    // the staging close: ONE stream wait for the queued copies, then copy out
    void taps_out(float * cur, float * ffn_inp, float * shexp, float * moe_res,
                  int32_t * sel, float * wts, ggml_backend_t be);
    ~M1Layer();
};

// ---------------------------------------------------------------------------
// tier A — the reference's own per-layer checksums, parsed out of the
// builds/hip-fork/bin/moe-logits-dump stderr with LLAMA_MOE_SPLIT_DUMP=1.
// ---------------------------------------------------------------------------
struct RefChecksum { double sum = 0, sumsq = 0; float f4[4] = {0,0,0,0}; bool have = false; };
struct RefDump {
    // [step][layer * N_KINDS + kind]; kinds 0..2 are the tensors the U2 run dumped, 3/4 the
    // residual-path taps U2b added to the example's dump filter (ffn_inp = the FFN input
    // inpSA+attn_out, l_out = the layer output that feeds the next layer).  A dump without
    // 3/4 (every log predating U2b) simply leaves those slots !have.
    static constexpr int N_KINDS = 5;
    std::vector<std::vector<RefChecksum>> v;
    int n_dec_steps = 0;
    int n_layers    = 0;
    int n_evals     = 0;
    bool parse(const char * path);
};

// The host router (M0's gen_ids_and_weights) — the near-tie probe of plan §Q4 risk 3.
void m1_host_router(const float * gate_inp, const float * x, int nt,
                    std::vector<int32_t> & ids, std::vector<float> & w,
                    std::vector<float> * logits_out = nullptr);

// ---------------------------------------------------------------------------
// M1/Q1 — THE DAMAGE PROBE (calibration only, default OFF).
//
// Scale ONE named F32 layer weight by (1 + eps) as it is uploaded from the file — host side,
// before the bytes reach the backend, so it damages the CPU path and the GPU-core path alike
// and touches nothing else (no graph edit, no op substitution: the run that follows is the
// product path computing a slightly wrong model).
//
// Matched by tensor-name SUFFIX, so "ffn_norm.weight" damages all 36 layers' ffn_norm (a
// global, physically meaningful gain on the FFN input, which also rescales the router logits
// and therefore the routing temperature) while "blk.12.ffn_norm.weight" damages one layer.
// The tensor must be F32 and must arrive in ONE chunk covering the whole tensor; a suffix that
// matches NO tensor fails the run (m1_damage_hits() == 0 in the caller) — a probe that damages
// nothing is not a probe.
// ---------------------------------------------------------------------------
void m1_set_damage(const char * suffix, double eps);
long m1_damage_hits();   // tensors actually scaled since the last m1_set_damage()
// the two primitives, exported because the OUTPUT HEAD is built in m1_engine.cpp and its
// output_norm.weight is the one damage site with a first-order, monotone quality response
bool m1_damage_match(const ggml_tensor * t);
void m1_damage_set(ggml_tensor * t, const uint8_t * bytes, size_t offset, size_t nbytes);

// checksum helpers (F32 host tensors)
struct Sums { double sum = 0, sumsq = 0; float f4[4] = {0,0,0,0}; };
Sums sum_of(const float * v, size_t n);
double max_abs_diff(const float * a, const float * b, size_t n);
