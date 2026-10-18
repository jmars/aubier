// engine/m1_layer.cpp — M1/U2: ONE layer's forward.  See engine/m1_layer.h for the contract.
//
// Every op below is transcribed from the reference and carries the source line it came from.
// The transcription target (MEASURED, read this session):
//
//   src/llama-fork/src/models/deepseek2.cpp      llama_model_deepseek2::graph::graph()
//   src/llama-fork/src/llama-graph.cpp           build_attn_mha / build_moe_ffn / build_ffn /
//                                                build_norm / build_inp_embd
//   src/llama-fork/src/llama-kv-cache.cpp        cpy_k (the KV write) + get_k (the K view)
//   src/llama-fork/ggml/src/ggml-cpu/ops.cpp     ggml_rope_yarn_corr_dims / rope_yarn
//
// The ONE deliberate deviation from the reference, and it is named: this file allocates the cache
// at n_kv_max and masks the not-yet-written cells with -INFINITY, while the reference builds its
// mask at the KV cache's PADDED cell count.  That padding is NOT "the current position + 1" on
// this fork and it is not a function of any flag: get_n_kv() pads by max(n_pad = 1, 256) — the
// PR-16812 graph-reuse padding, llama-kv-cache.cpp:1283-1290 — so the reference's mask width is a
// CONSTANT 256 while fewer than 257 cells are used and 512 thereafter.  It reads 256 here because
// the cache is first padded to 256, not because the gate passes -b 256.
// ggml_soft_max adds the mask value to the score and exponentiates
// (exp(-INF - max) == 0), and adding an exact 0.0f to a running F32 sum is exact, so the masked
// ENTRIES agree bitwise.  The WIDTH does not, and that is not inert: the length of the V*P
// reduction changes ggml's SIMD/FMA accumulation, which left a ~4e-8 relative difference in the
// attention value kqv at every cell in U2 — normally absorbed, but amplified into a visible,
// single-cell divergence at (L4,p4) by the Q8_1 quantisation of the q8_0 attn_v_b matmul's src1.
// NOTE the exact-zero ENTRIES do not make the WEIGHTS bitwise at every cell either: measured in
// U2c, at a 10-wide mask the kq_soft_max tap differs from the reference at 12 of the 54 compared
// cells (<= 1.3e-8 relative, at the positions whose weights are spread rather than peaked), i.e.
// 36 of the 42 cells U2b's table used — a second, smaller width-sensitive path, removed by the
// same fix (results/m1-u2b.md §3.4.1).
// The control was run in U2b and it is asymmetric: kq_soft_max/kqv/kqv_mla/kqv_out are bitwise
// identical to the reference at every compared cell only when n_kv_max matches the reference's
// mask width — its PADDED CACHE width, 256 for this gate's 10 positions, NOT its -b — which is
// what --kv-max sets (results/m1-u2b.md §3.4-§3.5, logs/engine/m1-u2b-kv256.dump vs
// logs/engine/m1-u2b-core2.dump).
//
// CORRECTION (U2c, comment only — no behaviour change): an earlier version of this comment said
// the reference's mask width "follows its ubatch, -b 256" and that the reference masks at "n_kv =
// current position + 1"; both are wrong for this fork — see the padding paragraph above.  256
// merely COINCIDES with this gate's -b.  Consequence for U3: --kv-max must pin the REFERENCE'S
// BUFFER WIDTH (256 for the first gate), and every comparison must follow it to 512 if a gate
// context ever exceeds 255 used cells; the check that notices is the reference's own dumped
// kq_soft_max shape — if it is not [256,1,32] the width moved and --kv-max must be re-derived.

#include "m1_layer.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ---------------------------------------------------------------------------
// the attention scalars
// ---------------------------------------------------------------------------
M1AttnParams m1_attn_params() {
    M1AttnParams p;
    const float factor = 1.0f / p.freq_scale;          // llama-context.cpp:185

    auto get_mscale = [](float scale, float mscale) {
        return scale <= 1.0f ? 1.0f : (0.1f * mscale * logf(scale) + 1.0f);
    };

    // llama-context.cpp:191-217.  hparams.rope_yarn_log_mul = 1.0 (the GGUF's 0.1 is cancelled
    // by [TAG_DEEPSEEK2_YARN_LOG_MUL_FIX]), so the `!= 0` branch runs; arch is MISTRAL4, not
    // DEEPSEEK2, so the mscale = mscale_all_dims special case does NOT apply.
    const float mscale_all_dims = L_YARN_LOG_MUL;
    float yarn_attn = get_mscale(factor, 1.0f) / get_mscale(factor, mscale_all_dims);
    yarn_attn *= 1.0f / (1.0f + 0.1f * logf(factor));   // llama-context.cpp:214
    p.attn_factor = yarn_attn * L_ROPE_ATTN_FACTOR;     // llama-context.cpp:217

    // deepseek2.cpp:468-471
    p.attn_factor_org = p.attn_factor * (1.0f + 0.1f * logf(1.0f / p.freq_scale));
    p.mscale          = p.attn_factor_org * (1.0f + 0.1f * L_YARN_LOG_MUL * logf(1.0f / p.freq_scale));
    p.kq_scale        = 1.0f * p.mscale * p.mscale / sqrtf((float) L_HEAD_K_MLA);

    // what the rope KERNEL actually multiplies cos/sin by (ggml-cpu/ops.cpp rope_yarn:6250)
    p.rope_mscale_kernel = p.attn_factor * (1.0f + 0.1f * logf(1.0f / p.freq_scale));

    // ggml_rope_yarn_corr_dims (ggml-cpu/ops.cpp): the YaRN ramp bounds
    auto corr_dim = [&](float n_rot) {
        return (float) ((double) L_ROT *
                        log((double) L_ROPE_N_CTX_ORIG / ((double) n_rot * 2.0 * M_PI)) /
                        (2.0 * log((double) L_ROPE_FREQ_BASE)));
    };
    p.corr_dims[0] = std::max(0.0f, floorf(corr_dim(p.beta_fast)));
    p.corr_dims[1] = std::min((float) (L_ROT - 1), ceilf(corr_dim(p.beta_slow)));
    return p;
}

const char * m1_attn_params_str(char * buf, size_t n) {
    const M1AttnParams p = m1_attn_params();
    std::snprintf(buf, n,
        "freq_base=%.6g freq_scale=%.8g ext_factor=%.6g beta_fast=%.6g beta_slow=%.6g "
        "n_ctx_orig=%lld rope_mode=%lld | yarn_attn_factor(cparams)=%.9g  "
        "attn_factor_org=%.9g  mscale=%.9g  kq_scale=%.9g  rope_mscale_in_kernel=%.9g  "
        "corr_dims=[%.0f,%.0f]",
        p.freq_base, p.freq_scale, p.ext_factor, p.beta_fast, p.beta_slow,
        (long long) L_ROPE_N_CTX_ORIG, (long long) L_ROPE_MODE,
        p.attn_factor, p.attn_factor_org, p.mscale, p.kq_scale, p.rope_mscale_kernel,
        p.corr_dims[0], p.corr_dims[1]);
    return buf;
}

// ---------------------------------------------------------------------------
// helper: create one per-layer tensor in ctx and hand back its source bytes
// ---------------------------------------------------------------------------
namespace {

// one backend_tensor_set chunk: the file bytes for one SEGMENT of the tensor, at `offset`
// inside it.  A flat (ungrouped) tensor has exactly one chunk at offset 0 — the pre-U4e
// behaviour; a grouped one has one chunk per member, laid out in the members' expert order,
// which reconstructs the flat byte layout EXACTLY (the segments are contiguous by rule 1 of
// taillis/layout.md §2 and every member carries the expert rows of its own range).
struct SrcChunk { size_t offset; const uint8_t * bytes; size_t nbytes; };

struct Src { ggml_tensor * t; std::vector<SrcChunk> chunks; };

ggml_tensor * make_layer_tensor(GGUFModel & m, int layer, int idx, ggml_context * ctx,
                                std::vector<SrcChunk> & chunks_out) {
    const TensorSpec & sp = LAYER_TENSOR_SPECS[idx];
    const std::string name = layer_tensor_name(layer, sp.suffix);
    LogicalTensor lt = m.logical(name.c_str());
    if (lt.segs.size() > 1) {
        // U4e: the format groups only the two expert tensors (the surface specs' grouped
        // rows); anything else multi-segment is a malformed file.
        CHECK(idx == LW_GATE_UP_EXPS || idx == LW_DOWN_EXPS,
              "%s: grouped with %zu segments — the format groups only the expert tensors",
              name.c_str(), lt.segs.size());
    }
    const ggml_type type = lt.segs[0].type;
    for (const Segment & s : lt.segs) {
        // U4e, THE ORACLE REFUSAL (deliberate — never a fallback, never a silent type pick):
        // a grouped tensor whose members do not share ONE type has no single type, so the
        // monolithic expert tensor the ORACLE arm's ggml_mul_mat_id reads cannot exist (a
        // ggml tensor has one type).  The STORE arm loads the same file: M1Layer::build
        // skips these very tensors when no oracle is requested, and the stores slice
        // per segment with per-segment types (engine/store.h).
        CHECK(s.type == type,
              "%s: grouped file with MIXED member types (%s, then %s at experts [%d,%d)) — "
              "no single type exists, so the ORACLE arm's monolithic expert tensor cannot be "
              "built; run this file through the store arm (--arm-store), whose slices carry "
              "per-segment types",
              name.c_str(), ggml_type_name(type), ggml_type_name(s.type), s.lo, s.hi);
    }
    ggml_tensor * t = nullptr;
    switch (sp.n_ne) {
        case 1: t = ggml_new_tensor_1d(ctx, type, lt.ne[0]); break;
        case 2: t = ggml_new_tensor_2d(ctx, type, lt.ne[0], lt.ne[1]); break;
        case 3: t = ggml_new_tensor_3d(ctx, type, lt.ne[0], lt.ne[1], lt.ne[2]); break;
        default: FATAL("%s: unsupported n_ne %d", name.c_str(), sp.n_ne);
    }
    ggml_set_name(t, name.c_str());
    size_t off = 0;
    for (const Segment & s : lt.segs) {
        chunks_out.push_back({ off, s.bytes, s.nbytes });
        off += s.nbytes;
    }
    CHECK(ggml_nbytes(t) == off,
          "%s: ggml_nbytes %zu != file %zu across %zu segment(s) (type %s)", name.c_str(),
          ggml_nbytes(t), off, lt.segs.size(), ggml_type_name(type));
    return t;
}

// one-time sanity check that LayerTensorIdx lines up with U1's checklist order
void check_spec_order() {
    static const char * want[LW_N] = {
        "attn_norm.weight", "attn_q_a.weight", "attn_q_a_norm.weight", "attn_q_b.weight",
        "attn_kv_a_mqa.weight", "attn_kv_a_norm.weight", "attn_k_b.weight", "attn_v_b.weight",
        "attn_output.weight", "ffn_norm.weight", "ffn_gate_inp.weight", "ffn_gate_up_exps.weight",
        "ffn_down_exps.weight", "ffn_gate_shexp.weight", "ffn_up_shexp.weight", "ffn_down_shexp.weight",
    };
    static bool done = false;
    if (done) return;
    for (int i = 0; i < LW_N; ++i) {
        CHECK(std::strcmp(LAYER_TENSOR_SPECS[i].suffix, want[i]) == 0,
              "LayerTensorIdx %d is '%s', checklist %d is '%s'",
              i, want[i], i, LAYER_TENSOR_SPECS[i].suffix);
    }
    done = true;
}

// H3/U3: the attention-chain frontier LEVEL (monotonic: higher = more of the chain expanded).
// FULL/ROUTER map to -1 (the whole chain is always expanded there).  The enum's NUMERIC values
// are not monotonic (CORE_STOP_ATTN = 2 sits BELOW the bisection stops 3..10 while naming the
// LARGEST frontier), so the expand order is read off this map, not off the enum's values.
static int attn_level(int s) {
    switch (s) {
        case M1Layer::CORE_STOP_KVWRITE:  return 0;
        case M1Layer::CORE_STOP_Q_A:      return 1;
        case M1Layer::CORE_STOP_Q_NORM:   return 2;
        case M1Layer::CORE_STOP_Q_B:      return 3;
        case M1Layer::CORE_STOP_Q_NOPE:   return 4;
        case M1Layer::CORE_STOP_QCUR:     return 5;
        case M1Layer::CORE_STOP_KQ:       return 6;
        case M1Layer::CORE_STOP_KQV:      return 7;
        case M1Layer::CORE_STOP_KQV_MLA:  return 8;
        case M1Layer::CORE_STOP_KQV_OUT:  return 9;
        case M1Layer::CORE_STOP_ATTN:     return 10;
        default:                          return -1;  // CORE_STOP_FULL / CORE_STOP_ROUTER
    }
}

} // namespace

// ---------------------------------------------------------------------------
// M1/Q1 — the damage probe's state and its two helpers (see m1_layer.h).  The state is a
// PROCESS global set once, before any layer is built: it is a calibration switch, not a
// per-layer parameter, so it never has to be threaded through the store list or the probes.
// ---------------------------------------------------------------------------
static std::string g_dmg_suffix;
static double      g_dmg_eps  = 0.0;
static long        g_dmg_hits = 0;

void m1_set_damage(const char * suffix, double eps) {
    g_dmg_suffix = suffix ? suffix : "";
    g_dmg_eps    = eps;
    g_dmg_hits   = 0;
}

long m1_damage_hits() { return g_dmg_hits; }

// does this tensor carry the damage target's name?  (suffix match on the layer-qualified name,
// so a suffix can address one layer or all of them)
bool m1_damage_match(const ggml_tensor * t) {
    if (g_dmg_suffix.empty() || t == nullptr) return false;
    const char * nm = ggml_get_name(t);
    if (nm == nullptr) return false;
    const size_t n = g_dmg_suffix.size(), m = std::strlen(nm);
    return m >= n && std::strcmp(nm + (m - n), g_dmg_suffix.c_str()) == 0;
}

// upload one chunk of a damaged tensor: the F32 bytes scaled by (1 + eps) in a host copy.  The
// file's own bytes are never modified (they are a read-only mmap of the GGUF).
void m1_damage_set(ggml_tensor * t, const uint8_t * bytes, size_t offset, size_t nbytes) {
    CHECK(t->type == GGML_TYPE_F32,
          "damage probe: %s is %s — only an F32 tensor can be scaled in place meaningfully "
          "(a quantized tensor's blocks would be corrupted, not scaled)",
          ggml_get_name(t), ggml_type_name(t->type));
    CHECK(offset == 0 && nbytes == ggml_nbytes(t),
          "damage probe: %s arrives in %zu B at offset %zu of %zu B — the probe needs the WHOLE "
          "tensor in one chunk", ggml_get_name(t), nbytes, offset, ggml_nbytes(t));
    CHECK(nbytes % sizeof(float) == 0, "damage probe: %s has %zu bytes, not a whole number of f32",
          ggml_get_name(t), nbytes);
    std::vector<uint8_t> tmp(nbytes);
    std::memcpy(tmp.data(), bytes, nbytes);
    float * f = reinterpret_cast<float *>(tmp.data());
    const double scale = 1.0 + g_dmg_eps;
    const size_t n = nbytes / sizeof(float);
    for (size_t i = 0; i < n; ++i) f[i] = (float) ((double) f[i] * scale);
    ggml_backend_tensor_set(t, tmp.data(), offset, nbytes);
    ++g_dmg_hits;
    std::printf("# damage probe: %s x (1 + %.17g) applied to %zu f32\n",
                ggml_get_name(t), g_dmg_eps, n);
}

M1Layer::~M1Layer() {
    // the backend is owned by main and SHARED — not freed here (M0's rule)
    if (buf) ggml_backend_buffer_free(buf);
    if (host_buf) ggml_backend_buffer_free(host_buf);   // U5: the pinned staging (owns nothing else)
    if (ev_core) ggml_backend_event_free(ev_core);      // U5
    std::free(hraw);                                    // U5: the malloc fallback (null normally)
    if (ctx) ggml_free(ctx);
}

void M1Layer::build(GGUFModel & m, int layer_, ggml_backend_t backend,
                    int64_t n_kv_max_, int nt_, int n_res_, bool gpu_core_,
                    bool want_oracle_, bool fold_resident_, int fold_slots, int core_stop,
                    int attn_expr, bool m4_wkb_, bool m4_kqkv_,
                    bool m4_kvwrite_, bool m4_wvb_) {
    check_spec_order();
    CHECK(attn_expr >= 0 && attn_expr <= 1, "attn_expr %d out of range [0,1]", attn_expr);
    CHECK(attn_expr == 0 || nt_ == 1, "attn_expr %d needs nt == 1 (see m1_layer.h)", attn_expr);
    // H3/U4: the wk_b custom-op hatch is GPU-core-only (the CPU arm is the numerics oracle and
    // must keep the reference mul_mat).  The kernel is hardcoded nt == 1, so for nt > 1 (which
    // only the timing probe's nt=4 variant builds) the reference mul_mat is kept — this is a
    // build-time guard, not runtime shape dispatch.
    CHECK(!m4_wkb_ || gpu_core_, "--m4-wkb needs --gpu-core (the CPU arm keeps the reference mul_mat)");
    m4_wkb = m4_wkb_ && nt_ == 1;
    // H3/U5: the fused attention-core hatch is GPU-core-only (the CPU arm is the numerics
    // oracle and must keep the reference kq/softmax/kqv sequence).  The kernel is hardcoded
    // nt == 1, so for nt > 1 (only the timing probe's nt=4 variant builds) the reference
    // sequence is kept — a build-time guard, not runtime shape dispatch.
    CHECK(!m4_kqkv_ || gpu_core_, "--m4-kqkv needs --gpu-core (the CPU arm keeps the reference attention)");
    m4_kqkv = m4_kqkv_ && nt_ == 1;
    // H3/U11: the two remaining attention phases follow the SAME rule — GPU-core only (the CPU
    // arm is the numerics oracle and must keep the reference op sequence), nt == 1, and a
    // build-time guard rather than runtime shape dispatch.
    CHECK(!m4_kvwrite_ || gpu_core_, "--m4-kvwrite needs --gpu-core (the CPU arm keeps the reference KV write)");
    m4_kvwrite = m4_kvwrite_ && nt_ == 1;
    CHECK(!m4_wvb_ || gpu_core_, "--m4-wvb needs --gpu-core (the CPU arm keeps the reference wv_b mul_mat)");
    m4_wvb = m4_wvb_ && nt_ == 1;
    // the caller owns this invariant (m1_engine.cpp refuses --gpu-core without --arm-store);
    // re-checked because the expert-tensor skip below depends on it
    CHECK(!gpu_core_ || !want_oracle_, "gpu-core builds no layer-expert tensors: no oracle");
    // U8: the fold requires the GPU-core path (the resident experts live on the core backend)
    // and a single-token batch (the id-clamp lookup uses ggml_get_rows, whose shape contract
    // needs a->ne[2] == b->ne[1] == nt == 1).  The engine only ever builds nt == 1.
    CHECK(!fold_resident_ || gpu_core_, "fold_resident needs gpu_core");
    CHECK(!fold_resident_ || nt_ == 1, "fold_resident needs nt == 1 (the loop's batch)");
    CHECK(!fold_resident_ || n_res_ >= 1, "fold_resident needs n_res >= 1");
    // U9: the fold's slot count is a TIMING knob (the A/B that sizes the amplification), not a
    // product parameter: -1 (default) keeps the landed 4-slot fold; 0..4 builds a c-slot block
    // whose numerics are WRONG by construction (slots c..3 are dropped — the c<4 runs are
    // timing-only).  0 builds no resident block at all (the "empty" variant).
    CHECK(!fold_resident_ || fold_slots >= -1 && fold_slots <= (int) N_USED,
          "fold_slots %d out of range [0,%d]", fold_slots, (int) N_USED);
    fold_slots_ = fold_slots < 0 ? (int) N_USED : fold_slots;
    layer = layer_; n_kv_max = n_kv_max_; nt = nt_; n_res = n_res_;
    CHECK(nt >= 1, "nt must be >= 1");
    CHECK(n_kv_max >= nt, "n_kv_max %lld < nt %d", (long long) n_kv_max, nt);

    // The whole layer (weights + KV cache + every intermediate) lives in ONE ggml context with
    // ONE backend buffer, so no tensor is ever a view of a foreign allocation and the graph can
    // be built once and re-run with new inputs (the M0 execution model).
    ggml_init_params ip = { 64u*1024u*1024u, nullptr, /* no_alloc */ true };
    ctx = ggml_init(ip);
    CHECK(ctx, "ggml_init failed for layer %d", layer);

    std::vector<Src> srcs;
    for (int i = 0; i < LW_N; ++i) {
        // U8 --fold-resident: the RESIDENT slice's expert weights are loaded into the layer
        // buffer (a PARTIAL [0,n_res) tensor, flat files only) so the fold's mul_mat_id in
        // gf_core has them on the core backend.  The CPU store still slices the file for the
        // misses; nothing is duplicated (the VRAM store is skipped by the caller in fold mode).
        if (fold_resident_ && (i == LW_GATE_UP_EXPS || i == LW_DOWN_EXPS)) {
            const TensorSpec & sp = LAYER_TENSOR_SPECS[i];
            const std::string name = layer_tensor_name(layer, sp.suffix);
            LogicalTensor lt = m.logical(name.c_str());
            CHECK(lt.segs.size() == 1,
                  "%s: fold-resident needs a FLAT (single-segment) expert tensor, got %zu segments "
                  "(grouped files keep the store path — the fold is gated off there)",
                  name.c_str(), lt.segs.size());
            const Segment & sg = lt.segs[0];
            const size_t row    = ggml_row_size(sg.type, lt.ne[0]);
            const size_t stride = row * (size_t) lt.ne[1];          // bytes per expert
            const size_t nbytes = stride * (size_t) n_res;          // the [0,n_res) experts
            CHECK(sg.nbytes >= nbytes,
                  "%s: segment has %zu bytes, need %zu for %d resident experts",
                  name.c_str(), sg.nbytes, nbytes, n_res);
            ggml_tensor * t = ggml_new_tensor_3d(ctx, sg.type, lt.ne[0], lt.ne[1], n_res);
            ggml_set_name(t, name.c_str());
            w[i] = t;
            wbytes += nbytes;
            Src s;
            s.t = t;
            s.chunks.push_back({ 0, sg.bytes, nbytes });
            srcs.push_back(std::move(s));
            continue;
        }
        // U4c --gpu-core, and U4e: every pure-store-arm build (--arm-store without the
        // oracle).  The expert weights are not resident in the layer buffer — the store arm
        // multiplies through the stores' own slices, and the ORACLE graph (the only reader
        // of the layer's own copies) is not built.  This skip is also what lets a MIXED-type
        // grouped file load on the store arm: no monolithic expert tensor is ever requested.
        // gu_lt/dn_lt below still point at the file, which is what the stores slice.
        if (!want_oracle_ && (i == LW_GATE_UP_EXPS || i == LW_DOWN_EXPS)) {
            w[i] = nullptr;
            continue;
        }
        Src s;
        s.t = make_layer_tensor(m, layer, i, ctx, s.chunks);
        w[i] = s.t;
        for (const SrcChunk & c : s.chunks) wbytes += c.nbytes;
        srcs.push_back(std::move(s));
    }
    gu_lt = m.logical(layer_tensor_name(layer, LAYER_TENSOR_SPECS[LW_GATE_UP_EXPS].suffix).c_str());
    dn_lt = m.logical(layer_tensor_name(layer, LAYER_TENSOR_SPECS[LW_DOWN_EXPS].suffix).c_str());

    // ---- inputs ----------------------------------------------------------
    g_x    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H_DIM, nt);       ggml_set_input(g_x);
    g_pos  = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, nt);              ggml_set_input(g_pos);
    g_kidx = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, nt);              ggml_set_input(g_kidx);
    g_mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, n_kv_max, nt, 1, 1); ggml_set_input(g_mask);
    ggml_set_name(g_x, "m1_x"); ggml_set_name(g_pos, "m1_pos");
    ggml_set_name(g_kidx, "m1_kidx"); ggml_set_name(g_mask, "m1_mask");

    // ---- the per-layer KV cache: [320, n_kv_max] F16, the compressed K -------------
    // llama-kv-cache.cpp: K row = n_embd_k_gqa = kv_lora_rank + n_rot = 320 (MEASURED: the
    // reference logs "n_rot_k = 64, n_embd_head_k_all = 320"), V = a VIEW of rows [0,256).
    kcache = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, L_K_ROW, n_kv_max);
    ggml_set_name(kcache, "m1_kcache");

    // ---- the op sequence (see the file header for the transcription target) --------
    // attn_norm: build_norm(inpL, attn_norm, LLM_NORM_RMS)
    ggml_tensor * cur = ggml_mul(ctx, ggml_rms_norm(ctx, g_x, L_RMS_EPS), w[LW_ATTN_NORM]);
    ggml_set_name(cur, "attn_norm");

    // q path: wq_a -> attn_q_a_norm -> wq_b   (deepseek2.cpp)
    g_q_a = ggml_mul_mat(ctx, w[LW_Q_A], cur);
    g_q_norm = ggml_mul(ctx, ggml_rms_norm(ctx, g_q_a, L_RMS_EPS), w[LW_Q_A_NORM]);   // H3/U3: kept as a tap
    g_q   = ggml_mul_mat(ctx, w[LW_Q_B], g_q_norm);

    const M1AttnParams ap = m1_attn_params();

    ggml_tensor * q3 = ggml_reshape_3d(ctx, g_q, L_HEAD_K_MLA, L_N_HEAD, nt);
    ggml_tensor * q_nope = ggml_view_3d(ctx, q3, L_NOPE, L_N_HEAD, nt, q3->nb[1], q3->nb[2], 0);
    ggml_tensor * q_pe   = ggml_view_3d(ctx, q3, L_ROT,  L_N_HEAD, nt, q3->nb[1], q3->nb[2],
                                        ggml_row_size(q3->type, L_NOPE));
    q_pe = ggml_rope_ext(ctx, q_pe, g_pos, nullptr, L_ROT, L_ROPE_MODE, L_ROPE_N_CTX_ORIG,
                         ap.freq_base, ap.freq_scale, ap.ext_factor, ap.attn_factor,
                         ap.beta_fast, ap.beta_slow);

    // kv path: wkv_a_mqa -> split {kv_cmpr | k_pe} -> rope(k_pe) -> attn_kv_a_norm(kv_cmpr)
    //
    // H3/U11: the flag-ON form is ONE op (GGML_OP_M4_PHASE phase 0) for the whole group — the
    // gemv, the norm, the rope, the Kcur concat and the set_rows scatter.  g_kv_cmpr_pe /
    // g_kv_cmpr / g_k_pe do not exist as graph tensors in that arm (they are taps of the
    // reference sequence, and the CPU oracle arm is the only reader of them).
    if (m4_kvwrite) {
        g_kv_cmpr_pe = g_kv_cmpr = g_k_pe = nullptr;
        g_kcur = ggml_m4_phase_kvwrite(ctx, cur, w[LW_KV_A_MQA], w[LW_KV_A_NORM], kcache,
                                       g_kidx, g_pos, ap.freq_base, ap.freq_scale,
                                       ap.ext_factor, ap.attn_factor, ap.beta_fast,
                                       ap.beta_slow, L_RMS_EPS, (int) L_ROT,
                                       (int) L_ROPE_N_CTX_ORIG);
    } else {
    ggml_tensor * kv_cmpr_pe = g_kv_cmpr_pe = ggml_mul_mat(ctx, w[LW_KV_A_MQA], cur);
    ggml_tensor * kv_cmpr = ggml_view_2d(ctx, kv_cmpr_pe, L_KV_RANK, nt,
                                         ggml_row_size(kv_cmpr_pe->type, L_K_ROW), 0);
    ggml_tensor * k_pe = ggml_view_3d(ctx, kv_cmpr_pe, L_ROT, 1, nt,
                                      ggml_row_size(kv_cmpr_pe->type, L_K_ROW),
                                      ggml_row_size(kv_cmpr_pe->type, L_K_ROW),
                                      ggml_row_size(kv_cmpr_pe->type, L_KV_RANK));
    g_k_pe = ggml_rope_ext(ctx, k_pe, g_pos, nullptr, L_ROT, L_ROPE_MODE, L_ROPE_N_CTX_ORIG,
                           ap.freq_base, ap.freq_scale, ap.ext_factor, ap.attn_factor,
                           ap.beta_fast, ap.beta_slow);
    g_kv_cmpr = ggml_mul(ctx, ggml_rms_norm(ctx, kv_cmpr, L_RMS_EPS), w[LW_KV_A_NORM]);
    }

    // MLA absorption: wk_b over the nope half, heads broadcast
    if (m4_wkb) {
        // H3/U4: the custom-op hatch.  ggml_m4_wkb consumes the UN-permuted nope view
        // directly (the kernel reads the head-stride-128 layout) and produces the already
        // [256, 32, nt] layout the concat wants — the two permutes of the reference expansion
        // are replaced by the op's own fixed shape (nt == 1, asserted by the builder).
        g_q_nope_abs = ggml_m4_wkb(ctx, w[LW_K_B], q_nope);          // [256, 32, nt]
    } else {
        ggml_tensor * q_nope_perm = ggml_permute(ctx, q_nope, 0, 2, 1, 3);   // [64, nt, 32]
        g_q_nope_abs = ggml_mul_mat(ctx, w[LW_K_B], q_nope_perm);            // [256, nt, 32]
        g_q_nope_abs = ggml_permute(ctx, g_q_nope_abs, 0, 2, 1, 3);          // [256, 32, nt]
    }

    // Qcur = concat(absorbed nope, roped pe); Kcur = concat(normed kv_cmpr, roped k_pe).
    // NOTE the concat order is nope-FIRST for K: 'Kcur = ggml_concat(ctx0, kv_cmpr, k_pe, 0)'
    // (deepseek2.cpp) — V is kcache rows [0,256) and MUST be kv_cmpr.
    g_qcur = ggml_concat(ctx, g_q_nope_abs, q_pe, 0);                        // [320, 32, nt]
    if (!m4_kvwrite) {
        ggml_tensor * kv_cmpr3 = ggml_reshape_3d(ctx, g_kv_cmpr, L_KV_RANK, 1, nt);
        g_kcur = ggml_concat(ctx, kv_cmpr3, g_k_pe, 0);                      // [320, 1, nt]

        // KV write — llama-kv-cache.cpp:cpy_k == ggml_set_rows(cache, k_cur, k_idxs)
        g_kv_write = ggml_set_rows(ctx, kcache, ggml_reshape_2d(ctx, g_kcur, L_K_ROW, nt), g_kidx);
    } else {
        // the fused KV-write phase already scattered the cache row in place; g_kcur IS its
        // output (the values the row received) and g_kv_write aliases it, so the graph still
        // expands the write FIRST and in the same place (see the expand order below).
        g_kv_write = g_kcur;
    }

    // attention — build_attn_mha, non-flash-attn path
    ggml_tensor * q4 = ggml_view_4d(ctx, g_qcur, L_K_ROW, L_N_HEAD, nt, 1,
                                    g_qcur->nb[1], g_qcur->nb[2], g_qcur->nb[2] * (size_t) nt, 0);
    if (m4_kqkv) {
        // H3/U5: the fused attention core — ONE op (GGML_OP_M4_KQKV) replaces the kq mul_mat,
        // the softmax and the kqv mul_mat + vT copy.  The kernel reads the KV cache and the
        // Qcur view directly (byte-explicit layout, one head per block) and produces the
        // [256, nt, 32] g_kqv the wv_b matmul consumes.  The softmax tap g_kq no longer
        // exists as a separate tensor (the bisection's KQ frontier collapses into KQV).
        g_kq  = nullptr;
        g_kqv = ggml_m4_kqkv(ctx, kcache, q4, g_mask, ap.kq_scale);   // [256, nt, 32]
    } else {
        const size_t rs_row = ggml_row_size(GGML_TYPE_F16, L_K_ROW);
        ggml_tensor * k_view = ggml_view_4d(ctx, kcache, L_K_ROW, 1, n_kv_max, 1,
                                            rs_row, rs_row, rs_row * (size_t) n_kv_max, 0);
        ggml_tensor * v_view = ggml_view_4d(ctx, kcache, L_KV_RANK, 1, n_kv_max, 1,
                                            rs_row, rs_row, rs_row * (size_t) n_kv_max, 0);
        ggml_tensor * qp = ggml_permute(ctx, q4, 0, 2, 1, 3);        // [320, nt, 32, 1]
        ggml_tensor * kp = ggml_permute(ctx, k_view, 0, 2, 1, 3);    // [320, n_kv_max, 1, 1]
        ggml_tensor * vp = ggml_permute(ctx, v_view, 0, 2, 1, 3);    // [256, n_kv_max, 1, 1]
        // v_trans = v->nb[1] > v->nb[2] == false here (both are the row stride), so the reference
        // takes the "!v_trans" branch and transposes v (llama-graph.cpp:3137).
        // H3/U2: this CONT CANNOT be removed.  Contracting over n_kv needs the transposed view as
        // mul_mat's src0, and ggml_mul_mat refuses that by construction — ggml.c:3356
        // GGML_ASSERT(!ggml_is_transposed(a)) fired when the graph was built without the copy.
        ggml_tensor * vT = ggml_cont(ctx, ggml_transpose(ctx, vp));  // [n_kv_max, 256]
        ggml_tensor * kq = ggml_mul_mat(ctx, kp, qp);                // [n_kv_max, nt, 32, 1]
        g_kq = ggml_soft_max_ext(ctx, kq, g_mask, ap.kq_scale, 0.0f);
        g_kqv = ggml_mul_mat(ctx, vT, g_kq);                         // [256, nt, 32, 1]
    }
    // H3/U11: the MLA value absorption gemv (`kqv_mla`, ~16.6 us/layer in the H3/U3 map) as
    // ONE op — the same shape class as wk_b, which the shipped GGML_OP_M4_WKB takes from 19 to
    // 128 GB/s.  The dst keeps the reference mul_mat's ne[] layout, so everything below is
    // unchanged.
    ggml_tensor * kqv = g_kqv_mla = m4_wvb
        ? ggml_m4_phase_wvb(ctx, w[LW_V_B], g_kqv)
        : ggml_mul_mat(ctx, w[LW_V_B], g_kqv);  // [128, nt, 32, 1]  (v_mla)
    // H3/U2 attn_expr >= 1: with nt == 1 the head recombination needs NO copy.  kqv is
    // contiguous [128, 1, 32]; permute(0,2,1,3) is [128, 32, 1] and cont_2d flattens it to
    // [4096, 1] — which is the SAME byte order as kqv (the swapped dim has extent 1), so
    // ggml_reshape_2d names that tensor without a kernel.  The reference expansion is kept
    // for nt > 1, where the copy is real (checked at the top of build()).
    ggml_tensor * attn = nullptr;
    if (attn_expr >= 1) {
        attn = g_kqv_out = ggml_reshape_2d(ctx, kqv, L_V_HEAD * L_N_HEAD, nt);   // [4096, nt]
    } else {
        attn = ggml_permute(ctx, kqv, 0, 2, 1, 3);               // [128, 32, nt, 1]
        attn = g_kqv_out = ggml_cont_2d(ctx, attn, L_V_HEAD * L_N_HEAD, nt);   // [4096, nt]
    }
    g_attn_out = ggml_mul_mat(ctx, w[LW_ATTN_OUT], attn);        // [H, nt]

    // residual + ffn_norm
    g_ffn_inp = ggml_add(ctx, g_attn_out, g_x);
    g_cur = ggml_mul(ctx, ggml_rms_norm(ctx, g_ffn_inp, L_RMS_EPS), w[LW_FFN_NORM]);

    // router — build_moe_ffn (gating softmax, top-4, norm_w, w_scale 1.0, no exp_probs_b)
    g_logits = ggml_mul_mat(ctx, w[LW_GATE_INP], g_cur);                       // [128, nt]
    g_probs  = ggml_soft_max(ctx, g_logits);
    g_sel    = ggml_argsort_top_k(ctx, g_probs, N_USED);                       // [4, nt] I32
    {
        ggml_tensor * probs3 = ggml_reshape_3d(ctx, g_probs, 1, N_EXPERT, nt);
        ggml_tensor * wts = ggml_get_rows(ctx, probs3, g_sel);                 // [1,4,nt]
        ggml_tensor * w2  = ggml_reshape_2d(ctx, wts, N_USED, nt);
        ggml_tensor * wsum = ggml_clamp(ctx, ggml_sum_rows(ctx, w2), L_W_SUM_MIN, INFINITY);
        g_wts = ggml_reshape_3d(ctx, ggml_div(ctx, w2, wsum), 1, N_USED, nt);
    }

    // shared expert — build_ffn(up_shexp, gate_shexp, down_shexp, SILU, PAR), added AFTER the MoE
    {
        ggml_tensor * up   = ggml_mul_mat(ctx, w[LW_UP_SHEXP],   g_cur);       // [FF, nt]
        ggml_tensor * gate = ggml_mul_mat(ctx, w[LW_GATE_SHEXP], g_cur);       // [FF, nt]
        ggml_tensor * h    = ggml_swiglu_split(ctx, gate, up);                 // [FF, nt]
        g_shexp = ggml_mul_mat(ctx, w[LW_DOWN_SHEXP], h);                      // [H, nt]
    }

    // U8 — THE RESIDENT FOLD (gpu_core + gpu_store + flat file): the resident slice's MoE is
    // folded INTO gf_core so the per-layer VRAM-store sync boundary disappears.  The ids are
    // clamped to [0,n_res) and out-of-range slots' weights are zeroed ON DEVICE through two
    // tiny lookup tables (ggml_get_rows over remap/mask — no ggml_cast/clamp, neither is a
    // CUDA-supported op here).  The resident part therefore never round-trips its ids/weights/
    // activations to the host; the CPU store still needs them there, so they still cross the
    // bus once (the taps).  NOT byte-identical to the store arm: the fold always accumulates
    // 4 slots (zero-weight slots add exact +0.0, but the mul_mat_id kernel dispatch and the
    // accumulation ORDER change — U4e) — measured and reported, never claimed.
    if (fold_resident_ && fold_slots_ > 0) {
        const int fs = fold_slots_;   // U9: the slot count (N_USED when not overridden)
        CHECK(w[LW_GATE_UP_EXPS] && w[LW_DOWN_EXPS], "fold_resident: resident experts not loaded");
        g_remap      = ggml_new_tensor_3d(ctx, GGML_TYPE_I32, 1, N_EXPERT, 1);
        g_remap_mask = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, N_EXPERT, 1);
        ggml_set_name(g_remap, "m1_remap"); ggml_set_name(g_remap_mask, "m1_remap_mask");

        // [1,fs,nt] -> [fs,nt] clamped ids; [1,fs,nt] weight mask applied to the top-fs weights.
        // U9: the top-fs SLOTS of g_sel/g_wts (top-k order) — the c<4 runs drop slots fs..3, so
        // their numerics are WRONG by construction (timing-only); the per-slot MMVQ work is the
        // same regardless of which experts are indexed.
        ggml_tensor * selv = ggml_view_2d(ctx, g_sel, fs, nt, g_sel->nb[1], 0);
        ggml_tensor * wtsv = ggml_view_3d(ctx, g_wts, 1, fs, nt, g_wts->nb[1], g_wts->nb[2], 0);
        ggml_tensor * ids_res = ggml_reshape_2d(ctx, ggml_get_rows(ctx, g_remap, selv), fs, nt);
        ggml_tensor * wts_res = ggml_mul(ctx, wtsv, ggml_get_rows(ctx, g_remap_mask, selv));

        ggml_tensor * cur3 = ggml_reshape_3d(ctx, g_cur, H_DIM, 1, nt);
        ggml_tensor * gu = ggml_mul_mat_id(ctx, w[LW_GATE_UP_EXPS], cur3, ids_res);  // [2*FF,fs,nt]
        ggml_tensor * gv = ggml_view_3d(ctx, gu, FF_DIM, fs, nt, gu->nb[1], gu->nb[2], 0);
        ggml_tensor * uv = ggml_view_3d(ctx, gu, FF_DIM, fs, nt, gu->nb[1], gu->nb[2],
                                        FF_DIM * gu->nb[0]);
        ggml_tensor * h  = ggml_swiglu_split(ctx, gv, uv);                          // [FF,fs,nt]
        ggml_tensor * d  = ggml_mul_mat_id(ctx, w[LW_DOWN_EXPS], h, ids_res);       // [H,fs,nt]
        ggml_prec_set_src(d, GGML_PREC_F32, 1);   // MISTRAL4 F32 src1, as the oracle sets it
        ggml_tensor * dm = ggml_mul(ctx, d, wts_res);
        ggml_tensor * acc = ggml_view_2d(ctx, dm, H_DIM, nt, dm->nb[2], 0);
        for (int e = 1; e < fs; ++e) {
            ggml_tensor * col = ggml_view_2d(ctx, dm, H_DIM, nt, dm->nb[2], e * dm->nb[1]);
            acc = ggml_add(ctx, acc, col);
        }
        g_moe_res = acc;
    }
    // U9: fold_resident with fold_slots == 0 builds NO resident block (g_moe_res stays null) —
    // the "empty" variant whose core bracket is the c=0 fast path's cost.

    // the monolithic expert block — build_moe_ffn with the FUSED gate_up tensor.
    // U4c --gpu-core: built only when the layer carries its own expert weights (the oracle
    // arm); with the flag the layer has none (they live in the stores) and gf_oracle stays
    // null — the engine refuses that arm in this mode.  U8: in fold mode the layer DOES carry
    // the RESIDENT expert weights, but the oracle arm must still not be built (its monolithic
    // mul_mat_id would index the partial tensor with global ids) — the fold arm has no oracle.
    if (want_oracle_ && w[LW_GATE_UP_EXPS] && w[LW_DOWN_EXPS]) {
        ggml_tensor * cur3 = ggml_reshape_3d(ctx, g_cur, H_DIM, 1, nt);
        ggml_tensor * gu = ggml_mul_mat_id(ctx, w[LW_GATE_UP_EXPS], cur3, g_sel);  // [2*FF,4,nt]
        ggml_tensor * gv = ggml_view_3d(ctx, gu, FF_DIM, N_USED, nt, gu->nb[1], gu->nb[2], 0);
        ggml_tensor * uv = ggml_view_3d(ctx, gu, FF_DIM, N_USED, nt, gu->nb[1], gu->nb[2],
                                        FF_DIM * gu->nb[0]);
        ggml_tensor * h  = ggml_swiglu_split(ctx, gv, uv);                         // [FF,4,nt]
        ggml_tensor * d  = ggml_mul_mat_id(ctx, w[LW_DOWN_EXPS], h, g_sel);        // [H,4,nt]
        // MISTRAL4 forces F32 src1 for the down matmul (llama-graph.cpp, build_moe_ffn).
        // MEASURED (M0): the CPU backend ignores it (Q8_K src1 unconditionally) — set for
        // fidelity with the reference graph, not because the CPU honours it.
        ggml_prec_set_src(d, GGML_PREC_F32, 1);
        ggml_tensor * dm = ggml_mul(ctx, d, g_wts);
        ggml_tensor * acc = ggml_view_2d(ctx, dm, H_DIM, nt, dm->nb[2], 0);
        for (int e = 1; e < (int) N_USED; ++e) {
            ggml_tensor * col = ggml_view_2d(ctx, dm, H_DIM, nt, dm->nb[2], e * dm->nb[1]);
            acc = ggml_add(ctx, acc, col);
        }
        g_moe = acc;
    }

    // ---- one backend buffer for the whole context --------------------------------
    buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    CHECK(buf, "backend alloc failed for layer %d (%s)%s", layer, ggml_backend_name(backend),
          gpu_core_ ? " — the layer core did not fit in VRAM; lower --n-res or run without --gpu-core"
                    : "");
    CHECK(!gpu_core_ || wbytes > 0, "gpu-core layer %d lost its weights", layer);

    for (const Src & s : srcs)
        for (const SrcChunk & c : s.chunks) {
            if (m1_damage_match(s.t)) m1_damage_set(s.t, c.bytes, c.offset, c.nbytes);
            else                      ggml_backend_tensor_set(s.t, c.bytes, c.offset, c.nbytes);
        }
    ggml_backend_tensor_memset(kcache, 0, 0, ggml_nbytes(kcache));

    // U8: the fold's two lookup tables (set once, read every token by ggml_get_rows on device).
    // U9: g_remap is null when fold_slots == 0 (no resident block) — skip the set in that case.
    if (g_remap) {
        std::vector<int32_t> remap((size_t) N_EXPERT);
        std::vector<float>   mask((size_t) N_EXPERT);
        for (int e = 0; e < (int) N_EXPERT; ++e) {
            remap[e] = (e < n_res) ? e : (n_res - 1);   // clamp out-of-range ids to a valid one
            mask[e]  = (e < n_res) ? 1.0f : 0.0f;       // ...and zero their weight
        }
        ggml_backend_tensor_set(g_remap,      remap.data(), 0, sizeof(int32_t) * N_EXPERT);
        ggml_backend_tensor_set(g_remap_mask, mask.data(),  0, sizeof(float)   * N_EXPERT);
    }

    // ---- U5: the pinned staging for the GPU-core path ------------------------------
    // One pinned host allocation per layer holding the four graph inputs and the five
    // store-arm taps; every transfer then queues with ggml_backend_tensor_{set,get}_async and
    // ONE ggml_backend_synchronize covers a whole layer (see m1_layer.h).  Built ONLY for
    // gpu_core: the CPU-only path never allocates it and never calls the async methods.
    if (gpu_core_) {
        const size_t a = 64;   // carve alignment: every section 64-byte aligned (kidx is i64)
        auto pad_up = [&](size_t v) { return (v + a - 1) / a * a; };
        const size_t sz_x    = pad_up((size_t) H_DIM * nt * 4);
        const size_t sz_pos  = pad_up((size_t) nt * 4);
        const size_t sz_kidx = pad_up((size_t) nt * 8);
        const size_t sz_mask = pad_up((size_t) n_kv_max * nt * 4);
        const size_t sz_H    = pad_up((size_t) H_DIM * nt * 4);
        const size_t sz_sel  = pad_up((size_t) N_USED * nt * 4);
        const size_t total   = sz_x + sz_pos + sz_kidx + sz_mask
                             + sz_H * 4 /* cur, ffn_inp, shexp, moe_res */ + sz_sel * 2 /* sel, wts */;
        ggml_backend_dev_t dev = ggml_backend_get_device(backend);
        ggml_backend_buffer_type_t hbuft = dev ? ggml_backend_dev_host_buffer_type(dev) : nullptr;
        host_buf = hbuft ? ggml_backend_buft_alloc_buffer(hbuft, total) : nullptr;
        char * base = host_buf ? (char *) ggml_backend_buffer_get_base(host_buf) : nullptr;
        // fall back to plain malloc'd staging when no pinned buffer type exists: the async
        // calls stay valid (pageable host memory costs latency, never correctness)
        const bool pinned = base != nullptr;
        if (!pinned) { base = (char *) std::malloc(total); hraw = base; }
        CHECK(base, "U5 staging alloc failed for layer %d", layer);
        char * p = base;
        hx       = (float *)    p; p += sz_x;
        hpos     = (int32_t *)  p; p += sz_pos;
        hkidx    = (int64_t *)  p; p += sz_kidx;
        hmask    = (float *)    p; p += sz_mask;
        hcur     = (float *)    p; p += sz_H;
        hffn_inp = (float *)    p; p += sz_H;
        hshexp   = (float *)    p; p += sz_H;
        hmoe_res = (float *)    p; p += sz_H;
        hsel     = (int32_t *)  p; p += sz_sel;
        hwts     = (float *)    p; p += sz_sel;
        hstage = true;
        ev_core = dev ? ggml_backend_event_new(dev) : nullptr;   // the core/staging timer split
    }

    // ---- the cgraphs ------------------------------------------------------------
    // The KV write is expanded FIRST: ggml_set_rows writes the cache in place and the attention
    // reads it through a view, so the write is not an ancestor of the read — the CPU backend
    // executes in node-list order, and llama.cpp relies on exactly this ordering too
    // (llama-graph.cpp:3416 expands cpy_k before build_attn_mha).
    gf_core = ggml_new_graph_custom(ctx, 4096, false);
    // H3/U3 — the attention chain is expanded as ONE monotonic prefix order (each frontier a
    // superset of the previous), so the probe prices kernel groups by subtraction.  The FIRST
    // expand (g_kv_write) pulls the whole kv path + attn_norm; g_kv_cmpr/g_k_pe/g_kcur are
    // already inside that chain, so the three separate expand calls the H3/U1 code used for
    // them were no-ops and are dropped.  The write MUST stay first (the attention reads the
    // cache through a view — see the comment above the graph section).
    ggml_build_forward_expand(gf_core, g_kv_write);
    const int al = attn_level(core_stop);
    const bool full_attn = al < 0;      // FULL/ROUTER: the whole chain, exactly as H3/U1 built it
    if (full_attn || al >= 1) ggml_build_forward_expand(gf_core, g_q_a);
    if (full_attn || al >= 2) ggml_build_forward_expand(gf_core, g_q_norm);
    if (full_attn || al >= 3) ggml_build_forward_expand(gf_core, g_q);
    if (full_attn || al >= 4) ggml_build_forward_expand(gf_core, g_q_nope_abs);
    if (full_attn || al >= 5) ggml_build_forward_expand(gf_core, g_qcur);
    // H3/U5: with the fused attention core, g_kq is nullptr (the softmax tap is gone) and the
    // fused op is expanded at the KQV frontier — the KQ/KQV split collapses into one kernel.
    if ((full_attn || al >= 6) && g_kq) ggml_build_forward_expand(gf_core, g_kq);
    if (full_attn || al >= 7) ggml_build_forward_expand(gf_core, g_kqv);
    if (full_attn || al >= 8) ggml_build_forward_expand(gf_core, g_kqv_mla);
    if (full_attn || al >= 9) ggml_build_forward_expand(gf_core, g_kqv_out);
    if (full_attn || al >= 10) ggml_build_forward_expand(gf_core, g_attn_out);
    // H3/U1: the probe's phase stops.  CORE_STOP_FULL (the product) expands exactly the list
    // this file has always expanded; the shorter stops cut the same list at a named frontier
    // so the probe can price attention / router / shared-expert by subtraction.  H3/U3's finer
    // attention stops (>= CORE_STOP_ATTN) cut INSIDE the attention chain and expand no router.
    if (!stop_is_attn_only(core_stop)) {
        ggml_build_forward_expand(gf_core, g_ffn_inp);
        ggml_build_forward_expand(gf_core, g_cur);
        ggml_build_forward_expand(gf_core, g_logits);
        ggml_build_forward_expand(gf_core, g_probs);
        ggml_build_forward_expand(gf_core, g_sel);
        ggml_build_forward_expand(gf_core, g_wts);
        if (core_stop == CORE_STOP_FULL) {
            ggml_build_forward_expand(gf_core, g_shexp);
            if (g_moe_res) ggml_build_forward_expand(gf_core, g_moe_res);   // U8: the resident fold
        }
    }

    // U4c --gpu-core: no expert weights in this layer => no oracle graph (the arm is refused).
    // U8: in fold mode the layer carries the RESIDENT experts, but the oracle arm must still not
    // be built (its monolithic mul_mat_id would index the partial tensor with global ids).
    if (!want_oracle_ || !w[LW_GATE_UP_EXPS] || !w[LW_DOWN_EXPS]) return;

    gf_oracle = ggml_new_graph_custom(ctx, 4096, false);
    ggml_build_forward_expand(gf_oracle, g_kv_write);
    ggml_build_forward_expand(gf_oracle, g_q_a);
    ggml_build_forward_expand(gf_oracle, g_kv_cmpr);
    ggml_build_forward_expand(gf_oracle, g_k_pe);
    ggml_build_forward_expand(gf_oracle, g_q_nope_abs);
    ggml_build_forward_expand(gf_oracle, g_qcur);
    ggml_build_forward_expand(gf_oracle, g_kcur);
    ggml_build_forward_expand(gf_oracle, g_kq);
    ggml_build_forward_expand(gf_oracle, g_kqv);
    ggml_build_forward_expand(gf_oracle, g_attn_out);
    ggml_build_forward_expand(gf_oracle, g_ffn_inp);
    ggml_build_forward_expand(gf_oracle, g_cur);
    ggml_build_forward_expand(gf_oracle, g_logits);
    ggml_build_forward_expand(gf_oracle, g_probs);
    ggml_build_forward_expand(gf_oracle, g_sel);
    ggml_build_forward_expand(gf_oracle, g_wts);
    ggml_build_forward_expand(gf_oracle, g_shexp);
    ggml_build_forward_expand(gf_oracle, g_moe);
}

void M1Layer::set_input(const float * x, const int32_t * pos, const int64_t * kidx,
                        const int64_t * n_kv_per_token) {
    ggml_backend_tensor_set(g_x, x, 0, sizeof(float) * (size_t) H_DIM * nt);
    ggml_backend_tensor_set(g_pos, pos, 0, sizeof(int32_t) * nt);
    ggml_backend_tensor_set(g_kidx, kidx, 0, sizeof(int64_t) * nt);
    std::vector<float> mask((size_t) n_kv_max * nt, 0.0f);
    for (int t = 0; t < nt; ++t)
        for (int64_t i = n_kv_per_token[t]; i < n_kv_max; ++i)
            mask[(size_t) i + (size_t) n_kv_max * t] = -INFINITY;
    ggml_backend_tensor_set(g_mask, mask.data(), 0, sizeof(float) * mask.size());
}

void M1Layer::read(const ggml_tensor * t, void * dst) const {
    ggml_backend_tensor_get(t, dst, 0, ggml_nbytes(t));
}

// ---- U5: the GPU-core staging path (see m1_layer.h for the contract) ----------------
void M1Layer::set_input_async(const float * x, const int32_t * pos, const int64_t * kidx,
                              const int64_t * n_kv_per_token, ggml_backend_t be) {
    CHECK(hstage, "set_input_async on a layer built without the U5 staging (gpu_core only)");
    std::memcpy(hx, x, sizeof(float) * (size_t) H_DIM * nt);
    std::memcpy(hpos, pos, sizeof(int32_t) * nt);
    std::memcpy(hkidx, kidx, sizeof(int64_t) * nt);
    for (int t = 0; t < nt; ++t)
        for (int64_t i = 0; i < n_kv_max; ++i)
            hmask[(size_t) i + (size_t) n_kv_max * t] =
                i < n_kv_per_token[t] ? 0.0f : -INFINITY;   // the exact bytes set_input writes
    ggml_backend_tensor_set_async(be, g_x,    hx,    0, sizeof(float)   * (size_t) H_DIM * nt);
    ggml_backend_tensor_set_async(be, g_pos,  hpos, 0, sizeof(int32_t)  * nt);
    ggml_backend_tensor_set_async(be, g_kidx, hkidx, 0, sizeof(int64_t)  * nt);
    ggml_backend_tensor_set_async(be, g_mask, hmask, 0, sizeof(float) * (size_t) n_kv_max * nt);
}

void M1Layer::queue_taps(ggml_backend_t be) {
    CHECK(hstage, "queue_taps on a layer built without the U5 staging (gpu_core only)");
    ggml_backend_tensor_get_async(be, g_cur,     hcur,     0, sizeof(float)  * (size_t) H_DIM * nt);
    ggml_backend_tensor_get_async(be, g_ffn_inp, hffn_inp, 0, sizeof(float)  * (size_t) H_DIM * nt);
    ggml_backend_tensor_get_async(be, g_shexp,   hshexp,   0, sizeof(float)  * (size_t) H_DIM * nt);
    if (g_moe_res)   // U8: the resident fold's output (the VRAM store's old result, now in-core)
        ggml_backend_tensor_get_async(be, g_moe_res, hmoe_res, 0, sizeof(float) * (size_t) H_DIM * nt);
    ggml_backend_tensor_get_async(be, g_sel,     hsel,     0, sizeof(int32_t) * (size_t) N_USED * nt);
    ggml_backend_tensor_get_async(be, g_wts,     hwts,     0, sizeof(float)  * (size_t) N_USED * nt);
}

void M1Layer::taps_out(float * cur, float * ffn_inp, float * shexp, float * moe_res,
                       int32_t * sel, float * wts, ggml_backend_t be) {
    CHECK(hstage, "taps_out on a layer built without the U5 staging (gpu_core only)");
    ggml_backend_synchronize(be);          // the staging's one stream wait (copies only: the
    std::memcpy(cur,     hcur,     sizeof(float)  * (size_t) H_DIM * nt);   // core wait already
    std::memcpy(ffn_inp, hffn_inp, sizeof(float)  * (size_t) H_DIM * nt);   // happened at the
    std::memcpy(shexp,   hshexp,   sizeof(float)  * (size_t) H_DIM * nt);   // event)
    if (moe_res && g_moe_res)   // U8: the fold's resident output (null in the non-fold gpu_core path)
        std::memcpy(moe_res, hmoe_res, sizeof(float) * (size_t) H_DIM * nt);
    std::memcpy(sel,     hsel,     sizeof(int32_t) * (size_t) N_USED * nt);
    std::memcpy(wts,     hwts,     sizeof(float)  * (size_t) N_USED * nt);
}

// ---------------------------------------------------------------------------
// tier A — the reference's own per-(layer, decode step) checksums
// ---------------------------------------------------------------------------
bool RefDump::parse(const char * path) {
    FILE * f = std::fopen(path, "r");
    if (!f) return false;
    v.clear();
    n_dec_steps = 0;
    n_layers = (int) N_LAYERS;

    std::vector<std::vector<RefChecksum>> evals;
    std::vector<char> complete;   // this eval has all 36 layers of ffn_out (=> a decode step)
    char line[1024];

    auto push_eval = [&]() {
        evals.push_back(std::vector<RefChecksum>((size_t) N_LAYERS * N_KINDS));
    };
    push_eval();

    while (std::fgets(line, sizeof line, f)) {
        if (std::strstr(line, "get_logits_ith")) { push_eval(); continue; }
        const char * p = std::strstr(line, "[moe_dump] ");
        if (!p) continue;
        p += std::strlen("[moe_dump] ");
        char name[128];
        if (std::sscanf(p, "%127s", name) != 1) continue;

        int kind = -1;
        const char * dash = nullptr;
        if (!std::strncmp(name, "ffn_moe_out-", 12))      { kind = 0; dash = name + 12; }
        else if (!std::strncmp(name, "ffn_shexp-", 10))   { kind = 1; dash = name + 10; }
        else if (!std::strncmp(name, "ffn_out-", 8))      { kind = 2; dash = name + 8; }
        else if (!std::strncmp(name, "ffn_inp-", 8))      { kind = 3; dash = name + 8; }
        else if (!std::strncmp(name, "l_out-", 6))        { kind = 4; dash = name + 6; }
        else continue;
        const int il = std::atoi(dash);
        if (il < 0 || il >= (int) N_LAYERS) continue;

        const char * ps = std::strstr(p, "sum=");
        const char * pq = std::strstr(p, "sumsq=");
        const char * pf = std::strstr(p, "first4=");
        if (!ps || !pq || !pf) continue;

        RefChecksum & c = evals.back()[(size_t) il * N_KINDS + kind];
        c.sum   = std::strtod(ps + 4, nullptr);
        c.sumsq = std::strtod(pq + 6, nullptr);
        std::sscanf(pf + 7, "%f %f %f %f", &c.f4[0], &c.f4[1], &c.f4[2], &c.f4[3]);
        c.have  = true;
    }
    std::fclose(f);
    n_evals = (int) evals.size();

    // keep every eval that carries all 36 layers of ffn_out: those are the decode steps, in
    // evaluation order (== position order).  The incomplete ones are dropped and reported as
    // dropped.  In the gate's `-ub 1` geometry the PREFILL eval DOES carry all 36 layers (one
    // ubatch per prompt token, last-write-wins => P(n_prompt-1)), so it is KEPT as v[0] and the
    // tier-A alignment DEPENDS on keeping it (engine step k == P(n_prompt-1+k)).  A batched
    // prefill (ub > 1) would carry only the cropped last layer and be dropped here, shifting the
    // table — the gate pins -ub 1 for that reason.  MEASURED on the N=128 artifact
    // logs/engine/m1-u3/gate1/ref-p2.stderr.txt: all 36 ffn_out layer ids appear before the
    // first "get_logits_ith", i.e. inside eval 0.
    int dropped = 0;
    for (size_t e = 0; e < evals.size(); ++e) {
        int have = 0;
        for (int il = 0; il < (int) N_LAYERS; ++il)
            if (evals[e][(size_t) il * N_KINDS + 2].have) ++have;
        if (have == (int) N_LAYERS) v.push_back(evals[e]);
        else ++dropped;
    }
    n_dec_steps = (int) v.size();
    std::fprintf(stderr, "# ref dump: %d evals parsed (%d dropped as prefill/cropped), "
                         "%d decode steps x %d layers x %d tensors\n",
                 n_evals, dropped, n_dec_steps, (int) N_LAYERS, (int) N_KINDS);
    return n_dec_steps > 0;
}

// ---------------------------------------------------------------------------
// the host router — M0's gen_ids_and_weights verbatim (the near-tie probe)
// ---------------------------------------------------------------------------
void m1_host_router(const float * gate_inp, const float * x, int nt,
                    std::vector<int32_t> & ids, std::vector<float> & w,
                    std::vector<float> * logits_out) {
    ids.assign((size_t) N_USED * nt, 0);
    w.assign((size_t) N_USED * nt, 0.0f);
    if (logits_out) logits_out->assign((size_t) N_EXPERT * nt, 0.0f);
    for (int t = 0; t < nt; ++t) {
        float logits[N_EXPERT];
        for (int k = 0; k < (int) N_EXPERT; ++k) {
            double s = 0.0;
            const float * g = gate_inp + (size_t) k * H_DIM;   // gate_inp is [H, N_EXPERT]
            for (int64_t j = 0; j < H_DIM; ++j) s += (double) g[j] * x[j + H_DIM * t];
            logits[k] = (float) s;
        }
        if (logits_out)
            for (int k = 0; k < (int) N_EXPERT; ++k) (*logits_out)[k + N_EXPERT * t] = logits[k];
        float mx = *std::max_element(logits, logits + N_EXPERT);
        double z = 0.0; float p[N_EXPERT];
        for (int k = 0; k < (int) N_EXPERT; ++k) { p[k] = std::exp(logits[k] - mx); z += p[k]; }
        for (int k = 0; k < (int) N_EXPERT; ++k) p[k] /= (float) z;
        int idx[N_EXPERT];
        std::iota(idx, idx + N_EXPERT, 0);
        std::partial_sort(idx, idx + N_USED, idx + N_EXPERT,
                          [&](int a, int b) { return p[a] > p[b]; });
        for (int e = 0; e < (int) N_USED; ++e) {
            ids[e + N_USED * t] = idx[e];
            w[e + N_USED * t]   = p[idx[e]];
        }
    }
}

Sums sum_of(const float * v, size_t n) {
    Sums s;
    for (size_t i = 0; i < n; ++i) { s.sum += v[i]; s.sumsq += (double) v[i] * v[i]; }
    for (size_t i = 0; i < 4 && i < n; ++i) s.f4[i] = v[i];
    return s;
}

double max_abs_diff(const float * a, const float * b, size_t n) {
    double m = 0.0;
    for (size_t i = 0; i < n; ++i) m = std::max(m, std::fabs((double) a[i] - (double) b[i]));
    return m;
}
