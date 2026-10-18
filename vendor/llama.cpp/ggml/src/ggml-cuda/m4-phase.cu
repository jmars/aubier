#include "m4-phase.cuh"

// MISTRAL-SMALL-4-ONLY.  Everything here is HARDCODED to the model's geometry: the dims, the
// quant types, the block sizes, the head layout.  There is no generality, no fallback and no
// runtime shape dispatch; the builders (ggml.c) and the launchers below ASSERT the contract.
//
// ---------------------------------------------------------------------------------------
// PHASE 0 — M4_PHASE_KVWRITE.  The phase group the H3/U3 bisection prices at ~29.4 us/layer:
//   attn_norm            (RMS_NORM + MUL, already one fused kernel — NOT part of this op)
//   wkv_a_mqa            Q8_0 [4096, 320] gemv                  1.39 MiB
//   attn_kv_a_norm       RMS_NORM over rows [0,256) + MUL
//   k_pe rope            rope_ext over rows [256,320), n_dims 64, NORMAL mode, YaRN
//   Kcur = concat(kv_cmpr, k_pe)                                  [320, 1]
//   set_rows             scatter that row into the F16 KV cache
// The op replaces the last five with TWO launches: a gemv (grid 320 = one block per output
// row) and one small tail block that norms, ropes and scatters.  The reference's six kernels
// become two, and the three intermediate tensors never round-trip.
//
//   src0 = cur        F32  [4096, nt]      (the attn_norm output; nt == 1 asserted)
//   src1 = wkv_a_mqa  Q8_0 [4096, 320]
//   src2 = kv_a_norm  F32  [256]
//   src3 = kcache     F16  [320, n_kv]     (written IN PLACE, exactly as ggml_set_rows does)
//   src4 = kidx       I64  [nt]            (the cache row to write)
//   src5 = pos        I32  [nt]            (the rope position)
//   dst  = Kcur       F32  [320, nt]       (the post-norm, post-rope row = the written values)
// op_params: [0]=phase i32, then f32 at byte offsets 4,8,12,16,20,24 = freq_base, freq_scale,
//            ext_factor, attn_factor, beta_fast, beta_slow; 28 = eps; i32 at 32,36 = n_dims,
//            n_ctx_orig.
//
// ---------------------------------------------------------------------------------------
// PHASE 1 — M4_PHASE_WVB.  The MLA value absorption gemv (`kqv_mla` in the H3/U3 map, ~16.6
// us/layer = ~70 GB/s).  Same shape class as wk_b (which the shipped GGML_OP_M4_WKB takes from
// 19 GB/s to 128 GB/s) — stock MMVQ puts 256 threads behind each of the 4096 outputs.
//   src0 = attn_v_b  Q8_0 [256, 128, 32]   (ne0 = K = 256, ne1 = 128 outputs, ne2 = 32 heads)
//   src1 = g_kqv     F32  [256, 1, 32]     (head c at x + c*256)
//   dst  = kqv_mla   F32  [128, 1, 32]     (head c at out + c*128)
//
// NUMERICS: both phases replicate the stock MMVQ path's Q8_1 quantisation of the F32
// activation (quantize_q8_1: scale amax/127, fp16 d, roundf) rather than an exact-fp32 dot —
// the same choice GGML_OP_M4_WKB documents (L57).  The reduction ASSOCIATION still differs
// (it is a tree over threads, not MMVQ's per-warp dp4a), so the token line is chaotic and the
// quality instrument is the gate (L57/L58) — never output equality.

// =========================================================================================
// PHASE 1 — wv_b: Q8_0 [256,128,32] x F32 [256,1,32] -> F32 [128,1,32]
// =========================================================================================
// One warp per output: lane k owns element k of every Q8_0 block, so each of the 8 blocks is
// read as one coalesced 32-byte transaction (the Q8_0 qs of a block are contiguous, but only
// 2-byte aligned because of the leading fp16 d — hence byte loads with a coalesced WARP
// footprint rather than vector loads per thread).
// block = 256 threads = 8 warps = 8 outputs of ONE head; grid = 32 heads x 16 groups.
static __global__ void m4_phase_wvb_kernel(const uint8_t * __restrict__ w,
                                           const float   * __restrict__ x,
                                           float         * __restrict__ out) {
    __shared__ int8_t xq[8][32];   // the head's 256-float input as 8 Q8_1 blocks
    __shared__ float  xd[8];

    const int c    = blockIdx.x >> 4;              // head 0..31
    const int grp  = blockIdx.x & 15;              // group of 8 outputs
    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;

    if (tid < 8) {                                 // stage the head's input, quantised
        const float * xb = x + c * 256 + tid * 32;
        float amax = 0.0f;
#pragma unroll
        for (int k = 0; k < 32; ++k) amax = fmaxf(amax, fabsf(xb[k]));
        const float d  = amax / 127.0f;
        const float id = d ? 1.0f / d : 0.0f;          // quantize_q8_1's form (a divide is slow)
        xd[tid] = __half2float(__float2half(d));
#pragma unroll
        for (int k = 0; k < 32; ++k) xq[tid][k] = (int8_t) (int) roundf(xb[k] * id);
    }
    __syncthreads();

    const int h = grp * 8 + warp;                  // output row 0..127
    const uint8_t * wrow = w + (size_t) (c * 128 + h) * 272;   // 8 blocks x 34 bytes

    float acc = 0.0f;
#pragma unroll
    for (int b = 0; b < 8; ++b) {
        const float wd = __half2float(*(const __half *) (wrow + b * 34));
        const int   q  = (int) (int8_t) wrow[b * 34 + 2 + lane];
        acc += wd * xd[b] * (float) (q * (int) xq[b][lane]);
    }
    acc = warp_reduce_sum(acc);
    if (lane == 0) out[c * 128 + h] = acc;
}

// =========================================================================================
// PHASE 0 — wkv_a gemv: Q8_0 [4096,320] x F32 [4096,1] -> F32 [320,1]
// =========================================================================================
// One block (128 threads = 4 warps) per output row; thread t owns Q8_0 block t of the row
// (34 bytes of weight) AND the 32 input floats it needs, so the input is quantised exactly
// once per (output, block) with no cross-thread dependency.
static __global__ void m4_phase_kvwrite_gemv(const float   * __restrict__ x,
                                             const uint8_t * __restrict__ w,
                                             float         * __restrict__ out) {
    const int o   = blockIdx.x;                    // output row 0..319
    const int tid = threadIdx.x;                   // Q8_0 block index 0..127

    const float   * xb = x + tid * 32;
    const uint8_t * wb = w + (size_t) o * (128 * 34) + (size_t) tid * 34;

    float amax = 0.0f;
#pragma unroll
    for (int k = 0; k < 32; ++k) amax = fmaxf(amax, fabsf(xb[k]));
    const float d   = amax / 127.0f;
    const float id  = d ? 1.0f / d : 0.0f;
    const float d16 = __half2float(__float2half(d));
    const float wd  = __half2float(*(const __half *) wb);
    const int8_t * q = (const int8_t *) (wb + 2);

    int sum = 0;
#pragma unroll
    for (int k = 0; k < 32; ++k) sum += (int) q[k] * (int) roundf(xb[k] * id);
    float acc = wd * d16 * (float) sum;

    acc = warp_reduce_sum(acc);
    __shared__ float sv[4];
    if ((tid & 31) == 0) sv[tid >> 5] = acc;
    __syncthreads();
    if (tid == 0) out[o] = sv[0] + sv[1] + sv[2] + sv[3];
}

// =========================================================================================
// PHASE 0 — the tail: attn_kv_a_norm + k_pe rope + the cache scatter.  ONE block (320
// threads = 10 full warps; warp t/32 holds rows [32t, 32t+32)).
// =========================================================================================
static __global__ void m4_phase_kvwrite_tail(float         * __restrict__ kcur,   // F32  [320]
                                             const float   * __restrict__ nw,     // F32  [256]
                                             __half        * __restrict__ kcache, // F16  [320, n_kv]
                                             const int64_t * __restrict__ kidx,   // I64  [1]
                                             const int32_t * __restrict__ posv,   // I32  [1]
                                             const float     eps,
                                             const float     freq_scale,
                                             const float     ext_factor,
                                             const float     attn_factor,
                                             const float     theta_scale,
                                             const float     corr0,
                                             const float     corr1) {
    const int tid = threadIdx.x;    // 0..320 (10 full warps)
    __shared__ float sred[10];

    // The cache row and the rope position are read ON DEVICE from the graph's input tensors: a
    // host read of `kidx->data[0]` in the launcher would be a read of DEVICE memory (it "works"
    // only on a card whose BAR maps VRAM into the process, and it is a silent-wrong scatter on
    // one that does not).
    const int64_t row = kidx[0];
    const int     pos = (int) posv[0];

    // ---- attn_kv_a_norm over rows [0,256) (rms_norm_f32 x mul: rsqrtf, then scale*x*w) ----
    float v = 0.0f, ss = 0.0f;
    if (tid < 256) { v = kcur[tid]; ss = v * v; }
    const float t = warp_reduce_sum(ss);
    if ((tid & 31) == 0) sred[tid >> 5] = t;
    __syncthreads();
    if (tid == 0) {
        float s = 0.0f;
        for (int w = 0; w < 10; ++w) s += sred[w];
        sred[0] = s;                       // serial over the 10 warp partials (order is OURS)
    }
    __syncthreads();
    const float scale = rsqrtf(sred[0] / 256.0f + eps);

    // ---- k_pe rope (rows [256,320) = 32 pairs; NORMAL mode, n_offs 0, no freq_factors) ----
    // This is ggml-cuda/rope.cu's rope_norm + rope_yarn, transcribed for the fixed geometry:
    //   theta_extrap = pos * theta_scale^p;  theta = freq_scale*theta_extrap (interp) mixed
    //   with theta_extrap by the YaRN ramp;  mscale = attn_factor * (1 + 0.1*log(1/freq_scale))
    //   cos/sin are multiplied by mscale; the pair is rotated (x0*c - x1*s, x0*s + x1*c).
    float x0 = 0.0f, x1 = 0.0f, co = 0.0f, si = 0.0f;
    const bool is_rope = (tid >= 256 && tid < 288);
    if (is_rope) {
        const int p = tid - 256;
        x0 = kcur[256 + 2 * p];
        x1 = kcur[257 + 2 * p];
        const float theta_extrap = (float) pos * powf(theta_scale, (float) p);
        float mscale = attn_factor;
        float theta  = freq_scale * theta_extrap;
        if (ext_factor != 0.0f) {
            // rope_yarn_ramp(low, high, i0): y = (i0/2 - low) / max(0.001, high - low)
            const float y = ((float) p - corr0) / fmaxf(0.001f, corr1 - corr0);
            const float ramp_mix = (1.0f - fminf(1.0f, fmaxf(0.0f, y))) * ext_factor;
            theta = theta * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
            mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
        }
        co = cosf(theta) * mscale;
        si = sinf(theta) * mscale;
    }

    // ---- write out: the normed rows [0,256), the roped rows [256,320), then the F16 row ----
    if (tid < 256) kcur[tid] = scale * v * nw[tid];
    if (is_rope) {
        kcur[256 + 2 * (tid - 256) + 0] = x0 * co - x1 * si;
        kcur[256 + 2 * (tid - 256) + 1] = x0 * si + x1 * co;
    }
    __syncthreads();
    kcache[row * 320 + tid] = __float2half(kcur[tid]);
}

// =========================================================================================
// the dispatcher
// =========================================================================================
void ggml_cuda_m4_phase(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int32_t phase = ggml_get_op_params_i32(dst, 0);
    cudaStream_t stream = ctx.stream();

    if (phase == 1) {   // ---- wv_b (kqv_mla) ----
        const ggml_tensor * w = dst->src[0];
        const ggml_tensor * x = dst->src[1];

        GGML_ASSERT(w->type == GGML_TYPE_Q8_0);
        GGML_ASSERT(w->ne[0] == 256 && w->ne[1] == 128 && w->ne[2] == 32 && w->ne[3] == 1);
        GGML_ASSERT(x->type == GGML_TYPE_F32);
        GGML_ASSERT(x->ne[0] == 256 && x->ne[1] == 1 && x->ne[2] == 32 && x->ne[3] == 1);
        GGML_ASSERT(x->nb[1] == 1024 && x->nb[2] == 1024);
        GGML_ASSERT(dst->type == GGML_TYPE_F32);
        GGML_ASSERT(dst->ne[0] == 128 && dst->ne[1] == 1 && dst->ne[2] == 32 && dst->ne[3] == 1);

        m4_phase_wvb_kernel<<<32 * 16, 256, 0, stream>>>(
            (const uint8_t *) w->data, (const float *) x->data, (float *) dst->data);
        return;
    }

    if (phase == 0) {   // ---- kvwrite ----
        const ggml_tensor * cur    = dst->src[0];
        const ggml_tensor * w      = dst->src[1];
        const ggml_tensor * nw     = dst->src[2];
        const ggml_tensor * kcache = dst->src[3];
        const ggml_tensor * kidx   = dst->src[4];
        const ggml_tensor * pos    = dst->src[5];

        GGML_ASSERT(cur->type == GGML_TYPE_F32);
        GGML_ASSERT(cur->ne[0] == 4096 && cur->ne[1] == 1 && cur->ne[2] == 1 && cur->ne[3] == 1);
        GGML_ASSERT(w->type == GGML_TYPE_Q8_0);
        GGML_ASSERT(w->ne[0] == 4096 && w->ne[1] == 320 && w->ne[2] == 1 && w->ne[3] == 1);
        GGML_ASSERT(nw->type == GGML_TYPE_F32 && nw->ne[0] == 256);
        GGML_ASSERT(kcache->type == GGML_TYPE_F16);
        GGML_ASSERT(kcache->ne[0] == 320 && kcache->ne[2] == 1 && kcache->ne[3] == 1);
        GGML_ASSERT(kidx->type == GGML_TYPE_I64 && kidx->ne[0] == 1);
        GGML_ASSERT(pos->type == GGML_TYPE_I32 && pos->ne[0] == 1);
        GGML_ASSERT(dst->type == GGML_TYPE_F32);
        GGML_ASSERT(dst->ne[0] == 320 && dst->ne[1] == 1 && dst->ne[2] == 1 && dst->ne[3] == 1);

        // op_params are an int32 ARRAY (the get/set helpers index it, they are not byte
        // offsets) — see the layout comment on ggml_m4_phase_kvwrite in ggml.c.
        const float freq_base   = ggml_get_op_params_f32(dst, 1);
        const float freq_scale  = ggml_get_op_params_f32(dst, 2);
        const float ext_factor  = ggml_get_op_params_f32(dst, 3);
        const float attn_factor = ggml_get_op_params_f32(dst, 4);
        const float beta_fast   = ggml_get_op_params_f32(dst, 5);
        const float beta_slow   = ggml_get_op_params_f32(dst, 6);
        const float eps         = ggml_get_op_params_f32(dst, 7);
        const int   n_dims      = ggml_get_op_params_i32(dst, 8);
        const int   n_ctx_orig  = ggml_get_op_params_i32(dst, 9);
        GGML_ASSERT(n_dims == 64);

        float corr_dims[2];
        ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims);
        const float theta_scale = powf(freq_base, -2.0f / (float) n_dims);

        m4_phase_kvwrite_gemv<<<320, 128, 0, stream>>>(
            (const float *) cur->data, (const uint8_t *) w->data, (float *) dst->data);
        // kidx/pos go to the DEVICE as pointers; the tail block reads them there.  Reading them
        // here (kidx->data[0]) would be a host read of device memory: correct only on a card
        // whose BAR maps VRAM into the process, a silent-wrong scatter everywhere else.
        m4_phase_kvwrite_tail<<<1, 320, 0, stream>>>(
            (float *) dst->data, (const float *) nw->data, (__half *) kcache->data,
            (const int64_t *) kidx->data, (const int32_t *) pos->data,
            eps, freq_scale, ext_factor, attn_factor, theta_scale,
            corr_dims[0], corr_dims[1]);
        return;
    }

    GGML_ABORT("GGML_OP_M4_PHASE: unknown phase %d", (int) phase);
}
