#include "m4-wkb.cuh"

// MISTRAL-SMALL-4-ONLY: the specialized MLA absorption gemv.  The shape, type and geometry are
// HARDCODED (the user directive — no generality, no fallback, no runtime dispatch):
//
//   wk_b      = Q8_0 [64, 256, 32]  (src0, attn_k_b.weight, 0.53 MiB)
//   q_nope    = F32  [64, 32, 1]    (src1, the nope half of q_b, head-stride 128 floats)
//   out       = F32  [256, 32, 1]   (dst, one 256-float block per head)
//
// 8192 outputs = 256 blocks x 32 threads; one thread per output; each thread does a 64-dot
// (2 Q8_0 blocks, 34 bytes each) against the head's 64-float input vector.  This is the exact
// kernel measured standalone at WARM 4.05 us / 128 GB/s (engine/wkb_bench.hip, H3-U3).
//
// NUMERICS: this MUST reproduce the stock MMVQ path, not the "exact fp32" dot.  The stock
// mul_mat for Q8_0 src0 quantizes the F32 src1 to Q8_1 (scale amax/127, int8 quants) and dots
// int8*int8 in int32 — see ggml-cuda/quantize.cu quantize_q8_1 and vecdotq.cuh
// vec_dot_q8_0_q8_1.  The exact-fp32 dot differs from that by ~amax/254 per element (~1e-3),
// which compounds through the softmax + 36 layers + KV cache into a broken token line; so we
// replicate the Q8_1 quantization bit-for-bit here (device roundf, __float2half, int32 sum).

static __device__ __forceinline__ float m4_wkb_q8_0_q8_1_dot(
        const uint8_t * __restrict__ wrow, const float * __restrict__ xc) {
    // weight block 0: fp16 d at +0, int8 qs at +2; block 1: d at +34, qs at +36
    const float wd0 = __half2float(*(const __half *) wrow);
    const float wd1 = __half2float(*(const __half *) (wrow + 34));
    const int8_t * q0 = (const int8_t *) (wrow + 2);
    const int8_t * q1 = (const int8_t *) (wrow + 36);

    float acc = 0.0f;
#pragma unroll
    for (int b = 0; b < 2; ++b) {
        const float * xb = xc + b * 32;
        const int8_t * qb = (b == 0) ? q0 : q1;
        const float wd = (b == 0) ? wd0 : wd1;

        float amax = 0.0f;
#pragma unroll
        for (int k = 0; k < 32; ++k) {
            amax = fmaxf(amax, fabsf(xb[k]));
        }

        const float d = amax / 127.0f;
        const float d16 = __half2float(__float2half(d));
        int sum = 0;
#pragma unroll
        for (int k = 0; k < 32; ++k) {
            const int q = (amax == 0.0f) ? 0 : (int) roundf(xb[k] / d);
            sum += (int) qb[k] * q;
        }
        acc += wd * d16 * (float) sum;
    }
    return acc;
}

static __global__ void m4_wkb_kernel(const uint8_t * __restrict__ w,
                                     const float * __restrict__ x,
                                     float * __restrict__ out) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;   // 0..8191
    const int c = tid >> 8;          // head 0..31
    const int h = tid & 255;         // output row 0..255

    // byte-explicit Q8_0 layout: 64 values = 2 blocks x 34 bytes (half d + 32 int8)
    const uint8_t * wrow = w + (size_t)(c * 256 + h) * 68;
    // the real input is the nope half of g_q [4096] = 32 heads x 128 floats; head c's
    // 64 nope values sit at x[c*128 + k] (stride 128, NOT contiguous).
    const float * xc = x + c * 128;

    out[tid] = m4_wkb_q8_0_q8_1_dot(wrow, xc);
}

void ggml_cuda_m4_wkb(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * w = dst->src[0];
    const ggml_tensor * x = dst->src[1];

    GGML_ASSERT(w->type == GGML_TYPE_Q8_0);
    GGML_ASSERT(w->ne[0] == 64 && w->ne[1] == 256 && w->ne[2] == 32 && w->ne[3] == 1);
    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(x->ne[0] == 64 && x->ne[1] == 32 && x->ne[2] == 1 && x->ne[3] == 1);
    GGML_ASSERT(x->nb[1] == 512);   // 128-float head stride (the engine's q_nope view)
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->ne[0] == 256 && dst->ne[1] == 32 && dst->ne[2] == 1 && dst->ne[3] == 1);

    const dim3 grid(256), block(32);
    m4_wkb_kernel<<<grid, block, 0, ctx.stream()>>>(
        (const uint8_t *) w->data, (const float *) x->data, (float *) dst->data);
}
