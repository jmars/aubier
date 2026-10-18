#include "m4-kqkv.cuh"

// MISTRAL-SMALL-4-ONLY: the fused attention core (K^T.Q scores + causal softmax + V-weighted
// contraction).  The HEAD GEOMETRY is HARDCODED (the user directive — no generality, no
// fallback, no runtime shape dispatch), but the SEQUENCE LENGTH IS NOT: the reduction bound is
// `n_kv = kcache->ne[1]` (the actual window), so the SAME op serves a 256-key decode window and
// a full 262 144-key context.  The causal mask (already in the `mask` tensor: 0.0 / -INFINITY)
// carries the effective length, so `eff_len` never appears as a constant — a shorter-than-window
// prefix simply masks every key past it, and the masked cells contribute exact zeros.
//
//   kcache = F16 [320, n_kv_max]  (src0; K = columns [0,320), V = columns [0,256) of the SAME row)
//   q      = F32 [320, 32, 1]     (src1; the engine's g_qcur, head stride 320 floats)
//   mask   = F32 [n_kv_max, 1]    (src2; 0.0 / -INFINITY, the causal mask)
//   out    = F32 [256, 1, 32]     (dst; one 256-float block per head)
//
// GEOMETRY: block = 256 threads; thread tid is BOTH the score lane for KV position (base + tid)
// AND the output dim (tid) for the contraction.  TILE SIZE = 256 = block size (one key per thread
// per tile) is a TUNING constant, not a semantic one.  Any window length is legal: the last tile
// is always a partial handled by an `n_tile` mask.
//
// THE KV CACHE IS SHARED BY ALL 32 HEADS (that is what MLA's compressed latent *is* — one F16
// row of 320 halfs per token serves every head), so the same bytes are consumed 32 times per
// token.  Two things follow, and both are in this file:
//
//  1. SPLIT-K (H3-U7).  A single block per head left 32 blocks on 28 workgroup slots working
//     through tiles SERIALLY, and the kernel was latency-bound at ~6.2 GiB/s regardless of the
//     window (H3-U6 measured it linear in n_kv: 48.6 ms/layer at 262 144 keys).  The reduction is
//     now split across workgroups: pass 1 produces one partial `(m, l, o)` per (head, split) with
//     the SAME online-softmax loop, pass 2 MERGES the splits with the same online-softmax rule
//     (m* = max m_i; l* = sum l_i*exp(m_i-m*); o* = sum o_i*exp(m_i-m*); out = o*/l*), which is
//     exact up to fp association.  The split count is a FUNCTION OF THE WINDOW, never a constant
//     that assumes one: `splits = min(M4_KQKV_MAX_SPLITS, n_tiles)`.
//  2. HEADS PER BLOCK (H3-U7).  Because one key row feeds every head, a block that carries FOUR
//     heads reads each 16-byte K chunk once and uses it four times, and each V element once and
//     uses it four times — the 32x fetch amplification becomes 8x.  It is bit-identical to the
//     one-head form: the per-head dot keeps the same element order and the same accumulator
//     grouping.
//
// KERNELS ARE KEPT SEPARATE (H3-U6's lesson): compiling the tile loop into the flat kernel cost
// ~300 VGPR spills and regressed the product's 256-key decode path.  `m4_kqkv_kernel_flat` is the
// untouched L62 geometry for the decode width (n_kv <= 256) and the split path serves every wider
// window.  Spill counts are checked on every build (see results/m1-h3-u7.md §4).

#define M4_KQKV_TILE 256
// The measured knee of the split sweep on this card (RX 9070, 28 workgroup slots): past it every
// extra split is an empty block plus a longer merge.  A TUNING constant, not a semantic one.
#define M4_KQKV_MAX_SPLITS 28
// Heads per block.  4 measured faster than 2 and 8 at 8192 / 65536 / 262 144 keys.
#define M4_KQKV_HEADS 4
#define M4_KQKV_HGROUPS (32 / M4_KQKV_HEADS)
// one partial record per (head, split): m, l, then 256 output-dim accumulators
#define M4_PART_STRIDE (2 + 256)

// The merge pass needs somewhere to put its partials.  A STATIC DEVICE ARRAY, not an allocation:
// the op must stay capture-safe (a captured graph cannot allocate).  Consequence: the op assumes
// the engine's single-stream use — it is not re-entrant across two concurrent streams.
static __device__ float g_m4_kqkv_part[32 * M4_KQKV_MAX_SPLITS * M4_PART_STRIDE];

// The address the kernels get.  Resolved with hipGetSymbolAddress, NOT by naming the symbol in
// host code: on this toolchain a `static __device__` array named as a host expression resolves to
// a BOGUS address and the kernel faults (measured: a hang inside hipDeviceSynchronize, while the
// same kernel fed a hipGetSymbolAddress pointer runs correctly).  Cached, so the steady state is
// one load and no API call — the op stays capture-safe (no allocation, no stream operation).
static float * m4_kqkv_part_ptr() {
    static float * p = nullptr;
    if (p == nullptr) {
        void * q = nullptr;
        CUDA_CHECK(hipGetSymbolAddress(&q, HIP_SYMBOL(g_m4_kqkv_part)));
        p = (float *) q;
    }
    return p;
}

// NUMERICS: this is a numerics-CHANGING fusion (the reduction association of both dots differs
// from the stock mul_mat's tiled/SIMD order, and online softmax accumulates the denominator
// instead of a single-pass normalise), so it is gated by the QUALITY instrument (--score +
// scripts/nll_eval.py), never by byte-identity — the token line is chaotic under a 1-ulp-class
// perturbation (measured: a one-ulp change in one kernel reaches O(1) in the logits by position
// 100).  The softmax is numerically stable: the running max is subtracted
// before every exp, so a long window cannot overflow (max-subtraction is mandatory, not
// cosmetic — a kernel that overflows on a long window is a latent bug a short run cannot expose).

// ---- the L62 dot: 4-byte half2 loads, two accumulators.  Used ONLY by the flat decode path,
// whose register budget (0 spills) is what keeps the 256-key width at L62 speed. ----
static __device__ __forceinline__ float m4_kqkv_dot(
        const half * __restrict__ kcol, const float * __restrict__ qs) {
    float acc0 = 0.0f, acc1 = 0.0f;
    const __half2 * kc2 = (const __half2 *) kcol;
    const float2  * qs2 = (const float2 *) qs;
#pragma unroll
    for (int k = 0; k < 160; k += 2) {
        const half2  a = kc2[k];
        const half2  b = kc2[k + 1];
        const float2 x = qs2[k];
        const float2 y = qs2[k + 1];
        acc0 += __half2float(a.x) * x.x + __half2float(a.y) * x.y;
        acc1 += __half2float(b.x) * y.x + __half2float(b.y) * y.y;
    }
    return acc0 + acc1;
}

typedef union { uint4 u; half2 h[4]; } m4_u4h;

// ---- the 16-byte dot: same arithmetic, one 16-byte load per 8 dims.  A lane's 640-byte key row
// sits in a different 128-byte line from its 31 neighbours either way, so the wavefront count
// follows the load WIDTH — 4x fewer instructions and 4x fewer wavefronts than the half2 form for
// the same bytes.  That is the difference between this kernel and H3-U6's. ----
static __device__ __forceinline__ float m4_kqkv_dot_v(
        const half * __restrict__ kcol, const float * __restrict__ qs) {
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const uint4  * r  = (const uint4 *)  kcol;   // 40 x 16 B = 640 B = 320 halfs
    const float2 * q2 = (const float2 *) qs;     // 160 x 8 B
#pragma unroll
    for (int i = 0; i < 40; ++i) {
        m4_u4h cv;
        cv.u = r[i];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float2 kf = __half22float2(cv.h[j]);
            const float2 qf = q2[i * 4 + j];
            acc[j] += kf.x * qf.x + kf.y * qf.y;
        }
    }
    return (acc[0] + acc[1]) + (acc[2] + acc[3]);
}

// Contract one tile: out[d = tid] contribution sum_j wts[j] * V[base+j, d], and ls = the tile's
// sum of wts (every thread computes the SAME sum in the SAME order, so the running denominator
// stays bit-identical across threads — the final division uses one shared denominator).  The
// full-tile arm is the L62 geometry and is fully unrolled; the partial arm handles the last
// (possibly odd-length) tile.
static __device__ __forceinline__ float m4_kqkv_contract(
        const half * __restrict__ kcache, const float * wts,
        const int base, const int n_tile, const int tid, float & ls) {
    float o0 = 0.0f, o1 = 0.0f;
    ls = 0.0f;
    if (n_tile == M4_KQKV_TILE) {
#pragma unroll
        for (int j = 0; j < M4_KQKV_TILE; j += 2) {
            const float w0 = wts[j], w1 = wts[j + 1];
            o0 += w0 * __half2float(kcache[(size_t)(base + j)     * 320 + tid]);
            o1 += w1 * __half2float(kcache[(size_t)(base + j + 1) * 320 + tid]);
            ls += w0 + w1;
        }
    } else {
        int j = 0;
        for (; j + 1 < n_tile; j += 2) {
            const float w0 = wts[j], w1 = wts[j + 1];
            o0 += w0 * __half2float(kcache[(size_t)(base + j)     * 320 + tid]);
            o1 += w1 * __half2float(kcache[(size_t)(base + j + 1) * 320 + tid]);
            ls += w0 + w1;
        }
        if (j < n_tile) {   // odd tail (the last partial tile's length need not be even)
            const float w0 = wts[j];
            o1 += w0 * __half2float(kcache[(size_t)(base + j) * 320 + tid]);
            ls += w0;
        }
    }
    return o0 + o1;
}

// ---- the FLAT kernel (n_kv <= 256, the decode width): the exact L62 geometry, untouched. ----
// Online softmax over ONE tile degenerates to single-pass (no rescale), and the contraction is
// fully unrolled with a literal base 0.  This kernel must NOT regress L62's 256-key speed, so it
// is kept register-lean: it is the ONLY kernel here that uses the half2 dot, because the 16-byte
// dot's 40-deep unrolled load chain does not fit in its register budget alongside the unrolled
// contraction.
static __global__ void m4_kqkv_kernel_flat(const half * __restrict__ kcache,
                                           const float * __restrict__ q,
                                           const float * __restrict__ mask,
                                           float kq_scale,
                                           const int n_kv,
                                           float * __restrict__ out) {
    const int h   = blockIdx.x;
    const int tid = threadIdx.x;

    __shared__ float qs[320];
    __shared__ float red[32];
    __shared__ float wts[M4_KQKV_TILE];

    const float * qh = q + h * 320;
    for (int k = tid; k < 320; k += 256) {
        qs[k] = qh[k];
    }
    __syncthreads();

    const int n_tile = n_kv;   // <= M4_KQKV_TILE (the launcher guarantees it)
    const float s = (tid < n_tile)
        ? m4_kqkv_dot(kcache + (size_t) tid * 320, qs) * kq_scale + mask[tid]
        : -INFINITY;

    const float m = block_reduce<block_reduce_method::MAX>(s, red);
    const float e = (s > -INFINITY) ? expf(s - m) : 0.0f;
    wts[tid] = e;
    __syncthreads();

    float ls;
    const float o = m4_kqkv_contract(kcache, wts, 0, n_tile, tid, ls);
    out[h * 256 + tid] = o / ls;
}

// ---- SPLIT-K pass 1: one block per (head group, split), M4_KQKV_HEADS heads carried together. ----
// The tile range is a BALANCED partition of the window's tiles (equal counts, the remainder given
// to the low splits), so a window of n_tiles >= splits uses every workgroup exactly once.
static __global__ void m4_kqkv_kernel_split_partial(const half * __restrict__ kcache,
                                                    const float * __restrict__ q,
                                                    const float * __restrict__ mask,
                                                    float kq_scale,
                                                    const int n_kv,
                                                    const int splits,
                                                    float * __restrict__ part) {
    const int hg   = blockIdx.x;   // head group (M4_KQKV_HEADS heads)
    const int sp   = blockIdx.y;   // split index
    const int tid  = threadIdx.x;
    const int n_tiles = (n_kv + M4_KQKV_TILE - 1) / M4_KQKV_TILE;
    const int base_t  = n_tiles / splits;
    const int rem     = n_tiles % splits;
    const int t0      = sp * base_t + min(sp, rem);
    const int t1      = t0 + base_t + (sp < rem ? 1 : 0);

    __shared__ float qs[M4_KQKV_HEADS][320];
    __shared__ float red[M4_KQKV_HEADS][32];
    __shared__ float wts[M4_KQKV_HEADS][M4_KQKV_TILE];

    for (int hh = 0; hh < M4_KQKV_HEADS; ++hh)
        for (int k = tid; k < 320; k += 256) qs[hh][k] = q[(size_t)(hg * M4_KQKV_HEADS + hh) * 320 + k];
    __syncthreads();

    float m_prev[M4_KQKV_HEADS], l[M4_KQKV_HEADS], o[M4_KQKV_HEADS];
#pragma unroll
    for (int hh = 0; hh < M4_KQKV_HEADS; ++hh) { m_prev[hh] = -INFINITY; l[hh] = 0.0f; o[hh] = 0.0f; }

    for (int t = t0; t < t1; ++t) {
        const int base = t * M4_KQKV_TILE;
        const int n_tile = min(M4_KQKV_TILE, n_kv - base);
        const bool have = (n_tile == M4_KQKV_TILE) || (tid < n_tile);

        // ---- scores: ONE 16-byte K load per 8 dims, used by ALL M4_KQKV_HEADS heads ----
        float dacc[M4_KQKV_HEADS][2];
#pragma unroll
        for (int hh = 0; hh < M4_KQKV_HEADS; ++hh) { dacc[hh][0] = 0.0f; dacc[hh][1] = 0.0f; }
        if (have) {
            const uint4 * r = (const uint4 *)(kcache + (size_t)(base + tid) * 320);
#pragma unroll
            for (int i = 0; i < 40; ++i) {
                m4_u4h cv;
                cv.u = r[i];
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const float2 kf = __half22float2(cv.h[j]);
#pragma unroll
                    for (int hh = 0; hh < M4_KQKV_HEADS; ++hh) {
                        const float * qq = qs[hh] + i * 8 + j * 2;
                        dacc[hh][j & 1] += kf.x * qq[0] + kf.y * qq[1];
                    }
                }
            }
        }
        float s[M4_KQKV_HEADS];
#pragma unroll
        for (int hh = 0; hh < M4_KQKV_HEADS; ++hh) {
            const float d = dacc[hh][0] + dacc[hh][1];
            s[hh] = have ? d * kq_scale + mask[base + tid] : -INFINITY;
            // block max per head — each head owns its own red[] row, so no cross-head race and no
            // extra barrier inside the loop
            float v = s[hh];
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffull, v, off, 32));
            if (tid % 32 == 0) red[hh][tid / 32] = v;
        }
        __syncthreads();
#pragma unroll
        for (int hh = 0; hh < M4_KQKV_HEADS; ++hh) {
            float v = (tid % 32 < 8) ? red[hh][tid % 32] : -INFINITY;
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffull, v, off, 32));
            // Rescale only when the max advances.  The guard skips expf(-inf - -inf) == NaN, which
            // an all-masked tile following an all-masked prefix would produce — impossible for the
            // causal mask (key 0 is always valid), but it keeps the kernel correct for ANY mask.
            if (v > m_prev[hh]) {
                const float alpha = expf(m_prev[hh] - v);
                o[hh] *= alpha;
                l[hh] *= alpha;
                m_prev[hh] = v;
            }
            wts[hh][tid] = (s[hh] > -INFINITY) ? expf(s[hh] - m_prev[hh]) : 0.0f;
        }
        __syncthreads();

        // ---- contraction: ONE 2-byte V load per key, used by ALL M4_KQKV_HEADS heads ----
        const int nj = (n_tile + 1) & ~1;
        for (int j = 0; j < nj; j += 2) {
            const bool odd = (j + 1 < n_tile);
            const float v0 = __half2float(kcache[(size_t)(base + j) * 320 + tid]);
            const float v1 = odd ? __half2float(kcache[(size_t)(base + j + 1) * 320 + tid]) : 0.0f;
#pragma unroll
            for (int hh = 0; hh < M4_KQKV_HEADS; ++hh) {
                const float w0 = wts[hh][j];
                const float w1 = odd ? wts[hh][j + 1] : 0.0f;
                o[hh] += w0 * v0 + w1 * v1;
                l[hh] += w0 + w1;
            }
        }
        __syncthreads();   // wts is overwritten by the next tile
    }

#pragma unroll
    for (int hh = 0; hh < M4_KQKV_HEADS; ++hh) {
        float * ph = part + (size_t)((hg * M4_KQKV_HEADS + hh) * splits + sp) * M4_PART_STRIDE;
        if (tid == 0) { ph[0] = m_prev[hh]; ph[1] = l[hh]; }
        ph[2 + tid] = o[hh];
    }
}

// ---- SPLIT-K pass 2: merge the splits per head with the online-softmax rule. ----
// m* = max_i m_i ; l* = sum_i l_i*exp(m_i - m*) ; o* = sum_i o_i*exp(m_i - m*) ; out = o*/l*.
// Every thread walks the same split order with the same arithmetic, so all 256 output dims share
// one denominator exactly as the single-block kernel did.
static __global__ void m4_kqkv_kernel_split_combine(const float * __restrict__ part,
                                                    const int splits,
                                                    float * __restrict__ out) {
    const int h   = blockIdx.x;
    const int tid = threadIdx.x;
    const float * ph = part + (size_t) h * splits * M4_PART_STRIDE;

    float m_star = -INFINITY;
    for (int s = 0; s < splits; ++s) m_star = fmaxf(m_star, ph[s * M4_PART_STRIDE + 0]);

    float l_star = 0.0f, o_star = 0.0f;
    for (int s = 0; s < splits; ++s) {
        const float m_s = ph[s * M4_PART_STRIDE + 0];
        if (m_s > -INFINITY) {
            const float a = expf(m_s - m_star);
            l_star += ph[s * M4_PART_STRIDE + 1] * a;
            o_star += ph[s * M4_PART_STRIDE + 2 + tid] * a;
        }
    }
    out[h * 256 + tid] = (m_star > -INFINITY) ? o_star / l_star : 0.0f;
}

void ggml_cuda_m4_kqkv(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * kcache = dst->src[0];
    const ggml_tensor * q      = dst->src[1];
    const ggml_tensor * mask   = dst->src[2];

    GGML_ASSERT(kcache->type == GGML_TYPE_F16);
    // Head geometry pinned (Mistral Small 4); the KV width is DYNAMIC — the reduction's bound is
    // kcache->ne[1] and the mask carries the effective length, so a wider cache is honoured, not
    // silently sliced to a 256 window.
    GGML_ASSERT(kcache->ne[0] == 320 && kcache->ne[1] >= 1 && kcache->ne[2] == 1 && kcache->ne[3] == 1);
    GGML_ASSERT(q->type == GGML_TYPE_F32);
    GGML_ASSERT(q->ne[0] == 320 && q->ne[1] == 32 && q->ne[2] == 1 && q->ne[3] == 1);
    GGML_ASSERT(mask->type == GGML_TYPE_F32);
    GGML_ASSERT(mask->ne[0] == kcache->ne[1] && mask->ne[1] == 1 && mask->ne[2] == 1 && mask->ne[3] == 1);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->ne[0] == 256 && dst->ne[1] == 1 && dst->ne[2] == 32 && dst->ne[3] == 1);

    float kq_scale = 1.0f;
    memcpy(&kq_scale, (const float *) dst->op_params + 0, sizeof(float));

    const int n_kv = (int) kcache->ne[1];
    const dim3 block(256);

    if (n_kv <= M4_KQKV_TILE) {
        // the decode width: the single-tile flat path, one block per head (the L62 geometry)
        m4_kqkv_kernel_flat<<<dim3(32), block, 0, ctx.stream()>>>(
            (const half *) kcache->data, (const float *) q->data,
            (const float *) mask->data, kq_scale, n_kv, (float *) dst->data);
    } else {
        // splits: a function of the WINDOW — one split per tile until the workgroup slots run out
        const int n_tiles = (n_kv + M4_KQKV_TILE - 1) / M4_KQKV_TILE;
        const int splits  = n_tiles < M4_KQKV_MAX_SPLITS ? n_tiles : M4_KQKV_MAX_SPLITS;
        m4_kqkv_kernel_split_partial<<<dim3(M4_KQKV_HGROUPS, splits), block, 0, ctx.stream()>>>(
            (const half *) kcache->data, (const float *) q->data,
            (const float *) mask->data, kq_scale, n_kv, splits, m4_kqkv_part_ptr());
        m4_kqkv_kernel_split_combine<<<dim3(32), block, 0, ctx.stream()>>>(
            m4_kqkv_part_ptr(), splits, (float *) dst->data);
    }
}
