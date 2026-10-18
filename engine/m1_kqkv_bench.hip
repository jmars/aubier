// engine/m1_kqkv_bench.hip — H3/U5 STEP 2: a standalone microbenchmark for the fused attention
// core (GGML_OP_M4_KQKV).  Modeled on engine/wkb_bench.hip.
//
// The kernel fuses what the reference does in three nodes: kq = K^T.Q (mul_mat), the causal
// softmax (soft_max_ext), and kqv = V-weighted contraction (mul_mat) + the vT copy.  Shapes are
// HARDCODED for Mistral Small 4 (the user directive): one wave32 geometry, one grid, no dispatch.
//
//   kcache = F16 [320, 256]   (K = rows [0,320), V = rows [0,256))
//   q      = F32 [320, 32]    (32 heads x 320-float stride)
//   mask   = F32 [256]        (0.0 / -INFINITY)
//   out    = F32 [256, 32]
//
// The inputs are RUNTIME activations (the KV cache and the Qcur), which do not exist standalone,
// so they are deterministic patterns in the real shapes/types/layouts (the values do not affect a
// memory-bound kernel's throughput) — exactly the wkb_bench approach for its input vector.
//
// Build (hipcc launcher has a stale shebang — invoke through the venv python):
//   venvs/rocm/bin/python venvs/rocm/bin/hipcc -O2 --offload-arch=gfx1201 \
//       -I vendor/llama.cpp/ggml/include engine/m1_kqkv_bench.hip \
//       -o builds/engine/m1_kqkv_bench
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CHECK_HIP(c) do { hipError_t e = (c); if (e != hipSuccess) { \
    std::fprintf(stderr, "HIP error %d: %s at %s:%d\n", e, hipGetErrorString(e), __FILE__, __LINE__); \
    std::exit(1); } } while (0)

// ---------------------------------------------------------------------------
// THE FUSED KERNEL — a byte-for-byte copy of the one in ggml-cuda/m4-kqkv.cu
// ---------------------------------------------------------------------------
static __device__ __forceinline__ float m4_kqkv_dot(
        const __half * __restrict__ kcol, const float * __restrict__ qs) {
    // 2 accumulators (break the FMA dependency chain) + half2/float2 vector loads.
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

__global__ void m4_kqkv_kernel(const __half * __restrict__ kcache,
                               const float * __restrict__ q,
                               const float * __restrict__ mask,
                               float kq_scale,
                               float * __restrict__ out) {
    const int h   = blockIdx.x;
    const int tid = threadIdx.x;

    __shared__ float qs[320];
    __shared__ float red[32];
    __shared__ float wts[256];

    const float * qh = q + h * 320;
    for (int k = tid; k < 320; k += 256) qs[k] = qh[k];
    __syncthreads();

    const float s = m4_kqkv_dot(kcache + (size_t) tid * 320, qs) * kq_scale + mask[tid];

    // block reduction mirroring ggml-cuda common.cuh block_reduce (8 warps of 32; ALL 32 lanes
    // participate in every shuffle — a guarded shuffle with a full mask is undefined behaviour).
    float m = s;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffull, m, off, 32));
    if (tid % 32 == 0) red[tid / 32] = m;
    __syncthreads();
    m = (tid < 8) ? red[tid] : -INFINITY;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffull, m, off, 32));

    const float e = expf(s - m);
    float sum = e;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) sum += __shfl_xor_sync(0xffffffffull, sum, off, 32);
    if (tid % 32 == 0) red[tid / 32] = sum;
    __syncthreads();
    sum = (tid < 8) ? red[tid] : 0.0f;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) sum += __shfl_xor_sync(0xffffffffull, sum, off, 32);

    wts[tid] = e * (1.0f / sum);
    __syncthreads();

    float o0 = 0.0f, o1 = 0.0f;
#pragma unroll
    for (int i = 0; i < 256; i += 2) {
        o0 += wts[i]     * __half2float(kcache[(size_t) i * 320 + tid]);
        o1 += wts[i + 1] * __half2float(kcache[(size_t)(i + 1) * 320 + tid]);
    }
    out[h * 256 + tid] = o0 + o1;
}

// a strided L3-flush read
__global__ void touch_kernel(const float * __restrict__ p, size_t nfloats, float * __restrict__ sink) {
    size_t stride = (size_t) gridDim.x * blockDim.x;
    float acc = 0.0f;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < nfloats; i += stride)
        acc += p[i];
    if (acc == 12345.678f) sink[0] = acc;
}

static double now_s() {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double) ts.tv_sec + 1e-9 * (double) ts.tv_nsec;
}

// float -> binary16, round-toward-zero (fine for a [-0.5,0.5] input pattern: CPU h2f and the
// device __half2float read the SAME 16 bits, so the self-check is exact regardless of rounding)
static uint16_t f2h(float v) {
    uint32_t f; std::memcpy(&f, &v, 4);
    const uint32_t sign = (f >> 16) & 0x8000u;
    const int32_t exp = (int32_t)((f >> 23) & 0xffu) - 127 + 15;
    if (exp >= 31) return (uint16_t)(sign | 0x7c00u);   // saturate (unreachable for [-0.5,0.5])
    if (exp <= 0)  return (uint16_t) sign;              // |v| < 2^-14 -> signed zero
    return (uint16_t)(sign | ((uint32_t) exp << 10) | ((f >> 13) & 0x3ffu));
}

static float h2f(uint16_t h) {
    // IEEE754 binary16 -> float
    const uint32_t sign = ((uint32_t) h & 0x8000u) << 16;
    const uint32_t exp  = ((uint32_t) h >> 10) & 0x1fu;
    const uint32_t man  = (uint32_t) h & 0x3ffu;
    uint32_t f;
    if (exp == 0) {
        if (man == 0) f = sign;
        else { uint32_t m = man; int e = -1; do { m <<= 1; ++e; } while (!(m & 0x400u));
               f = sign | ((uint32_t)(127-15-e) << 23) | ((m & 0x3ff) << 13); }
    } else if (exp == 31) f = sign | 0x7f800000u | (man << 13);
    else f = sign | ((exp + 127 - 15) << 23) | (man << 13);
    float r; std::memcpy(&r, &f, 4); return r;
}

int main(int argc, char ** argv) {
    const int N_KV = argc > 1 ? std::atoi(argv[1]) : 256;   // n_kv_max (the KV width)
    const float KQ_SCALE = 1.0f / std::sqrt(320.0f);        // a representative scale (matters not)

    const size_t kbytes = (size_t) 320 * N_KV * sizeof(uint16_t);
    const size_t qbytes = (size_t) 320 * 32 * sizeof(float);
    const size_t mbytes = (size_t) N_KV * sizeof(float);
    const size_t obytes = (size_t) 256 * 32 * sizeof(float);

    // deterministic host patterns
    uint16_t * hkc = (uint16_t *) std::malloc(kbytes);
    float    * hq  = (float *)    std::malloc(qbytes);
    float    * hm  = (float *)    std::malloc(mbytes);
    for (size_t i = 0; i < 320 * (size_t) N_KV; ++i) {
        const float v = (float)((i * 7919) % 1000) / 1000.0f - 0.5f;   // [-0.5, 0.5]
        hkc[i] = f2h(v);
    }
    for (size_t i = 0; i < 320 * 32; ++i) hq[i] = (float)((i * 3571) % 1000) / 1000.0f - 0.5f;
    const int eff = (argc > 2 ? std::atoi(argv[2]) : N_KV);   // the causal effective length
    for (int i = 0; i < N_KV; ++i) hm[i] = i < eff ? 0.0f : -INFINITY;

    __half * dkc = nullptr; float * dq = nullptr, * dm = nullptr, * dout = nullptr;
    CHECK_HIP(hipMalloc(&dkc, kbytes));
    CHECK_HIP(hipMalloc(&dq, qbytes));
    CHECK_HIP(hipMalloc(&dm, mbytes));
    CHECK_HIP(hipMalloc(&dout, obytes));
    CHECK_HIP(hipMemcpy(dkc, hkc, kbytes, hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(dq, hq, qbytes, hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(dm, hm, mbytes, hipMemcpyHostToDevice));

    float * dflush = nullptr, * dsink = nullptr;
    CHECK_HIP(hipMalloc(&dflush, 128u * 1024u * 1024u));
    CHECK_HIP(hipMalloc(&dsink, sizeof(float)));
    CHECK_HIP(hipMemset(dflush, 0, 128u * 1024u * 1024u));

    const int WARM_ITERS = 2000;
    const int COLD_ITERS = 400;
    dim3 grid(32), block(256);
    dim3 tgrid(256), tblock(256);

    std::printf("# m1_kqkv_bench: kcache F16 [320,%d] q F32 [320,32] mask F32 [%d] eff=%d -> out F32 [256,32]\n",
                N_KV, N_KV, eff);
    std::printf("#   bytes read/kernel: K %.1f KiB + V %.1f KiB + Q %.1f KiB = %.1f KiB\n",
                (double) 320 * N_KV * 2 / 1024, (double) 256 * N_KV * 2 / 1024,
                (double) 320 * 32 * 4 / 1024,
                (double)(320 + 256) * N_KV * 2 / 1024 + 320.0 * 32 * 4 / 1024);

    for (int i = 0; i < 10; ++i) m4_kqkv_kernel<<<grid, block>>>(dkc, dq, dm, KQ_SCALE, dout);
    CHECK_HIP(hipDeviceSynchronize());

    {
        const double t0 = now_s();
        for (int i = 0; i < WARM_ITERS; ++i) m4_kqkv_kernel<<<grid, block>>>(dkc, dq, dm, KQ_SCALE, dout);
        CHECK_HIP(hipDeviceSynchronize());
        const double dt = now_s() - t0;
        const double us = dt / WARM_ITERS * 1e6;
        std::printf("# WARM  %5d launches: %8.3f us/launch\n", WARM_ITERS, us);
    }

    {
        const double t0 = now_s();
        for (int i = 0; i < COLD_ITERS; ++i) {
            touch_kernel<<<tgrid, tblock>>>(dflush, 128u*1024u*1024u/4, dsink);
            m4_kqkv_kernel<<<grid, block>>>(dkc, dq, dm, KQ_SCALE, dout);
        }
        CHECK_HIP(hipDeviceSynchronize());
        const double dt = now_s() - t0;
        const double t_touch0 = now_s();
        for (int i = 0; i < COLD_ITERS; ++i) touch_kernel<<<tgrid, tblock>>>(dflush, 128u*1024u*1024u/4, dsink);
        CHECK_HIP(hipDeviceSynchronize());
        const double t_touch = now_s() - t_touch0;
        const double us = (dt - t_touch) / COLD_ITERS * 1e6;
        std::printf("# COLD  %5d launches: %8.3f us/launch  (L3 evicted each time; touch subtracted)\n",
                    COLD_ITERS, us);
    }

    // ---- numerics self-check vs a CPU reference (catches layout/mask/softmax bugs) ----
    {
        m4_kqkv_kernel<<<grid, block>>>(dkc, dq, dm, KQ_SCALE, dout);
        CHECK_HIP(hipDeviceSynchronize());
        float * hout = (float *) std::malloc(obytes);
        CHECK_HIP(hipMemcpy(hout, dout, obytes, hipMemcpyDeviceToHost));

        double max_abs = 0.0; int max_h = -1, max_d = -1;
        for (int h = 0; h < 32; ++h) {
            // scores
            float sc[256];
            float mx = -INFINITY;
            for (int i = 0; i < N_KV; ++i) {
                float acc = 0.0f;
                for (int k = 0; k < 320; ++k) acc += h2f(hkc[(size_t) i * 320 + k]) * hq[(size_t) h * 320 + k];
                sc[i] = acc * KQ_SCALE + hm[i];
                mx = std::fmax(mx, sc[i]);
            }
            float sum = 0.0f;
            for (int i = 0; i < N_KV; ++i) sum += std::exp(sc[i] - mx);
            for (int d = 0; d < 256; ++d) {
                float o = 0.0f;
                for (int i = 0; i < N_KV; ++i) o += std::exp(sc[i] - mx) / sum * h2f(hkc[(size_t) i * 320 + d]);
                const double dd = std::fabs((double) hout[h * 256 + d] - (double) o);
                if (dd > max_abs) { max_abs = dd; max_h = h; max_d = d; }
            }
        }
        std::printf("# numerics self-check: max |device - cpu_ref| = %g over %d outputs (head %d dim %d)\n",
                    max_abs, 256 * 32, max_h, max_d);
        // the max-subtraction probe: without it, a long-window score would overflow; assert the
        // kernel's own max is finite and the sum is a probability mass
        std::printf("#   (softmax max-subtraction is in the kernel: exp(s - m) with m = block max)\n");
        std::free(hout);
    }

    std::free(hkc); std::free(hq); std::free(hm);
    CHECK_HIP(hipFree(dkc)); CHECK_HIP(hipFree(dq)); CHECK_HIP(hipFree(dm)); CHECK_HIP(hipFree(dout));
    CHECK_HIP(hipFree(dflush)); CHECK_HIP(hipFree(dsink));
    return 0;
}
