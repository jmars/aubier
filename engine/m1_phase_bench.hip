// engine/m1_phase_bench.hip — H3/U11: the standalone microbenchmark for the TWO fused attention
// phases (the gate instrument: a kernel is kept only if it clears >= 1.5x on its OWN phase here
// AND the in-situ core bracket falls).
//
//   PHASE 0 kvwrite : wkv_a_mqa Q8_0 [4096,320] x cur F32 [4096] -> 320 outputs, then in ONE
//                     small block: attn_kv_a_norm over rows [0,256), the k_pe rope over rows
//                     [256,320) (n_dims 64, NORMAL mode, YaRN) and the F16 scatter into the
//                     KV cache row.  TWO launches, exactly as the shipped op issues them.
//   PHASE 1 wvb     : attn_v_b Q8_0 [256,128,32] x g_kqv F32 [256,1,32] -> F32 [128,1,32].
//
// The kernels are COPIES of the shipped ones (ggml-cuda/m4-phase.cu) — the shared-memory
// reduction shapes differ slightly (the shipped file uses warp_reduce_sum), which makes this rig
// marginally pessimistic and changes only float association.  Shapes/types/geometry are the
// model's, hardcoded.  The weights are the REAL bytes from blk.0 (gguf.h); the activations do not
// exist standalone, so they are a deterministic pattern (a Q8_0 gemv's throughput does not depend
// on them).
//
// Build:
//   venvs/rocm/bin/python venvs/rocm/bin/hipcc -O2 --offload-arch=gfx1201 \
//       -I vendor/llama.cpp/ggml/include engine/m1_phase_bench.hip \
//       -L builds/hip/bin -lggml-base -Wl,-rpath,builds/hip/bin -o builds/engine/m1_phase_bench
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <gguf.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define CHECK_HIP(c) do { hipError_t e = (c); if (e != hipSuccess) { \
    std::fprintf(stderr, "HIP error %d: %s at %s:%d\n", e, hipGetErrorString(e), __FILE__, __LINE__); \
    std::exit(1); } } while (0)

static double now_s() { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
                        return (double) ts.tv_sec + 1e-9 * (double) ts.tv_nsec; }

// =========================================================================================
// PHASE 1 — wv_b: one warp per output, lane k owns element k of every Q8_0 block.
// =========================================================================================
static __global__ void phase_wvb_kernel(const uint8_t * __restrict__ w,
                                        const float   * __restrict__ x,
                                        float         * __restrict__ out) {
    __shared__ int8_t xq[8][32];
    __shared__ float  xd[8];
    __shared__ float  red[8][32];

    const int c    = blockIdx.x >> 4;              // head 0..31
    const int grp  = blockIdx.x & 15;              // group of 8 outputs
    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;

    if (tid < 8) {
        const float * xb = x + c * 256 + tid * 32;
        float amax = 0.0f;
#pragma unroll
        for (int k = 0; k < 32; ++k) amax = fmaxf(amax, fabsf(xb[k]));
        const float d = amax / 127.0f;
        const float id = d ? 1.0f / d : 0.0f;
        xd[tid] = __half2float(__float2half(d));
#pragma unroll
        for (int k = 0; k < 32; ++k) xq[tid][k] = (int8_t) (int) roundf(xb[k] * id);
    }
    __syncthreads();

    const int h = grp * 8 + warp;
    const uint8_t * wrow = w + (size_t) (c * 128 + h) * 272;
    float acc = 0.0f;
#pragma unroll
    for (int b = 0; b < 8; ++b) {
        const float wd = __half2float(*(const __half *) (wrow + b * 34));
        const int   q  = (int) (int8_t) wrow[b * 34 + 2 + lane];
        acc += wd * xd[b] * (float) (q * (int) xq[b][lane]);
    }
    red[warp][lane] = acc;
    __syncthreads();
    if (lane == 0) {
        float s = 0.0f;
        for (int k = 0; k < 32; ++k) s += red[warp][k];
        out[c * 128 + h] = s;
    }
}

// =========================================================================================
// PHASE 0a — wkv_a gemv: one block (128 threads) per output row, thread t owns Q8_0 block t.
// =========================================================================================
static __global__ void phase_kvwrite_gemv(const float   * __restrict__ x,
                                          const uint8_t * __restrict__ w,
                                          float         * __restrict__ out) {
    const int o   = blockIdx.x;
    const int tid = threadIdx.x;
    __shared__ float sv[128];

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
    sv[tid] = wd * d16 * (float) sum;
    __syncthreads();
    for (int s = 64; s > 0; s >>= 1) {
        if (tid < s) sv[tid] += sv[tid + s];
        __syncthreads();
    }
    if (tid == 0) out[o] = sv[0];
}

// =========================================================================================
// PHASE 0b — the tail: norm rows [0,256), rope rows [256,320), scatter the F16 cache row.
// One block of 320 threads (10 full warps).
// =========================================================================================
static __global__ void phase_kvwrite_tail(float       * __restrict__ kcur,
                                          const float * __restrict__ nw,
                                          __half      * __restrict__ kcache,
                                          const int64_t row, const int pos, const float eps,
                                          const float freq_scale, const float ext_factor,
                                          const float attn_factor, const float theta_scale,
                                          const float corr0, const float corr1) {
    const int tid = threadIdx.x;

    float v = 0.0f, ss = 0.0f;
    if (tid < 256) { v = kcur[tid]; ss = v * v; }
    // a shared-memory tree (no warp intrinsics: ROCm requires a 64-bit shuffle mask here)
    __shared__ float ssum[320];
    ssum[tid] = ss;
    __syncthreads();
    for (int s = 256; s > 0; s >>= 1) {
        if (tid < s) ssum[tid] += ssum[tid + s];
        __syncthreads();
    }
    const float scale = rsqrtf(ssum[0] / 256.0f + eps);

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
            const float y = ((float) p - corr0) / fmaxf(0.001f, corr1 - corr0);
            const float ramp_mix = (1.0f - fminf(1.0f, fmaxf(0.0f, y))) * ext_factor;
            theta = theta * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
            mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
        }
        co = cosf(theta) * mscale;
        si = sinf(theta) * mscale;
    }
    if (tid < 256) kcur[tid] = scale * v * nw[tid];
    if (is_rope) {
        kcur[256 + 2 * (tid - 256) + 0] = x0 * co - x1 * si;
        kcur[256 + 2 * (tid - 256) + 1] = x0 * si + x1 * co;
    }
    __syncthreads();
    kcache[row * 320 + tid] = __float2half(kcur[tid]);
}

// a strided L3-flush read (see wkb_bench.hip)
static __global__ void touch_kernel(const float * __restrict__ p, size_t nfloats, float * __restrict__ sink) {
    size_t stride = (size_t) gridDim.x * blockDim.x;
    float acc = 0.0f;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < nfloats; i += stride) acc += p[i];
    if (acc == 12345.678f) sink[0] = acc;
}

// ---------------------------------------------------------------------------------------
static uint8_t * load_tensor(const char * path, const char * name, size_t * out_n, int * out_type) {
    gguf_init_params p = { /*no_alloc*/ true, /*ctx*/ nullptr };
    gguf_context * ctx = gguf_init_from_file(path, p);
    if (!ctx) { std::fprintf(stderr, "gguf_init_from_file failed\n"); std::exit(1); }
    const int64_t id = gguf_find_tensor(ctx, name);
    if (id < 0) { std::fprintf(stderr, "tensor %s not found\n", name); std::exit(1); }
    *out_type = (int) gguf_get_tensor_type(ctx, id);
    const size_t nbytes = gguf_get_tensor_size(ctx, id);
    const size_t t_off  = gguf_get_tensor_offset(ctx, id);
    const size_t d_off  = gguf_get_data_offset(ctx);
    const int fd = open(path, O_RDONLY);
    if (fd < 0) { std::perror("open"); std::exit(1); }
    struct stat st; fstat(fd, &st);
    uint8_t * map = (uint8_t *) mmap(nullptr, (size_t) st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (map == MAP_FAILED) { std::perror("mmap"); std::exit(1); }
    uint8_t * out = (uint8_t *) std::malloc(nbytes);
    std::memcpy(out, map + d_off + t_off, nbytes);
    munmap(map, (size_t) st.st_size);
    close(fd);
    gguf_free(ctx);
    *out_n = nbytes;
    return out;
}

static float f16_to_f32(uint16_t h) {
    uint32_t s = ((uint32_t) h & 0x8000u) << 16, e = ((uint32_t) h >> 10) & 0x1fu, m = (uint32_t) h & 0x3ffu;
    uint32_t f;
    if (e == 0) {
        if (m == 0) f = s;
        else { uint32_t mm = m; int ee = -1; do { mm <<= 1; ++ee; } while (!(mm & 0x400u));
               f = s | ((uint32_t) (127 - 15 - ee) << 23) | ((mm & 0x3ffu) << 13); }
    } else if (e == 31) f = s | 0x7f800000u | (m << 13);
    else f = s | ((e + 127u - 15u) << 23) | (m << 13);
    float r; std::memcpy(&r, &f, 4); return r;
}

// the CPU Q8_0 x Q8_1 dot the shipped kernels replicate (quantize_q8_1: amax/127, fp16 d)
static float cpu_q8_0_q8_1_dot(const uint8_t * wrow, int nblocks, const float * x) {
    float acc = 0.0f;
    for (int b = 0; b < nblocks; ++b) {
        const uint8_t * wb = wrow + b * 34;
        const float * xb = x + b * 32;
        float amax = 0.0f;
        for (int k = 0; k < 32; ++k) amax = fmaxf(amax, fabsf(xb[k]));
        const float d  = amax / 127.0f;
        const float id = d ? 1.0f / d : 0.0f;
        // the fp16 round-trip of d, as the hardware does
        float d16 = 0.0f;
        { __half h = __float2half(d); d16 = __half2float(h); }
        const float wd = f16_to_f32(*(const uint16_t *) wb);
        int sum = 0;
        for (int k = 0; k < 32; ++k) sum += (int) (int8_t) wb[2 + k] * (int) roundf(xb[k] * id);
        acc += wd * d16 * (float) sum;
    }
    return acc;
}

// L64: every numeric diff asserts finiteness on BOTH sides before comparing.
static double max_abs_diff_finite(const float * a, const float * b, size_t n, const char * what) {
    for (size_t i = 0; i < n; ++i)
        if (!std::isfinite(a[i]) || !std::isfinite(b[i])) {
            std::printf("# FAIL %s: NON-FINITE at %zu (device %g, cpu %g) — the diff is not computed\n",
                        what, i, a[i], b[i]);
            std::exit(1);
        }
    double m = 0.0;
    for (size_t i = 0; i < n; ++i) m = std::fmax(m, std::fabs((double) a[i] - (double) b[i]));
    return m;
}

int main(int argc, char ** argv) {
    const char * model = argc > 1 ? argv[1] : "models/Mistral-Small-4-119B-2603-UD-IQ2_XXS.gguf";

    size_t n_wkv = 0, n_wvb = 0, n_norm = 0;
    int t1 = -1, t2 = -1, t3 = -1;
    uint8_t * wkv  = load_tensor(model, "blk.0.attn_kv_a_mqa.weight", &n_wkv,  &t1);
    uint8_t * wvb  = load_tensor(model, "blk.0.attn_v_b.weight",      &n_wvb,  &t2);
    uint8_t * hnorm= load_tensor(model, "blk.0.attn_kv_a_norm.weight",&n_norm, &t3);
    std::printf("# real weights from blk.0: kv_a_mqa %zu B (type %d), v_b %zu B (type %d), "
                "kv_a_norm %zu B (type %d)\n", n_wkv, t1, n_wvb, t2, n_norm, t3);

    // ---- device buffers -------------------------------------------------------------
    const size_t cur_bytes   = 4096 * sizeof(float);
    const size_t kcur_bytes  = 320 * sizeof(float);
    const size_t cache_bytes = 320 * 256 * sizeof(uint16_t);
    const size_t x_bytes     = 256 * 32 * sizeof(float);   // g_kqv [256,1,32]
    const size_t out_bytes   = 128 * 32 * sizeof(float);   // kqv_mla [128,1,32]

    uint8_t *d_wkv = nullptr, *d_wvb = nullptr, *d_cache = nullptr;
    float *d_cur = nullptr, *d_kcur = nullptr, *d_x = nullptr, *d_out = nullptr, *d_norm = nullptr;
    CHECK_HIP(hipMalloc(&d_wkv, n_wkv));
    CHECK_HIP(hipMalloc(&d_wvb, n_wvb));
    CHECK_HIP(hipMalloc(&d_norm, n_norm));
    CHECK_HIP(hipMalloc(&d_cache, cache_bytes));
    CHECK_HIP(hipMalloc(&d_cur, cur_bytes));
    CHECK_HIP(hipMalloc(&d_kcur, kcur_bytes));
    CHECK_HIP(hipMalloc(&d_x, x_bytes));
    CHECK_HIP(hipMalloc(&d_out, out_bytes));
    CHECK_HIP(hipMemcpy(d_wkv, wkv, n_wkv, hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_wvb, wvb, n_wvb, hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_norm, hnorm, n_norm, hipMemcpyHostToDevice));
    CHECK_HIP(hipMemset(d_cache, 0, cache_bytes));

    float * h_cur = (float *) std::malloc(cur_bytes);
    for (int i = 0; i < 4096; ++i) h_cur[i] = (float) ((i * 7919) % 1000) / 1000.0f - 0.5f;
    CHECK_HIP(hipMemcpy(d_cur, h_cur, cur_bytes, hipMemcpyHostToDevice));
    float * h_x = (float *) std::malloc(x_bytes);
    for (int i = 0; i < 256 * 32; ++i) h_x[i] = (float) ((i * 6151) % 997) / 997.0f - 0.5f;
    CHECK_HIP(hipMemcpy(d_x, h_x, x_bytes, hipMemcpyHostToDevice));

    float * dflush = nullptr, * dsink = nullptr;
    CHECK_HIP(hipMalloc(&dflush, 128u * 1024u * 1024u));
    CHECK_HIP(hipMalloc(&dsink, sizeof(float)));
    CHECK_HIP(hipMemset(dflush, 0, 128u * 1024u * 1024u));

    // the rope scalars, the model's own (engine printout: corr_dims [12,25], rope_mscale 1)
    const float freq_base = 10000.0f, freq_scale = 1.0f / 128.0f, ext_factor = 1.0f;
    const float beta_fast = 32.0f, beta_slow = 1.0f, eps = 9.999999974752427e-07f;
    const int   n_dims = 64, n_ctx_orig = 8192;
    const float attn_factor = 0.673308611f;   // yarn_attn: the engine's printed value
    const float theta_scale = powf(freq_base, -2.0f / (float) n_dims);
    auto corr_dim = [&](float beta) {
        return (float) ((double) n_dims * log((double) n_ctx_orig / ((double) beta * 2.0 * M_PI)) /
                        (2.0 * log((double) freq_base)));
    };
    const float corr0 = std::fmax(0.0f, floorf(corr_dim(beta_fast)));
    const float corr1 = std::fmin((float) (n_dims - 1), ceilf(corr_dim(beta_slow)));
    const int   pos = 27;
    std::printf("# rope: theta_scale=%.9g corr_dims=[%.0f,%.0f] pos=%d\n", theta_scale, corr0, corr1, pos);

    const int WARM = 2000, COLD = 400;
    dim3 tgrid(256), tblock(256);

    // ================= PHASE 1: wvb =================
    for (int i = 0; i < 10; ++i) phase_wvb_kernel<<<512, 256>>>(d_wvb, d_x, d_out);
    CHECK_HIP(hipDeviceSynchronize());
    double us_wvb_w = 0.0, us_wvb_c = 0.0;
    {
        const double t0 = now_s();
        for (int i = 0; i < WARM; ++i) phase_wvb_kernel<<<512, 256>>>(d_wvb, d_x, d_out);
        CHECK_HIP(hipDeviceSynchronize());
        us_wvb_w = (now_s() - t0) / WARM * 1e6;
    }
    {
        const double t0 = now_s();
        for (int i = 0; i < COLD; ++i) {
            touch_kernel<<<tgrid, tblock>>>(dflush, 128u*1024u*1024u/4, dsink);
            phase_wvb_kernel<<<512, 256>>>(d_wvb, d_x, d_out);
        }
        CHECK_HIP(hipDeviceSynchronize());
        const double dt = now_s() - t0;
        const double t1_ = now_s();
        for (int i = 0; i < COLD; ++i) touch_kernel<<<tgrid, tblock>>>(dflush, 128u*1024u*1024u/4, dsink);
        CHECK_HIP(hipDeviceSynchronize());
        us_wvb_c = (dt - (now_s() - t1_)) / COLD * 1e6;
    }
    std::printf("# PHASE wvb    WARM %8.3f us/launch  %7.1f GB/s   |  COLD %8.3f us  %7.1f GB/s "
                "(1.11 MiB of weight)\n",
                us_wvb_w, (double) n_wvb / 1e9 / (us_wvb_w * 1e-6),
                us_wvb_c, (double) n_wvb / 1e9 / (us_wvb_c * 1e-6));

    // ---- wvb numerics: one launch vs the CPU reference over the SAME bytes ----
    {
        phase_wvb_kernel<<<512, 256>>>(d_wvb, d_x, d_out);
        CHECK_HIP(hipDeviceSynchronize());
        float * hout = (float *) std::malloc(out_bytes);
        CHECK_HIP(hipMemcpy(hout, d_out, out_bytes, hipMemcpyDeviceToHost));
        float * ref = (float *) std::malloc(out_bytes);
        for (int c = 0; c < 32; ++c)
            for (int h = 0; h < 128; ++h)
                ref[c * 128 + h] = cpu_q8_0_q8_1_dot(wvb + ((size_t) (c * 128 + h) * 256) / 32 * 34,
                                                      8, h_x + c * 256);
        const double m = max_abs_diff_finite(hout, ref, 128 * 32, "wvb");
        double relmax = 0.0;
        for (int i = 0; i < 128 * 32; ++i)
            relmax = std::fmax(relmax, std::fabs((double) hout[i] - (double) ref[i]) /
                                       std::fmax(1e-9, std::fabs((double) ref[i])));
        std::printf("# wvb numerics: max |device - cpu_ref| = %.6g (max rel %.3g) over 4096 outputs; "
                    "finite on both sides\n", m, relmax);
        std::free(hout); std::free(ref);
    }

    // ================= PHASE 0: kvwrite (two launches, as shipped) =================
    for (int i = 0; i < 10; ++i) {
        phase_kvwrite_gemv<<<320, 128>>>(d_cur, d_wkv, d_kcur);
        phase_kvwrite_tail<<<1, 320>>>(d_kcur, d_norm, (__half *) d_cache, 5, pos, eps,
                                       freq_scale, ext_factor, attn_factor, theta_scale, corr0, corr1);
    }
    CHECK_HIP(hipDeviceSynchronize());
    double us_kv_w = 0.0, us_kv_c = 0.0, us_gemv_only = 0.0;
    {
        const double t0 = now_s();
        for (int i = 0; i < WARM; ++i)
            phase_kvwrite_gemv<<<320, 128>>>(d_cur, d_wkv, d_kcur);
        CHECK_HIP(hipDeviceSynchronize());
        us_gemv_only = (now_s() - t0) / WARM * 1e6;
    }
    {
        const double t0 = now_s();
        for (int i = 0; i < WARM; ++i) {
            phase_kvwrite_gemv<<<320, 128>>>(d_cur, d_wkv, d_kcur);
            phase_kvwrite_tail<<<1, 320>>>(d_kcur, d_norm, (__half *) d_cache, 5, pos, eps,
                                           freq_scale, ext_factor, attn_factor, theta_scale, corr0, corr1);
        }
        CHECK_HIP(hipDeviceSynchronize());
        us_kv_w = (now_s() - t0) / WARM * 1e6;
    }
    {
        const double t0 = now_s();
        for (int i = 0; i < COLD; ++i) {
            touch_kernel<<<tgrid, tblock>>>(dflush, 128u*1024u*1024u/4, dsink);
            phase_kvwrite_gemv<<<320, 128>>>(d_cur, d_wkv, d_kcur);
            phase_kvwrite_tail<<<1, 320>>>(d_kcur, d_norm, (__half *) d_cache, 5, pos, eps,
                                           freq_scale, ext_factor, attn_factor, theta_scale, corr0, corr1);
        }
        CHECK_HIP(hipDeviceSynchronize());
        const double dt = now_s() - t0;
        const double t1_ = now_s();
        for (int i = 0; i < COLD; ++i) touch_kernel<<<tgrid, tblock>>>(dflush, 128u*1024u*1024u/4, dsink);
        CHECK_HIP(hipDeviceSynchronize());
        us_kv_c = (dt - (now_s() - t1_)) / COLD * 1e6;
    }
    std::printf("# PHASE kvwrite WARM %8.3f us/token (gemv alone %8.3f, tail %6.3f)  %7.1f GB/s  |  "
                "COLD %8.3f us  %7.1f GB/s  (1.39 MiB of weight)\n",
                us_kv_w, us_gemv_only, us_kv_w - us_gemv_only,
                (double) n_wkv / 1e9 / (us_kv_w * 1e-6),
                us_kv_c, (double) n_wkv / 1e9 / (us_kv_c * 1e-6));

    // ---- kvwrite numerics: gemv + norm + rope + the F16 cache row ----
    {
        phase_kvwrite_gemv<<<320, 128>>>(d_cur, d_wkv, d_kcur);
        phase_kvwrite_tail<<<1, 320>>>(d_kcur, d_norm, (__half *) d_cache, 5, pos, eps,
                                       freq_scale, ext_factor, attn_factor, theta_scale, corr0, corr1);
        CHECK_HIP(hipDeviceSynchronize());
        float * got = (float *) std::malloc(kcur_bytes);
        uint16_t * gcache = (uint16_t *) std::malloc(320 * sizeof(uint16_t));
        CHECK_HIP(hipMemcpy(got, d_kcur, kcur_bytes, hipMemcpyDeviceToHost));
        CHECK_HIP(hipMemcpy(gcache, d_cache + (size_t) 5 * 320 * sizeof(uint16_t), 320 * sizeof(uint16_t),
                            hipMemcpyDeviceToHost));
        const float * nw = (const float *) hnorm;
        float * ref = (float *) std::malloc(kcur_bytes);
        // the gemv
        float proj[320];
        for (int o = 0; o < 320; ++o)
            proj[o] = cpu_q8_0_q8_1_dot(wkv + (size_t) o * 128 * 34, 128, h_cur);
        // attn_kv_a_norm over rows [0,256)
        float ss = 0.0f;
        for (int i = 0; i < 256; ++i) ss += proj[i] * proj[i];
        const float scale = 1.0f / sqrtf(ss / 256.0f + eps);
        for (int i = 0; i < 256; ++i) ref[i] = scale * proj[i] * nw[i];
        // the k_pe rope over rows [256,320)
        for (int p = 0; p < 32; ++p) {
            const float x0 = proj[256 + 2 * p], x1 = proj[257 + 2 * p];
            const float theta_extrap = (float) pos * powf(theta_scale, (float) p);
            const float y = ((float) p - corr0) / fmaxf(0.001f, corr1 - corr0);
            const float ramp_mix = (1.0f - fminf(1.0f, fmaxf(0.0f, y))) * ext_factor;
            const float theta = (freq_scale * theta_extrap) * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
            const float mscale = attn_factor * (1.0f + 0.1f * logf(1.0f / freq_scale));
            const float co = cosf(theta) * mscale, si = sinf(theta) * mscale;
            ref[256 + 2 * p]     = x0 * co - x1 * si;
            ref[257 + 2 * p]     = x0 * si + x1 * co;
        }
        const double m = max_abs_diff_finite(got, ref, 320, "kvwrite kcur");
        double mrot = 0.0, mnorm = 0.0;
        for (int i = 0; i < 256; ++i) mnorm = std::fmax(mnorm, std::fabs((double) got[i] - (double) ref[i]));
        for (int i = 256; i < 320; ++i) mrot = std::fmax(mrot, std::fabs((double) got[i] - (double) ref[i]));
        // the cache row is F16: compare in f16 space (an exact f16 round-trip of the reference)
        int ncache_eq = 0; double mcache = 0.0;
        for (int i = 0; i < 320; ++i) {
            const uint16_t want = __half_as_ushort(__float2half(ref[i]));
            if (want == gcache[i]) ++ncache_eq;
            mcache = std::fmax(mcache, std::fabs((double) f16_to_f32(want) - (double) f16_to_f32(gcache[i])));
        }
        std::printf("# kvwrite numerics: max |device - cpu_ref| = %.6g over 320 (norm half %.3g, rope half %.3g); "
                    "finite on both sides\n", m, mnorm, mrot);
        std::printf("# kvwrite cache row: %d/320 F16 halves bit-identical to the reference, max |d| %.3g "
                    "(the f16 store absorbs the association difference)\n", ncache_eq, mcache);
        std::free(got); std::free(gcache); std::free(ref);
    }

    std::free(h_cur); std::free(h_x); std::free(wkv); std::free(wvb); std::free(hnorm);
    return 0;
}
