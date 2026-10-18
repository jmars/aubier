// engine/wkb_bench.hip — H3/U3 STEP 2: a standalone microbenchmark for ONE specialized kernel.
//
// The kernel is the MLA absorption gemv, the dominant kernel the H3/U3 bisection identified:
//   wk_b  = Q8_0 [64, 256, 32]  x  q_nope_perm  = F32 [64, 1, 32]   ->  out  = F32 [256, 1, 32]
// i.e. 32 heads, each a [256 x 64] Q8_0 gemv.  Every shape, type and block geometry here is
// HARDCODED for Mistral Small 4 (the user directive): one wave size (wave32), one grid, one
// block.  No dispatch, no generality, no fallback.
//
// The weight is read from the REAL model (blk.0.attn_k_b.weight via gguf.h); the input vector
// is runtime activations, which do not exist standalone, so it is a deterministic pattern
// (the values do not affect Q8_0 gemv throughput).
//
// Build (hipcc launcher has a stale shebang — invoke through the venv python):
//   venvs/rocm/bin/python venvs/rocm/bin/hipcc -O2 --offload-arch=gfx1201 \
//       -I vendor/llama.cpp/ggml/include engine/wkb_bench.hip \
//       -L builds/hip/bin -lggml-base -Wl,-rpath,builds/hip/bin \
//       -o builds/engine/wkb_bench
#include <hip/hip_runtime.h>
#include <gguf.h>

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

// ---------------------------------------------------------------------------
// the Q8_0 block, exactly as ggml-common.h defines it (34 bytes: half delta + 32 int8)
// ---------------------------------------------------------------------------
struct block_q8_0 { uint16_t d; int8_t qs[32]; };   // d stored as raw half bits

static __device__ __forceinline__ float half2float(uint16_t h) {
    // portable IEEE754 binary16 -> float (no half intrinsic needed)
    const uint32_t sign = ((uint32_t) h & 0x8000u) << 16;
    const uint32_t exp  = ((uint32_t) h >> 10) & 0x1fu;
    const uint32_t man  = (uint32_t) h & 0x3ffu;
    uint32_t f;
    if (exp == 0) {
        if (man == 0) f = sign;
        else {   // subnormal: normalize
            uint32_t m = man; int e = -1;
            do { m <<= 1; ++e; } while (!(m & 0x400u));
            m &= 0x3ffu;
            f = sign | ((uint32_t) (127 - 15 - e) << 23) | (m << 13);
        }
    } else if (exp == 31) {
        f = sign | 0x7f800000u | (man << 13);   // inf / nan
    } else {
        f = sign | ((exp + (127u - 15u)) << 23) | (man << 13);
    }
    return __uint_as_float(f);
}

// ---------------------------------------------------------------------------
// THE SPECIALIZED KERNEL — Q8_0 [64, 256, 32] x F32 [64, 1, 32] -> F32 [256, 1, 32]
// 8192 outputs = 256 blocks x 32 threads; one thread per output; each thread does a 64-dot
// (2 Q8_0 blocks) with the head's 64-float input vector (shared across the head's 256 rows).
// ---------------------------------------------------------------------------
__global__ void wkb_gemv_kernel(const uint8_t * __restrict__ w,
                                const float * __restrict__ x,
                                float * __restrict__ out) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;   // 0..8191
    if (tid >= 8192) return;
    const int c = tid >> 8;          // head 0..31
    const int h = tid & 255;         // output row 0..255
    // byte-explicit Q8_0 layout: 64 values = 2 blocks x 34 bytes (half d + 32 int8)
    const uint8_t * wrow = w + (size_t)(c * 256 + h) * 68;
    // the REAL input is the nope half of g_q [4096] = 32 heads x 128 (head_k_mla); head c's
    // 64 nope values sit at g_q[c*128 + k] (stride 128, NOT contiguous).
    const float * xc = x + c * 128;

    const float d0 = half2float(*(const uint16_t *) wrow);
    const float d1 = half2float(*(const uint16_t *) (wrow + 34));
    const int8_t * q0 = (const int8_t *) (wrow + 2);
    const int8_t * q1 = (const int8_t *) (wrow + 36);
    float acc = 0.0f;
#pragma unroll
    for (int k = 0; k < 32; ++k) acc += d0 * (float) q0[k] * xc[k];
#pragma unroll
    for (int k = 0; k < 32; ++k) acc += d1 * (float) q1[k] * xc[32 + k];
    out[tid] = acc;
}

// a strided L3-flush read: touches the whole buffer and folds into one word so the compiler
// cannot elide the loads.
__global__ void touch_kernel(const float * __restrict__ p, size_t nfloats, float * __restrict__ sink) {
    size_t stride = (size_t) gridDim.x * blockDim.x;
    float acc = 0.0f;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < nfloats; i += stride)
        acc += p[i];
    if (acc == 12345.678f) sink[0] = acc;   // never true; keeps the reads
}

// debug: dump the device's raw reads for ONE element (head c, row h)
__global__ void dbg_kernel(const uint8_t * __restrict__ w, const float * __restrict__ x,
                           float * __restrict__ out, int c, int h) {
    if ((blockIdx.x * blockDim.x + threadIdx.x) != 0) return;
    const uint8_t * wrow = w + (size_t)(c * 256 + h) * 68;
    const float * xc = x + c * 128;
    float a0 = 0.0f, a1 = 0.0f;
    for (int k = 0; k < 32; ++k) a0 += (float)((const int8_t *)(wrow + 2))[k] * xc[k];
    for (int k = 0; k < 32; ++k) a1 += (float)((const int8_t *)(wrow + 36))[k] * xc[32 + k];
    out[0] = a0; out[1] = a1;
    out[2] = (float) *(const uint16_t *) wrow;          // d0 raw bits as float
    out[3] = (float) *(const uint16_t *) (wrow + 34);   // d1 raw bits
    out[4] = xc[0];  out[5] = xc[32];
    out[6] = (float)((const int8_t *)(wrow + 2))[0];
    out[7] = (float)((const int8_t *)(wrow + 36))[0];
}

// ---------------------------------------------------------------------------
// the measurement
// ---------------------------------------------------------------------------
static double now_s() {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double) ts.tv_sec + 1e-9 * (double) ts.tv_nsec;
}

static void load_real_weight(const char * path, uint8_t ** out, size_t * out_n) {
    gguf_init_params p = { /*no_alloc*/ true, /*ctx*/ nullptr };
    gguf_context * ctx = gguf_init_from_file(path, p);
    if (!ctx) { std::fprintf(stderr, "gguf_init_from_file failed\n"); std::exit(1); }
    const int64_t id = gguf_find_tensor(ctx, "blk.0.attn_k_b.weight");
    if (id < 0) { std::fprintf(stderr, "tensor not found\n"); std::exit(1); }
    if (gguf_get_tensor_type(ctx, id) != GGML_TYPE_Q8_0) {
        std::fprintf(stderr, "unexpected type %d\n", (int) gguf_get_tensor_type(ctx, id));
        std::exit(1);
    }
    const size_t nbytes = gguf_get_tensor_size(ctx, id);
    const size_t t_off  = gguf_get_tensor_offset(ctx, id);
    const size_t d_off  = gguf_get_data_offset(ctx);
    *out_n = nbytes;

    const int fd = open(path, O_RDONLY);
    if (fd < 0) { std::perror("open"); std::exit(1); }
    struct stat st; fstat(fd, &st);
    uint8_t * map = (uint8_t *) mmap(nullptr, (size_t) st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (map == MAP_FAILED) { std::perror("mmap"); std::exit(1); }
    *out = (uint8_t *) std::malloc(nbytes);
    std::memcpy(*out, map + d_off + t_off, nbytes);
    munmap(map, (size_t) st.st_size);
    close(fd);
    gguf_free(ctx);
}

int main(int argc, char ** argv) {
    const char * model = argc > 1 ? argv[1] : "models/Mistral-Small-4-119B-2603-UD-IQ2_XXS.gguf";
    uint8_t * wbytes = nullptr; size_t wn = 0;
    load_real_weight(model, &wbytes, &wn);
    std::printf("# wk_b real weight: %zu bytes (Q8_0 [64,256,32]) from %s\n", wn, model);

    const size_t xbytes = 4096 * sizeof(float);  // g_q[4096], stride-128 head layout
    const size_t obytes = 256 * 32 * sizeof(float);
    uint8_t * dw = nullptr; float * dx = nullptr, * dout = nullptr;
    CHECK_HIP(hipMalloc(&dw, wn));
    CHECK_HIP(hipMalloc(&dx, xbytes));
    CHECK_HIP(hipMalloc(&dout, obytes));
    CHECK_HIP(hipMemcpy(dw, wbytes, wn, hipMemcpyHostToDevice));

    float * hx = (float *) std::malloc(xbytes);
    for (int i = 0; i < 4096; ++i) hx[i] = (float) ((i * 7919) % 1000) / 1000.0f - 0.5f;
    CHECK_HIP(hipMemcpy(dx, hx, xbytes, hipMemcpyHostToDevice));

    // 128 MiB flush buffer (the RX 9070 L3 is ~64 MiB; 2x is a real eviction)
    float * dflush = nullptr, * dsink = nullptr;
    CHECK_HIP(hipMalloc(&dflush, 128u * 1024u * 1024u));
    CHECK_HIP(hipMalloc(&dsink, sizeof(float)));
    CHECK_HIP(hipMemset(dflush, 0, 128u * 1024u * 1024u));

    const int WARM_ITERS = 2000;
    const int COLD_ITERS = 400;
    dim3 grid(256), block(32);
    dim3 tgrid(256), tblock(256);   // touch: 256*256 threads

    for (int i = 0; i < 10; ++i) wkb_gemv_kernel<<<grid, block>>>(dw, dx, dout);
    CHECK_HIP(hipDeviceSynchronize());

    // ---- WARM: back-to-back launches, no flush ----
    {
        const double t0 = now_s();
        for (int i = 0; i < WARM_ITERS; ++i) wkb_gemv_kernel<<<grid, block>>>(dw, dx, dout);
        CHECK_HIP(hipDeviceSynchronize());
        const double dt = now_s() - t0;
        const double us = dt / WARM_ITERS * 1e6;
        const double gib = (double) wn / (1024.0*1024.0*1024.0);
        std::printf("# WARM  %5d launches: %8.3f us/launch   %7.1f GB/s\n",
                    WARM_ITERS, us, gib / (us * 1e-6));
    }

    // ---- COLD: evict L3 (touch 128 MiB) before every gemv ----
    {
        const double t0 = now_s();
        for (int i = 0; i < COLD_ITERS; ++i) {
            touch_kernel<<<tgrid, tblock>>>(dflush, 128u*1024u*1024u/4, dsink);
            wkb_gemv_kernel<<<grid, block>>>(dw, dx, dout);
        }
        CHECK_HIP(hipDeviceSynchronize());
        const double dt = now_s() - t0;
        // subtract the touch kernel's own time so the number is the gemv's, not the flush's
        const double t_touch0 = now_s();
        for (int i = 0; i < COLD_ITERS; ++i) touch_kernel<<<tgrid, tblock>>>(dflush, 128u*1024u*1024u/4, dsink);
        CHECK_HIP(hipDeviceSynchronize());
        const double t_touch = now_s() - t_touch0;
        const double us = (dt - t_touch) / COLD_ITERS * 1e6;
        const double gib = (double) wn / (1024.0*1024.0*1024.0);
        std::printf("# COLD  %5d launches: %8.3f us/launch   %7.1f GB/s  (L3 evicted each time; "
                    "touch subtracted)\n", COLD_ITERS, us, gib / (us * 1e-6));
    }

    // ---- numerics self-check: one launch vs a CPU dequantize reference (catches silent-wrong
    // layout/scale errors, the class this unit is routed for) ----
    {
        wkb_gemv_kernel<<<grid, block>>>(dw, dx, dout);
        CHECK_HIP(hipDeviceSynchronize());
        float * hout = (float *) std::malloc(obytes);
        CHECK_HIP(hipMemcpy(hout, dout, obytes, hipMemcpyDeviceToHost));
        // CPU reference over the SAME bytes
        auto f16 = [](uint16_t h) {
            uint32_t s = ((uint32_t) h & 0x8000u) << 16, e = ((uint32_t) h >> 10) & 0x1f, m = (uint32_t) h & 0x3ff;
            uint32_t f;
            if (e == 0) { if (m == 0) f = s; else { uint32_t mm = m; int ee = -1; do { mm <<= 1; ++ee; } while (!(mm & 0x400)); f = s | ((uint32_t)(127-15-ee) << 23) | ((mm & 0x3ff) << 13); } }
            else if (e == 31) f = s | 0x7f800000u | (m << 13);
            else f = s | ((e + 127 - 15) << 23) | (m << 13);
            return *reinterpret_cast<float *>(&f);
        };
        double max_abs = 0.0; int max_c = -1, max_h = -1; float max_got = 0, max_ref = 0;
        for (int c = 0; c < 32; ++c) for (int h = 0; h < 256; ++h) {
            const uint8_t * wrow = wbytes + ((size_t)(c * 256 + h) * 64) / 32 * 34;
            const float d0 = f16(*reinterpret_cast<const uint16_t *>(wrow));
            const float d1 = f16(*reinterpret_cast<const uint16_t *>(wrow + 34));
            float acc = 0.0f;
            for (int k = 0; k < 32; ++k) acc += d0 * (float)((const int8_t *) wrow)[2 + k] * hx[c * 128 + k];
            for (int k = 0; k < 32; ++k) acc += d1 * (float)((const int8_t *) wrow)[36 + k] * hx[c * 128 + 32 + k];
            const float got = hout[c * 256 + h];
            const double d = std::fabs((double) got - (double) acc);
            if (d > max_abs) { max_abs = d; max_c = c; max_h = h; max_got = got; max_ref = acc; }
        }
        std::printf("# numerics self-check: max |device - cpu_ref| = %g over 8192 outputs (at head %d row %d: "
                    "device=%g cpu=%g)\n", max_abs, max_c, max_h, max_got, max_ref);
        // focused dump for the worst element
        {
            const int c = max_c, h = max_h;
            const uint8_t * wrow = wbytes + ((size_t)(c * 256 + h) * 64) / 32 * 34;
            std::printf("#   debug head=%d row=%d d0=%g d1=%g\n", c, h,
                        f16(*reinterpret_cast<const uint16_t *>(wrow)),
                        f16(*reinterpret_cast<const uint16_t *>(wrow + 34)));
            float a0 = 0, a1 = 0;
            for (int k = 0; k < 32; ++k) a0 += f16(*reinterpret_cast<const uint16_t *>(wrow)) * (float)((const int8_t *) wrow)[2 + k] * hx[c * 128 + k];
            for (int k = 0; k < 32; ++k) a1 += f16(*reinterpret_cast<const uint16_t *>(wrow + 34)) * (float)((const int8_t *) wrow)[36 + k] * hx[c * 128 + 32 + k];
            std::printf("#   debug cpu block0=%g block1=%g total=%g  device=%g\n", a0, a1, a0 + a1, max_got);
            // which (c,h) in the DEVICE array holds the cpu value? scan for a near match
            float best = 0; int bc = -1, bh = -1; double bd = 1e30;
            for (int cc = 0; cc < 32; ++cc) for (int hh = 0; hh < 256; ++hh) {
                double dd = std::fabs((double) hout[cc * 256 + hh] - (double) max_ref);
                if (dd < bd) { bd = dd; bc = cc; bh = hh; best = hout[cc * 256 + hh]; }
            }
            std::printf("#   debug nearest device value to cpu=%g: device[head %d row %d]=%g (d=%g)\n",
                        max_ref, bc, bh, best, bd);
            // dump the device's raw reads for the worst element (a0/a1 = raw q*x sums, no d)
            {
                float dbg[8] = {0}; float * ddbg = nullptr;
                CHECK_HIP(hipMalloc(&ddbg, sizeof(dbg)));
                dbg_kernel<<<1, 1>>>(dw, dx, ddbg, max_c, max_h);
                CHECK_HIP(hipDeviceSynchronize());
                CHECK_HIP(hipMemcpy(dbg, ddbg, sizeof(dbg), hipMemcpyDeviceToHost));
                std::printf("#   debug device raw: a0=%g a1=%g d0bits=%g d1bits=%g xc0=%g xc32=%g q00=%g q10=%g\n",
                            dbg[0], dbg[1], dbg[2], dbg[3], dbg[4], dbg[5], dbg[6], dbg[7]);
                float ca0 = 0, ca1 = 0;
                for (int k = 0; k < 32; ++k) ca0 += (float)((const int8_t *) wrow)[2 + k] * hx[max_c * 128 + k];
                for (int k = 0; k < 32; ++k) ca1 += (float)((const int8_t *) wrow)[36 + k] * hx[max_c * 128 + 32 + k];
                std::printf("#   debug cpu raw   : a0=%g a1=%g d0bits=%g d1bits=%g xc0=%g xc32=%g q00=%g q10=%g\n",
                            ca0, ca1, (float) *reinterpret_cast<const uint16_t *>(wrow),
                            (float) *reinterpret_cast<const uint16_t *>(wrow + 34),
                            hx[max_c * 128], hx[max_c * 128 + 32],
                            (float)((const int8_t *) wrow)[2], (float)((const int8_t *) wrow)[36]);
                CHECK_HIP(hipFree(ddbg));
            }
        }
        std::free(hout);
    }

    std::free(hx); std::free(wbytes);
    CHECK_HIP(hipFree(dw)); CHECK_HIP(hipFree(dx)); CHECK_HIP(hipFree(dout));
    CHECK_HIP(hipFree(dflush)); CHECK_HIP(hipFree(dsink));
    return 0;
}
