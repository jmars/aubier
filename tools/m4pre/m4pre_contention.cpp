// m4pre_contention.cpp — M4-PRE: does the streaming expert path (pinned H2D copy of
// expert bytes) ADD to or SUBTRACT from the CPU's expert-compute DRAM reads?
//
// The three arms, all reading the SAME ~580 MiB of expert bytes:
//   cpu  : T threads stream the real quantized expert bytes out of the model file
//          (mmap'd, page-cache resident) with a pure read/accumulate loop. This is a
//          measurement of the DRAM READ DEMAND of an expert-shaped consumer, NOT of
//          compute: llama.cpp's CPU MoE kernel is memory-bound, so the demand is the
//          quantity that matters. Honest statement is in results/p-m4pre.md.
//   copy : a pinned host->device copy of the same volume on one HIP stream
//          (hipMemcpyAsync + hipStreamSynchronize), which the SDMA engine services by
//          reading host DRAM over PCIe.
//   both : the two run concurrently, started off one barrier; each arm timestamps
//          itself, so we get each arm's rate WHILE the other is running.
//
// Build: venvs/rocm/bin/hipcc -O2 --offload-arch=gfx1201 -o builds/m4pre/bin/m4pre_contention \
//            tools/m4pre/m4pre_contention.cpp -lpthread
// Run:   scripts/m4pre_contention.sh

#include <hip/hip_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <mutex>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <vector>

#define HIP_CHECK(call)                                                                  \
    do {                                                                                 \
        hipError_t _e = (call);                                                          \
        if (_e != hipSuccess) {                                                          \
            fprintf(stderr, "HIP error %d (%s) at %s:%d\n", (int) _e,                    \
                    hipGetErrorString(_e), __FILE__, __LINE__);                          \
            exit(2);                                                                     \
        }                                                                                \
    } while (0)

using clk = std::chrono::steady_clock;

// ------------------------------------------------------------------ region model

struct Span {
    const uint8_t * p;
    size_t len;
};

static std::vector<Span> g_spans;
static std::vector<size_t> g_prefix; // g_prefix[i] = start byte of span i; back() = total
static size_t g_total = 0;

static void add_span(const uint8_t * p, size_t len) {
    if (len == 0) return;
    g_prefix.push_back(g_total);
    g_spans.push_back({p, len});
    g_total += len;
}

// read the byte interval [b0,b1) of the concatenated region space
static inline uint64_t read_range(size_t b0, size_t b1, size_t * covered = nullptr) {
    uint64_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;
    // BUG FIXED (caught by an impossible 108 GiB/s anon row): lower_bound on a list of
    // span STARTS returns end() whenever b0 falls inside the LAST span, which silently
    // skipped that worker's whole read. Use the last start <= b0.
    if (b0 >= g_total) return 0;
    size_t i = (size_t)(std::upper_bound(g_prefix.begin(), g_prefix.end(), b0) -
                        g_prefix.begin()) - 1;
    if (i >= g_spans.size()) return 0;
    size_t off = b0 - g_prefix[i];
    for (; i < g_spans.size(); ++i) {
        const uint8_t * p = g_spans[i].p + off;
        size_t avail = g_spans[i].len - off;
        off = 0;
        size_t want = std::min(avail, b1 - b0);
        if (covered) *covered += want;
        b0 += want;
        size_t k = 0;
        for (; k + 64 <= want; k += 64) {
            const uint64_t * q = (const uint64_t *) (p + k);
            a0 += q[0]; a1 += q[1];
            a2 ^= q[2]; a3 += q[3];
            a0 ^= q[4]; a1 += q[5];
            a2 += q[6]; a3 ^= q[7];
        }
        for (; k + 8 <= want; k += 8) a0 += *(const uint64_t *) (p + k);
        for (; k < want; ++k) a1 += p[k];
        if (b0 >= b1) break;
    }
    return a0 + a1 + a2 + a3;
}

// ------------------------------------------------------------------ barrier

// A condition_variable broadcast wakes 16 threads over ~ms, which is a fatal
// fraction of the ~14 ms being measured (measured: min/max worker time differed by
// 7 ms on the first run). Spin instead: every worker is already on its own core, so
// the release skew is sub-microsecond.
struct Gate {
    std::atomic<int> ready{0};
    std::atomic<bool> go{false};
    void arrive() { ready.fetch_add(1, std::memory_order_acq_rel); }
    void start(int n) {
        while (ready.load(std::memory_order_acquire) < n) { }
        go.store(true, std::memory_order_release);
    }
    void wait() {
        while (!go.load(std::memory_order_acquire)) { }
    }
};

// ------------------------------------------------------------------ args

struct Args {
    std::string source = "file";     // file | anon
    std::string file;
    std::string regions;             // "off:len,off:len,..."  (file offsets, absolute)
    std::string regions_file;
    std::string mode = "both";       // cpu | copy | both
    std::string copy_source = "pinned"; // pinned | file
    std::string label = "row";
    int threads = 16;
    int passes = 7;
    int cpu_reps = 1;   // repeat the CPU arm's slice N times per pass (control arms)
    int dur_ms = 0;     // fixed-duration mode: both arms run for the SAME wall window and
                        // report bytes/window. The volume mode gives each arm the whole
                        // volume, so the faster arm finishes first and its rate is an
                        // average over a partly-uncontended window -- the summed rate is
                        // then not an instantaneous combined rate. This mode fixes that.
    size_t cpu_mib = 0; // restrict the CPU arm to the first N MiB (0 = the whole volume).
                        // Lets the copy arm keep the full volume while the CPU arm's
                        // working set drops into cache: isolates DRAM contention from
                        // "any CPU activity perturbs the copy".
    size_t chunk_kib = 512;
    size_t anon_mib = 0;
    int device = 0;
};

static size_t parse_size(const std::string & s) {
    return (size_t) strtoull(s.c_str(), nullptr, 10);
}

static std::vector<std::pair<size_t, size_t>> parse_regions(const std::string & s,
                                                            const std::string & f) {
    std::vector<std::pair<size_t, size_t>> v;
    std::string all = s;
    if (!f.empty()) {
        FILE * fp = fopen(f.c_str(), "r");
        if (!fp) { perror(f.c_str()); exit(2); }
        char buf[512];
        while (fgets(buf, sizeof buf, fp)) {
            char * h = strchr(buf, '#'); if (h) *h = 0;
            size_t o, l;
            if (sscanf(buf, "%zu %zu", &o, &l) == 2) v.push_back({o, l});
        }
        fclose(fp);
    }
    if (!all.empty()) {
        size_t pos = 0;
        while (pos < all.size()) {
            size_t c = all.find(',', pos);
            std::string tok = all.substr(pos, c == std::string::npos ? std::string::npos : c - pos);
            size_t col = tok.find(':');
            if (col != std::string::npos)
                v.push_back({parse_size(tok.substr(0, col)), parse_size(tok.substr(col + 1))});
            if (c == std::string::npos) break;
            pos = c + 1;
        }
    }
    return v;
}

// ------------------------------------------------------------------ helpers

static double load1() {
    FILE * f = fopen("/proc/loadavg", "r");
    if (!f) return -1;
    double a = -1; if (fscanf(f, "%lf", &a) != 1) a = -1; fclose(f);
    return a;
}

static long meminfo_field(const char * key) {
    FILE * f = fopen("/proc/meminfo", "r");
    if (!f) return -1;
    char line[256];
    long v = -1;
    while (fgets(line, sizeof line, f)) {
        if (strncmp(line, key, strlen(key)) == 0) { sscanf(line + strlen(key), " %ld", &v); break; }
    }
    fclose(f);
    return v;
}

struct Arm {
    double sec = 1e30;   // fastest worker's time for its (equal) slice
    double sec_last = 0; // slowest worker's time = when the arm actually finished
    size_t bytes = 0;    // this arm's own byte volume
    int workers = 1;
};

int main(int argc, char ** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : ""; };
        if      (k == "--source")       a.source = next();
        else if (k == "--file")         a.file = next();
        else if (k == "--regions")      a.regions = next();
        else if (k == "--regions-file") a.regions_file = next();
        else if (k == "--mode")         a.mode = next();
        else if (k == "--copy-source")  a.copy_source = next();
        else if (k == "--label")        a.label = next();
        else if (k == "--threads")      a.threads = atoi(next().c_str());
        else if (k == "--passes")       a.passes = atoi(next().c_str());
        else if (k == "--cpu-reps")     a.cpu_reps = atoi(next().c_str());
        else if (k == "--cpu-mib")      a.cpu_mib = parse_size(next());
        else if (k == "--dur-ms")       a.dur_ms = atoi(next().c_str());
        else if (k == "--chunk-kib")    a.chunk_kib = parse_size(next());
        else if (k == "--anon-mib")     a.anon_mib = parse_size(next());
        else if (k == "--device")       a.device = atoi(next().c_str());
        else { fprintf(stderr, "unknown arg %s\n", k.c_str()); return 2; }
    }
    if (a.threads < 1) a.threads = 1;
    if (a.passes < 1) a.passes = 1;

    const bool need_cpu  = (a.mode == "cpu"  || a.mode == "both");
    const bool need_copy = (a.mode == "copy" || a.mode == "both");
    if (a.mode != "cpu" && a.mode != "copy" && a.mode != "both") {
        fprintf(stderr, "bad --mode %s\n", a.mode.c_str()); return 2;
    }

    // ---- build the region set
    void * map = nullptr;
    size_t map_len = 0;
    int fd = -1;
    if (a.source == "file") {
        auto regs = parse_regions(a.regions, a.regions_file);
        if (regs.empty()) { fprintf(stderr, "no regions given\n"); return 2; }
        fd = open(a.file.c_str(), O_RDONLY);
        if (fd < 0) { perror(a.file.c_str()); return 2; }
        struct stat st;
        if (fstat(fd, &st) != 0) { perror("fstat"); return 2; }
        size_t fsize = (size_t) st.st_size;
        // mmap whole file once, read-only, exactly as llama.cpp mmaps weights
        map_len = (fsize + 4095) & ~(size_t) 4095;
        map = mmap(nullptr, map_len, PROT_READ, MAP_SHARED, fd, 0);
        if (map == MAP_FAILED) { perror("mmap"); return 2; }
        for (auto & r : regs) {
            if (r.first + r.second > fsize) {
                fprintf(stderr, "region %zu+%zu past EOF %zu\n", r.first, r.second, fsize);
                return 2;
            }
            add_span((const uint8_t *) map + r.first, r.second);
        }
    } else {
        if (a.anon_mib == 0) { fprintf(stderr, "--anonymous needs --anon-mib\n"); return 2; }
        // a single anonymous region, matching the file case's byte volume
        map_len = a.anon_mib << 20;
        map = mmap(nullptr, map_len, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (map == MAP_FAILED) { perror("mmap anon"); return 2; }
        memset(map, 0x5a, map_len);
        add_span((const uint8_t *) map, map_len);
    }
    if (g_total == 0) { fprintf(stderr, "zero-length region set\n"); return 2; }

    // ---- banner
    hipDeviceProp_t prop;
    HIP_CHECK(hipSetDevice(a.device));
    HIP_CHECK(hipGetDeviceProperties(&prop, a.device));
    char link[128] = "";
    {
        FILE * f = fopen("/sys/class/drm/card1/device/current_link_speed", "r");
        char a1[64] = "", a2[64] = "";
        if (f) { if (fgets(a1, sizeof a1, f)) a1[strcspn(a1, "\n")] = 0; fclose(f); }
        f = fopen("/sys/class/drm/card1/device/current_link_width", "r");
        if (f) { if (fgets(a2, sizeof a2, f)) a2[strcspn(a2, "\n")] = 0; fclose(f); }
        snprintf(link, sizeof link, "%s x%s", a1, a2);
    }
    const double l_before = load1();
    printf("# m4pre label=%s mode=%s source=%s copy_source=%s threads=%d chunk_kib=%zu passes=%d\n",
           a.label.c_str(), a.mode.c_str(), a.source.c_str(), a.copy_source.c_str(),
           a.threads, a.chunk_kib, a.passes);
    printf("# device: %s  pcie=%s  nproc=%ld  load1_before=%.2f  MemAvailable_kB=%ld  Cached_kB=%ld\n",
           prop.name, link, sysconf(_SC_NPROCESSORS_ONLN), l_before,
           meminfo_field("MemAvailable:"), meminfo_field("Cached:"));
    printf("# region bytes = %zu (%.1f MiB) in %zu spans\n", g_total,
           (double) g_total / 1048576.0, g_spans.size());
    fflush(stdout);

    // ---- warm pass: force page residency (file case) / touch (anon case). Timed so a
    // cold page-cache is visible in the log rather than silently inside a measurement.
    const auto tw0 = clk::now();
    volatile uint64_t sink = 0;
    sink += read_range(0, g_total);
    const double warm_sec = std::chrono::duration<double>(clk::now() - tw0).count();
    printf("# warm pass (1 thread, untimed-by-arms): %.3f s = %.2f GiB/s  [cold-page-cache would be s-scale]\n",
           warm_sec, (double) g_total / 1073741824.0 / warm_sec);
    fflush(stdout);

    // ---- coverage self-check: the T-way split must read every byte EXACTLY once.
    // This is the check that catches a worker silently reading nothing (the failure that
    // produced an impossible 108 GiB/s row before the lower_bound fix).
    {
        size_t full = 0;
        read_range(0, g_total, &full);
        for (int T : {1, 2, 3, 4, 6, 8, 12, 16, 20, 24, 32, 48}) {
            const size_t per = g_total / (size_t) T, rem = g_total % (size_t) T;
            size_t sum = 0;
            for (int t = 0; t < T; ++t) {
                const size_t b0 = (size_t) t * per + std::min((size_t) t, rem);
                const size_t b1 = b0 + per + (t < (int) rem ? 1 : 0);
                read_range(b0, b1, &sum);
            }
            if (sum != g_total || full != g_total) {
                printf("# FATAL: coverage check FAILED at T=%d: full=%zu split_sum=%zu total=%zu\n",
                       T, full, sum, g_total);
                return 3;
            }
        }
        printf("# coverage check: T-way splits of {1,2,3,4,6,8,12,16,20,24,32,48} each read"
               " exactly %zu bytes (once each) -- OK\n", g_total);
        fflush(stdout);
    }

    // ---- pinned host copy source (a separate DRAM allocation holding the same bytes)
    void * host = nullptr;
    const size_t chunk = a.chunk_kib * 1024;
    void * dev = nullptr;
    hipStream_t stream = nullptr;
    if (need_copy) {
        HIP_CHECK(hipMalloc(&dev, chunk));
        HIP_CHECK(hipStreamCreate(&stream));
        if (a.copy_source == "pinned") {
            HIP_CHECK(hipHostMalloc(&host, g_total, hipHostMallocDefault));
            // copy the real expert bytes in, so the copy arm reads real data (not zeros):
            // zero pages could conceivably be optimised, real bytes cannot.
            size_t o = 0;
            for (auto & s : g_spans) { memcpy((uint8_t *) host + o, s.p, s.len); o += s.len; }
        } else if (a.copy_source == "pageable") {
            // pageable host memory holding the same real bytes — this is what llama.cpp
            // actually copies from (mmap'd weights); post-hardware-fix P-K measured
            // pageable == pinned on this link, so this arm tests whether the *source's
            // pageability* changes DRAM contention (a bounce buffer would double it).
            if (posix_memalign(&host, 4096, g_total) != 0) { perror("posix_memalign"); return 2; }
            size_t o = 0;
            for (auto & s : g_spans) { memcpy((uint8_t *) host + o, s.p, s.len); o += s.len; }
        } else {
            fprintf(stderr, "bad --copy-source %s\n", a.copy_source.c_str()); return 2;
        }
        // warm the copy path
        for (size_t o = 0; o < g_total; o += chunk) {
            size_t n = std::min(chunk, g_total - o);
            HIP_CHECK(hipMemcpyAsync(dev, (const uint8_t *) host + o, n,
                                     hipMemcpyHostToDevice, stream));
        }
        HIP_CHECK(hipStreamSynchronize(stream));
        fflush(stdout);
    }

    // ---- run
    // window mode: the copy arm runs full-volume passes for --dur-ms and then raises
    // this flag; the CPU arm loops 1 MiB blocks until it sees it. Both arms are then
    // measured over the SAME wall window, which the volume mode cannot do (the faster
    // arm finishes first, so its rate average would span a partly-uncontended window).
    std::atomic<bool> dur_done{false};
    std::vector<double> cpu_secs, cpu_secs_last, copy_secs;
    std::vector<size_t> copy_bytes_per_pass;
    size_t cpu_bytes_per_pass = 0;
    for (int pass = 0; pass < a.passes; ++pass) {
        dur_done.store(false);
        Gate gate;
        Arm cpu_arm, copy_arm;
        std::vector<std::thread> th;
        std::vector<uint64_t> results(need_cpu ? a.threads : 0, 0);

        const int nworkers = (need_cpu ? a.threads : 0) + (need_copy ? 1 : 0);
        cpu_arm.bytes = 0;   // filled from the workers' own byte counts (= g_total when
                             // the volume mode reads every region exactly once)
        cpu_arm.workers = a.threads;
        copy_arm.bytes = g_total;
        copy_arm.workers = 1;
        const size_t cpu_hi = (a.cpu_mib && need_cpu) ? std::min(a.cpu_mib << 20, g_total) : g_total;
        if (need_cpu) {
            const size_t per = cpu_hi / (size_t) a.threads;
            const size_t rem = cpu_hi % (size_t) a.threads;
            for (int t = 0; t < a.threads; ++t) {
                const size_t b0 = (size_t) t * per + std::min((size_t) t, rem);
                const size_t b1 = b0 + per + (t < (int) rem ? 1 : 0);
                th.emplace_back([&, t, b0, b1] {
                    volatile uint64_t sink_local = 0;
                    gate.arrive();
                    gate.wait();
                    const auto t0 = clk::now();
                    uint64_t br = 0;
                    double dt;
                    if (a.dur_ms > 0) {
                        // walk the slice in 1 MiB blocks, wrapping, until the window ends
                        const auto deadline = t0 + std::chrono::milliseconds(a.dur_ms);
                        const size_t blk = 1u << 20;
                        size_t pos = b0;
                        while (true) {
                            size_t n = std::min(blk, b1 - pos);
                            sink_local += read_range(pos, pos + n);
                            br += n;
                            pos += n;
                            if (pos >= b1) pos = b0;
                            // with a copy arm present the window is the copy arm's; without
                            // one there is nothing to align to and we use the clock
                            if (need_copy ? dur_done.load(std::memory_order_acquire)
                                          : (clk::now() >= deadline)) break;
                        }
                        dt = std::chrono::duration<double>(clk::now() - t0).count();
                    } else {
                        // BUG FIXED: this accumulated read_range()'s returned DATA SUM (a
                        // checksum), not a byte count, so the reported GiB/s for every
                        // volume-mode CPU row was garbage (one row printed 8.75e11 B in
                        // 13 ms). The timing was always correct; only the byte accounting
                        // was wrong.
                        for (int rep = 0; rep < a.cpu_reps; ++rep) {
                            sink_local += read_range(b0, b1);
                            br += (b1 - b0);
                        }
                        dt = std::chrono::duration<double>(clk::now() - t0).count();
                    }
                    results[t] = br;
                    (void) br;
                    if (dt < cpu_arm.sec) cpu_arm.sec = dt;
                    if (dt > cpu_arm.sec_last) cpu_arm.sec_last = dt;
                });
            }
        }
        if (need_copy) {
            th.emplace_back([&] {
                gate.arrive();
                gate.wait();
                const auto t0 = clk::now();
                size_t moved = 0;
                if (a.dur_ms > 0) {
                    // One sync per FULL-VOLUME pass: a sync per chunk was measured at
                    // ~2.5 ms of overhead (64 MiB chunk: 9.6 GiB/s vs 26.7 async), and no
                    // sync at all lets the issue loop outrun the engine by 1000x (it queued
                    // 1.4 TB in 50 ms). One sync per 581 MiB keeps the queue bounded to one
                    // pass and costs ~0.
                    const auto deadline = t0 + std::chrono::milliseconds(a.dur_ms);
                    long passes = 0;
                    while (true) {
                        for (size_t o = 0; o < g_total; o += chunk) {
                            size_t n = std::min(chunk, g_total - o);
                            hipError_t e = hipMemcpyAsync(dev, (const uint8_t *) host + o, n,
                                                          hipMemcpyHostToDevice, stream);
                            if (e != hipSuccess) {
                                fprintf(stderr, "# copy issue failed: %s\n", hipGetErrorString(e));
                                break;
                            }
                        }
                        const auto p0 = clk::now();
                        hipStreamSynchronize(stream);
                        const double pdt = std::chrono::duration<double>(clk::now() - p0).count();
                        if (getenv("M4PRE_PASSES"))
                            fprintf(stderr, "#   copy pass %ld: sync_drain=%.5f s\n", passes + 1, pdt);
                        moved += g_total;
                        ++passes;
                        if (clk::now() >= deadline) break;
                    }
                    if (getenv("M4PRE_DEBUG"))
                        fprintf(stderr, "# dur copy: passes=%ld moved=%zu\n", passes, moved);
                } else {
                    for (size_t o = 0; o < g_total; o += chunk) {
                        size_t n = std::min(chunk, g_total - o);
                        hipMemcpyAsync(dev, (const uint8_t *) host + o, n,
                                       hipMemcpyHostToDevice, stream);
                    }
                    hipStreamSynchronize(stream);
                    moved = g_total;
                }
                const auto t1 = clk::now();
                copy_arm.sec = copy_arm.sec_last =
                    std::chrono::duration<double>(t1 - t0).count();
                copy_arm.bytes = moved;
                if (a.dur_ms > 0) dur_done.store(true, std::memory_order_release);
            });
        }
        gate.start(nworkers);
        for (auto & x : th) x.join();
        uint64_t acc = 0; for (auto r : results) acc += r;
        sink += acc;
        if (need_cpu) {
            cpu_arm.bytes = (size_t) acc;
            cpu_bytes_per_pass = (size_t) acc;
            cpu_secs.push_back(cpu_arm.sec);
            cpu_secs_last.push_back(cpu_arm.sec_last);
        }
        if (need_copy) { copy_secs.push_back(copy_arm.sec); copy_bytes_per_pass.push_back(copy_arm.bytes); }
    }
    (void) sink;

    const double gib = (double) g_total / 1073741824.0;
    // Throughput of an arm = its own bytes / the time the ARM took to finish. For the
    // CPU arm the finisher is the slowest of T equal-slice workers; sec_min is the
    // fastest worker, reported separately as a best-case bound (they converge once the
    // barrier skew is gone). The copy arm is a single worker, so min == max.
    auto report = [&](const char * arm, std::vector<double> & first, std::vector<double> & last,
                      size_t bytes, int workers) {
        std::vector<double> f(first), l(last);
        std::sort(f.begin(), f.end());
        std::sort(l.begin(), l.end());
        const double fmin = f.front(), fmed = f[f.size() / 2];
        const double lmin = l.front(), lmed = l[l.size() / 2];
        const double bg = (double) bytes / 1073741824.0;
        const double lmax = l.back();
        printf("%-24s bytes=%zu workers=%d  GiBps_completion=%.3f (sec=%.5f)  GiBps_med=%.3f (sec=%.5f)"
               "  GiBps_bestthread=%.3f  GiBps_worst=%.3f  sec_min=%.5f sec_max=%.5f  n=%zu\n",
               arm, bytes, workers, bg / lmin, lmin, bg / lmed, lmed,
               bg / fmin, bg / lmax, fmin, lmax, l.size());
    };
    const size_t cpu_bytes = cpu_bytes_per_pass ? cpu_bytes_per_pass : g_total;
    if (a.mode == "cpu")
        report("cpu_alone", cpu_secs, cpu_secs_last, cpu_bytes, a.threads);
    if (a.mode == "copy") {
        std::vector<size_t> cb(copy_bytes_per_pass);
        std::sort(cb.begin(), cb.end());
        report("copy_alone", copy_secs, copy_secs, cb.empty() ? g_total : cb.front(), 1);
    }
    if (a.mode == "both") {
        report("cpu_concurrent", cpu_secs, cpu_secs_last, cpu_bytes, a.threads);
        {
            std::vector<size_t> cb(copy_bytes_per_pass);
            std::sort(cb.begin(), cb.end());
            report("copy_concurrent", copy_secs, copy_secs, cb.empty() ? g_total : cb.front(), 1);
        }
        // Both arms cover the SAME byte volume, so if they finish together the pair has
        // moved 2 x g_total bytes in the joint window: the sustainable combined DRAM
        // read rate is g_total/time_cpu + g_total/time_copy. That sum is the quantity the
        // balance point depends on: T_total = M / (r_cpu + r_copy).
        {
            std::vector<double> c(cpu_secs_last), p(copy_secs);
            std::sort(c.begin(), c.end()); std::sort(p.begin(), p.end());
            std::vector<size_t> cb(copy_bytes_per_pass);
            std::sort(cb.begin(), cb.end());
            const double cb_b = (double) (cb.empty() ? g_total : cb.front());
            const double rc = (double) cpu_bytes / 1073741824.0 / c.front();
            const double rp = cb_b / 1073741824.0 / p.front();
            printf("combined_concurrent       GiBps_sum=%.3f  (cpu_completion %.3f + copy %.3f)"
                   "  sec_cpu=%.5f sec_copy=%.5f  ratio_copy_over_cpu=%.3f\n",
                   rc + rp, rc, rp, c.front(), p.front(), rp / rc);
        }
    }
    printf("# load1_after=%.2f\n", load1());

    if (need_copy) { hipFree(dev); hipStreamDestroy(stream); }
    return 0;
}
