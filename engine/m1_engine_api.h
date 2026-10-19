// engine/m1_engine_api.h — THE SERVER'S ENGINE SEAM, and the ONE sampler (server plan §D/§F).
//
// Two implementations sit behind this interface and BOTH drive the same routing layer:
//
//   FakeEngine  (engine/m1_server.cpp, always built)   — the wire-protocol test engine of U3:
//               CPU-only, ggml-free, no GPU, no gpu.lock. It is what scripts/engine_gate_http.sh
//               gates, and it stays the default so a server built without the engine TU still
//               refuses --engine real loudly rather than substituting a fake for a real model.
//   RealEngine  (engine/m1_real.cpp, built with -DM1_ENGINE_REAL) — M1Session (engine/
//               m1_engine.cpp, the SAME open/step the single-shot binary drives) + m1_tok +
//               m1_chat. This is the product.
//
// WHY THIS HEADER EXISTS AT ALL: the seam was declared inside m1_server.cpp for U3, when there
// was one implementation. The real engine is a second TU, so the interface (and the sampler that
// plan §D says must be MOVED into the engine rather than re-implemented) has to be visible to
// both. The sampler below is U3's code, moved verbatim: splitmix64 + argmax + the
// temperature/top_k/top_p walk. There is exactly one copy, so the fake's gate arms and the
// product's generation cannot drift apart.
//
// Everything here is dependency-free on purpose: a fake-only build must keep linking no ggml
// object at all (scripts/engine_gate_http.sh asserts it with ldd).

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// RNG + sampler. splitmix64 is the per-request stream (plan §D: seeded per request, never
// shared, so a request's output is reproducible from its seed).
// ---------------------------------------------------------------------------
inline uint64_t m1_splitmix64(uint64_t & x) {
    x += 0x9E3779B97F4A7C15ull;
    uint64_t z = x;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

struct Sampling {
    double   temperature = 0.0;  // <= 0 => GREEDY. The server's default is greedy (plan §D);
                                 // note this is NOT OpenAI's 1.0 default, and it is deliberate:
                                 // greedy must be the product's default path.
    double   top_p       = 1.0;  // 1.0 = off
    int      top_k       = 0;    // 0 = off (extension, llama.cpp naming)
    uint64_t seed        = 0;
};

// The per-request RNG stream (plan §D): (--seed, the request's own seed, the request ordinal).
// Per-request and never shared, so a sampling request's output is reproducible from those three
// inputs and two requests cannot consume each other's stream. Greedy never touches it.
inline uint64_t m1_request_rng(uint64_t base_seed, uint64_t req_seed, long long ordinal) {
    uint64_t x = base_seed ^ (req_seed * 0x9E3779B97F4A7C15ull) ^
                 ((uint64_t) ordinal * 0x2545F4914F6CDD1Dull);
    return m1_splitmix64(x);
}

// llama's greedy, transcribed (engine/m1_engine.cpp:193-197, which itself transcribes
// llama-sampler.cpp:1053-1060): the candidate array is walked in vocab-id order and `best` moves
// only on a STRICTLY greater logit, so the LOWEST id wins a tie. The single-shot binary's decode
// path calls exactly this rule, which is why greedy id-parity against it is a meaningful test.
inline int32_t m1_argmax_row(const float * row, int n) {
    int32_t best = 0;
    for (int i = 1; i < n; ++i) {
        if (row[i] > row[best]) best = i;  // strict '>': lowest id wins a tie (llama's greedy)
    }
    return best;
}

// A temperature at or below this is GREEDY, not a tiny temperature. Below it, row[i]/T overflows to
// +inf, mx = inf, p[0] = exp(inf - inf) = NaN, the `u < acc` walk never matches and the function
// returns idx[keep-1] = the LOWEST-logit token — a silently wrong answer produced by an accepted
// request (the validator only rejects T < 0). The plan's own equivalence (T = 1e-9 == argmax) is
// what this threshold implements, and it is one rule for both engines because both call this.
inline constexpr double M1_SAMPLE_MIN_TEMPERATURE = 1e-6;

inline int32_t m1_sample_row(const float * row, int n, const Sampling & sp, uint64_t & rng) {
    if (!(sp.temperature > M1_SAMPLE_MIN_TEMPERATURE)) return m1_argmax_row(row, n);
    std::vector<int> idx((size_t) n);
    for (int i = 0; i < n; ++i) idx[(size_t) i] = i;
    std::sort(idx.begin(), idx.end(), [row](int a, int b) {
        if (row[a] != row[b]) return row[a] > row[b];
        return a < b;
    });
    int keep = n;
    if (sp.top_k > 0 && sp.top_k < keep) keep = sp.top_k;
    std::vector<double> p((size_t) keep);
    const double mx = (double) row[idx[0]] / sp.temperature;
    double sum = 0.0;
    for (int i = 0; i < keep; ++i) {
        p[(size_t) i] = std::exp((double) row[idx[(size_t) i]] / sp.temperature - mx);
        sum += p[(size_t) i];
    }
    if (sp.top_p < 1.0) {
        double cum = 0.0;
        int n_keep = keep;
        for (int i = 0; i < keep; ++i) {
            cum += p[(size_t) i] / sum;
            if (cum >= sp.top_p) { n_keep = i + 1; break; }  // the first index reaching the mass
        }
        if (n_keep < keep) {
            keep  = n_keep;
            sum   = 0.0;
            for (int i = 0; i < keep; ++i) sum += p[(size_t) i];
        }
    }
    const uint64_t r = m1_splitmix64(rng);
    const double u = (double) (r >> 11) * (1.0 / 9007199254740992.0);  // [0,1)
    double acc = 0.0;
    for (int i = 0; i < keep; ++i) {
        acc += p[(size_t) i] / sum;
        if (u < acc) return (int32_t) idx[(size_t) i];
    }
    return (int32_t) idx[(size_t) (keep - 1)];
}

// ---------------------------------------------------------------------------
// the engine's boot configuration
// ---------------------------------------------------------------------------
// The REAL engine's fields below mirror the single-shot binary's own flag names one for one
// (engine/m1_main.cpp), because the loop's configuration is LoopArgs itself: the server fills the
// SAME struct the binary fills, so an id-parity run needs no mapping layer and cannot diverge
// through one. The fake engine reads only kv_max / fake_delay_ms.
struct EngineCfg {
    std::string model_path;          // the GGUF (default: DEFAULT_MODEL, m1_session.h)
    int         kv_max        = 4096;  // plan R4: a NUMERIC parameter (plan §0 C4) — an id-parity
                                       // comparison against builds/engine/m1 needs the SAME value
    int         fake_delay_ms = 0;     // the fake's per-token sleep (concurrency arms)
    // ---- the real engine (m1_session.h's LoopArgs, flag for flag) ----
    bool        arm_store  = false;    // --arm-store   (the product store list; false = oracle arm)
    bool        gpu_store  = false;    // --gpu-store   (the [0,n_res) store range on ROCm0)
    bool        gpu_core   = false;    // --gpu-core    (the layer core + emb + head on ROCm0)
    int         n_res      = 64;       // --n-res      (== the m1 binary's default)
    int         threads    = 16;       // --threads    (the CPU backend's thread count)
    bool        rotate     = false;    // --rotate     (2Q admission control)
    int         rot_admit_k = 4;       // --rot-admit-k
    // --rotate-verify: read every swapped slot back and memcmp it. The refresh path has always
    // honoured it; since U5FIX the per-request RESET's replayed rows are read back too, so the
    // bytes a reset writes to VRAM are measured rather than argued.
    bool        rotate_verify = false;
    std::string res_path;              // --resident-set (the per-layer ordered resident sets)
    // was --n-res given EXPLICITLY? The set's own count IS the resident count, and a conflicting
    // pair is a CONFIGURATION error: the single-shot binary FATALs on it (m1_main.cpp:124-134) and
    // the server must refuse the same way rather than silently preferring one of the two, because
    // the store split is a numeric parameter (a different membership is a different rounding).
    bool        n_res_given = false;
    // ---- per-request state (plan §E) ----
    // 0 = none: THE CONTROL ARM. The position counter still restarts at 0, but the rotation's
    //     membership/policy is left drifted, so "request B == a fresh boot" is EXPECTED TO FAIL.
    //     It exists to prove the reset is what makes parity hold; it is never the default.
    // 1 = exact (the default): m1_session_reset_rotation(), i.e. the VRAM membership replayed back
    //     to its boot permutation plus the 2Q policy rebuilt. Measured and printed per request.
    int         reset      = 1;
    uint64_t    seed       = 0;        // --seed: the base of every request's RNG stream
    std::string request_log;           // --request-log DIR: req-<n>.tokens (the m1 file format)
    int         chat_bos   = 0;        // --chat-bos: M1_CHAT_BOS_TEMPLATE (0) | _TOKENIZER (1)
    bool        verbose    = false;    // --verbose: the per-request engine log lines
};

// A chat message exactly as the HTTP layer hands it over (the engine owns the render).
struct ChatMsg {
    std::string role;
    std::string content;
};

// A CLIENT-caused refusal from the engine's encode step. `status` is 400 for a request the client
// can fix and 500 for one it cannot (a refusal by the pinned template's own date ladder).
struct EncError {
    int         status = 400;
    std::string message;
    std::string param;
    std::string code;
};

typedef std::function<bool(int32_t)> TokenSink;  // false => stop at this position boundary
typedef std::function<bool()>        CancelFn;

struct GenResult {
    std::vector<int32_t> ids;
    const char * finish = "length";  // "stop" (EOS) | "length" | "cancelled"
    // Empty unless the ENGINE refused mid-request — a state violation it cannot serve through
    // (an out-of-range position, a budget it was not supposed to receive). The router answers 500
    // for a non-streaming request and ends a stream, and logs it: never a silent wrong answer.
    std::string error;
};

struct M1Engine {
    virtual ~M1Engine() {}
    virtual const char * name() const = 0;
    virtual int n_vocab() const = 0;
    virtual int kv_max() const = 0;
    virtual int32_t eos_id() const = 0;
    virtual std::string token_text(int32_t id) const = 0;
    virtual std::string detokenize(const std::vector<int32_t> & ids) const = 0;
    // tokenize TEXT (the /v1/completions prompt). The engine applies the tokenizer's own BOS
    // policy, which for this model means one leading id 1.
    virtual std::vector<int32_t> encode(const std::string & text) const = 0;
    // THE CHAT PROMPT. The engine owns the render because only the real engine knows the model's
    // pinned template (the fake's documented placeholder ids are its own). false => *err says why,
    // and the router answers with err->status — a CLIENT-caused failure must never reach the
    // engine's FATALs (plan §G, U2's note 5).
    virtual bool encode_chat(const std::vector<ChatMsg> & msgs, std::vector<int32_t> & out,
                             EncError & err) const = 0;
    // Called ONCE per request, under the single-flight lock and BEFORE any response byte: asserts
    // the per-request ENTRY invariants (position 0, the rotation at its boot state). false =>
    // `why` names the violation and the router answers 500 — never a silent continue (plan §G1).
    virtual bool request_begin(std::string * why) { (void) why; return true; }
    virtual GenResult generate(const std::vector<int32_t> & prompt, int max_tokens,
                               const Sampling & sp, const TokenSink & sink,
                               const CancelFn & cancel) = 0;
};
