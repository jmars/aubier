// engine/m1_server.cpp — the OpenAI-compatible surface for Mistral Small 4. Server plan §F: this
// file is the WIRE PROTOCOL (JSON, HTTP, SSE, error codes, budget, stop strings, usage) and the
// routing layer; the ENGINE behind the seam below is the fake one here or, in a build made with
// -DM1_ENGINE_REAL, the real one in engine/m1_real.cpp (M1Session + m1_tok + m1_chat).
//
// TWO BUILD SHAPES, one source:
//   scripts/engine_build_http.sh  builds/engine/m1_server_fake  — the fake engine only. MEASURED
//       property, asserted by scripts/engine_gate_http.sh: its ldd names no ggml object at all, it
//       never takes scripts/gpu.lock, and the whole wire-protocol gate runs on a loaded box with
//       no GPU. That build REFUSES --engine real loudly rather than substituting the fake.
//   scripts/engine_build_server.sh builds/engine/m1_server      — the product: this file + the
//       engine TU (ggml, ROCm0 rpath, --disable-new-dtags), which TAKES scripts/gpu.lock ITSELF at
//       boot and holds it for its lifetime (plan R1, settled by the user).
//
// THE SEAM (engine/m1_engine_api.h) is what makes both builds possible and keeps U5 honest:
//
//     n_vocab() / kv_max() / eos_id()          static properties
//     encode(text) -> ids                       /v1/completions prompts (m1_tok in the real build)
//     encode_chat(msgs) -> ids, or a refusal    /v1/chat/completions (the pinned renderer; the
//                                               ENGINE owns it, because only it knows the template)
//     token_text(id) -> bytes                   per-token piece; MAY END MID-CODEPOINT
//     detokenize(ids) -> bytes                  the whole text of an id list
//     request_begin() -> why                    the per-request ENTRY invariant, checked before a
//                                               single response byte is written
//     generate(prompt_ids, max_tokens, sampling, sink, cancel) -> {ids, finish_reason}
//
// `generate` reports every produced token through `sink` and asks `cancel` between positions
// only — never inside one (plan §E: an aborted request must not leave pending store work).
//
// B3 — THE BATCH ASSEMBLER (--batch-slots N > 1, plan §D): the same seam's batch_* methods
// (m1_engine_api.h) driven by ONE worker thread over N slots — a joiner prefills alone, then
// every live slot advances one position per engine step; a departed slot is padded inside its
// own region.  The single-flight path below is untouched and remains the whole server at the
// default N=1 (Bar A: byte-identical to the pre-B3 server).

//
// THE FAKE ENGINE, documented so a test can predict it exactly (scripts/engine_gate_http.sh
// asserts the literal expected text, not merely self-consistency):
//   * vocabulary of 1024 ids; ids 0..7 are the special tokens of the real vocab, ids 16..25 are
//     the pieces below, every other id is "" (the fake's encode() only emits ids 64..95).
//   * greedy (temperature <= 0) output is POSITION-DETERMINED and prompt-independent:
//     token i = FAKE_CYCLE[i % 10].  Text of 10 tokens = "Hello world!\n你你stopabcé".
//     Two of the cycle's pieces are a codepoint SPLIT ACROSS TOKENS (0xE4 then 0xBD 0xA0), so
//     the SSE writer's UTF-8 tail buffer is on the tested path, not merely present.
//   * EMITTED TEXT IS ALWAYS VALID UTF-8: a token may carry a bare continuation byte (a real
//     byte-level tokenizer can sample one), and raw invalid bytes would make a JSON body
//     invalid; such bytes are emitted as U+FFFD and the JSON parser refuses them on the way in.
//   * sampling (temperature > 0) builds a deterministic row from (seed, position) and runs the
//     SAME temperature / top_k / top_p algorithm the real engine uses: the sampler lives ONCE, in
//     engine/m1_engine_api.h, and is shared by both implementations (plan §D's "move it into the
//     engine, do not re-implement it twice").
//
// WHAT THIS FILE DOES **NOT** DO, and why it is loud about it:
//   * it does not take scripts/gpu.lock for the fake engine (the fake touches no GPU, and the
//     gate asserts /proc/<pid>/fd holds no lock fd). The REAL engine takes the lock itself at
//     boot — in main(), before the session is built — and holds it for the process's lifetime.
//   * the FAKE engine's chat prompt render is a PLACEHOLDER (see render_chat_prompt); the REAL
//     engine renders through engine/m1_chat.cpp, which is hash-pinned and FATALs on a mismatch.
//   * the fake engine has no state to reset (that is why its request_begin() is the default);
//     the real engine asserts its per-request entry invariant and replays the rotation.
//
// THE RULE THIS FILE EXISTS TO ENFORCE, above all others: a KNOWN parameter at a non-default
// value must be refused 422 BY NAME. Silently ignoring a sampling parameter is a wrong answer
// the client cannot see, so the unsupported table below is checked before anything else.
//
// usage: builds/engine/m1_server [--port 0] [--port-file F] [--max-conns 32] [...]
//        curl 127.0.0.1:<port>/v1/models   # the port is printed at boot

#include "m1_engine_api.h"   // the seam + the ONE sampler (shared with engine/m1_real.cpp)
#include "m1_http.h"
#include "m1_json.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <condition_variable>
#include <time.h>
#include <vector>

#include <fcntl.h>
#include <sys/file.h>
#include <sys/socket.h>
#include <unistd.h>

#define FATAL(...) do { \
    std::fprintf(stderr, "FATAL: " __VA_ARGS__); \
    std::fprintf(stderr, "\n"); \
    std::exit(1); \
} while (0)

#define CHECK(cond, ...) do { if (!(cond)) FATAL(__VA_ARGS__); } while (0)

// ---------------------------------------------------------------------------
// UTF-8 helpers. The server NEVER needs to validate a codepoint; it needs one property only:
// where the last COMPLETE codepoint in a byte buffer ends, so an SSE chunk can never split one
// (plan §F: "partial UTF-8 buffered (<=3 bytes)").
// ---------------------------------------------------------------------------
static size_t utf8_decode(const char * s, size_t n, size_t i, unsigned & cp) {
    const unsigned char c = (unsigned char) s[i];
    size_t need = 1;
    if (c < 0x80u) { cp = c; return 1; }
    if ((c & 0xE0u) == 0xC0u) { need = 2; cp = c & 0x1Fu; }
    else if ((c & 0xF0u) == 0xE0u) { need = 3; cp = c & 0x0Fu; }
    else if ((c & 0xF8u) == 0xF0u) { need = 4; cp = c & 0x07u; }
    else { cp = c; return 1; }  // a stray continuation byte: emit it alone (never swallow it)
    if (i + need > n) { cp = c; return 1; }
    for (size_t k = 1; k < need; ++k) {
        const unsigned char cc = (unsigned char) s[i + k];
        if ((cc & 0xC0u) != 0x80u) { cp = c; return 1; }
        cp = (cp << 6) | (cc & 0x3Fu);
    }
    return need;
}

// Length of the longest prefix of s (of at most `limit` bytes) that ends on a codepoint
// boundary. Used for TWO hold-backs that must compose: only complete codepoints may be emitted,
// and the last (max_stop_len - 1) bytes must be held back for stop matching. Subtracting the
// stop hold-back straight from a codepoint boundary can land INSIDE a codepoint, so the
// subtraction is followed by this floor — that bug split a 3-byte codepoint into raw bytes and
// made the JSON body invalid UTF-8 (caught by U3-arm-15).
static size_t utf8_complete_prefix(const std::string & s, size_t limit) {
    if (limit > s.size()) limit = s.size();
    size_t i = 0, last = 0;
    while (i < limit) {
        const unsigned char c = (unsigned char) s[i];
        size_t need = 1;
        if (c < 0x80u) need = 1;
        else if ((c & 0xE0u) == 0xC0u) need = 2;
        else if ((c & 0xF0u) == 0xE0u) need = 3;
        else if ((c & 0xF8u) == 0xF0u) need = 4;
        else need = 1;  // a stray continuation byte: a 1-byte unit (sanitised on emission)
        if (need > 1) {
            if (i + need > limit) break;  // incomplete at the limit: stop below the lead byte
            bool ok = true;
            for (size_t k = 1; k < need; ++k) {
                if (((unsigned char) s[i + k] & 0xC0u) != 0x80u) { ok = false; break; }
            }
            if (!ok) { i += 1; last = i; continue; }  // invalid sequence: a 1-byte unit
        }
        i += need;
        last = i;
    }
    return last;
}

static size_t utf8_complete_prefix(const std::string & s) {
    return utf8_complete_prefix(s, s.size());
}

// Copy [s, s+n) to out, replacing every byte that is not part of a well-formed UTF-8 sequence
// with U+FFFD. Emission boundaries are codepoint boundaries for VALID sequences, but a byte-level
// tokenizer can hand us a bare continuation byte (a byte-fallback token sampled without its lead
// byte), and a bare invalid byte inside a JSON string makes the RESPONSE invalid JSON. The
// replacement is the only honest option: JSON has no escape for a non-Unicode byte.
static void utf8_sanitize_append(std::string & out, const char * s, size_t n) {
    static const char REPL[] = "\xEF\xBF\xBD";  // U+FFFD
    size_t i = 0;
    while (i < n) {
        const unsigned char c = (unsigned char) s[i];
        size_t need = 0;
        if (c < 0x80u) need = 1;
        else if ((c & 0xE0u) == 0xC0u) need = 2;
        else if ((c & 0xF0u) == 0xE0u) need = 3;
        else if ((c & 0xF8u) == 0xF0u) need = 4;
        bool good = need > 0 && i + need <= n;
        if (good) {
            for (size_t k = 1; k < need; ++k) {
                if (((unsigned char) s[i + k] & 0xC0u) != 0x80u) { good = false; break; }
            }
        }
        if (!good) {
            out.append(REPL, 3);
            ++i;
            continue;
        }
        out.append(s + i, need);
        i += need;
    }
}

// ---------------------------------------------------------------------------
// THE ENGINE SEAM, the sampler and the boot configuration live in engine/m1_engine_api.h: both
// implementations (this file's fake and engine/m1_real.cpp's real one) must see ONE interface and
// ONE sampler, and the header is dependency-free so this fake-only build keeps linking no ggml.
// ---------------------------------------------------------------------------



// ---- the fake vocabulary (documented in the file header) ----
static const char * fake_piece(int32_t id) {
    switch (id) {
        case 0:  return "<unk>";
        case 1:  return "<s>";
        case 2:  return "</s>";
        case 3:  return "[INST]";
        case 4:  return "[/INST]";
        case 5:  return "<pad>";
        case 6:  return "[SYSTEM_PROMPT]";
        case 7:  return "[/SYSTEM_PROMPT]";
        case 16: return "Hello";
        case 17: return " world";
        case 18: return "!";
        case 19: return "\n";
        case 20: return "\xE4\xBD\xA0";      // 你 — one whole codepoint in one token
        case 21: return "\xC3\xA9";          // é
        case 22: return "\xE4";              // 你 SPLIT: the leading byte of one codepoint
        case 23: return "\xBD\xA0";          // ... and its continuation bytes, in the next token
        case 24: return "stop";
        case 25: return "abc";
        default: return "";
    }
}

static const int32_t FAKE_CYCLE[] = {16, 17, 18, 19, 20, 22, 23, 24, 25, 21};
static const int     FAKE_CYCLE_N = (int) (sizeof(FAKE_CYCLE) / sizeof(FAKE_CYCLE[0]));

// the fake's chat prompt (defined with its documentation further down, where the real renderer's
// absence is explained)
static std::string render_chat_prompt(const std::vector<ChatMsg> & messages);

struct FakeEngine : M1Engine {
    EngineCfg cfg;
    explicit FakeEngine(const EngineCfg & c) : cfg(c) {
        f_n = c.batch_slots < 1 ? 1 : c.batch_slots;
        fslots.assign((size_t) f_n, FSlot());
    }

    const char * name() const override { return "fake"; }
    int n_vocab() const override { return 1024; }
    int kv_max() const override { return cfg.kv_max; }
    int32_t eos_id() const override { return 2; }

    std::string token_text(int32_t id) const override { return fake_piece(id); }

    std::string detokenize(const std::vector<int32_t> & ids) const override {
        std::string s;
        for (int32_t id : ids) s += fake_piece(id);
        return s;
    }

    // One id per CODE POINT, in [64,96). Deterministic, so a test can predict prompt_tokens.
    std::vector<int32_t> encode(const std::string & text) const override {
        std::vector<int32_t> out;
        size_t i = 0;
        while (i < text.size()) {
            unsigned cp = 0;
            const size_t len = utf8_decode(text.data(), text.size(), i, cp);
            out.push_back((int32_t) (64 + (int) ((((uint64_t) cp) * 2654435761ull) % 32)));
            i += len;
        }
        return out;
    }

    // THE CHAT PROMPT, fake-side: the placeholder render below, encoded exactly as before (so the
    // wire-protocol gate's arms keep their documented prompt_tokens). The REAL engine's chat
    // render is engine/m1_chat.cpp behind its hash pin; this implementation exists to prove the
    // ROUTE, never to be the product's ids.
    bool encode_chat(const std::vector<ChatMsg> & msgs, std::vector<int32_t> & out,
                     EncError & err) const override {
        (void) err;   // the placeholder cannot refuse: it renders whatever the parser accepted
        out = encode(render_chat_prompt(msgs));
        return true;
    }

    // The row the fake's sampler sees at `step` — the cycle ids hashed from (seed, step), with
    // this step's greedy id pinned at 2.0.  ONE construction shared by the solo pick() and the
    // batched row fill below, so the fake's batched greedy chain IS the solo chain by
    // construction (the engine-side M-dispatch divergence L90 measured does not exist here).
    void fake_row(float * row, int n, uint64_t seed, int step) const {
        std::fill(row, row + n, -30.0f);
        const int32_t greedy = FAKE_CYCLE[(size_t) (step % FAKE_CYCLE_N)];
        uint64_t x = seed ^ 0x9E3779B97F4A7C15ull ^ ((uint64_t) step << 32);
        for (int k = 0; k < FAKE_CYCLE_N; ++k) {
            const uint64_t h = m1_splitmix64(x);
            row[(size_t) FAKE_CYCLE[k]] = (float) ((double) (h >> 40) / 16777216.0 * 2.0 - 1.0);
        }
        row[(size_t) greedy] = 2.0f;
    }

    int32_t pick(int step, const Sampling & sp) const {
        const int32_t greedy = FAKE_CYCLE[(size_t) (step % FAKE_CYCLE_N)];
        if (!(sp.temperature > 0.0)) return greedy;
        // The fake's generatable set is the cycle (plus EOS is never sampled: every other id,
        // EOS included, sits far below the cut), so a sampled chain is a re-ordering of the
        // same pieces and no arm can be surprised by an accidental EOS.
        std::vector<float> row((size_t) n_vocab(), -30.0f);
        fake_row(row.data(), n_vocab(), sp.seed, step);
        uint64_t seed_state = sp.seed;
        uint64_t rng = m1_splitmix64(seed_state) ^ ((uint64_t) step * 0x2545F4914F6CDD1Dull);
        return m1_sample_row(row.data(), n_vocab(), sp, rng);
    }

    GenResult generate(const std::vector<int32_t> & prompt, int max_tokens, const Sampling & sp,
                       const TokenSink & sink, const CancelFn & cancel) override {
        (void) prompt;  // the fake's greedy chain is position-determined (documented)
        GenResult r;
        for (int step = 0; step < max_tokens; ++step) {
            if (cancel && cancel()) { r.finish = "cancelled"; return r; }
            if (cfg.fake_delay_ms > 0) {
                struct timespec ts;
                ts.tv_sec  = cfg.fake_delay_ms / 1000;
                ts.tv_nsec = (long) (cfg.fake_delay_ms % 1000) * 1000000L;
                nanosleep(&ts, nullptr);
            }
            const int32_t tok = pick(step, sp);
            if (tok == eos_id()) { r.finish = "stop"; return r; }
            r.ids.push_back(tok);
            if (sink && !sink(tok)) { r.finish = "cancelled"; return r; }
        }
        r.finish = "length";
        return r;
    }

    // ---- B3: the fake's minimal batched mode (plan §H: the assembler must be gateable
    // WITHOUT a GPU).  No KV cache and no regions IN FACT, but the SAME CONTRACT
    // (m1_engine_api.h): per-slot position/budget bookkeeping, join/depart, and a per-slot
    // row whose greedy argmax is the SOLO chain's pick(sample index) — so the fake's batched
    // greedy answer EQUALS its single-flight answer, and every assembler bug (wrong slot,
    // wrong attribution, a pad advancing, a lost position) shows in the ids.  The per-request
    // stream is SLOT-INDEPENDENT (m1_request_rng(0, req_seed, 0)): a bar must not depend on
    // which slot a request happens to land in.
    struct FSlot {
        int st = 0;          // 0 = never joined, 1 = live, 2 = departed
        int pos = 0;         // the next position a live row feeds
        int n_prompt = 0;
        int max_pos = 0;     // the last position the budgeted walk ever feeds
        uint64_t seed = 0;   // the request's sp.seed (the fake row hashes it per step)
    };
    int             f_n = 1;
    std::vector<FSlot> fslots;

    int n_batch_slots() const override { return f_n; }
    int batch_slot_cap() const override { return f_n <= 1 ? cfg.kv_max : cfg.kv_max / f_n; }

    bool batch_join(int slot, int n_prompt, int max_tokens, uint64_t req_seed,
                    uint64_t * rng, std::string * err) override {
        if (slot < 0 || slot >= f_n) {
            if (err) *err = "batch_join: slot " + std::to_string(slot) + " out of range [0," +
                            std::to_string(f_n) + ")";
            return false;
        }
        const long long cap = batch_slot_cap();
        if ((long long) n_prompt + max_tokens > cap) {
            if (err) *err = "batch_join: prompt " + std::to_string(n_prompt) + " + max_tokens " +
                            std::to_string(max_tokens) + " exceeds the per-slot region " +
                            std::to_string(cap) + " — the router's clamp should have prevented this";
            return false;
        }
        FSlot & f = fslots[(size_t) slot];
        f.st = 0;
        f.pos = 0;
        f.n_prompt = n_prompt;
        f.max_pos = n_prompt + max_tokens - 2;   // positions n_prompt-1 .. this one carry rows
        f.seed = req_seed;
        if (rng) *rng = m1_request_rng(0, req_seed, 0);
        return true;
    }

    // TEST HOOK (B3FIX A1's probe), deliberately NOT a feature: M1_BATCH_POISON_STEP=1 makes
    // the fake refuse its FIRST composed step, driving the assembler's poison path on demand
    // (poison is otherwise unreachable in a correct build — by contract it means an
    // assembler bug).  Announced at boot like every other hook; off unless the env says so.
    static bool mutant_poison_step() {
        const char * m = std::getenv("M1_BATCH_POISON_STEP");
        return m && std::strcmp(m, "1") == 0;
    }

    bool batch_step(const int32_t * tok, const bool * live, float * rows,
                    std::string * err) override {
        (void) tok;   // the fake's chain is position-determined and prompt-independent (documented)
        if (mutant_poison_step()) {
            static bool fired = false;
            if (!fired) {
                fired = true;
                if (err) *err = "MUTANT M1_BATCH_POISON_STEP: the first composed step is refused";
                return false;
            }
        }
        // the fake's per-token sleep applies to the batched step too (once per STEP, the
        // engine-step cost analogue): without it the concurrency arms' windows are empty and
        // --fake-delay-ms would only mean anything on the single-flight path.
        if (cfg.fake_delay_ms > 0) {
            struct timespec ts;
            ts.tv_sec  = cfg.fake_delay_ms / 1000;
            ts.tv_nsec = (long) (cfg.fake_delay_ms % 1000) * 1000000L;
            nanosleep(&ts, nullptr);
        }
        const int nv = n_vocab();
        for (int t = 0; t < f_n; ++t) {
            FSlot & f = fslots[(size_t) t];
            float * row = rows + (size_t) t * (size_t) nv;
            std::fill(row, row + nv, 0.0f);   // pad rows are UNSPECIFIED by contract; keep them tame
            if (!live[t]) continue;
            if (f.st == 0) f.st = 1;
            if (f.pos > f.max_pos) {
                if (err) *err = "fake batch_step: slot " + std::to_string(t) +
                                " walked past its budgeted positions (an assembler bug)";
                return false;
            }
            // sample index of position p = p - n_prompt + 1 (position n_prompt-1's row is the
            // first sampled id — the same mapping the real engine's solo decode uses).  The
            // earlier prefill positions carry NO sample: their row is never read by the
            // assembler, and the index would be negative here.
            if (f.pos >= f.n_prompt - 1)
                fake_row(row, nv, f.seed, f.pos - f.n_prompt + 1);
            ++f.pos;
        }
        return true;
    }

    void batch_depart(int slot) override {
        if (slot >= 0 && slot < f_n) fslots[(size_t) slot].st = 2;
    }
    void batch_log(int, const std::vector<int32_t> &) override {}   // the fake keeps no request log
};

#ifdef M1_ENGINE_REAL
M1Engine * m1_engine_create_real(const EngineCfg & cfg);  // engine/m1_real.cpp (U5)
#endif

static M1Engine * make_engine(const EngineCfg & cfg, bool real) {
#ifdef M1_ENGINE_REAL
    if (real) return m1_engine_create_real(cfg);
#else
    if (real) {
        FATAL("--engine real was requested but this binary was built WITHOUT the real engine "
              "(rebuild with -DM1_ENGINE_REAL and the U5 sources); the fake engine cannot be "
              "silently substituted for a real one");
    }
#endif
    (void) real;
    return new FakeEngine(cfg);
}

// ---------------------------------------------------------------------------
// errors + responses
// ---------------------------------------------------------------------------
struct ApiError {
    int         status  = 400;
    std::string type    = "invalid_request_error";
    std::string message;
    std::string param;    // empty => null
    std::string code;     // empty => null
    std::string extra;    // extra JSON members INSIDE "error" (no leading comma)
    std::string headers;  // extra response headers, e.g. "Allow: POST\r\n"
};

static void send_json(m1http::Conn & c, int status, const std::string & body,
                      const char * extra_headers = nullptr) {
    if (!c.head(status, "application/json", extra_headers, (long long) body.size())) return;
    c.put(body);
}

static void send_api_error(m1http::Conn & c, const ApiError & e) {
    const std::string body =
        m1http::error_body(e.type.c_str(), e.message.c_str(), e.param.empty() ? nullptr : e.param.c_str(),
                           e.code.empty() ? nullptr : e.code.c_str(), e.extra.empty() ? nullptr : e.extra.c_str());
    send_json(c, e.status, body, e.headers.empty() ? nullptr : e.headers.c_str());
}

static void append_usage(std::string & b, long long prompt, long long completion) {
    b += "{\"prompt_tokens\":";
    m1json::append_int(b, prompt);
    b += ",\"completion_tokens\":";
    m1json::append_int(b, completion);
    b += ",\"total_tokens\":";
    m1json::append_int(b, prompt + completion);
    b += "}";
}

// ---------------------------------------------------------------------------
// stop strings + the UTF-8 tail buffer
//
// Two independent hold-backs, both required by plan §F:
//   * STOP DISAMBIGUATION: a stop string (<= 4 of them) may span tokens, so the last
//     (max_stop_len - 1) bytes are held back until they can no longer start a match.
//   * CODEPOINT COMPLETENESS: only complete codepoints are emitted, so a chunk never splits one
//     (a 3-byte tail at most). The bytes are matched for stop strings BEFORE either hold-back,
//     because the match is on the detokenized TEXT, not on what has been flushed.
// ---------------------------------------------------------------------------
// TEST HOOK, deliberately NOT a feature. M1_SERVER_MUTANT=split_utf8 disables the codepoint
// tail buffer, so a token whose bytes end mid-codepoint is flushed raw and an SSE chunk splits
// a UTF-8 sequence. The gate REQUIRES its UTF-8 arm to fail on this (the asymmetry proof for
// the one property that cannot be established by reading the code). Off unless asked for.
static bool mutant_split_utf8() {
    const char * m = std::getenv("M1_SERVER_MUTANT");
    return m && std::strcmp(m, "split_utf8") == 0;
}

struct StopMatcher {
    std::vector<std::string> stops;
    size_t max_len  = 0;
    std::string text;      // every byte produced so far
    size_t emitted  = 0;   // bytes already handed to the client
    bool   matched  = false;
    size_t match_at = 0;

    void init(const std::vector<std::string> & s) {
        stops.clear();
        for (const std::string & x : s) {
            if (!x.empty()) stops.push_back(x);
        }
        max_len = 0;
        for (const std::string & x : stops) max_len = std::max(max_len, x.size());
    }

    // Appends newly emittable bytes to out. Returns true once a stop string matched: the caller
    // must stop generating and report finish_reason "stop".
    bool push(const std::string & piece, std::string & out) {
        const size_t prev = text.size();
        text += piece;
        if (!matched && max_len > 0) {
            const size_t back = max_len - 1;
            const size_t from = prev > back ? prev - back : 0;
            size_t best = std::string::npos;
            for (const std::string & s : stops) {
                const size_t p = text.find(s, from);
                if (p != std::string::npos && (best == std::string::npos || p < best)) best = p;
            }
            if (best != std::string::npos) {
                matched  = true;
                match_at = best;
            }
        }
        if (matched) {
            if (match_at > emitted) {
                utf8_sanitize_append(out, text.data() + emitted, match_at - emitted);
                emitted = match_at;
            }
            return true;
        }
        const size_t safe = utf8_complete_prefix(text);
        const size_t keep = max_len > 0 ? max_len - 1 : 0;
        // the stop hold-back must not land inside a codepoint
        const size_t end  = mutant_split_utf8()
                                ? text.size()  // MUTANT: emit everything, ignoring the boundary
                                : utf8_complete_prefix(text, safe > keep ? safe - keep : 0);
        if (end > emitted) {
            utf8_sanitize_append(out, text.data() + emitted, end - emitted);
            emitted = end;
        }
        return false;
    }

    // Flush what is left at end-of-generation. A trailing partial codepoint is DROPPED (and
    // reported by the caller): emitting half a codepoint would be the bug this buffering
    // exists to prevent.
    void finish(std::string & out, bool * dropped_partial) {
        if (dropped_partial) *dropped_partial = false;
        if (matched) return;
        const size_t safe = utf8_complete_prefix(text);
        if (safe > emitted) {
            utf8_sanitize_append(out, text.data() + emitted, safe - emitted);
            emitted = safe;
        }
        if (dropped_partial) *dropped_partial = emitted < text.size();
    }
};

// ---------------------------------------------------------------------------
// the server object
// ---------------------------------------------------------------------------
struct Server {
    M1Engine *                      engine = nullptr;
    std::string                     model_name = "mistral-small-4";
    int                             queue_wait_ms = 30000;
    std::timed_mutex                flight;   // SINGLE FLIGHT: one engine, one KV, one rotation
    std::atomic<unsigned long long> requests{0};
    bool                            verbose = false;
    // B3 — THE BATCH ASSEMBLER (--batch-slots N > 1, plan §D).  At the default 1 `pool` is
    // null and the single-flight path above IS the whole server, byte for byte as it was: the
    // assembler is reached only through explicit configuration, exactly like every other
    // default-off mechanism in this engine.  --queue-wait-ms governs only the single-flight
    // wait; at N > 1 seating is bounded by batch_window_ms instead.
    int                             batch_slots = 1;
    int                             batch_window_ms = 10;
    struct BatchPool *              pool = nullptr;
};

// ---------------------------------------------------------------------------
// request validation
//
// One rule dominates: a KNOWN parameter at a non-default value is either SERVED or refused 422
// BY NAME. It is never silently ignored (plan §F). Unknown keys ARE ignored — that is what
// lets ordinary OpenAI clients talk to us at all.
// ---------------------------------------------------------------------------
struct Validator {
    const m1json::Value & root;
    ApiError &            err;

    const m1json::Value * at(const char * k) const { return root.find(k); }

    bool fail(int status, const char * type, std::string msg, const char * param, const char * code) {
        err.status  = status;
        err.type    = type ? type : "invalid_request_error";
        err.message = std::move(msg);
        if (param) err.param = param;
        if (code) err.code = code;
        return false;
    }
    bool bad_type(const char * param, const char * want) {
        return fail(400, "invalid_request_error", std::string("'") + param + "' must be " + want,
                    param, "invalid_type");
    }
    bool bad_value(const char * param, const char * want) {
        return fail(400, "invalid_request_error", std::string("'") + param + "' " + want, param,
                    "invalid_value");
    }
    bool unsupported(const char * param, const char * detail = nullptr) {
        err.status = 422;
        err.type   = "invalid_request_error";
        err.param  = param;
        err.code   = "unsupported_parameter";
        err.message = std::string("'") + param + "' is recognized but not supported by this "
                      "server";
        if (detail) {
            err.message += " (";
            err.message += detail;
            err.message += ")";
        }
        err.message += "; a non-default value is refused by name rather than silently ignored";
        return false;
    }
};

// "is this known-but-unsupported parameter at its default value?"
static bool param_nondefault(const std::string & k, const m1json::Value & v) {
    long long n = 0;
    double    d = 0.0;
    if (k == "n") return !(m1json::get_int(v, n) && n == 1);
    if (k == "tools" || k == "functions" || k == "tool_choice" || k == "function_call" ||
        k == "parallel_tool_calls")
        return !v.is_null();
    if (k == "logprobs") {  // boolean in the chat API, a count in the legacy completions API
        if (v.is_bool()) return v.boolean;
        return !(m1json::get_int(v, n) && n == 0);
    }
    if (k == "top_logprobs") return !(m1json::get_int(v, n) && n == 0);
    if (k == "logit_bias") return !(v.is_object() && v.obj.empty());
    if (k == "frequency_penalty" || k == "presence_penalty")
        return !(m1json::get_number(v, d) && d == 0.0);
    if (k == "repetition_penalty") return !(m1json::get_number(v, d) && d == 1.0);
    if (k == "echo") return v.is_bool() && v.boolean;
    if (k == "best_of") return !(m1json::get_int(v, n) && n == 1);
    if (k == "suffix") return !(v.is_string() && v.str.empty());
    if (k == "response_format") {
        const m1json::Value * t = v.find("type");
        return !(v.is_object() && t && t->is_string() && t->str == "text");
    }
    return true;
}

static const char * kUnsupportedNames[] = {
    "n", "tools", "functions", "tool_choice", "function_call", "parallel_tool_calls",
    "logprobs", "top_logprobs", "logit_bias", "frequency_penalty", "presence_penalty",
    "repetition_penalty", "echo", "best_of", "suffix", "response_format",
};

static bool check_unsupported(Validator & V) {
    for (const char * name : kUnsupportedNames) {
        const m1json::Value * v = V.at(name);
        if (!v) continue;
        if (!param_nondefault(name, *v)) continue;
        return V.unsupported(name);
    }
    return true;
}

struct GenRequest {
    std::vector<int32_t>     prompt;
    long long                max_tokens = 0;      // AFTER the clamp
    long long                requested  = -1;     // -1 = unspecified
    bool                     clamped    = false;
    Sampling                 sp;
    std::vector<std::string> stops;
    bool                     stream = false;
    bool                     include_usage = false;
    bool                     chat = true;
};

// shared parameter subset: sampling, max_tokens, stop, stream, stream_options, seed
static bool parse_common(Validator & V, GenRequest & g) {
    if (!check_unsupported(V)) return false;

    const m1json::Value * v = nullptr;
    if ((v = V.at("temperature"))) {
        double d = 0.0;
        if (!m1json::get_number(*v, d)) return V.bad_type("temperature", "a number");
        if (d < 0.0) return V.bad_value("temperature", "must be >= 0 (0 selects greedy)");
        g.sp.temperature = d;
    }
    if ((v = V.at("top_p"))) {
        double d = 0.0;
        if (!m1json::get_number(*v, d)) return V.bad_type("top_p", "a number");
        if (!(d > 0.0) || d > 1.0) return V.bad_value("top_p", "must be in (0, 1]");
        g.sp.top_p = d;
    }
    if ((v = V.at("top_k"))) {
        long long n = 0;
        if (!m1json::get_int(*v, n)) return V.bad_type("top_k", "an integer");
        if (n < 0) return V.bad_value("top_k", "must be >= 0 (0 disables the cut)");
        g.sp.top_k = (int) std::min<long long>(n, 1 << 20);
    }
    if ((v = V.at("seed"))) {
        long long n = 0;
        if (!m1json::get_int(*v, n)) return V.bad_type("seed", "an integer");
        if (n < 0) return V.bad_value("seed", "must be >= 0");
        g.sp.seed = (uint64_t) n;
    }
    {
        const m1json::Value * a = V.at("max_tokens");
        const m1json::Value * b = V.at("max_completion_tokens");
        long long ma = -1, mb = -1;
        if (a) {
            if (!m1json::get_int(*a, ma)) return V.bad_type("max_tokens", "an integer");
            if (ma < 1) return V.bad_value("max_tokens", "must be >= 1");
        }
        if (b) {
            if (!m1json::get_int(*b, mb)) return V.bad_type("max_completion_tokens", "an integer");
            if (mb < 1) return V.bad_value("max_completion_tokens", "must be >= 1");
        }
        if (a && b && ma != mb)
            return V.fail(400, "invalid_request_error",
                          "'max_tokens' and 'max_completion_tokens' disagree; send exactly one",
                          "max_tokens", "invalid_value");
        g.requested = b ? mb : (a ? ma : -1);
    }
    if ((v = V.at("stream"))) {
        if (!v->is_bool()) return V.bad_type("stream", "a boolean");
        g.stream = v->boolean;
    }
    if ((v = V.at("stream_options"))) {
        if (!v->is_object()) return V.bad_type("stream_options", "an object");
        for (const auto & kv : v->obj) {
            if (kv.first == "include_usage") {
                if (!kv.second.is_bool()) return V.bad_type("stream_options.include_usage", "a boolean");
                g.include_usage = kv.second.boolean;
            } else {
                return V.unsupported(("stream_options." + kv.first).c_str());
            }
        }
        if (g.include_usage && !g.stream)
            return V.bad_value("stream_options.include_usage", "requires 'stream': true");
    }
    if ((v = V.at("stop"))) {
        std::vector<std::string> raw;
        if (v->is_string()) {
            raw.push_back(v->str);
        } else if (v->is_array()) {
            if (v->arr.size() > 4)
                return V.unsupported("stop", "at most 4 stop strings are supported");
            for (const m1json::Value & e : v->arr) {
                if (!e.is_string()) return V.bad_type("stop[]", "a string");
                raw.push_back(e.str);
            }
        } else {
            return V.bad_type("stop", "a string or an array of strings");
        }
        for (const std::string & s : raw) {
            if (!s.empty()) g.stops.push_back(s);
        }
        if (g.stops.size() > 4) return V.unsupported("stop", "at most 4 non-empty stop strings");
    }
    return true;
}

// `model` is VALIDATED against the single configured name: there is no routing, and a client
// that asks for something else must be told so (404), never quietly served the one model.
static bool check_model(Validator & V, const Server & s) {
    const m1json::Value * v = V.at("model");
    if (!v) return true;  // absent => the only model is implied
    if (!v->is_string()) return V.bad_type("model", "a string");
    if (v->str != s.model_name) {
        return V.fail(404, "invalid_request_error",
                      "model '" + v->str + "' is not served by this server (it serves exactly one: '" +
                          s.model_name + "', and model names are never routed)",
                      "model", "model_not_found");
    }
    return true;
}

static bool parse_messages(Validator & V, std::vector<ChatMsg> & out) {
    const m1json::Value * v = V.at("messages");
    if (!v) return V.fail(400, "invalid_request_error", "'messages' is required", "messages", nullptr);
    if (!v->is_array() || v->arr.empty())
        return V.fail(400, "invalid_request_error", "'messages' must be a non-empty array", "messages",
                      "invalid_type");
    for (const m1json::Value & m : v->arr) {
        if (!m.is_object())
            return V.fail(400, "invalid_request_error", "each message must be an object", "messages",
                          "invalid_type");
        const m1json::Value * role = m.find("role");
        if (!role || !role->is_string())
            return V.bad_type("messages[].role", "a string");
        if (role->str != "system" && role->str != "user" && role->str != "assistant") {
            if (role->str == "tool" || role->str == "function")
                return V.unsupported("tools", "tool/function messages imply tool calling");
            return V.fail(422, "invalid_request_error",
                          "message role '" + role->str + "' is not supported (system, user and "
                          "assistant only)",
                          "messages[].role", "unsupported_role");
        }
        ChatMsg msg;
        msg.role = role->str;
        const m1json::Value * content = m.find("content");
        if (!content || content->is_null()) {
            msg.content.clear();
        } else if (content->is_string()) {
            msg.content = content->str;
        } else if (content->is_array()) {
            // a parts array: text parts are flattened, anything else is refused BY NAME
            for (const m1json::Value & part : content->arr) {
                if (!part.is_object())
                    return V.fail(400, "invalid_request_error",
                                  "each 'content' part must be an object", "messages[].content",
                                  "invalid_type");
                const m1json::Value * t = part.find("type");
                if (!t || !t->is_string()) return V.bad_type("messages[].content.type", "a string");
                if (t->str == "text") {
                    const m1json::Value * x = part.find("text");
                    if (!x || !x->is_string()) return V.bad_type("messages[].content.text", "a string");
                    msg.content += x->str;
                } else if (t->str == "image_url" || t->str == "input_image" ||
                           t->str == "image") {
                    return V.unsupported("image_url", "this server is text-only");
                } else {
                    return V.unsupported(t->str.c_str(), "unknown content part type");
                }
            }
        } else {
            return V.bad_type("messages[].content", "a string or an array of content parts");
        }
        out.push_back(std::move(msg));
    }
    return true;
}

// ---------------------------------------------------------------------------
// PLACEHOLDER chat render.
//
// The product renderer is engine/m1_chat.cpp (U2/U5): Mistral Small 4's [INST] template,
// rendered exactly, with a FATAL on a template-hash mismatch and one BOS (plan §C). This
// placeholder exists ONLY so the plumbing has prompt text to encode while the engine is fake.
// The ids it produces are NOT the product's ids; U5 replaces this function body and nothing
// downstream of it changes. It is documented precisely so the gate can predict prompt_tokens:
//   text = concat over messages of (role + ": " + content + "\n")
// ---------------------------------------------------------------------------
static std::string render_chat_prompt(const std::vector<ChatMsg> & messages) {
    std::string s;
    for (const ChatMsg & m : messages) {
        s += m.role;
        s += ": ";
        s += m.content;
        s += "\n";
    }
    return s;
}

// ---------------------------------------------------------------------------
// the budget pre-check (plan §F). It MUST happen before the engine is entered: the real
// engine's own kv_max refusal is a FATAL that kills the process (m1_engine.cpp:2126-2129 via
// gguf_model.h's FATAL), so a client-caused over-length prompt has to be answered here.
// ---------------------------------------------------------------------------
static bool apply_budget(M1Engine * e, GenRequest & g, ApiError & err) {
    const long long n_prompt = (long long) g.prompt.size();
    const long long kv = (long long) e->kv_max();
    if (n_prompt >= kv) {
        err.status  = 400;
        err.type    = "invalid_request_error";
        err.param   = "prompt";
        err.code    = "prompt_too_long";
        err.message = "prompt has " + std::to_string(n_prompt) + " tokens, which is not less than "
                      "the engine's kv_max (" + std::to_string(kv) + ")";
        err.extra = "\"prompt_tokens\":" + std::to_string(n_prompt) + ",\"kv_max\":" + std::to_string(kv);
        return false;
    }
    const long long room = kv - n_prompt;
    if (g.requested < 0 || g.requested > room) {
        if (g.requested >= 0) {
            g.clamped = true;
            std::fprintf(stderr,
                         "m1_server: max_tokens CLAMPED %lld -> %lld (prompt %lld + max_tokens "
                         "would exceed kv_max %lld); finish_reason will be \"length\"\n",
                         g.requested, room, n_prompt, kv);
        }
        g.max_tokens = room;
    } else {
        g.max_tokens = g.requested;
    }
    return true;
}

// ---------------------------------------------------------------------------
// B3 — THE PER-SLOT BUDGET (plan §D): apply_budget's arithmetic against the slot's REGION
// (kv_max / N) instead of the global window.  The 400's extra members name the slot region
// ({prompt_tokens, kv_max_slot}) and the clamp is region-relative; everything else mirrors
// apply_budget exactly.  NEVER CALLED at batch_slots == 1 — that path is apply_budget above,
// untouched and byte-identical to the pre-B3 server (Bar A).
// ---------------------------------------------------------------------------
static bool apply_budget_slot(Server & srv, GenRequest & g, ApiError & err) {
    M1Engine * e = srv.engine;
    const long long n_prompt = (long long) g.prompt.size();
    const long long kv_slot  = (long long) e->batch_slot_cap();
    if (n_prompt >= kv_slot) {
        err.status  = 400;
        err.type    = "invalid_request_error";
        err.param   = "prompt";
        err.code    = "prompt_too_long";
        err.message = "prompt has " + std::to_string(n_prompt) + " tokens, which is not less than "
                      "the per-slot region (" + std::to_string(kv_slot) + " = kv_max " +
                      std::to_string(e->kv_max()) + " / " + std::to_string(srv.batch_slots) +
                      " slots)";
        err.extra = "\"prompt_tokens\":" + std::to_string(n_prompt) + ",\"kv_max_slot\":" +
                    std::to_string(kv_slot);
        return false;
    }
    const long long room = kv_slot - n_prompt;
    if (g.requested < 0 || g.requested > room) {
        if (g.requested >= 0) {
            g.clamped = true;
            std::fprintf(stderr,
                         "m1_server: max_tokens CLAMPED %lld -> %lld (prompt %lld + max_tokens "
                         "would exceed the per-slot region %lld); finish_reason will be \"length\"\n",
                         g.requested, room, n_prompt, kv_slot);
        }
        g.max_tokens = room;
    } else {
        g.max_tokens = g.requested;
    }
    return true;
}

// ---------------------------------------------------------------------------
// SSE output
// ---------------------------------------------------------------------------
static std::string chunk_prefix(const std::string & id, long long created, const std::string & model,
                                bool chat) {
    std::string b = "{\"id\":";
    m1json::append_string(b, id);
    b += ",\"object\":";
    b += chat ? "\"chat.completion.chunk\"" : "\"text_completion\"";
    b += ",\"created\":";
    m1json::append_int(b, created);
    b += ",\"model\":";
    m1json::append_string(b, model);
    b += ",\"choices\":[";
    return b;
}

static bool sse_data(m1http::Conn & c, const std::string & payload) {
    std::string s = "data: ";
    s += payload;
    s += "\n\n";
    return c.put(s);
}

static bool sse_role_chunk(m1http::Conn & c, const std::string & id, long long created,
                           const std::string & model, bool chat) {
    if (!chat) return true;  // the completions shape has no role delta
    std::string b = chunk_prefix(id, created, model, chat);
    b += "{\"index\":0,\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}";
    return sse_data(c, b);
}

static bool sse_content_chunk(m1http::Conn & c, const std::string & id, long long created,
                              const std::string & model, bool chat, const std::string & text) {
    std::string b = chunk_prefix(id, created, model, chat);
    if (chat) {
        b += "{\"index\":0,\"delta\":{\"content\":";
        m1json::append_string(b, text);
        b += "},\"finish_reason\":null}]}";
    } else {
        b += "{\"index\":0,\"text\":";
        m1json::append_string(b, text);
        b += ",\"logprobs\":null,\"finish_reason\":null}]}";
    }
    return sse_data(c, b);
}

static bool sse_final_chunk(m1http::Conn & c, const std::string & id, long long created,
                            const std::string & model, bool chat, const char * finish) {
    std::string b = chunk_prefix(id, created, model, chat);
    if (chat) b += "{\"index\":0,\"delta\":{},\"finish_reason\":";
    else      b += "{\"index\":0,\"text\":\"\",\"logprobs\":null,\"finish_reason\":";
    m1json::append_string(b, finish);
    b += "}]}";
    return sse_data(c, b);
}

static bool sse_usage_chunk(m1http::Conn & c, const std::string & id, long long created,
                            const std::string & model, bool chat, long long prompt,
                            long long completion) {
    std::string b = chunk_prefix(id, created, model, chat);
    b += "],\"usage\":";
    append_usage(b, prompt, completion);
    b += "}";
    return sse_data(c, b);
}

// TEST HOOK, deliberately NOT a feature. M1_SERVER_MUTANT=drop_done removes the [DONE]
// terminator so scripts/engine_gate_http.sh can prove its own framing check FAILS on a broken
// server (the gate runs its framing arm against this and REQUIRES the failure). It is off
// unless the environment says otherwise, and it says so loudly at boot.
static bool mutant_drop_done() {
    const char * m = std::getenv("M1_SERVER_MUTANT");
    return m && std::strcmp(m, "drop_done") == 0;
}

static bool sse_done(m1http::Conn & c) {
    if (mutant_drop_done()) return true;  // MUTANT: the SSE terminator is dropped
    return sse_data(c, "[DONE]");
}

// ---------------------------------------------------------------------------
// THE ONE CANCEL PREDICATE AND THE ONE finish_reason DECISION (U5FIX 3d).
//
// The two generation drivers — run_generation below and do_debug_tokens — used to carry separate
// copies, and they had DRIFTED: /debug/tokens polled only the shutdown flag, so a client that
// vanished during a long /debug/tokens generation was never cancelled and held the single-flight
// engine to completion (up to kv_max positions), where the same disconnect on the OpenAI routes
// aborts at the next position boundary. It also fabricated a finish_reason for a
// shutdown-cancelled request, where run_generation deliberately emits none — a shutdown is not the
// client's "length". Both decisions live here now, so the routes cannot disagree again.
// ---------------------------------------------------------------------------
static CancelFn make_cancel(m1http::Conn & c) {
    return [&c]() -> bool { return m1http::stopping() || !c.alive(); };
}

// Fills *finish, or returns false when there is NO honest finish_reason to report: "cancelled" with
// no stop-string match means shutdown (or a client that is gone), and then the caller must emit
// nothing rather than invent a reason the client would read as a complete answer.
static bool decide_finish(const StopMatcher & sm, const GenResult & res, const char ** finish) {
    if (!sm.matched && std::strcmp(res.finish, "cancelled") == 0) return false;
    *finish = (sm.matched || std::strcmp(res.finish, "stop") == 0) ? "stop" : "length";
    return true;
}

// ---------------------------------------------------------------------------
// the generation driver: single flight, budget already applied, both response shapes
// ---------------------------------------------------------------------------
static void log_done(const Server & srv, unsigned long long no, const m1http::Request & rq,
                     const char * finish, long long n_prompt, long long n_completion,
                     bool stream) {
    if (!srv.verbose) return;
    std::fprintf(stderr,
                 "m1_server: req %llu %s %s -> 200 prompt=%lld completion=%lld finish=%s%s\n",
                 no, rq.method.c_str(), rq.path.c_str(), n_prompt, n_completion, finish,
                 stream ? " (stream)" : "");
}

// THE ONE 503 SHAPE (B3 reuses it verbatim for a request that cannot be seated within
// --batch-window-ms — the same status/type/code/Retry-After, the message names who holds the
// engine).  `slots` <= 0 prints the single-flight message, byte-identical to the pre-B3
// server's two inline copies, which this replaces.
static void send_engine_busy(m1http::Conn & c, int slots) {
    ApiError e;
    e.status  = 503;
    e.type    = "server_error";
    e.message = slots > 0
                    ? "engine busy: all " + std::to_string(slots) +
                          " batch slots are held; retry after 1s"
                    : "engine busy: another request holds the single-flight engine; retry after 1s";
    e.code    = "engine_busy";
    e.headers = "Retry-After: 1\r\n";
    send_api_error(c, e);
}

// (B3) defined below, after BatchPool: the batched driver run_generation dispatches to.
static void run_batch_generation(Server & srv, m1http::Conn & c, const m1http::Request & rq,
                                 unsigned long long no, GenRequest & g, bool debug_route);

static void run_generation(Server & srv, m1http::Conn & c, const m1http::Request & rq,
                           unsigned long long no, GenRequest & g) {
    // B3: at --batch-slots > 1 the single-flight mutex does not exist; the assembler owns the
    // engine.  Everything above this line (validation, budget, encode) already ran.
    if (srv.pool) {
        run_batch_generation(srv, c, rq, no, g, /*debug_route=*/false);
        return;
    }
    // ---- single flight. A second request waits a bounded time and is then refused with
    // 503 + Retry-After, never run concurrently against the one engine (plan §E).
    std::unique_lock<std::timed_mutex> lock(srv.flight, std::defer_lock);
    bool got = false;
    if (srv.queue_wait_ms <= 0) {
        got = lock.try_lock();
    } else {
        got = lock.try_lock_for(std::chrono::milliseconds(srv.queue_wait_ms));
    }
    if (!got) {
        send_engine_busy(c, 0);
        return;
    }

    const long long n_prompt = (long long) g.prompt.size();
    const std::string id = (g.chat ? "chatcmpl-" : "cmpl-") + std::to_string(no);
    const long long created = (long long) ::time(nullptr);

    // THE PER-REQUEST ENTRY INVARIANT (plan §G1), checked BEFORE a single response byte: the engine
    // must start this request from its boot state (position 0, the rotation clean). A violation is a
    // loud 500 — the request is not served, because serving it would be a silently wrong answer.
    {
        std::string why;
        if (!srv.engine->request_begin(&why)) {
            ApiError e;
            e.status  = 500;
            e.type    = "server_error";
            e.message = "the engine's per-request state is not clean, refusing to generate: " + why +
                        ". This is not a client error: the engine is unusable until the server is "
                        "restarted (the state is only repaired by a successful per-request reset)";
            e.code    = "engine_state_dirty";
            std::fprintf(stderr, "m1_server: req %llu ENGINE STATE REFUSAL (RESTART THE SERVER): %s\n",
                         no, why.c_str());
            send_api_error(c, e);
            return;
        }
    }

    StopMatcher sm;
    sm.init(g.stops);
    std::string text;  // non-stream accumulation
    bool client_gone = false;

    if (g.stream) {
        if (!c.head(200, "text/event-stream",
                    "Cache-Control: no-cache\r\nX-Accel-Buffering: no\r\n", -1))
            return;
        // the FIRST chunk carries delta.role (plan §F); a failure here means the client is
        // already gone, so there is nothing to cancel or report.
        if (!sse_role_chunk(c, id, created, srv.model_name, g.chat)) return;
    }

    const TokenSink sink = [&](int32_t tok) -> bool {
        if (m1http::stopping()) return false;
        std::string delta;
        const bool matched = sm.push(srv.engine->token_text(tok), delta);
        if (!delta.empty()) {
            if (g.stream) {
                if (!sse_content_chunk(c, id, created, srv.model_name, g.chat, delta)) {
                    client_gone = true;
                    return false;
                }
            } else {
                text += delta;
            }
        }
        if (!c.alive()) {
            client_gone = true;
            return false;
        }
        return !matched;
    };
    const CancelFn cancel = make_cancel(c);

    const GenResult res = srv.engine->generate(g.prompt, (int) g.max_tokens, g.sp, sink, cancel);

    // A MID-REQUEST ENGINE REFUSAL (the engine's own "unreachable" guards): answer 500 when no body
    // has been committed, and end an already-open stream WITHOUT a finish_reason — a fabricated one
    // would be a lie the client cannot see.
    if (!res.error.empty()) {
        std::fprintf(stderr, "m1_server: req %llu ENGINE REFUSED: %s\n", no, res.error.c_str());
        if (!g.stream) {
            ApiError e;
            e.status  = 500;
            e.type    = "server_error";
            e.message = "the engine refused to generate this request: " + res.error;
            e.code    = "engine_refused";
            send_api_error(c, e);
        }
        return;
    }

    std::string tail;
    bool dropped_partial = false;
    sm.finish(tail, &dropped_partial);
    if (!tail.empty()) {
        if (g.stream) {
            if (!sse_content_chunk(c, id, created, srv.model_name, g.chat, tail)) return;
        } else {
            text += tail;
        }
    }
    if (dropped_partial)
        std::fprintf(stderr, "m1_server: req %llu: dropped a trailing partial UTF-8 sequence\n", no);
    if (client_gone || !c.alive()) return;  // the client vanished mid-stream (plan §G2)
    const char * finish = "length";
    if (!decide_finish(sm, res, &finish)) return;   // shutdown: no honest finish_reason exists
    // completion_tokens counts every id the engine PRODUCED before it stopped, including the id
    // that completed a stop string (whose tail bytes were then dropped). The engine produced
    // them, so the count is honest; the text is shorter than the count by the stop string.
    const long long n_completion = (long long) res.ids.size();

    if (g.stream) {
        if (!sse_final_chunk(c, id, created, srv.model_name, g.chat, finish)) return;
        if (g.include_usage) {
            if (!sse_usage_chunk(c, id, created, srv.model_name, g.chat, n_prompt, n_completion)) return;
        }
        sse_done(c);
        log_done(srv, no, rq, finish, n_prompt, n_completion, true);
        return;
    }

    std::string b = "{\"id\":";
    m1json::append_string(b, id);
    b += ",\"object\":";
    b += g.chat ? "\"chat.completion\"" : "\"text_completion\"";
    b += ",\"created\":";
    m1json::append_int(b, created);
    b += ",\"model\":";
    m1json::append_string(b, srv.model_name);
    b += ",\"choices\":[{\"index\":0,";
    if (g.chat) {
        b += "\"message\":{\"role\":\"assistant\",\"content\":";
        m1json::append_string(b, text);
        b += "},\"finish_reason\":";
    } else {
        b += "\"text\":";
        m1json::append_string(b, text);
        b += ",\"logprobs\":null,\"finish_reason\":";
    }
    m1json::append_string(b, finish);
    b += "}],\"usage\":";
    append_usage(b, n_prompt, n_completion);
    b += "}";
    send_json(c, 200, b);
    log_done(srv, no, rq, finish, n_prompt, n_completion, false);
}

// B3FIX A2 — the batch worker's per-chunk send deadline.  m1http sets no send timeout, so a
// stream client that stops reading could otherwise park the worker in send_all indefinitely;
// with the deadline the write fails, the conn dies, and the slot departs at that position
// boundary (padded).  10 s for ONE small SSE chunk is generous by orders of magnitude — a
// live reader drains it in milliseconds — and it also bounds the request thread's own tail
// writes on the same conn, which fail the same honest way (no body to a client that is gone).
static constexpr int kBatchSendTimeoutMs = 10000;

// ---------------------------------------------------------------------------
// B3 — THE BATCH ASSEMBLER (plan §D).  N slots, ONE worker thread stepping the engine once
// per batch position, per-slot condition variables, --batch-window-ms as both the formation
// window (an empty pool holds the first prefill up to W ms so a burst can share steps) and
// the seating deadline (all N slots busy for W ms => the single-flight server's own 503
// shape).  This struct exists ONLY at srv.batch_slots > 1; the single-flight path above is
// the whole server otherwise.
//
// THE DISCIPLINE (plan §B/§D — exactly the B2 rig's, with the row composition moved into the
// engine so it cannot drift per engine):
//   * a joiner PREFILLS ALONE: while its prompt walks, every other slot is padded or FROZEN
//     (an idempotent replay of its last fed position — same tok, same pos, same window, so
//     the KV write lands on the same cell with the same bytes and a waiting slot's state
//     cannot move);
//   * then it joins the decode batch: every live slot advances ONE position of its own
//     sequence per engine step;
//   * a departing slot (EOS, stop string, length, cancel, client gone) is PADDED at a
//     position boundary — a one-cell window inside its OWN region; live slots are untouched;
//   * no chunked prefill, no prefill/decode interleave inside one step, no compaction: a
//     freed region is claimed by the next joiner, which rewrites every cell it attends.
//
// THREADING.  The worker owns every slot's walk; a request thread owns its conn from claim
// to publish (the SSE head) and again after the wake (the tail) — while it is parked the
// worker is the only writer, so the conn is never touched by two threads at once.  All
// state changes are under `mu`.  A client that stops READING can park the worker inside a
// socket write exactly as it parks the single-flight engine today (m1http sets no send
// timeout) — with N requests on it instead of one; that is the named cost of this design,
// not a regression the wire layer can fix.
// ---------------------------------------------------------------------------
struct BatchPool {
    Server &   srv;
    M1Engine * eng;        // == srv.engine
    const int  n;
    const int  window_ms;

    struct Slot {
        enum St { FREE, PREPARING, SEATED, PREFILL, DECODE, FINISHED };
        St st = FREE;
        std::condition_variable cv;   // this slot's request thread parks here
        // ---- request context (the request thread fills between claim and publish) ----
        GenRequest          g;        // BY VALUE: the caller's frame parks with us
        m1http::Conn *      conn = nullptr;
        std::string         rid;      // "chatcmpl-<no>" / "cmpl-<no>"
        long long           created = 0;
        unsigned long long  no = 0;
        bool                debug_route = false;
        // ---- walk state (the worker's) ----
        int      fed = 0;            // prompt ids fed so far
        int32_t  tok_in = 0;         // the id the next decode step feeds
        uint64_t rng = 0;            // this request's own stream (plan §E)
        StopMatcher sm;
        std::string text;            // non-stream accumulation
        bool client_gone = false;
        // ---- result (the worker fills; the request thread reads after the wake) ----
        std::vector<int32_t> ids;
        const char * finish = "length";   // the RAW finish: "stop" | "length" | "cancelled"
        std::string error;                // non-empty => 500 (non-stream) / end the stream
    };
    std::vector<Slot>     slots;
    std::mutex            mu;
    std::condition_variable pool_cv;     // seat() waiters + the worker's formation wait
    std::thread           worker;
    bool                  stop_flag = false;   // set once, from shutdown_and_join()
    bool                  worker_done = false; // A1: set under mu after join(); nobody else can
                                               // finish a slot past this point
    bool                  poisoned  = false;   // an engine refusal: nothing honest can be served
    std::string           poison_why;
    std::chrono::steady_clock::time_point pending_since{};   // the formation window's start
    // the occupancy instrument — an OBSERVATION on this boot, never a product number
    unsigned long long steps = 0, live_sum = 0, joins = 0, departures = 0;

    BatchPool(Server & s, int window)
        : srv(s), eng(s.engine), n(s.batch_slots), window_ms(window),
          slots((size_t) s.batch_slots) {}

    Slot & slot(int s) { return slots[(size_t) s]; }

    void start() { worker = std::thread([this] { work(); }); }

    // ---- the request thread's side --------------------------------------------------------
    // Claim a FREE slot within window_ms.  Returns the slot index with st == PREPARING (the
    // worker cannot see it yet); -1 = the window expired (the caller sends the 503 shape);
    // -2 = the pool is poisoned or shutting down (the caller sends the 500 / nothing).
    int claim(const GenRequest & g, m1http::Conn & c, unsigned long long no, bool debug_route) {
        const auto t0 = std::chrono::steady_clock::now();
        std::unique_lock<std::mutex> lk(mu);
        for (;;) {
            if (poisoned || stop_flag) return -2;
            for (int s = 0; s < n; ++s) {
                Slot & q = slots[(size_t) s];
                if (q.st != Slot::FREE) continue;
                q.st = Slot::PREPARING;
                q.g = g;   // the copy keeps prompt/sp/stops alive past the park
                q.conn = &c;
                q.rid = (g.chat ? "chatcmpl-" : "cmpl-") + std::to_string(no);
                q.created = (long long) ::time(nullptr);
                q.no = no;
                q.debug_route = debug_route;
                // a FULL reset — a stale field here would leak into the next request
                q.fed = 0;
                q.tok_in = 0;
                q.rng = 0;
                q.text.clear();
                q.client_gone = false;
                q.ids.clear();
                q.finish = "length";
                q.error.clear();
                // a FULL reset — a stale field here would leak into the next request.  The
                // stop matcher needs a whole-fresh object: init() alone does NOT clear
                // matched/text/emitted (the single-flight path always constructs a fresh one,
                // so it never had to).
                q.sm = StopMatcher();
                q.sm.init(g.stops);
                return s;
            }
            if (window_ms <= 0) return -1;
            const auto deadline = t0 + std::chrono::milliseconds(window_ms);
            pool_cv.wait_until(lk, deadline);   // the re-scan on the next loop sees a late free
            if (std::chrono::steady_clock::now() >= deadline) return -1;
        }
    }

    // Make the prepared slot visible to the worker (the SSE head, if any, is already written).
    // A1: if the pool died (poison, or the worker exited) between claim and here, the worker's
    // sweeps have already finished this slot as PREPARING — seating it now would overwrite
    // FINISHED into a pool nobody will ever step, and wait_done would park the handler forever.
    // Finish it instead: q.error set, so the tail answers the 500 the poisoned pool owes it
    // (a stream just ends — no honest finish_reason exists, exactly like the sweep).
    // TEST HOOK (B3FIX A1's probe), deliberately NOT a feature: M1_BATCH_PUBLISH_DELAY_MS=N
    // sleeps N ms BEFORE seating a slot, holding the request thread in the claim->publish
    // window (the window that otherwise contains only the fast SSE head writes, too narrow
    // to hit a sweep by scheduling).  Announced at boot; off unless the env says so.
    static int publish_delay_ms() {
        const char * m = std::getenv("M1_BATCH_PUBLISH_DELAY_MS");
        return m ? std::atoi(m) : 0;
    }

    void publish(int s) {
        {
            const int pd = publish_delay_ms();
            if (pd > 0) {
                struct timespec ts;
                ts.tv_sec  = pd / 1000;
                ts.tv_nsec = (long) (pd % 1000) * 1000000L;
                nanosleep(&ts, nullptr);
            }
        }
        {
            std::lock_guard<std::mutex> lk(mu);
            Slot & q = slots[(size_t) s];
            if (q.st != Slot::PREPARING) return;   // already finished by a sweep; do not touch
            if (poisoned || stop_flag) {
                q.error = poisoned ? ("the batched engine was poisoned before this request " +
                                      std::to_string(q.no) + " could start: " + poison_why)
                                   : "the server is shutting down before this request can start";
                q.finish = "cancelled";
                q.st = Slot::FINISHED;
                q.cv.notify_all();
                return;
            }
            q.st = Slot::SEATED;
            // A2: from here the WORKER writes this conn; give every write a deadline so a
            // non-reading client cannot park the pool (see kBatchSendTimeoutMs above).
            struct timeval tv;
            tv.tv_sec  = kBatchSendTimeoutMs / 1000;
            tv.tv_usec = (kBatchSendTimeoutMs % 1000) * 1000;
            ::setsockopt(q.conn->fd(), SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof tv);
        }
        pool_cv.notify_all();
    }

    // A PREPARING slot whose SSE head write already failed: the worker never saw it.
    void abandon(int s) {
        {
            std::lock_guard<std::mutex> lk(mu);
            slots[(size_t) s].st = Slot::FREE;
        }
        pool_cv.notify_all();
    }

    // Park until the worker finishes this slot.  On return the worker has handed the slot
    // over and touches neither it nor the conn again.
    // A1 belt-and-braces: publish() and the worker's poison/exit sweeps now guarantee every
    // non-FREE slot reaches FINISHED, but a hang here costs a handler thread + a connection
    // forever, so the wait also observes `worker_done` (set only after the worker thread has
    // been joined, i.e. when nobody else can ever finish the slot) and finishes the slot
    // itself — only the request thread touches a slot past that point, so this is race-free.
    void wait_done(int s) {
        std::unique_lock<std::mutex> lk(mu);
        Slot & q = slots[(size_t) s];
        q.cv.wait(lk, [&] {
            if (q.st == Slot::FINISHED) return true;
            if (worker_done) {
                q.error = "the batch worker exited before this request finished";
                q.finish = "cancelled";
                q.st = Slot::FINISHED;
                return true;
            }
            return false;
        });
    }

    // The request thread is done reading the result; the slot is claimable again.
    void free_slot(int s) {
        {
            std::lock_guard<std::mutex> lk(mu);
            slots[(size_t) s].st = Slot::FREE;
        }
        pool_cv.notify_all();
    }

    void shutdown_and_join() {
        {
            std::lock_guard<std::mutex> lk(mu);
            stop_flag = true;
        }
        pool_cv.notify_all();
        if (worker.joinable()) worker.join();
        {   // wake any handler that parked on a slot the exit sweep already missed
            std::lock_guard<std::mutex> lk(mu);
            worker_done = true;
            for (auto & q : slots) q.cv.notify_all();
        }
    }

    // ---- the worker ------------------------------------------------------------------------
    // A5: EVERY departure/cancellation defers its req-<ordinal>.tokens write to
    // pending_logs — the solo path writes the artifact on every exit including cancelled
    // (m1_real.cpp finish_request), and the batch path now agrees; the writes are flushed
    // OUTSIDE mu (A2: no I/O under the lock).
    std::vector<std::pair<int, std::vector<int32_t>>> pending_logs;

    void cancel_slot_locked(int s, const char * finish, const std::string & error) {
        Slot & q = slots[(size_t) s];
        if (q.st == Slot::FREE || q.st == Slot::FINISHED) return;
        q.finish = finish;
        if (!error.empty()) q.error = error;
        q.st = Slot::FINISHED;
        pending_logs.emplace_back(s, q.ids);   // the engine skips slots that never joined
        q.cv.notify_all();
    }

    void poison_locked(const std::string & why) {
        poisoned = true;
        if (poison_why.empty()) poison_why = why;
        for (int s = 0; s < n; ++s)
            if (slots[(size_t) s].st != Slot::FREE && slots[(size_t) s].st != Slot::FINISHED)
                cancel_slot_locked(s, "cancelled", why);
    }

    void depart_locked(int s, const char * finish) {
        Slot & q = slots[(size_t) s];
        eng->batch_depart(s);   // engine-side: the one-cell pad from the next step
        ++departures;
        if (srv.verbose)
            std::fprintf(stderr,
                         "m1_server: req %llu done (batch slot %d): prompt=%zu produced=%zu "
                         "finish=%s\n",
                         q.no, s, q.g.prompt.size(), q.ids.size(), finish);
        cancel_slot_locked(s, finish, "");
    }

    // A3/A2: the one "is this client gone" predicate.  alive() is PASSIVE (it flips only
    // after a write has failed — mid-generation a dead client is invisible to it);
    // peer_gone() is the ACTIVE check (a RST is visible on the next peek).
    static bool client_dead(const Slot & q) {
        return !q.conn->alive() || q.conn->peer_gone();
    }

    // ONE sampled id's bookkeeping for slot s — runs UNDER mu, does NO I/O.  A departing
    // slot is finished and its log deferred; a live stream slot's delta is handed back for
    // the caller to write OUTSIDE the lock (A2).  Returns true when the request departs.
    // This is the solo path's per-position tail (m1_real.cpp's generate loop +
    // run_generation's sink), same order, same finish mapping: EOS is never emitted, the
    // stop matcher owns stop-vs-length, cancellation is decided at position boundaries only.
    bool sample_bookkeep(int s, int32_t got, std::string & delta_out) {
        Slot & q = slots[(size_t) s];
        if (got == eng->eos_id()) { depart_locked(s, "stop"); return true; }
        q.ids.push_back(got);
        std::string delta;
        const bool matched = q.sm.push(eng->token_text(got), delta);
        if (!delta.empty()) {
            if (q.g.stream) delta_out = std::move(delta);   // written outside mu by the caller
            else            q.text += delta;
        }
        if (client_dead(q)) {
            q.client_gone = true;
            depart_locked(s, "cancelled");
            return true;
        }
        if (matched) { depart_locked(s, "stop"); return true; }
        if ((long long) q.ids.size() >= q.g.max_tokens) { depart_locked(s, "length"); return true; }
        if (m1http::stopping()) { depart_locked(s, "cancelled"); return true; }
        q.tok_in = got;
        return false;
    }

    void work() {
        const int nv = eng->n_vocab();
        std::vector<float> rows((size_t) nv * (size_t) n, 0.0f);
        std::vector<int32_t> tok((size_t) n, 0);
        bool livebuf[8] = {false};   // the seam takes const bool*; n <= 8
        std::string err;
        for (;;) {
            int prefiller = -1;
            {
                std::unique_lock<std::mutex> lk(mu);
                // ---- wait for work, applying the formation window -----------------------------
                for (;;) {
                    if (poisoned) break;
                    prefiller = -1;
                    int ndecode = 0, nseated = 0;
                    for (int s = 0; s < n; ++s) {
                        const Slot::St st = slots[(size_t) s].st;
                        if (st == Slot::PREFILL) prefiller = s;
                        else if (st == Slot::DECODE) ++ndecode;
                        else if (st == Slot::SEATED) ++nseated;
                    }
                    if (prefiller >= 0 || ndecode > 0) break;   // a step is already due
                    if (nseated > 0) {
                        // FORMATION (plan §D): with nothing stepping, hold the first prefill up
                        // to window_ms so a burst can share the batch.  A running batch never
                        // waits: a request that seats mid-decode is joined at the next boundary.
                        if (pending_since == std::chrono::steady_clock::time_point{})
                            pending_since = std::chrono::steady_clock::now();
                        const auto dl = pending_since + std::chrono::milliseconds(window_ms);
                        if (window_ms > 0 && std::chrono::steady_clock::now() < dl) {
                            pool_cv.wait_until(lk, dl);
                            continue;
                        }
                        break;   // the window elapsed: the prefill starts
                    }
                    if (stop_flag) break;   // idle and told to stop
                    pool_cv.wait(lk);       // idle: park until a seat, a free, or the stop
                }
                if (poisoned) { poison_locked(poison_why); break; }
                bool any = prefiller >= 0;
                for (auto & q : slots)
                    any = any || q.st == Slot::DECODE || q.st == Slot::SEATED;
                if (stop_flag && !any) break;
                if (stop_flag) {   // stragglers at shutdown: no honest finish_reason exists
                    for (int s2 = 0; s2 < n; ++s2) cancel_slot_locked(s2, "cancelled", "");
                    break;
                }
                // ---- start a joiner's prefill (at most one prefill runs at a time) -------------
                if (prefiller < 0) {
                    int pick = -1;
                    for (int s = 0; s < n && pick < 0; ++s)
                        if (slots[(size_t) s].st == Slot::SEATED) pick = s;
                    if (pick >= 0) {
                        Slot & q = slots[(size_t) pick];
                        uint64_t rng = 0;
                        std::string jerr;
                        if (!eng->batch_join(pick, (int) q.g.prompt.size(), (int) q.g.max_tokens,
                                             q.g.sp.seed, &rng, &jerr)) {
                            std::fprintf(stderr, "m1_server: req %llu BATCH JOIN REFUSED: %s\n",
                                         q.no, jerr.c_str());
                            poison_locked(jerr);
                            break;
                        }
                        q.rng = rng;
                        q.st = Slot::PREFILL;
                        ++joins;
                        prefiller = pick;
                        pending_since = std::chrono::steady_clock::time_point{};
                    }
                }
                // ---- compose the step: the prefiller advances ALONE, else every DECODE row ----
                for (int s = 0; s < n; ++s) {
                    Slot & q = slots[(size_t) s];
                    const bool adv = prefiller >= 0 ? s == prefiller : q.st == Slot::DECODE;
                    livebuf[s] = adv;
                    if (adv)
                        tok[(size_t) s] = q.st == Slot::PREFILL
                                              ? q.g.prompt[(size_t) q.fed]
                                              : q.tok_in;
                }
            }   // mu released: the step runs against slots the request threads cannot move

            if (!eng->batch_step(tok.data(), livebuf, rows.data(), &err)) {
                std::fprintf(stderr, "m1_server: BATCH STEP REFUSED: %s\n", err.c_str());
                std::lock_guard<std::mutex> lk(mu);
                poison_locked(err);
                break;
            }

            // ---- post-step: BOOKKEEPING under mu, I/O outside it (A2) --------------------------
            // A stalled stream reader can park the worker only in the write loop below —
            // never holding mu — so claims, 503s and every other slot keep moving; the
            // send deadline (set at publish) bounds the park and departs the stalled slot.
            {
                std::unique_lock<std::mutex> lk(mu);
                ++steps;
                pending_since = std::chrono::steady_clock::time_point{};
                int nadv = 0;
                for (int s = 0; s < n; ++s)
                    if (livebuf[s]) ++nadv;
                live_sum += (unsigned long long) nadv;
                struct Write { int s; std::string delta; };
                std::vector<Write> writes;
                for (int s = 0; s < n; ++s) {
                    if (!livebuf[s]) continue;
                    Slot & q = slots[(size_t) s];
                    if (q.st == Slot::PREFILL) {
                        ++q.fed;
                        if (q.fed == (int) q.g.prompt.size()) {
                            // position n_prompt-1's row: the FIRST sampled id (the solo decode's
                            // entry position, m1_real.cpp's mapping)
                            const int32_t got = m1_sample_row(rows.data() + (size_t) s * (size_t) nv,
                                                              nv, q.g.sp, q.rng);
                            std::string delta;
                            if (!sample_bookkeep(s, got, delta)) q.st = Slot::DECODE;   // joins
                            if (!delta.empty()) writes.push_back({ s, std::move(delta) });
                        } else if (client_dead(q) || m1http::stopping()) {
                            // A3: a joiner whose client is gone (or a shutdown) leaves at the
                            // NEXT position boundary — the solo path checks every prefill
                            // position (m1_real.cpp's cancel between steps), and the batch no
                            // longer freezes the whole engine behind a dead client for the
                            // rest of its prompt.  Safe engine-side: the slot has advanced
                            // (last_pos >= 0), so depart's pad lands in its own region.
                            q.client_gone = q.client_gone || client_dead(q);
                            depart_locked(s, "cancelled");
                        }
                    } else if (q.st == Slot::DECODE) {
                        const int32_t got = m1_sample_row(rows.data() + (size_t) s * (size_t) nv,
                                                          nv, q.g.sp, q.rng);
                        std::string delta;
                        sample_bookkeep(s, got, delta);
                        if (!delta.empty()) writes.push_back({ s, std::move(delta) });
                    }
                }
                auto logs = std::move(pending_logs);
                pending_logs.clear();
                lk.unlock();

                // the worker is the ONLY conn writer between publish and FINISHED, so these
                // unlocked writes race nothing; a slot that departed WITH a pending delta
                // (stop string, length) still gets its final content — the request thread's
                // tail runs only after the FINISHED handoff.
                std::vector<int> write_failed;
                for (const Write & w : writes) {
                    const Slot & q = slots[(size_t) w.s];
                    if (!sse_content_chunk(*q.conn, q.rid, q.created, srv.model_name, q.g.chat,
                                           w.delta))
                        write_failed.push_back(w.s);
                }
                for (const auto & lg : logs) eng->batch_log(lg.first, lg.second);

                if (!write_failed.empty()) {
                    std::unique_lock<std::mutex> lk2(mu);
                    for (int s : write_failed) {
                        Slot & q = slots[(size_t) s];
                        if (q.st == Slot::FINISHED) continue;   // already departed another way
                        q.client_gone = true;
                        depart_locked(s, "cancelled");
                    }
                    auto logs2 = std::move(pending_logs);
                    pending_logs.clear();
                    lk2.unlock();
                    for (const auto & lg : logs2) eng->batch_log(lg.first, lg.second);
                }
            }
        }
        // exit: wake everything the loop left behind (a seat racing the stop flag), flush
        // the deferred request logs (A5: the artifact is written on every exit, including
        // cancels — solo-path parity), then print the observation line.
        std::vector<std::pair<int, std::vector<int32_t>>> logs;
        {
            std::lock_guard<std::mutex> lk(mu);
            for (int s2 = 0; s2 < n; ++s2) cancel_slot_locked(s2, "cancelled", "");
            logs = std::move(pending_logs);
            pending_logs.clear();
        }
        for (const auto & lg : logs) eng->batch_log(lg.first, lg.second);
        std::printf("# batch assembler shutdown: steps=%llu mean-live-slots=%.2f joins=%llu "
                    "departures=%llu (an OBSERVATION on this boot, not a product number)\n",
                    steps, steps ? (double) live_sum / (double) steps : 0.0, joins, departures);
        std::fflush(stdout);
    }
};

// ---------------------------------------------------------------------------
// B3 — the batched generation driver: claim → (SSE head) → publish → park while the worker
// walks the slot → the SAME tail run_generation runs (stop flush, finish decision, usage,
// the wire shapes).  The engine interaction is entirely the worker's; this thread touches
// the conn only before publish (the SSE head) and after the wake (the tail).
// ---------------------------------------------------------------------------
static void run_batch_generation(Server & srv, m1http::Conn & c, const m1http::Request & rq,
                                 unsigned long long no, GenRequest & g, bool debug_route) {
    BatchPool & pool = *srv.pool;
    const int s = pool.claim(g, c, no, debug_route);
    if (s == -1) {
        send_engine_busy(c, srv.batch_slots);   // the SAME 503 shape, the slots named
        return;
    }
    if (s < 0) {
        // poisoned (or shutting down): the same 500 the single-flight path answers its entry
        // invariant with — the engine cannot honestly serve this request.
        ApiError e;
        e.status  = 500;
        e.type    = "server_error";
        e.message = "the batched engine is not servable (a batch step was refused, or the server "
                    "is shutting down); the engine is unusable until the server is restarted";
        e.code    = "engine_state_dirty";
        {
            std::lock_guard<std::mutex> lk(pool.mu);
            if (!pool.poison_why.empty()) e.message += ": " + pool.poison_why;
        }
        send_api_error(c, e);
        return;
    }
    BatchPool::Slot & q = pool.slot(s);
    if (g.stream) {
        if (!c.head(200, "text/event-stream",
                    "Cache-Control: no-cache\r\nX-Accel-Buffering: no\r\n", -1)) {
            pool.abandon(s);
            return;
        }
        // the FIRST chunk carries delta.role (plan §F), written here so the worker's first
        // content chunk can never race it
        if (!sse_role_chunk(c, q.rid, q.created, srv.model_name, g.chat)) {
            pool.abandon(s);
            return;
        }
    }
    pool.publish(s);
    pool.wait_done(s);

    // ---- the tail: run_generation's, reading the slot's result ------------------------------
    if (!q.error.empty()) {
        std::fprintf(stderr, "m1_server: req %llu ENGINE REFUSED: %s\n", no, q.error.c_str());
        if (!g.stream) {
            ApiError e;
            e.status  = 500;
            e.type    = "server_error";
            e.message = "the engine refused to generate this request: " + q.error;
            e.code    = "engine_refused";
            send_api_error(c, e);
        }
        pool.free_slot(s);
        return;
    }
    std::string tail;
    bool dropped_partial = false;
    q.sm.finish(tail, &dropped_partial);
    if (!tail.empty()) {
        if (g.stream) {
            if (!sse_content_chunk(c, q.rid, q.created, srv.model_name, g.chat, tail)) {
                pool.free_slot(s);
                return;
            }
        } else {
            q.text += tail;
        }
    }
    if (dropped_partial)
        std::fprintf(stderr, "m1_server: req %llu: dropped a trailing partial UTF-8 sequence\n", no);
    if (q.client_gone || !c.alive()) {   // the client vanished mid-stream (plan §G2)
        pool.free_slot(s);
        return;
    }
    const char * finish = "length";
    {
        GenResult res;   // decide_finish reads exactly {finish}; the ids stay the slot's
        res.finish = q.finish;
        if (!decide_finish(q.sm, res, &finish)) {   // shutdown: no honest finish_reason exists
            pool.free_slot(s);
            return;
        }
    }
    const long long n_prompt = (long long) g.prompt.size();
    const long long n_completion = (long long) q.ids.size();

    if (debug_route) {
        std::string b = "{\"object\":\"debug.tokens\",\"ids\":[";
        for (size_t k = 0; k < q.ids.size(); ++k) {
            if (k) b += ",";
            m1json::append_int(b, q.ids[k]);
        }
        b += "],\"text\":";
        m1json::append_string(b, q.text);
        b += ",\"finish_reason\":";
        m1json::append_string(b, finish);
        b += ",\"usage\":";
        append_usage(b, n_prompt, n_completion);
        b += "}";
        send_json(c, 200, b);
        log_done(srv, no, rq, finish, n_prompt, n_completion, false);
        pool.free_slot(s);
        return;
    }

    if (g.stream) {
        if (!sse_final_chunk(c, q.rid, q.created, srv.model_name, g.chat, finish)) {
            pool.free_slot(s);
            return;
        }
        if (g.include_usage) {
            if (!sse_usage_chunk(c, q.rid, q.created, srv.model_name, g.chat, n_prompt,
                                 n_completion)) {
                pool.free_slot(s);
                return;
            }
        }
        sse_done(c);
        log_done(srv, no, rq, finish, n_prompt, n_completion, true);
        pool.free_slot(s);
        return;
    }

    std::string b = "{\"id\":";
    m1json::append_string(b, q.rid);
    b += ",\"object\":";
    b += g.chat ? "\"chat.completion\"" : "\"text_completion\"";
    b += ",\"created\":";
    m1json::append_int(b, q.created);
    b += ",\"model\":";
    m1json::append_string(b, srv.model_name);
    b += ",\"choices\":[{\"index\":0,";
    if (g.chat) {
        b += "\"message\":{\"role\":\"assistant\",\"content\":";
        m1json::append_string(b, q.text);
        b += "},\"finish_reason\":";
    } else {
        b += "\"text\":";
        m1json::append_string(b, q.text);
        b += ",\"logprobs\":null,\"finish_reason\":";
    }
    m1json::append_string(b, finish);
    b += "}],\"usage\":";
    append_usage(b, n_prompt, n_completion);
    b += "}";
    send_json(c, 200, b);
    log_done(srv, no, rq, finish, n_prompt, n_completion, false);
    pool.free_slot(s);
}

// ---------------------------------------------------------------------------
// route: POST /v1/chat/completions
// ---------------------------------------------------------------------------
static void do_chat(Server & srv, m1http::Conn & c, const m1http::Request & rq,
                    unsigned long long no) {
    m1json::Value     root;
    m1json::Error     jerr;
    if (!m1json::parse(rq.body, root, jerr)) {
        ApiError e;
        e.status  = 400;
        e.message = std::string("malformed JSON request body: ") + jerr.what + " at byte " +
                    std::to_string(jerr.offset);
        e.code = "invalid_json";
        e.extra = "\"offset\":" + std::to_string(jerr.offset);
        send_api_error(c, e);
        return;
    }
    ApiError err;
    Validator val{root, err};
    if (!root.is_object()) {
        val.fail(400, "invalid_request_error", "the request body must be a JSON object", nullptr,
                 "invalid_type");
        send_api_error(c, err);
        return;
    }
    GenRequest g;
    g.chat = true;
    std::vector<ChatMsg> msgs;
    if (!check_model(val, srv) || !parse_common(val, g) || !parse_messages(val, msgs)) {
        send_api_error(c, err);
        return;
    }
    // THE ENGINE OWNS THE CHAT PROMPT (engine/m1_engine_api.h): the real one renders through the
    // hash-pinned template, and the two refusals it can produce are answered HERE, before the
    // engine is entered — 400 for a conversation the template will not accept, 500 for a date its
    // own ladder refuses. Neither may reach the renderer's FATAL (plan §G3, U2's notes 2 and 5).
    std::vector<int32_t> prompt;
    EncError enc;
    if (!srv.engine->encode_chat(msgs, prompt, enc)) {
        ApiError e;
        e.status  = enc.status;
        e.type    = enc.status >= 500 ? "server_error" : "invalid_request_error";
        e.message = enc.message;
        e.param   = enc.param;
        e.code    = enc.code;
        send_api_error(c, e);
        return;
    }
    g.prompt = std::move(prompt);
    if (srv.pool ? !apply_budget_slot(srv, g, err) : !apply_budget(srv.engine, g, err)) {
        send_api_error(c, err);
        return;
    }
    run_generation(srv, c, rq, no, g);
}

// ---------------------------------------------------------------------------
// route: POST /v1/completions
// ---------------------------------------------------------------------------
static void do_completions(Server & srv, m1http::Conn & c, const m1http::Request & rq,
                           unsigned long long no) {
    m1json::Value root;
    m1json::Error jerr;
    if (!m1json::parse(rq.body, root, jerr)) {
        ApiError e;
        e.status  = 400;
        e.message = std::string("malformed JSON request body: ") + jerr.what + " at byte " +
                    std::to_string(jerr.offset);
        e.code  = "invalid_json";
        e.extra = "\"offset\":" + std::to_string(jerr.offset);
        send_api_error(c, e);
        return;
    }
    ApiError err;
    Validator val{root, err};
    if (!root.is_object()) {
        val.fail(400, "invalid_request_error", "the request body must be a JSON object", nullptr,
                 "invalid_type");
        send_api_error(c, err);
        return;
    }
    GenRequest g;
    g.chat = false;
    if (!check_model(val, srv) || !parse_common(val, g)) {
        send_api_error(c, err);
        return;
    }
    const m1json::Value * p = val.at("prompt");
    if (!p) {
        val.fail(400, "invalid_request_error", "'prompt' is required", "prompt", nullptr);
        send_api_error(c, err);
        return;
    }
    if (!p->is_string()) {
        if (p->is_array()) val.unsupported("prompt", "batch/array prompts are not supported");
        else val.bad_type("prompt", "a string");
        send_api_error(c, err);
        return;
    }
    g.prompt = srv.engine->encode(p->str);
    if (srv.pool ? !apply_budget_slot(srv, g, err) : !apply_budget(srv.engine, g, err)) {
        send_api_error(c, err);
        return;
    }
    run_generation(srv, c, rq, no, g);
}

// ---------------------------------------------------------------------------
// route: POST /debug/tokens  {ids:[...], max_tokens:N}
//
// THE DELIBERATE NON-OPENAI SEAM (plan §F). It drives the engine from an explicit id list, so
// U5's id-parity test against the m1 binary is exact and independent of the tokenizer: a
// tokenizer bug can never mask an engine bug. It is the same budget + engine path as the
// OpenAI routes; only the response shape differs (it returns the ids).
// ---------------------------------------------------------------------------
static void do_debug_tokens(Server & srv, m1http::Conn & c, const m1http::Request & rq,
                            unsigned long long no) {
    m1json::Value root;
    m1json::Error jerr;
    if (!m1json::parse(rq.body, root, jerr)) {
        ApiError e;
        e.status  = 400;
        e.message = std::string("malformed JSON request body: ") + jerr.what + " at byte " +
                    std::to_string(jerr.offset);
        e.code  = "invalid_json";
        e.extra = "\"offset\":" + std::to_string(jerr.offset);
        send_api_error(c, e);
        return;
    }
    ApiError err;
    Validator val{root, err};
    if (!root.is_object()) {
        val.fail(400, "invalid_request_error", "the request body must be a JSON object", nullptr,
                 "invalid_type");
        send_api_error(c, err);
        return;
    }
    GenRequest g;
    g.chat = true;
    if (!parse_common(val, g)) {
        send_api_error(c, err);
        return;
    }
    if (g.stream) {
        val.unsupported("stream", "/debug/tokens is a non-streaming seam");
        send_api_error(c, err);
        return;
    }
    const m1json::Value * ids = val.at("ids");
    if (!ids || !ids->is_array() || ids->arr.empty()) {
        val.fail(400, "invalid_request_error", "'ids' must be a non-empty array of integers", "ids",
                 "invalid_type");
        send_api_error(c, err);
        return;
    }
    const long long nv = (long long) srv.engine->n_vocab();
    for (const m1json::Value & e : ids->arr) {
        long long id = 0;
        if (!m1json::get_int(e, id)) {
            val.bad_type("ids[]", "an integer");
            send_api_error(c, err);
            return;
        }
        if (id < 0 || id >= nv) {
            // The real engine would treat an out-of-range id as a FATAL; catch it here.
            val.bad_value("ids[]", "is outside the vocabulary");
            err.extra = "\"n_vocab\":" + std::to_string(nv);
            send_api_error(c, err);
            return;
        }
        g.prompt.push_back((int32_t) id);
    }
    if (srv.pool) {
        // B3: the same budget against the slot's REGION, then the assembler.  This route is
        // the id-parity seam, so it rides the batched engine exactly like the OpenAI routes.
        if (!apply_budget_slot(srv, g, err)) {
            send_api_error(c, err);
            return;
        }
        run_batch_generation(srv, c, rq, no, g, /*debug_route=*/true);
        return;
    }
    if (!apply_budget(srv.engine, g, err)) {
        send_api_error(c, err);
        return;
    }

    std::unique_lock<std::timed_mutex> lock(srv.flight, std::defer_lock);
    bool got = srv.queue_wait_ms <= 0
                   ? lock.try_lock()
                   : lock.try_lock_for(std::chrono::milliseconds(srv.queue_wait_ms));
    if (!got) {
        send_engine_busy(c, 0);
        return;
    }

    // the same per-request ENTRY invariant as the OpenAI routes (plan §G1): this route is the
    // id-parity seam, so a dirty engine state must be refused here too, never measured through.
    {
        std::string why;
        if (!srv.engine->request_begin(&why)) {
            ApiError e;
            e.status  = 500;
            e.type    = "server_error";
            e.message = "the engine's per-request state is not clean, refusing to generate: " + why +
                        ". This is not a client error: the engine is unusable until the server is "
                        "restarted (the state is only repaired by a successful per-request reset)";
            e.code    = "engine_state_dirty";
            std::fprintf(stderr, "m1_server: req %llu ENGINE STATE REFUSAL (RESTART THE SERVER): %s\n",
                         no, why.c_str());
            send_api_error(c, e);
            return;
        }
    }

    StopMatcher sm;
    sm.init(g.stops);
    std::string text;
    const TokenSink sink = [&](int32_t tok) -> bool {
        if (m1http::stopping() || !c.alive()) return false;
        std::string delta;
        const bool matched = sm.push(srv.engine->token_text(tok), delta);
        text += delta;
        return !matched && !m1http::stopping() && c.alive();
    };
    const CancelFn cancel = make_cancel(c);
    const GenResult res = srv.engine->generate(g.prompt, (int) g.max_tokens, g.sp, sink, cancel);

    if (!res.error.empty()) {
        std::fprintf(stderr, "m1_server: req %llu ENGINE REFUSED: %s\n", no, res.error.c_str());
        ApiError e;
        e.status  = 500;
        e.type    = "server_error";
        e.message = "the engine refused to generate this request: " + res.error;
        e.code    = "engine_refused";
        send_api_error(c, e);
        return;
    }

    std::string tail;
    bool dropped_partial = false;
    sm.finish(tail, &dropped_partial);
    text += tail;
    const char * finish = "length";
    if (!decide_finish(sm, res, &finish)) {
        // cancelled by a shutdown or by a client that is gone: the ids/artifact were still written
        // by the engine's own request log, but there is no honest response body to send.
        std::fprintf(stderr, "m1_server: req %llu /debug/tokens cancelled before completion "
                             "(shutdown or dead client); no response body\n", no);
        return;
    }

    std::string b = "{\"object\":\"debug.tokens\",\"ids\":[";
    for (size_t k = 0; k < res.ids.size(); ++k) {
        if (k) b += ",";
        m1json::append_int(b, res.ids[(size_t) k]);
    }
    b += "],\"text\":";
    m1json::append_string(b, text);
    b += ",\"finish_reason\":";
    m1json::append_string(b, finish);
    b += ",\"usage\":";
    append_usage(b, (long long) g.prompt.size(), (long long) res.ids.size());
    b += "}";
    send_json(c, 200, b);
    log_done(srv, no, rq, finish, (long long) g.prompt.size(), (long long) res.ids.size(), false);
}

// ---------------------------------------------------------------------------
// routes that answer WITHOUT the engine (so /health stays live during a long generation)
// ---------------------------------------------------------------------------
static void do_models(Server & srv, m1http::Conn & c) {
    std::string b = "{\"object\":\"list\",\"data\":[{\"id\":";
    m1json::append_string(b, srv.model_name);
    b += ",\"object\":\"model\",\"created\":";
    m1json::append_int(b, (long long) ::time(nullptr));
    b += ",\"owned_by\":\"aubier\"}]}";
    send_json(c, 200, b);
}

static void do_health(Server & srv, m1http::Conn & c) {
    std::string b = "{\"status\":\"ok\",\"engine\":";
    m1json::append_string(b, srv.engine->name());
    b += ",\"model\":";
    m1json::append_string(b, srv.model_name);
    b += ",\"n_vocab\":";
    m1json::append_int(b, srv.engine->n_vocab());
    b += ",\"kv_max\":";
    m1json::append_int(b, srv.engine->kv_max());
    b += ",\"requests\":";
    m1json::append_int(b, (long long) srv.requests.load());
    b += "}";
    send_json(c, 200, b);
}

// ---------------------------------------------------------------------------
// the router
// ---------------------------------------------------------------------------
static void method_not_allowed(m1http::Conn & c, const char * allow) {
    ApiError e;
    e.status  = 405;
    e.type    = "invalid_request_error";
    e.message = std::string("method not allowed on this route (use ") + allow + ")";
    e.code    = "method_not_allowed";
    e.headers = std::string("Allow: ") + allow + "\r\n";
    send_api_error(c, e);
}

static void unknown_route(m1http::Conn & c, const std::string & path) {
    ApiError e;
    e.status  = 404;
    e.type    = "invalid_request_error";
    e.message = "unknown route '" + path + "' (this server serves /v1/chat/completions, "
                "/v1/completions, /v1/models, /health and /debug/tokens)";
    e.code    = "unknown_route";
    send_api_error(c, e);
}

static void handle_request(Server & srv, const m1http::Request & rq, m1http::Conn & c) {
    const unsigned long long no = ++srv.requests;
    const std::string & path = rq.path;
    if (path == "/v1/chat/completions") {
        if (rq.method != "POST") return method_not_allowed(c, "POST");
        if (srv.verbose) std::fprintf(stderr, "m1_server: req %llu POST /v1/chat/completions\n", no);
        do_chat(srv, c, rq, no);
        return;
    }
    if (path == "/v1/completions") {
        if (rq.method != "POST") return method_not_allowed(c, "POST");
        if (srv.verbose) std::fprintf(stderr, "m1_server: req %llu POST /v1/completions\n", no);
        do_completions(srv, c, rq, no);
        return;
    }
    if (path == "/debug/tokens") {
        if (rq.method != "POST") return method_not_allowed(c, "POST");
        if (srv.verbose) std::fprintf(stderr, "m1_server: req %llu POST /debug/tokens\n", no);
        do_debug_tokens(srv, c, rq, no);
        return;
    }
    if (path == "/v1/models") {
        if (rq.method != "GET") return method_not_allowed(c, "GET");
        do_models(srv, c);
        return;
    }
    if (path == "/health") {
        if (rq.method != "GET") return method_not_allowed(c, "GET");
        do_health(srv, c);
        return;
    }
    unknown_route(c, path);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
static void usage(const char * argv0) {
    std::printf(
        "usage: %s [options]\n"
        "  --port N               bind port (0 = ephemeral, the chosen port is printed) [0]\n"
        "  --port-file PATH       write the chosen port to PATH (one line)\n"
        "  --bind ADDR            bind address [127.0.0.1]\n"
        "  --max-conns N          in-flight connection cap, over-cap => 503 + Retry-After [32]\n"
        "  --queue-wait-ms N      how long a second request waits for the single-flight engine\n"
        "                         before 503 [30000].  At --batch-slots > 1 this does not apply:\n"
        "                         seating is bounded by --batch-window-ms instead\n"
        "  --batch-slots N        N > 1 turns the single-flight engine into an N-slot batch\n"
        "                         assembler (plan §D): one engine step advances every live request\n"
        "                         by one position of its own sequence, over fixed per-slot KV\n"
        "                         regions of kv_max/N cells.  A joiner PREFILLS ALONE (the batch\n"
        "                         pauses), then joins; a departed slot is padded.  At N >= 2 a\n"
        "                         batched answer is EXPECTED to differ from the single-flight\n"
        "                         answer for the same request (F16/F32 GEMM dispatch at M=N,\n"
        "                         LEDGER L90/L91) — gated by quality (L92), never id equality;\n"
        "                         the fused m4_* kernels are nt==1 and the batch runs STOCK.\n"
        "                         REFUSES --rotate (rotation under a batch is B4's unit) and\n"
        "                         kv_max/slots must be a multiple of 16.  [1 = single flight]\n"
        "  --batch-window-ms N    the assembler's deadline: how long an EMPTY pool holds the\n"
        "                         first prefill waiting for company (formation), and how long a\n"
        "                         request waits for a free slot before the engine_busy 503. [10]\n"
        "  --model-name NAME      the ONE served model id, everything else 404s [mistral-small-4]\n"
        "  --model PATH           the GGUF path [the pinned Mistral Small 4 file]\n"
        "  --engine fake|real     engine implementation [fake] (real needs -DM1_ENGINE_REAL,\n"
        "                         i.e. scripts/engine_build_server.sh)\n"
        "  --kv-max N             the engine's KV width [4096].  NUMERIC, not just capacity: the\n"
        "                         softmax reduces over the padded width, so an id-parity run\n"
        "                         against builds/engine/m1 must use the SAME --kv-max on both sides\n"
        "  --fake-kv-max N        the fake engine's kv_max (an alias for --kv-max)\n"
        "  --fake-delay-ms N      sleep N ms per generated token (a slow fake, for the\n"
        "                         concurrency arms) [0]\n"
        "  --read-timeout-ms N    per-connection read timeout [30000]\n"
        "  --drain-ms N           shutdown wait for in-flight handlers [10000]\n"
        "  --verbose              per-request log lines\n"
        "  --help\n"
        "\n"
        "THE REAL ENGINE's configuration — the single-shot binary's own flags, one for one\n"
        "  --arm-store | --arm-oracle  the product store list / the oracle arm [oracle]\n"
        "  --gpu-store            the [0,n_res) store range on ROCm0\n"
        "  --gpu-core             the layer core + embedding + head on ROCm0 (needs --arm-store)\n"
        "  --n-res N              resident experts per layer [64]\n"
        "  --threads N            the CPU backend's threads [16]\n"
        "  --rotate              2Q admission control over the VRAM membership\n"
        "  --rotate-verify       read every swapped slot back and memcmp it — the refreshes AND\n"
        "                        the per-request reset's replayed rows (costs a row read per slot)\n"
        "  --rot-admit-k N        the 2Q window's threshold [4]\n"
        "  --resident-set PATH    the per-layer ORDERED resident sets (implies --n-res)\n"
        "  --reset exact|none     per-request state reset [exact].  exact replays the VRAM\n"
        "                         membership + the 2Q policy back to boot (the reset cost is\n"
        "                         MEASURED and printed per request); none is the CONTROL arm —\n"
        "                         request B is then EXPECTED to differ from a fresh boot\n"
        "  --seed N               the base of every request's RNG stream [0]\n"
        "  --chat-bos template|tokenizer   one BOS (the template's) / llama-server's double BOS\n"
        "  --request-log DIR      write req-<n>.tokens per request (the m1 file format: u32 n,\n"
        "                         then n i32) — the artifact the id-parity test compares\n"
        "  --gpu-lock PATH        the exclusive GPU lock file [scripts/gpu.lock]\n"
        "  --lock-wait-ms N       how long to wait for it at boot, then exit 69 [30000]\n"
        "\n"
        "THE FAKE-ENGINE BUILD (builds/engine/m1_server_fake, scripts/engine_build_http.sh) is\n"
        "CPU-only, links no ggml and does NOT take scripts/gpu.lock — the wire-protocol gate runs\n"
        "without a GPU. THE REAL-ENGINE BUILD (builds/engine/m1_server,\n"
        "scripts/engine_build_server.sh) TAKES scripts/gpu.lock ITSELF at boot and HOLDS it for\n"
        "its lifetime: while a server is up no measurement row can take the lock (plan R1).\n",
        argv0);
}

// ---------------------------------------------------------------------------
// THE BOOT LOCK (plan R1, settled by the user): the server is a long-lived GPU tenant, so it takes
// scripts/gpu.lock itself and holds the fd for its whole lifetime — the mutual exclusion is
// structural, not a convention a caller must remember. Mirrors scripts/with_gpu_lock.sh's contract
// (the same file, the same exclusive flock, the same exit 69 when the resource is unavailable), plus
// a BOUNDED wait so a server started while a measurement row runs reports honestly instead of
// hanging.
// ---------------------------------------------------------------------------
#ifdef M1_ENGINE_REAL
static std::string default_gpu_lock_path() {
    // the project root is <binary dir>/../.. (builds/engine/m1_server), so a server started from any
    // CWD still locks the ONE file every script locks. CWD-relative remains the fallback.
    char buf[4096];
    const ssize_t n = ::readlink("/proc/self/exe", buf, sizeof buf - 1);
    if (n > 0) {
        buf[n] = '\0';
        std::string p(buf);
        const size_t slash = p.rfind('/');
        if (slash != std::string::npos) {
            p = p.substr(0, slash) + "/../../scripts/gpu.lock";
            if (::access(p.c_str(), F_OK) == 0) return p;
        }
    }
    return "scripts/gpu.lock";
}

// returns the held fd (kept open for the process's lifetime) or -1 after printing the reason
static int take_gpu_lock(const std::string & path, int wait_ms) {
    if (::access(path.c_str(), F_OK) != 0) {
        std::fprintf(stderr, "m1_server: FATAL: lock file %s missing (scaffold broken)\n",
                     path.c_str());
        return -1;
    }
    const int fd = ::open(path.c_str(), O_RDWR | O_CLOEXEC);
    if (fd < 0) {
        std::fprintf(stderr, "m1_server: FATAL: cannot open the GPU lock %s: %s\n", path.c_str(),
                     std::strerror(errno));
        return -1;
    }
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(wait_ms);
    for (;;) {
        if (::flock(fd, LOCK_EX | LOCK_NB) == 0) return fd;
        if (errno != EWOULDBLOCK && errno != EINTR) {
            std::fprintf(stderr, "m1_server: FATAL: flock(%s) failed: %s\n", path.c_str(),
                         std::strerror(errno));
            ::close(fd);
            return -1;
        }
        if (std::chrono::steady_clock::now() >= deadline) {
            std::fprintf(stderr,
                         "m1_server: FATAL: cannot take the exclusive GPU lock %s within %d ms — "
                         "another run holds it.\n  the GPU is a global serial resource and an "
                         "unlocked run can collide with a peer lane; exiting 69 (EX_UNAVAILABLE) "
                         "rather than serving without it\n", path.c_str(), wait_ms);
            ::close(fd);
            return -1;
        }
        struct timespec ts;
        ts.tv_sec = 0;
        ts.tv_nsec = 100L * 1000000L;   // 100 ms
        nanosleep(&ts, nullptr);
    }
}
#endif  // M1_ENGINE_REAL

int main(int argc, char ** argv) {
    EngineCfg   ecfg;
    std::string port_file;
    std::string bind_addr = "127.0.0.1";
    int         port = 0;
    int         max_conns = 32;
    int         read_timeout_ms = 30000;
    int         drain_ms = 10000;
    bool        real_engine = false;
    std::string gpu_lock;
    int         lock_wait_ms = 30000;
    Server      srv;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        const bool has_next = (i + 1 < argc);
        if (a == "--help" || a == "-h") { usage(argv[0]); return 0; }
        else if (a == "--port" && has_next) port = std::atoi(argv[++i]);
        else if (a == "--port-file" && has_next) port_file = argv[++i];
        else if (a == "--bind" && has_next) bind_addr = argv[++i];
        else if (a == "--max-conns" && has_next) max_conns = std::atoi(argv[++i]);
        else if (a == "--queue-wait-ms" && has_next) srv.queue_wait_ms = std::atoi(argv[++i]);
        else if (a == "--batch-slots" && has_next) srv.batch_slots = std::atoi(argv[++i]);
        else if (a == "--batch-window-ms" && has_next) srv.batch_window_ms = std::atoi(argv[++i]);
        else if (a == "--model-name" && has_next) srv.model_name = argv[++i];
        else if (a == "--model" && has_next) ecfg.model_path = argv[++i];
        else if (a == "--engine" && has_next) real_engine = (std::strcmp(argv[++i], "real") == 0);
        else if (a == "--kv-max" && has_next) ecfg.kv_max = std::atoi(argv[++i]);
        else if (a == "--fake-kv-max" && has_next) ecfg.kv_max = std::atoi(argv[++i]);
        else if (a == "--fake-delay-ms" && has_next) ecfg.fake_delay_ms = std::atoi(argv[++i]);
        else if (a == "--read-timeout-ms" && has_next) read_timeout_ms = std::atoi(argv[++i]);
        else if (a == "--drain-ms" && has_next) drain_ms = std::atoi(argv[++i]);
        else if (a == "--verbose") { srv.verbose = true; ecfg.verbose = true; }
        // ---- the real engine's product configuration (the m1 binary's flags, verbatim) ----
        else if (a == "--arm-store") ecfg.arm_store = true;
        else if (a == "--arm-oracle") ecfg.arm_store = false;
        else if (a == "--gpu-store") ecfg.gpu_store = true;
        else if (a == "--gpu-core") ecfg.gpu_core = true;
        else if (a == "--n-res" && has_next) { ecfg.n_res = std::atoi(argv[++i]);
                                                  ecfg.n_res_given = true; }
        else if (a == "--threads" && has_next) ecfg.threads = std::atoi(argv[++i]);
        else if (a == "--rotate") ecfg.rotate = true;
        else if (a == "--rotate-verify") ecfg.rotate_verify = true;
        else if (a == "--rot-admit-k" && has_next) ecfg.rot_admit_k = std::atoi(argv[++i]);
        else if (a == "--resident-set" && has_next) ecfg.res_path = argv[++i];
        else if (a == "--reset" && has_next) {
            const std::string mode = argv[++i];
            if (mode == "exact")      ecfg.reset = 1;
            else if (mode == "none")  ecfg.reset = 0;
            else FATAL("--reset %s: only 'exact' (the default) and 'none' (the CONTROL arm) exist; "
                       "'policy' is deliberately NOT shipped — it drops the membership replay, whose "
                       "equivalence to a fresh boot has never been measured, and the instrument that "
                       "could measure it (the paired per-position dNLL) has no verdict for it", mode.c_str());
        }
        else if (a == "--seed" && has_next) ecfg.seed = (uint64_t) std::strtoull(argv[++i], nullptr, 10);
        else if (a == "--chat-bos" && has_next) {
            const std::string m = argv[++i];
            if (m == "template")       ecfg.chat_bos = 0;
            else if (m == "tokenizer") ecfg.chat_bos = 1;
            else FATAL("--chat-bos %s: template|tokenizer", m.c_str());
        }
        else if (a == "--request-log" && has_next) ecfg.request_log = argv[++i];
        else if (a == "--gpu-lock" && has_next) gpu_lock = argv[++i];
        else if (a == "--lock-wait-ms" && has_next) lock_wait_ms = std::atoi(argv[++i]);
        else FATAL("unknown argument '%s' (try --help)", argv[i]);
    }
    CHECK(port >= 0 && port <= 65535, "bad --port %d", port);
    CHECK(max_conns >= 1, "--max-conns must be >= 1 (the accept loop refuses at the cap)");
    CHECK(ecfg.kv_max >= 2, "--kv-max must be >= 2");
    CHECK(ecfg.threads >= 1, "--threads must be >= 1");
    CHECK(ecfg.n_res >= 1, "--n-res must be >= 1");
    CHECK(lock_wait_ms >= 0, "--lock-wait-ms must be >= 0");
    CHECK(!srv.model_name.empty(), "--model-name must not be empty");
    // ---- B3: the assembler's configuration contract, refused at STARTUP with both flags
    // named — never a silent downgrade (the U4 lesson: a silent fallback presents as a
    // performance result that is not one).
    CHECK(srv.batch_slots >= 1 && srv.batch_slots <= 8,
          "--batch-slots %d is out of range [1,8] (the session's batch width cap, "
          "m1_engine.cpp open)", srv.batch_slots);
    CHECK(srv.batch_window_ms >= 0, "--batch-window-ms must be >= 0 (0 = no formation wait, "
          "no seating wait)");
    if (srv.batch_slots > 1) {
        CHECK(!ecfg.rotate,
              "--batch-slots %d REFUSES --rotate: rotation over the batch's union is B4's unit "
              "(the union-driven membership changes the store split's float accumulation, so "
              "per-sequence equality is dead by construction, plan §C) — drop --rotate or run "
              "--batch-slots 1", srv.batch_slots);
        CHECK(ecfg.kv_max % srv.batch_slots == 0,
              "--kv-max %d is not divisible by --batch-slots %d (the per-slot KV region is a "
              "FIXED contiguous block of kv_max/slots cells, plan §A option ii)",
              ecfg.kv_max, srv.batch_slots);
        CHECK((ecfg.kv_max / srv.batch_slots) % 16 == 0,
              "--kv-max %d / --batch-slots %d = %d cells per slot is not a multiple of 16: "
              "region bases must stay 16-cell aligned — B2FIX MEASURED that a 1-cell region "
              "translation is NOT bit-inert on the kq/kqv lane-blocked reductions while "
              "16-cell steps are inert", ecfg.kv_max, srv.batch_slots,
              ecfg.kv_max / srv.batch_slots);
    }
    ecfg.batch_slots = srv.batch_slots;

#ifdef M1_ENGINE_REAL
    // THE BOOT LOCK, before the engine touches the card: the fd is held for the process's lifetime
    // and never closed (the kernel releases it at exit, which is the whole point).
    int lock_fd = -1;
    if (real_engine) {
        if (gpu_lock.empty()) gpu_lock = default_gpu_lock_path();
        lock_fd = take_gpu_lock(gpu_lock, lock_wait_ms);
        if (lock_fd < 0) return 69;   // EX_UNAVAILABLE, exactly like scripts/with_gpu_lock.sh
        std::printf("m1_server: holding the exclusive GPU lock %s for this process's lifetime\n",
                    gpu_lock.c_str());
        std::fflush(stdout);
    }
#else
    (void) lock_wait_ms;
    if (!gpu_lock.empty()) FATAL("--gpu-lock only means something for the real engine");
#endif

    srv.engine = make_engine(ecfg, real_engine);
    std::printf("m1_server: engine=%s model=%s kv_max=%d n_vocab=%d max_conns=%d "
                "queue_wait_ms=%d\n",
                srv.engine->name(), srv.model_name.c_str(), srv.engine->kv_max(),
                srv.engine->n_vocab(), max_conns, srv.queue_wait_ms);
    if (std::strcmp(srv.engine->name(), "fake") == 0) {
        std::printf("m1_server: FAKE engine — CPU-only, no ggml, no GPU, and it does NOT take "
                    "scripts/gpu.lock (the wire protocol is what is under test)\n");
    }
    if (srv.batch_slots > 1) {
        // B3: build the assembler and start its worker BEFORE the first request can arrive.
        // The engine behind it is already N wide (its own boot line says so).
        srv.pool = new BatchPool(srv, srv.batch_window_ms);
        srv.pool->start();
        std::printf("m1_server: batch assembler ON: slots=%d window_ms=%d region=%d cells/slot "
                    "(kv_max %d / %d) — a new request PREFILLS ALONE (the batch pauses) and "
                    "joins at the next position boundary; a departed slot is padded; "
                    "rotation OFF (B4 owns it); --reset is inert here\n",
                    srv.batch_slots, srv.batch_window_ms, srv.engine->batch_slot_cap(),
                    srv.engine->kv_max(), srv.batch_slots);
    }
    if (const char * mut = std::getenv("M1_SERVER_MUTANT")) {
        std::printf("m1_server: *** TEST MUTANT ACTIVE: M1_SERVER_MUTANT=%s ***\n", mut);
    }
    if (const char * m1 = std::getenv("M1_BATCH_POISON_STEP"))
        std::printf("m1_server: *** TEST HOOK ACTIVE: M1_BATCH_POISON_STEP=%s ***\n", m1);
    if (const char * m2 = std::getenv("M1_BATCH_PUBLISH_DELAY_MS"))
        std::printf("m1_server: *** TEST HOOK ACTIVE: M1_BATCH_PUBLISH_DELAY_MS=%s ***\n", m2);
    std::fflush(stdout);

    m1http::Config hc;
    hc.bind_addr       = bind_addr.c_str();
    hc.port            = port;
    hc.max_conns       = max_conns;
    hc.read_timeout_ms = read_timeout_ms;
    hc.drain_ms        = drain_ms;
    hc.busy_body       = m1http::error_body(
        "server_error", "server busy: too many in-flight connections; retry after 1s", nullptr,
        "server_busy");
    hc.on_listen = [&](int chosen) {
        std::printf("listening on http://%s:%d\n", bind_addr.c_str(), chosen);
        std::fflush(stdout);
        if (!port_file.empty()) {
            FILE * f = std::fopen(port_file.c_str(), "w");
            if (!f) FATAL("cannot write the port file '%s'", port_file.c_str());
            std::fprintf(f, "%d\n", chosen);
            std::fclose(f);
        }
    };

    const int rc = m1http::serve(hc, [&srv](const m1http::Request & rq, m1http::Conn & c) {
        handle_request(srv, rq, c);
    }, nullptr);
    // B3: stop the assembler's worker (it cancels any parked request at a position boundary)
    // before the process unwinds.  DELIBERATELY NOT `delete srv.pool`, for the same reason as
    // the engine below: a handler that outlived the drain window would touch freed state.
    if (srv.pool) srv.pool->shutdown_and_join();
    std::printf("m1_server: shut down cleanly after %llu requests\n",
                (unsigned long long) srv.requests.load());
    // DELIBERATELY NOT `delete srv.engine` (U5FIX finding 4). m1http::serve() waits at most --drain-ms
    // for in-flight handlers, and a client that stops READING parks its worker inside send_all while
    // it still holds the engine; freeing here could then run the engine's destructor under that
    // worker. The engine holds no state the process needs at exit (RealEngine has no destructor and
    // never called m1_session_close even before this), so the free buys nothing and costs a race.
    return rc;
}
