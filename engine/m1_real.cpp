// engine/m1_real.cpp — THE REAL ENGINE behind the server's seam (server plan §H row U5).
//
// WHAT THIS FILE IS: the adapter between engine/m1_server.cpp's routing layer and the engine the
// single-shot binary runs. It adds NO loop of its own — it fills the SAME LoopArgs the binary
// fills (engine/m1_main.cpp), opens the SAME M1Session (engine/m1_engine.cpp: M1Session::open /
// step / reset_rotation), and drives it one position at a time. That is the whole reason the id
// parity test is meaningful: there is exactly one per-position body in this tree, and both
// binaries call it.
//
// THE POSITION MAPPING, STATED ONCE (it is where an off-by-one would silently produce a different
// chain). The single-shot decode loop (m1_engine.cpp loop_run) walks n_prompt + n_decode positions
// with `ref_pos0 = n_prompt - 1`, feeds ids[p] while it prefills and gen.back() afterwards, runs
// the head from ref_pos0 onward, and keeps the argmax of positions ref_pos0 .. ref_pos0+n_decode-1;
// the LAST walked position's row is computed and never sampled. So to produce G ids a caller needs
// exactly the positions 0 .. n_prompt-1+G-1, and its emitted ids are the binary's FIRST G ids. The
// budget the router applies (n_prompt + G <= kv_max) makes n_prompt-1+G <= kv_max-1, i.e. every
// position is inside [0, kv_max) — which m1_session_step() REFUSES rather than writing past the
// KV cache (U4FIX item 3b).
//
// PER-REQUEST STATE (plan §E). What must be reset: the position counter (this file's `pos`) and,
// with --reset=exact (the default), the rotation's VRAM membership + 2Q policy, replayed back to
// the boot permutation by m1_session_reset_rotation(). What must NOT be reset: the KV cache —
// forward() writes cell pos and set_input masks EVERY cell >= the live length with -INFINITY
// (engine/m1_layer.cpp:747-751, the async twin :767-775), and both attention expressions consume
// that mask (m4_kqkv :475, soft_max_ext :492), so at position p the layers attend exactly cells
// 0..p, all written by THIS request. A stale cell is unreadable by construction. The reset-parity
// arm of scripts/engine_gate_server.sh is what proves the rest of the claim: it would fail if the
// KV, the rotation or any other engine state leaked between requests.
//
// EVERY CLIENT-CAUSED FAILURE IS PRE-CHECKED IN THE ROUTER, NOT HERE (plan §G3/R7): the engine's
// invariant violations are FATALs that kill the process, so n_prompt >= kv_max, an out-of-vocab id,
// a non-alternating conversation and an unrenderable date are all answered before this file runs.
// This file refuses (returns an error, never a FATAL) on exactly the two things the router's own
// budget arithmetic makes unreachable — a prompt that does not fit the KV width, and a step()
// refusal — because "unreachable" is a claim that must fail loudly if it is ever wrong.

#define FATAL(...) do { \
    std::fprintf(stderr, "FATAL: " __VA_ARGS__); \
    std::fprintf(stderr, "\n"); \
    std::exit(1); \
} while (0)

#define CHECK(cond, ...) do { if (!(cond)) FATAL(__VA_ARGS__); } while (0)

#include "m1_engine_api.h"
#include "m1_session.h"

#include "m1_chat.h"   // the pinned renderer + its boot checks
#include "m1_tok.h"    // the byte-level BPE tokenizer/detokenizer

#include "gguf.h"      // gguf_find_key / gguf_get_val_str — the chat template's bytes

#include <cerrno>
#include <cstdlib>
#include <unistd.h>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

double now_s() {
    const auto t = std::chrono::steady_clock::now().time_since_epoch();
    return std::chrono::duration<double>(t).count();
}

// The GGUF string KV the chat renderer is pinned to. This vendored gguf.h exposes the string
// pointer but no length (gguf_get_val_strlen is absent), so the length is strlen() — and
// m1_chat_verify_template() re-checks it against the pinned 14121 bytes, so a truncated read cannot
// pass silently.
const char * gguf_string_key(gguf_context * g, const char * key) {
    const int64_t id = gguf_find_key(g, key);
    CHECK(id >= 0, "the GGUF carries no '%s' key — this is not the pinned model", key);
    CHECK(gguf_get_kv_type(g, id) == GGUF_TYPE_STRING, "'%s' is not a GGUF string", key);
    return gguf_get_val_str(g, id);
}

// write the m1 file format (engine/m1_engine.cpp write_logits / read_tokens_bin: u32 n, then n i32)
bool write_tokens_file(const std::string & path, const std::vector<int32_t> & toks) {
    std::FILE * f = std::fopen(path.c_str(), "wb");
    if (!f) {
        std::fprintf(stderr, "m1_server: cannot write the request log '%s': %s\n", path.c_str(),
                     std::strerror(errno));
        return false;
    }
    const uint32_t n = (uint32_t) toks.size();
    const bool ok = std::fwrite(&n, sizeof n, 1, f) == 1 &&
                    (n == 0 || std::fwrite(toks.data(), sizeof(int32_t), n, f) == n);
    if (std::fclose(f) != 0 || !ok) {
        std::fprintf(stderr, "m1_server: short write on the request log '%s'\n", path.c_str());
        return false;
    }
    return true;
}

// FAIL-ABILITY HOOK, deliberately NOT a feature: M1_SERVER_MUTANT=skip_reset makes the real engine
// SKIP its per-request reset while leaving the entry invariant ON, so the NEXT request must be
// refused 500 ("engine_state_dirty"). It is the asymmetry proof for the invariant — the arm that
// shows the check can fire — and it is announced at boot like m1_server.cpp's other mutants.
bool mutant_skip_reset() {
    const char * m = std::getenv("M1_SERVER_MUTANT");
    return m && std::strcmp(m, "skip_reset") == 0;
}

}  // namespace

// ---------------------------------------------------------------------------
// THE ENGINE
// ---------------------------------------------------------------------------
struct RealEngine : M1Engine {
    EngineCfg  cfg;
    LoopArgs   args;
    M1Session * sess = nullptr;
    M1Tok *     tok  = nullptr;

    std::vector<float> row;        // n_vocab logits (one position's head output)
    int         pos = 0;           // the next position this engine will walk; 0 == clean
    long long   ordinal = 0;       // request ordinal (the request log's req-<n>.tokens)

    // reset accounting (plan §H U5: the --reset=exact cost must be MEASURED and printed, never
    // projected)
    double    reset_last = 0.0, reset_max = 0.0, reset_total = 0.0;
    long long n_resets = 0;

    explicit RealEngine(const EngineCfg & c) : cfg(c) { boot(); }

    // ---- boot ----------------------------------------------------------------------------
    void boot() {
        // THE CONFIGURATION: the same LoopArgs the single-shot binary fills from its own flags.
        // Every switch is one of its flags, verbatim; nothing is inferred here.
        args.model      = cfg.model_path.empty() ? std::string(DEFAULT_MODEL) : cfg.model_path;
        args.kv_max     = cfg.kv_max;   // a NUMERIC parameter (plan §0 C4): parity needs the same
        args.seq_len    = cfg.kv_max;   // ... and no boot prompt is published (the prompt is per
                                        // request), so the session resolves the width from this
        args.threads    = cfg.threads;
        args.n_res      = cfg.n_res;
        args.do_store   = cfg.arm_store;
        args.gpu_store  = cfg.gpu_store;
        args.gpu_core   = cfg.gpu_core;
        args.rotate     = cfg.rotate;
        args.rot_admit_k = cfg.rot_admit_k;
        args.rot_admit_k_set = cfg.rotate;   // the server names the policy explicitly, so
                                             // --rotate's "needs --rotate" refusal reads an
                                             // honest flag; without --rotate it must stay unset
        args.rotate_verify = cfg.rotate_verify;
        args.res_path   = cfg.res_path;
        if (!cfg.res_path.empty()) {
            int rn = 0;
            CHECK(read_hotset_json(cfg.res_path.c_str(), args.res_hot, rn),
                  "cannot read --resident-set %s", cfg.res_path.c_str());
            // THE CONFLICT IS A CONFIGURATION ERROR, NOT A PREFERENCE (U5FIX 1). The single-shot
            // binary refuses this pair (m1_main.cpp:124-134) and the server must refuse it the same
            // way: the resident SET is the resident count, and the store split is a NUMERIC
            // parameter (a different membership is a different accumulation grouping, plan §E), so
            // silently preferring one of the two would change what the engine computes without
            // saying so. Same message shape as the binary's.
            if (cfg.n_res_given && cfg.n_res != rn)
                FATAL("--n-res %d conflicts with --resident-set %s (n_res=%d): the SET is the "
                      "resident count — drop --n-res or pass the matching value",
                      cfg.n_res, cfg.res_path.c_str(), rn);
            args.res_n = rn;
            args.n_res = rn;
        }

        if (!m1_session_open(args, &sess))
            FATAL("the engine session refused to open (see the FAIL line above); the server "
                  "cannot serve without it");
        CHECK(sess, "m1_session_open returned true with a null session");

        // THE TOKENIZER reads the vocab out of the SESSION's OWN open gguf_context, so the 61 GB
        // file is mapped and parsed once, by the engine, and nothing re-reads it.
        tok = M1Tok::load(m1_session_gctx(sess));
        CHECK(tok, "the tokenizer could not be loaded");

        // THE CHAT TEMPLATE PIN (U2's wiring note 1): both boot checks FATAL on a mismatch, which
        // is exactly right — a template change must be a reviewed code change, not a silent
        // behaviour change. They run before the first request, whatever the route mix is.
        const char * tmpl = gguf_string_key(m1_session_gctx(sess), "tokenizer.chat_template");
        m1_chat_verify_template(tmpl, std::strlen(tmpl));
        m1_chat_verify_default_system_message(tmpl, std::strlen(tmpl));
        // The verifiers are SILENT on success (they only FATAL on a mismatch), so the server says
        // the pin held in its own boot log — an operator reading a serving log should see that the
        // renderer's pin was checked, not have to infer it.
        std::printf("# chat template pin = sha256 %s len %zu, default_system_message %zu bytes — "
                    "VERIFIED against the GGUF (m1_chat_verify_template + _default_system_message)\n",
                    M1_CHAT_TEMPLATE_SHA256, (size_t) M1_CHAT_TEMPLATE_LEN,
                    (size_t) M1_CHAT_DEFAULT_SYSTEM_MESSAGE_LEN);
        // --request-log must be an EXISTING writable directory: a silently unwritable log would make
        // the parity evidence vanish while the answers still look fine.
        if (!cfg.request_log.empty()) {
            CHECK(::access(cfg.request_log.c_str(), W_OK | X_OK) == 0,
                  "--request-log %s is not a writable directory (mkdir it first): %s",
                  cfg.request_log.c_str(), std::strerror(errno));
        }

        const int nv = m1_session_n_vocab(sess);
        CHECK(nv > 0, "the session reports n_vocab %d", nv);
        CHECK(tok->id_to_text.size() == (size_t) nv,
              "the tokenizer's vocab has %zu entries, the head %d — the two must be the same file",
              tok->id_to_text.size(), nv);
        row.assign((size_t) nv, 0.0f);

        std::printf("m1_server: real engine ready: n_vocab=%d kv_max=%d eos=%d reset=%s "
                    "chat_bos=%s request_log=%s\n",
                    nv, m1_session_kv_max(sess), (int) tok->id_eos,
                    cfg.reset == 1 ? "exact (the VRAM membership is replayed per request)"
                                   : "none (CONTROL ARM: per-request parity is EXPECTED to fail)",
                    cfg.chat_bos == M1_CHAT_BOS_TEMPLATE ? "template (one BOS)"
                                                         : "tokenizer (llama-server's double BOS)",
                    cfg.request_log.empty() ? "(off)" : cfg.request_log.c_str());
        std::printf("# tokenizer = %s\n", tok->summary().c_str());
        if (mutant_skip_reset())
            std::printf("m1_server: *** TEST MUTANT ACTIVE: M1_SERVER_MUTANT=skip_reset — the "
                        "per-request reset is SKIPPED and the entry invariant must refuse the next "
                        "request ***\n");
        std::fflush(stdout);
    }

    const char * name() const override { return "real"; }
    int n_vocab() const override { return (int) row.size(); }
    int kv_max() const override { return m1_session_kv_max(sess); }
    int32_t eos_id() const override { return tok->id_eos; }

    std::string token_text(int32_t id) const override { return tok->piece(id); }
    std::string detokenize(const std::vector<int32_t> & ids) const override { return tok->decode(ids); }

    std::vector<int32_t> encode(const std::string & text) const override {
        // The tokenizer's own policy (this vocab: add_bos = true), which is what llama-server does
        // for a /v1/completions prompt. Special tokens in the text are parsed (they are part of
        // this model's own format), so a literal '[INST]' is one id, not seven.
        return tok->encode(text, tok->add_bos, /*parse_special=*/true);
    }

    // ---- the chat prompt: THE renderer, behind the pre-checks U2 required -------------------
    bool encode_chat(const std::vector<ChatMsg> & msgs, std::vector<int32_t> & out,
                     EncError & err) const override {
        // (1) THE DATE LADDER. The pinned template's own ladder REFUSES 1 March of a year outside
        // {2024,2028,2032,1970} (and 1 January outside {1970,2024..2032}), and render_chat() is a
        // FATAL there — a real bug in the model's template, reproduced rather than widened. A
        // client cannot fix it, so it is a 500 with the reason, not process death.
        // THE CLOCK, READ ONCE (U5FIX 3b). can_render() and render_chat() each read the wall clock
        // when now_ymd is null, so between them lies midnight: the pre-check could approve
        // 2026-03-01 23:59:59.9 and the render could then get 2026-03-02 ... and, on the days the
        // pinned template's own ladder refuses, render_chat() is a FATAL (a process death caused by
        // a client request). One date, both calls.
        M1ChatOpts opts;
        const std::string today = m1_chat_format_ymd(m1_chat_today());
        opts.now_ymd = today.c_str();
        std::string why;
        if (!m1_chat_can_render(opts, &why)) {
            err.status  = 500;
            err.message = "this server cannot render its chat template today: " + why;
            err.code    = "chat_template_refused";
            return false;
        }
        // (2) THE ALTERNATION. The template raises when user/assistant turns do not alternate
        // after the optional leading system message, and render_chat() reproduces the raise. That
        // IS client-caused, so it is a 400 here and the template is never asked.
        {
            int idx = 0;
            const size_t first = (!msgs.empty() && msgs[0].role == "system") ? 1 : 0;
            for (size_t i = 0; i < msgs.size(); ++i) {
                const std::string & role = msgs[i].role;
                if (i >= first && role == "system") {
                    err.status  = 400;
                    err.param   = "messages";
                    err.code    = "invalid_message_order";
                    err.message = "a system message is only allowed as messages[0] (message " +
                                  std::to_string(i) + " is 'system')";
                    return false;
                }
                if (role != "user" && role != "assistant") continue;  // 'tool' is refused by the
                                                                     // router (422 tools)
                const bool want_user = (idx % 2 == 0);
                if ((role == "user") != want_user) {
                    err.status  = 400;
                    err.param   = "messages";
                    err.code    = "invalid_message_order";
                    err.message = std::string("conversation roles must alternate user and "
                                              "assistant after the optional system message: "
                                              "message ") + std::to_string(i) + " is '" + role +
                                  "' where '" + (want_user ? "user" : "assistant") + "' was expected";
                    return false;
                }
                idx++;
            }
        }
        // (3) THE RENDER, then the tokenizer with the policy flag the renderer asks for: the
        // template ALREADY emits '<s>', so the product default (M1_CHAT_BOS_TEMPLATE) tokenizes
        // with add_special = false and the prompt carries exactly one BOS.
        std::vector<M1ChatMessage> cm;
        cm.reserve(msgs.size());
        for (const ChatMsg & m : msgs) cm.push_back(M1ChatMessage{m.role, m.content});
        const std::string text = render_chat(cm, opts);   // the SAME date the pre-check approved
        const int add_special = m1_chat_tokenizer_add_special((M1ChatBos) cfg.chat_bos);
        out = tok->encode(text, add_special, /*parse_special=*/true);
        if (out.empty()) {
            err.status  = 500;
            err.message = "the rendered chat prompt tokenized to ZERO ids";
            err.code    = "empty_prompt";
            return false;
        }
        // (4) THE BOS INVARIANT (U2's note 3), asserted per request: under the product policy the
        // prompt starts with exactly one BOS. A double BOS here means the tokenizer's own BOS was
        // added on top of the template's, i.e. the wrong policy reached the encoder.
        if (cfg.chat_bos == M1_CHAT_BOS_TEMPLATE &&
            !(out[0] == 1 && (out.size() < 2 || out[1] != 1))) {
            err.status  = 500;
            err.message = "the rendered chat prompt does not start with exactly one BOS (ids[0]=" +
                          std::to_string(out[0]) + ", ids[1]=" +
                          (out.size() > 1 ? std::to_string(out[1]) : "n/a") +
                          "); the tokenizer was called with the wrong add_special policy";
            err.code    = "bos_invariant";
            return false;
        }
        // U2's note 3: the first four ids are logged on EVERY chat request, so a prompt-shape
        // regression is visible in the server's own log without a debugger.
        std::fprintf(stderr,
                     "m1_server: chat prompt: %zu ids, add_special=%d, first4 = %d %d %d %d\n",
                     out.size(), add_special, out[0], out.size() > 1 ? out[1] : -1,
                     out.size() > 2 ? out[2] : -1, out.size() > 3 ? out[3] : -1);
        return true;
    }

    // ---- the per-request entry assertion (plan §G1) ----------------------------------------
    bool request_begin(std::string * why) override {
        if (pos != 0) {
            if (why) *why = "the engine's position counter is " + std::to_string(pos) +
                            ", expected 0 at request entry (a previous request did not complete "
                            "its per-request reset)";
            return false;
        }
        // --reset=none is the CONTROL ARM: it exists to be compared against a fresh boot, so it must
        // be allowed to RUN from the drifted state this check would refuse. The flag is explicit and
        // the boot line says so; the product mode (exact) always checks.
        if (cfg.reset != 1) return true;
        std::string w;
        if (!m1_session_rotation_is_boot(sess, &w)) {
            if (why) *why = "the rotation state is not at its boot state at request entry: " + w;
            return false;
        }
        return true;
    }

    // ---- generation -----------------------------------------------------------------------
    GenResult generate(const std::vector<int32_t> & prompt, int max_tokens, const Sampling & sp,
                       const TokenSink & sink, const CancelFn & cancel) override {
        GenResult r;
        const int n_prompt = (int) prompt.size();
        const int nv = n_vocab();
        const int kv = kv_max();
        // The router's budget pre-check makes both of these unreachable; they are here because
        // "unreachable" must fail loudly (and because step() would refuse the position one
        // position later, with a less useful message).
        if (n_prompt < 1 || max_tokens < 1) {
            r.error = "generate: empty prompt or max_tokens < 1 (the router's budget refused it)";
            refuse_request(r, n_prompt);
            return r;
        }
        if (n_prompt + max_tokens > kv) {
            r.error = "generate: prompt " + std::to_string(n_prompt) + " + max_tokens " +
                      std::to_string(max_tokens) + " exceeds kv_max " + std::to_string(kv) +
                      " — the router's clamp should have prevented this";
            refuse_request(r, n_prompt);
            return r;
        }

        const double t0 = now_s();
        pos = 0;
        // THE PREFILL: positions 0 .. n_prompt-2 carry no logits row (the binary skips the head
        // there too — `want_row = p >= ref_pos0` with ref_pos0 = n_prompt-1).
        for (int p = 0; p + 1 < n_prompt; ++p) {
            if (!m1_session_step(sess, prompt[(size_t) p], p, nullptr)) {
                r.error = "m1_session_step refused position " + std::to_string(p) + "/" +
                          std::to_string(n_prompt - 1);
                break;
            }
            pos = p + 1;
            if (cancel && cancel()) { r.finish = "cancelled"; break; }
        }
        if (r.error.empty() && std::strcmp(r.finish, "cancelled") != 0) {
            // THE DECODE: position n_prompt-1 feeds the prompt's last id and its row is the first
            // generated token; position n_prompt-1+i feeds the token generated at i-1.
            uint64_t rng = m1_request_rng(cfg.seed, sp.seed, ordinal);
            int32_t tok_in = prompt[(size_t) (n_prompt - 1)];
            for (int i = 0; i < max_tokens; ++i) {
                const int p = n_prompt - 1 + i;
                if (!m1_session_step(sess, tok_in, p, row.data())) {
                    r.error = "m1_session_step refused position " + std::to_string(p) +
                              " (kv_max " + std::to_string(kv) + ")";
                    break;
                }
                pos = p + 1;
                // THE SAMPLER: greedy by default, through the ONE implementation in
                // m1_engine_api.h whose greedy branch is llama's strict-'>' argmax — the same rule
                // the binary's decode path applies, which is what makes greedy parity exact.
                const int32_t got = m1_sample_row(row.data(), nv, sp, rng);
                if (got == tok->id_eos) { r.finish = "stop"; break; }
                r.ids.push_back(got);
                if (sink && !sink(got)) { r.finish = "cancelled"; break; }
                if (cancel && cancel()) { r.finish = "cancelled"; break; }
                tok_in = got;
            }
        }
        finish_request(r, n_prompt, t0);
        return r;
    }

    // A refusal that never touched the session: keep the RECORD honest (the artifact a U6 reader
    // counts against requests) without paying for a reset that has nothing to undo.
    void refuse_request(const GenResult & r, int n_prompt) {
        if (!cfg.request_log.empty())
            write_tokens_file(cfg.request_log + "/req-" + std::to_string(ordinal) + ".tokens", r.ids);
        std::fprintf(stderr,
                     "m1_server: req %lld REFUSED before the first position: prompt=%d %s\n",
                     ordinal, n_prompt, r.error.c_str());
        ++ordinal;
        pos = 0;
    }

    // ---- the per-request tail: the log, the reset, the loud accounting ---------------------
    void finish_request(const GenResult & r, int n_prompt, double t0) {
        const double dt = now_s() - t0;
        if (!cfg.request_log.empty()) {
            const std::string path = cfg.request_log + "/req-" + std::to_string(ordinal) + ".tokens";
            write_tokens_file(path, r.ids);
        }
        // THE RESET (plan §E). It runs on EVERY exit path that has STEPPED the session — stop,
        // length, cancel, a step refusal. The only exits that skip it are generate()'s two
        // pre-step budget refusals (refuse_request below), which return before the session is
        // touched at all: nothing was stepped, pos is already 0 and the rotation was never read. The KV needs
        // nothing (see the header comment); the rotation membership/policy is replayed when
        // --reset=exact, and left drifted by the --reset=none control arm.
        const double t1 = now_s();
        if (cfg.reset == 1 && !mutant_skip_reset()) m1_session_reset_rotation(sess);
        const double reset_s = now_s() - t1;
        pos = 0;
        reset_last = reset_s;
        reset_total += reset_s;
        ++n_resets;
        if (reset_s > reset_max) reset_max = reset_s;
        // the reset cost is MEASURED and printed, never projected (plan §H U5's last bullet)
        std::fprintf(stderr,
                     "m1_server: req %lld done: prompt=%d produced=%zu finish=%s gen=%.3fs "
                     "reset(%s)=%.3fs [reset max %.3fs over %lld request(s)]\n",
                     ordinal, n_prompt, r.ids.size(), r.finish, dt - reset_s,
                     cfg.reset == 1 ? "exact" : "none", reset_s, reset_max, n_resets);
        if (cfg.verbose) {
            std::fprintf(stderr, "m1_server: req %lld ids[0..7] =", ordinal);
            for (size_t i = 0; i < r.ids.size() && i < 8; ++i)
                std::fprintf(stderr, " %d", r.ids[i]);
            std::fprintf(stderr, "%s\n", r.ids.size() > 8 ? " ..." : "");
        }
        if (!r.error.empty())
            std::fprintf(stderr, "m1_server: req %lld ENGINE REFUSED: %s\n", ordinal, r.error.c_str());
        ++ordinal;
    }
};

// ---------------------------------------------------------------------------
// the factory the routing layer links (declared under -DM1_ENGINE_REAL in m1_server.cpp)
// ---------------------------------------------------------------------------
M1Engine * m1_engine_create_real(const EngineCfg & cfg) { return new RealEngine(cfg); }
