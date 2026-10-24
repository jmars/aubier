// tools/m1_batch_test.cpp — B2's TEST RIG: batch-of-N decode vs the stock-attention baseline.
//
// WHAT THIS IS (aubier batch plan §B, first shape; B2's brief): N sequences, one position each,
// per engine step, through the SAME session seam the single-shot binary and the server drive
// (M1Session::open + m1_session_step_batch).  A joiner PREFILLS ALONE (plan §B: the batch is
// paused; the already-joined slots replay their LAST position — re-running the same
// (tok, pos, base, len) rewrites the same KV cell with the same bytes and attends the same
// window, so a waiting slot's state cannot move), then joins the full-width decode batch.  A
// sequence that reaches its decode budget DEPARTS: from the next batch step it is PADDED — a
// ONE-CELL window inside its own region at its last live cell, kidx rewriting only that cell
// (plan §D).  No chunked prefill, no prefill/decode interleave, no speculation, no slot
// compaction, no assembler (B3), no rotation (the session refuses it at N > 1; B2 runs STATIC
// membership).
//
// THE KV LAYOUT (plan §A option ii): the session's kv_max stays ONE window; the rig partitions it
// into N fixed contiguous regions, base_s = s * (kv_max / N), capacity kv_max / N per slot.  The
// softmax/kqv reduction width is the FULL kv_max for every row — the same width the solo runs
// use, which is why every comparison here passes the SAME --kv-max to both arms.
//
// THE STOCK BASELINE: this rig FORCES the four m4_* hatches OFF (LoopArgs m4_* = 0, i.e.
// --no-m4-kvwrite --no-m4-wvb --no-m4-kqkv --no-m4-wkb) on EVERY run — the fused kernels are
// nt==1-gated (engine/m1_layer.cpp:283-296), so a batched step runs the stock op sequence, and
// every equality claim is made against solo runs carrying the same four flags.  Comparing a
// batch against the shipped fused path is not this unit's bar and is not attempted (those flags
// are REFUSED here so nobody thinks they ran one).
//
// OUTPUTS: per SLOT s, <out>-s<s>.logits and <out>-s<s>.logits.tokens in the single-shot binary's
// own format (u64 n_rows, u32 n_vocab, rows; .tokens = u32 n, ids) so `cmp` against a solo run
// is byte-for-byte.  --expect <slot>=<prefix> loads a solo run's pair and compares, reporting
// the FIRST divergent id (position + both ids) and, past that, the first differing logits byte
// (row + float offset + both values).  Exit 0 = every walk completed and every comparison
// passed; 1 = refusal or divergence.
//
// NOT THIS RIG'S JOB: the HTTP assembler (B3), rotation under a batch (B4, refused by the
// session), performance (no timing claims), kernel work.
//
// THE TWO COMPARISON FRAMES, and why both exist (B2's measured finding): the CPU backend's F16
// and F32 GEMMs accumulate in a DIFFERENT order at M=1 (gemv) than at M=nt (gemm) — measured at
// the single-op level, same weights and same input column — while every quantized dot product
// (q2_K..q8_0, iq4_xs) is per-column M-stable.  The stock attention's kq/kqv run against F16 KV
// views, so a width-N step can NEVER reproduce a width-1 solo run's bytes: the strict solo bar
// fails by kernel dispatch, not by plumbing.  The plumbing bar that CAN hold is at FIXED width:
// --solo-slot s runs ONLY slot s live (every other row an unjoined pad, same graphs, same
// dispatch), and its output must equal slot s's output from the all-live batch — that comparison
// sees every region/mask/kidx/pad/join mistake this rig exists to catch, with the dispatch
// constant.  Both frames are run; the solo frame's failure is the FINDING, the fixed-width
// frame's failure would be a bug.

#include "gguf_model.h"
#include "m1_session.h"
#include "m1_engine_api.h"     // the ONE sampler (greedy = llama's strict-'>' argmax)

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// read_tokens in the m1 binary's own format ("n_prompt <P>" then the ids) — the RIG's fixture
// reader (loop_run's reader is static in m1_engine.cpp and stays untouched; this is the same
// 20-line parse of the same documented format, not a second engine path).
static bool rig_read_tokens(const char * path, std::vector<int32_t> & ids, int & n_prompt) {
    FILE * f = std::fopen(path, "r");
    if (!f) { std::fprintf(stderr, "FAIL: cannot open token file %s\n", path); return false; }
    char key[64];
    int  np = -1;
    if (std::fscanf(f, "%63s %d", key, &np) != 2 || std::strcmp(key, "n_prompt") != 0) {
        std::fprintf(stderr, "FAIL: %s: expected first line 'n_prompt <N>'\n", path);
        std::fclose(f);
        return false;
    }
    int v;
    while (std::fscanf(f, "%d", &v) == 1) ids.push_back((int32_t) v);
    std::fclose(f);
    n_prompt = np;
    if (ids.empty() || np < 1 || np > (int) ids.size()) {
        std::fprintf(stderr, "FAIL: %s: %zu ids, n_prompt %d\n", path, ids.size(), np);
        return false;
    }
    return true;
}

// write_logits in the single-shot binary's own format (m1_engine.cpp's writer): u64 n, u32
// n_vocab, n rows, then <path>.tokens = u32 n + ids.  Byte-comparable with a solo run's files.
static bool rig_write_logits(const char * path, const std::vector<std::vector<float>> & rows,
                             const std::vector<int32_t> & toks) {
    if (rows.empty()) { std::printf("FAIL: no logits rows to write for %s\n", path); return false; }
    const uint64_t n  = (uint64_t) rows.size();
    const uint32_t nv = (uint32_t) rows[0].size();
    FILE * f = std::fopen(path, "wb");
    if (!f) { std::printf("FAIL: cannot write %s\n", path); return false; }
    std::fwrite(&n, sizeof n, 1, f);
    std::fwrite(&nv, sizeof nv, 1, f);
    for (const std::vector<float> & r : rows) {
        CHECK(r.size() == nv, "logits row size %zu != %u", r.size(), nv);
        std::fwrite(r.data(), sizeof(float), nv, f);
    }
    std::fclose(f);
    const std::string tp = std::string(path) + ".tokens";
    f = std::fopen(tp.c_str(), "wb");
    if (!f) { std::printf("FAIL: cannot write %s\n", tp.c_str()); return false; }
    const uint32_t nt = (uint32_t) toks.size();
    std::fwrite(&nt, sizeof nt, 1, f);
    std::fwrite(toks.data(), sizeof(int32_t), toks.size(), f);
    std::fclose(f);
    return true;
}

static bool rig_read_logits(const char * path, std::vector<std::vector<float>> & rows,
                            std::vector<int32_t> & toks) {
    FILE * f = std::fopen(path, "rb");
    if (!f) { std::printf("FAIL: cannot open the expected logits %s\n", path); return false; }
    uint64_t n = 0; uint32_t nv = 0;
    if (std::fread(&n, sizeof n, 1, f) != 1 || std::fread(&nv, sizeof nv, 1, f) != 1) {
        std::fclose(f); return false;
    }
    rows.assign((size_t) n, std::vector<float>((size_t) nv));
    for (uint64_t i = 0; i < n; ++i)
        if (std::fread(rows[(size_t) i].data(), sizeof(float), nv, f) != nv) {
            std::fclose(f); return false;
        }
    std::fclose(f);
    const std::string tp = std::string(path) + ".tokens";
    f = std::fopen(tp.c_str(), "rb");
    if (!f) { std::printf("FAIL: cannot open the expected tokens %s\n", tp.c_str()); return false; }
    uint32_t nt = 0;
    if (std::fread(&nt, sizeof nt, 1, f) != 1) { std::fclose(f); return false; }
    toks.assign((size_t) nt, 0);
    if (nt && std::fread(toks.data(), sizeof(int32_t), nt, f) != nt) { std::fclose(f); return false; }
    std::fclose(f);
    return true;
}

// ONE sequence's walk state.  The walk mirrors the solo binary's decode loop position for
// position: prefill 0..n_prompt-1 (the row at n_prompt-1 is step 0, sampled), then decode
// positions n_prompt..n_prompt+budget-1 (rows steps 1..budget-1, each sampled) and ONE final
// position n_prompt+budget-1 walked live whose row the solo binary also computes and never dumps.
//
// The state machine is the batch's whole scheduling discipline (plan B/D):
//   ST_UNJOINED — not yet prefilled: a ONE-CELL pad inside its OWN region (cell base+0, which
//                 its own prefill position 0 rewrites before anything reads it);
//   ST_PREFILL  — the joining sequence (phase A walks it alone; every OTHER slot is a pad or a
//                 frozen replay, so nothing else advances);
//   ST_WAIT     — prefilled, waiting for the batch to form: an IDEMPOTENT REPLAY of its last
//                 prefill position (same tok, same pos, same window => the set_rows rewrites the
//                 same cell with the same bytes, so its state cannot move);
//   ST_DECODE   — live in the full-width batch (phase B);
//   ST_DONE     — budget reached: plan D's pad (ONE live cell at its own region's last live
//                 cell; kidx rewrites that cell only).
struct Seq {
    enum St { ST_UNJOINED, ST_PREFILL, ST_WAIT, ST_DECODE, ST_SCORE, ST_DONE };
    St st = ST_UNJOINED;
    std::string path;
    std::vector<int32_t> ids;
    int n_prompt = 0;
    int budget   = 0;              // the decode row count (== the solo run's --n-decode)
    int64_t base = 0;              // this slot's region base (plan A option ii)
    int64_t cap  = 0;              // this slot's region capacity
    // ST_DONE's pad cell + token, factored so BOTH modes share the one case: decode pads at its
    // last live cell (base + n_prompt + budget - 1, token ids[n_prompt-1]); score pads at its
    // own last position (base + ids.size() - 1, token ids.back()).
    int64_t done_cell = 0;
    int32_t done_tok  = 0;
    int   walked = 0;              // live positions walked so far (prefill included)
    std::vector<int32_t> gen;      // the sampled ids (== the solo run's .tokens body)
    std::vector<std::vector<float>> rows;
    uint64_t rng = 0;              // this sequence's OWN stream (plan E: per-slot, never shared)
    std::string expect;            // optional solo-run prefix for the self-comparison
};

int main(int argc, char ** argv) {
    LoopArgs a;
    std::vector<std::string> token_files;
    int kv_max = 0, n_decode = 0;
    double temperature = 0.0;
    int per_seq_budgets[8] = {-1, -1, -1, -1, -1, -1, -1, -1};
    uint64_t base_seed = 0;
    bool pad_empty_mask = false, n_res_given = false;
    bool score_mode = false;
    int batch_n = -1;
    int solo_slot = -1, region_shift = 0;
    std::string out;
    std::vector<std::pair<int, std::string>> expects;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto need = [&](const char * opt) -> std::string {
            if (i + 1 >= argc) FATAL("%s needs a value", opt);
            return argv[++i];
        };
        auto need_int = [&](const char * opt) -> int {
            const std::string v = need(opt);
            char * e = nullptr;
            const long r = std::strtol(v.c_str(), &e, 10);
            if (e == v.c_str() || *e != '\0') FATAL("%s: '%s' is not an integer", opt, v.c_str());
            return (int) r;
        };
        if (arg == "--model")            a.model = need("--model");
        else if (arg == "--tokens")      token_files.push_back(need("--tokens"));
        else if (arg == "--kv-max")      kv_max = need_int("--kv-max");
        else if (arg == "--n-decode")    n_decode = need_int("--n-decode");
        else if (arg == "--decode") {    // --decode <slot>=<n>: that slot's decode budget
            const std::string v = need("--decode");
            int s = -1, n = -1;
            if (std::sscanf(v.c_str(), "%d=%d", &s, &n) != 2 || s < 0 || s >= 8)
                FATAL("--decode %s: want <slot>=<n> with slot in [0,8)", v.c_str());
            per_seq_budgets[s] = n;
        }
        else if (arg == "--out")         out = need("--out");
        else if (arg == "--threads")     a.threads = need_int("--threads");
        else if (arg == "--n-res")      { a.n_res = need_int("--n-res"); n_res_given = true; }
        else if (arg == "--arm-store")   a.do_store = true;
        else if (arg == "--gpu-store")   a.gpu_store = true;
        else if (arg == "--gpu-core")    a.gpu_core = true;
        else if (arg == "--resident-set") a.res_path = need("--resident-set");
        else if (arg == "--seed")        base_seed = std::strtoull(need("--seed").c_str(), nullptr, 10);
        else if (arg == "--temperature") temperature = std::atof(need("--temperature").c_str());
        else if (arg == "--expect") {    // --expect <slot>=<prefix>: compare against a solo run
            const std::string v = need("--expect");
            const size_t eq = v.find('=');
            if (eq == std::string::npos) FATAL("--expect %s: want <slot>=<prefix>", v.c_str());
            const int s = std::atoi(v.c_str());
            if (s < 0 || s >= 8) FATAL("--expect %s: slot out of range", v.c_str());
            expects.push_back({ s, v.substr(eq + 1) });
        }
        else if (arg == "--pad-empty-mask") pad_empty_mask = true;   // F2e's NaN-trap arm: pads
                                                                      // with n_kv = 0 (the session
                                                                      // REFUSES; run under the
                                                                      // no-check mutant to SEE it)
        else if (arg == "--score") score_mode = true;   // BQ: teacher-forced scoring of every
                                                        // slot's WHOLE file (the batched B arm of
                                                        // the paired-NLL instrument; the m1
                                                        // binary's --score is the A arm)
        else if (arg == "--batch") {   // score mode ONLY: the batch width, decoupled from the
                                       // file count (the corpus is 160 files; at N=2 that is 80
                                       // groups of 2, not a batch of 160)
            batch_n = need_int("--batch");
            if (batch_n < 1 || batch_n > 8) FATAL("--batch %d out of range [1,8]", batch_n);
        }
        else if (arg == "--solo-slot") {  // the FIXED-WIDTH solo control (see the header): only
                                          // this slot ever joins; every other row stays an
                                          // unjoined pad for the whole run
            solo_slot = need_int("--solo-slot");
            if (solo_slot < 0 || solo_slot >= 8) FATAL("--solo-slot %d out of range", solo_slot);
        }
        else if (arg == "--region-shift") {  // review finding 2: translate EVERY region by delta
            region_shift = need_int("--region-shift");
            if (region_shift < 0) FATAL("--region-shift %d: negative shifts are not needed for "
                                       "the bar (a delta > 0 already moves every base)", region_shift);
        }
        else if (arg == "--no-m4-wkb" || arg == "--no-m4-kqkv" || arg == "--no-m4-kvwrite"
                 || arg == "--no-m4-wvb" || arg == "--m4-wkb" || arg == "--m4-kqkv"
                 || arg == "--m4-kvwrite" || arg == "--m4-wvb")
            FATAL("%s: this rig FORCES the stock baseline (all four m4_* hatches OFF) on every "
                  "run — the batched build falls back to the stock op sequence anyway (nt==1 "
                  "gating) and every equality claim is made under the stock flags", arg.c_str());
        else FATAL("unknown arg '%s'", arg.c_str());
    }

    // ---- validation ------------------------------------------------------------------------
    // In score mode the batch width is --batch (decoupled from the file count: the file list
    // carries WHOLE GROUPS of N files, all walked in this one process); in decode mode N is the
    // file count, exactly as B2 built it.
    if (score_mode && batch_n < 0) {
        std::printf("FAIL: --score needs --batch N (the batch width); the --tokens list is the "
                    "CORPUS, walked in groups of N\n");
        return 1;
    }
    if (!score_mode && batch_n >= 0) {
        std::printf("FAIL: --batch is the score mode's width knob — decode mode's N IS the "
                    "--tokens file count\n");
        return 1;
    }
    const int N = score_mode ? batch_n : (int) token_files.size();
    const int n_files = (int) token_files.size();
    if (score_mode && n_files % N) {
        std::printf("FAIL: --score with %d --tokens file(s) at --batch %d — the list must carry "
                    "WHOLE groups of %d\n", n_files, N, N);
        return 1;
    }
    if (N < 1 || N > 8) { std::printf("FAIL: %d --tokens file(s) — the rig takes 1..8\n", N); return 1; }
    if (out.empty())   { std::printf("FAIL: --out <prefix> is required\n"); return 1; }
    if (kv_max <= 0)   { std::printf("FAIL: --kv-max %d — the batch needs an explicit window "
                                     "(the region arithmetic divides it)\n", kv_max); return 1; }
    if (score_mode && (n_decode >= 1 || solo_slot >= 0 || region_shift != 0 || temperature > 0.0)) {
        std::printf("FAIL: --score is the teacher-forced walk of each file's WHOLE length — "
                     "--n-decode/--solo-slot/--region-shift/temperature are decode-mode knobs and "
                     "are refused here so no arm half-scores a file\n");
        return 1;
    }
    if (!score_mode && n_decode < 1)  { std::printf("FAIL: --n-decode %d\n", n_decode); return 1; }
    if (kv_max % N)    { std::printf("FAIL: --kv-max %d not divisible by N=%d (fixed per-slot "
                                     "regions, plan A option ii)\n", kv_max, N); return 1; }
    // B3FIX B6: the SERVER refuses a region width that is not a multiple of 16
    // (m1_server.cpp --batch-slots) because B2FIX MEASURED that a non-16-aligned region base
    // is NOT bit-inert on the kq/kqv lane-blocked reductions (a 1-cell translation moves the
    // live window across SIMD lanes and changes the accumulation association), while
    // 16-cell steps are inert.  The rig — the certificate ORACLE — must refuse the same
    // widths, or a future certificate at an unaligned width folds the translation artifact
    // into the batched delta (a false DEGRADED, or an EQUIVALENT verdict for a
    // configuration the product REFUSES TO BOOT).
    if ((kv_max / N) % 16) {
        std::printf("FAIL: --kv-max %d / N=%d = %d cells per slot is not a multiple of 16 — the "
                    "server refuses this width (B2FIX: non-16-aligned region bases are not "
                    "bit-inert), and neither does this rig\n", kv_max, N, kv_max / N);
        return 1;
    }
    const int64_t slot_cap = kv_max / N;

    std::vector<Seq> seqs((size_t) N);
    for (int s = 0; s < N; ++s) {
        Seq & q = seqs[(size_t) s];
        if (!rig_read_tokens(token_files[(size_t) s].c_str(), q.ids, q.n_prompt)) return 1;
        q.path   = token_files[(size_t) s];
        q.budget = per_seq_budgets[s] >= 0 ? per_seq_budgets[s] : n_decode;
        q.base   = (int64_t) s * slot_cap + region_shift;
        q.cap    = slot_cap - region_shift;   // the shift eats capacity at the window's end
        // plan E: one stream per slot.  NOTE (review finding 4): this derivation —
        // m1_request_rng(base_seed, slot, 0) — is the RIG's own; the m1 SOLO binary has no
        // per-request stream to match it against, so a temperature>0 --expect comparison
        // against a solo run is NOT byte-provable without first proving stream parity.  Every
        // B2 arm ran GREEDY (temperature <= 0: the sampler never consumes the stream), so no
        // claim here rests on the stream; do not read a greedy-only identity proof as a general
        // sampled one.
        q.rng    = m1_request_rng(base_seed, (uint64_t) s, 0);
        if (!score_mode && q.budget < 1) { std::printf("FAIL: slot %d budget %d < 1\n", s, q.budget); return 1; }
        // plan D's budget refusal, rig-side: the per-slot region, not the global window.  In
        // --score mode the walk is the WHOLE file (the m1 binary's --score semantics: position p
        // feeds ids[p], its row predicts ids[p+1], the last position has no target), so the file
        // itself must fit the region and needs >= 2 ids to score one position.
        if (score_mode) {
            if (q.ids.size() < 2) {
                std::printf("FAIL: --score slot %d: %s carries %zu token(s) — needs >= 2 to score "
                            "one position\n", s, q.path.c_str(), q.ids.size());
                return 1;
            }
            if ((int64_t) q.ids.size() > q.cap) {
                std::printf("FAIL: --score slot %d: %zu tokens > the per-slot region %lld "
                            "(kv_max %d / %d) — the teacher-forced walk needs the WHOLE file in "
                            "its region\n", s, q.ids.size(), (long long) q.cap, kv_max, N);
                return 1;
            }
            q.done_cell = q.base + (int64_t) q.ids.size() - 1;
            q.done_tok  = q.ids.back();
        } else if ((int64_t) q.n_prompt + q.budget > q.cap) {
            std::printf("FAIL: slot %d: n_prompt %d + budget %d > the per-slot region %lld "
                        "(kv_max %d / %d) — refuse BEFORE the engine, exactly as the server's "
                        "400 arithmetic will (plan D)\n",
                        s, q.n_prompt, q.budget, (long long) q.cap, kv_max, N);
            return 1;
        } else {
            q.done_cell = q.base + q.n_prompt + q.budget - 1;
            q.done_tok  = q.ids[(size_t) q.n_prompt - 1];
        }
    }
    for (const auto & e : expects) {
        if (e.first >= N) { std::printf("FAIL: --expect %d=... but N=%d\n", e.first, N); return 1; }
        if (score_mode) {
            std::printf("FAIL: --expect with --score: the batched score dumps are compared "
                         "OFFLINE (cmp against the solo --score dumps + the paired-NLL "
                         "instrument) — the rig's in-process compare is the decode bar's\n");
            return 1;
        }
        seqs[(size_t) e.first].expect = e.second;
    }

    // ---- the session (the SAME open the m1 binary and the server call) ---------------------
    a.batch_nt = N;
    a.kv_max   = kv_max;
    // THE STOCK BASELINE, forced: the four tri-state hatches pinned to 0 = the reference op
    // sequence on every arm of every comparison this rig makes.
    a.m4_wkb = 0; a.m4_kqkv = 0; a.m4_kvwrite = 0; a.m4_wvb = 0;
    if (!a.res_path.empty()) {
        int rn = 0;
        if (!read_hotset_json(a.res_path.c_str(), a.res_hot, rn)) return 1;
        a.res_n = rn;
        if (n_res_given && a.n_res != rn)
            FATAL("--n-res %d conflicts with --resident-set %s (n_res=%d)", a.n_res, a.res_path.c_str(), rn);
        a.n_res = rn;
    }
    Sampling sp;
    sp.temperature = temperature;

    M1Session * S = nullptr;
    if (!m1_session_open(a, &S)) return 1;
    const int nv = m1_session_n_vocab(S);
    std::vector<float> lg((size_t) nv * (size_t) N, 0.0f);   // [n_vocab, N]

    if (score_mode)
        std::printf("=== B2 BATCH RIG — score walk: --batch %d, %d file(s) = %d group(s), ONE "
                    "process (one model load) ===\n", N, n_files, n_files / N);
    else
        std::printf("=== B2 BATCH RIG — %d sequence(s), kv window %d, %d-cell regions ===\n",
                    N, kv_max, (int) slot_cap);
    std::printf("# kv window %d, %d-cell regions\n", kv_max, (int) slot_cap);
    if (score_mode)
        std::printf("# mode = SCORE (teacher forcing: position p feeds ids[p], its row predicts "
                    "ids[p+1]; <out>-g<g>-s<s>.logits + .tokens carry the TARGET ids) — the "
                    "batched B arm of the paired-NLL instrument\n");
    std::printf("# stock baseline: --no-m4-kvwrite --no-m4-wvb --no-m4-kqkv --no-m4-wkb "
                "(forced on every run)\n");
    std::printf("# sampling = %s (the ONE sampler, m1_engine_api.h; one RNG stream per slot)\n",
                score_mode ? "none — SCORE is teacher-forced, the sampler never runs"
                           : (temperature > 0.0 ? "temperature" : "GREEDY (llama's strict-'>' argmax)"));
    if (region_shift)
        std::printf("# region shift = +%d (the SHIFT-SENSITIVITY arm, review finding 2: every "
                    "region translated by the same delta; outputs must stay byte-identical to the "
                    "unshifted run — a translation of disjoint regions is semantically inert)\n",
                    region_shift);
    for (int s = 0; s < N; ++s)
        std::printf("# slot %d: %s n_prompt=%d budget=%d region=[%lld,%lld) usable cap %lld\n", s,
                    seqs[(size_t) s].path.c_str(), seqs[(size_t) s].n_prompt,
                    seqs[(size_t) s].budget, (long long) seqs[(size_t) s].base,
                    (long long) (seqs[(size_t) s].base + slot_cap),
                    (long long) seqs[(size_t) s].cap);

    std::vector<int32_t> rtoks((size_t) N);
    std::vector<int>     rpos((size_t) N);
    std::vector<int64_t> rbase((size_t) N), rlen((size_t) N);
    // one engine step with whatever each row currently is (pad / replay / prefill / decode)
    auto step_rows = [&]() -> bool {
        for (int t = 0; t < N; ++t) {
            const Seq & q = seqs[(size_t) t];
            switch (q.st) {
            case Seq::ST_UNJOINED:
                // pad: ONE cell at the region's first cell, inside its own region only.  Its own
                // prefill position 0 later rewrites that cell (set_rows expands before the
                // attention reads), so the garbage is never read by anyone.
                rtoks[(size_t) t] = q.ids[0];
                rpos[(size_t) t]  = 0;
                rbase[(size_t) t] = q.base;
                rlen[(size_t) t]  = pad_empty_mask ? 0 : 1;
                break;
            case Seq::ST_WAIT:
                // the IDEMPOTENT REPLAY (see struct Seq): same token, same position, same window
                // as its last prefill step — the write lands on the same cell with the same
                // bytes, so a waiting slot's state cannot move while another slot prefills.
                rtoks[(size_t) t] = q.ids[(size_t) q.n_prompt - 1];
                rpos[(size_t) t]  = q.n_prompt - 1;
                rbase[(size_t) t] = q.base;
                rlen[(size_t) t]  = q.n_prompt;
                break;
            case Seq::ST_DONE: {
                // DEPARTED pad (plan D): ONE live cell at its own region's LAST live cell; kidx
                // rewrites that same cell (garbage in its own region only).  With
                // --pad-empty-mask the window is EMPTY (len 0) — the NaN-trap probe arm.  The
                // cell/token are mode-factored (see struct Seq): decode pads at n_prompt+budget-1,
                // score at its file's last position.
                rtoks[(size_t) t] = q.done_tok;
                rpos[(size_t) t]  = 0;                 // kidx = rbase + rpos = the pad cell
                rbase[(size_t) t] = q.done_cell;
                rlen[(size_t) t]  = pad_empty_mask ? 0 : 1;
                break;
            }
            case Seq::ST_SCORE:
                // BQ's teacher-forced row: feed the FILE's own token at its own position — the
                // m1 binary's --score line, batched.  len = walked+1 is the whole prefix, so the
                // mask is exactly the solo score run's mask inside this slot's region.
                rtoks[(size_t) t] = q.ids[(size_t) q.walked];
                rpos[(size_t) t]  = q.walked;
                rbase[(size_t) t] = q.base;
                rlen[(size_t) t]  = q.walked + 1;
                break;
            case Seq::ST_PREFILL:
                // the joining sequence: its own next prompt position, its own region.
                rtoks[(size_t) t] = q.ids[(size_t) q.walked];
                rpos[(size_t) t]  = q.walked;
                rbase[(size_t) t] = q.base;
                rlen[(size_t) t]  = q.walked + 1;
                break;
            case Seq::ST_DECODE:
                // live row: feed the last sampled id at the next position.
                rtoks[(size_t) t] = q.gen.back();
                rpos[(size_t) t]  = q.walked;
                rbase[(size_t) t] = q.base;
                rlen[(size_t) t]  = q.walked + 1;
                break;
            }
        }
        return m1_session_step_batch(S, N, rtoks.data(), rpos.data(), rbase.data(), rlen.data(),
                                     lg.data());
    };
    auto sample_row = [&](int t) -> int32_t {
        Seq & q = seqs[(size_t) t];
        return m1_sample_row(lg.data() + (size_t) t * (size_t) nv, nv, sp, q.rng);
    };

    if (score_mode) {
        // ---- SCORE MODE: every slot walks its file teacher-forced, group after group -----------
        // This is the m1 binary's --score loop over N rows: position p feeds ids[p] through the
        // SAME step seam, one row per position that HAS a target (p+1 < size), .tokens carrying
        // the TARGET ids so the dump is self-describing for the offline instrument — byte-format
        // identical to a solo --score dump, which is what the N=1 control cmps.  A slot that
        // reaches its last position still STEPS it (its row is discarded — no target), then
        // departs to the ST_DONE pad; with equal-length files (the quality corpus) every slot
        // finishes together and no pad ever fires, but the discipline is the same one decode
        // uses, so ragged files stay correct.
        //
        // MULTI-GROUP: the --tokens list carries GROUPS of N files (its size must be a multiple
        // of N); group g uses slots' files g*N+s, and ALL groups run in THIS ONE process — the
        // session stays open (one model load).  A new group is SAFE on a used region exactly as
        // plan D's joiner is: its walk rewrites every cell it reads (position p writes base+p
        // before attending [base, base+p], all rewritten this walk), so the next group's numbers
        // are identical to a fresh process's — the same property the ST_WAIT replay and the
        // permutation arm rest on.  On a box where the 32.5 GiB model pages in at ~30 MB/s this
        // is the difference between one load and 160.
        const int n_groups = n_files / N;
        for (int g = 0; g < n_groups; ++g) {
            for (int t = 0; t < N; ++t) {
                Seq & q = seqs[(size_t) t];
                const std::string & path = token_files[(size_t) g * N + t];
                q.ids.clear();   // rig_read_tokens APPENDS (its decode callers read each file
                                 // once into a fresh vector); a reused slot must start empty or
                                 // group g+1 walks group g's ids again — the smoke control caught
                                 // exactly that (53 rows for a 27-token file)
                if (!rig_read_tokens(path.c_str(), q.ids, q.n_prompt)) { m1_session_close(S); return 1; }
                q.path = path;
                q.rows.clear();
                q.gen.clear();
                q.walked = 0;
                q.st = Seq::ST_SCORE;
                if ((int64_t) q.ids.size() > q.cap) {
                    std::printf("FAIL: --score group %d slot %d: %s carries %zu tokens > the "
                                "per-slot region %lld\n", g, t, path.c_str(), q.ids.size(),
                                (long long) q.cap);
                    m1_session_close(S);
                    return 1;
                }
                q.done_cell = q.base + (int64_t) q.ids.size() - 1;
                q.done_tok  = q.ids.back();
            }
            bool any_live = true;
            while (any_live) {
                any_live = false;
                for (int t = 0; t < N; ++t)
                    if (seqs[(size_t) t].st == Seq::ST_SCORE) any_live = true;
                if (!any_live) break;
                if (!step_rows()) {
                    std::printf("FAIL: the batched step refused during group %d's score walk\n", g);
                    m1_session_close(S);
                    return 1;
                }
                for (int t = 0; t < N; ++t) {
                    Seq & q = seqs[(size_t) t];
                    if (q.st != Seq::ST_SCORE) continue;         // pads: nothing to advance
                    if (q.walked + 1 < (int) q.ids.size()) {
                        q.rows.emplace_back(lg.begin() + (size_t) t * (size_t) nv,
                                            lg.begin() + (size_t) (t + 1) * (size_t) nv);
                        q.gen.push_back(q.ids[(size_t) q.walked + 1]);   // the TARGET id
                        std::printf("p=%04d s%d tok=%d target=%d\n", q.walked, t,
                                    (int) q.ids[(size_t) q.walked],
                                    (int) q.ids[(size_t) q.walked + 1]);
                    } else {
                        std::printf("p=%04d s%d tok=%d (the last position: no target to score)\n",
                                    q.walked, t, (int) q.ids[(size_t) q.walked]);
                    }
                    q.walked++;
                    if (q.walked >= (int) q.ids.size()) q.st = Seq::ST_DONE;
                }
            }
            // per-group outputs, written BEFORE the next group reuses the slots (bounded RAM:
            // one group's rows, never the corpus)
            for (int t = 0; t < N; ++t) {
                Seq & q = seqs[(size_t) t];
                const std::string p = out + "-g" + std::to_string(g) + "-s" + std::to_string(t)
                                    + ".logits";
                if (!rig_write_logits(p.c_str(), q.rows, q.gen)) { m1_session_close(S); return 1; }
                std::printf("# group %d slot %d (%s): %d scored positions, first targets [", g, t,
                            q.path.c_str(), (int) q.gen.size());
                for (size_t k = 0; k < q.gen.size() && k < 6; ++k) std::printf(" %d", (int) q.gen[k]);
                std::printf(" ]\n");
            }
        }
    } else {

    // ---- PHASE A: each sequence prefills ALONE (plan B; the batch is paused) ---------------
    // Slot s is ST_PREFILL; slots < s are ST_WAIT (frozen replay); slots > s are ST_UNJOINED
    // pads.  Only the joining row ADVANCES — that is what "prefills alone" means through a
    // fixed-width graph: no other slot's position, window or KV cell moves.
    for (int s = 0; s < N; ++s) {
        if (solo_slot >= 0 && s != solo_slot) continue;   // the fixed-width solo control: never
        Seq & q = seqs[(size_t) s];                       // joins, stays an unjoined pad forever
        q.st = Seq::ST_PREFILL;
        for (int p = 0; p < q.n_prompt; ++p) {
            if (!step_rows()) {
                std::printf("FAIL: the batched step refused during slot %d's prefill at p=%d\n", s, p);
                m1_session_close(S);
                return 1;
            }
            q.walked = p + 1;
            if (p == q.n_prompt - 1) {           // the step-0 row: record + sample
                q.rows.emplace_back(lg.begin() + (size_t) s * (size_t) nv,
                                    lg.begin() + (size_t) (s + 1) * (size_t) nv);
                q.gen.push_back(sample_row(s));
            }
        }
        q.st = Seq::ST_WAIT;
        std::printf("# slot %d joined (prefill %d positions done; step-0 id %d)\n", s, q.n_prompt,
                    (int) q.gen.back());
    }

    // ---- PHASE B: full-width decode until every sequence departs ---------------------------
    // Only slots that JOINED go live (--solo-slot's never-joined controls stay pads throughout).
    for (int t = 0; t < N; ++t)
        if (seqs[(size_t) t].st == Seq::ST_WAIT) seqs[(size_t) t].st = Seq::ST_DECODE;
    bool any_live = true;
    while (any_live) {
        any_live = false;
        for (int t = 0; t < N; ++t)
            if (seqs[(size_t) t].st == Seq::ST_DECODE) any_live = true;
        if (!any_live) break;
        if (!step_rows()) {
            std::printf("FAIL: the batched step refused during decode\n");
            m1_session_close(S);
            return 1;
        }
        for (int t = 0; t < N; ++t) {
            Seq & q = seqs[(size_t) t];
            if (q.st != Seq::ST_DECODE) continue;         // pads: nothing to advance
            const int step = q.walked - (q.n_prompt - 1); // the row this position produced
            q.walked++;
            if (step >= 1 && step < q.budget) {           // rows steps 1..budget-1 kept (0 in A)
                q.rows.emplace_back(lg.begin() + (size_t) t * (size_t) nv,
                                    lg.begin() + (size_t) (t + 1) * (size_t) nv);
                q.gen.push_back(sample_row(t));
            }
            if (q.walked >= q.n_prompt + q.budget) {
                // the final position is walked (its row computed, never dumped — the solo
                // binary's own last step); the slot departs and is padded from the next step.
                q.st = Seq::ST_DONE;
                std::printf("# slot %d DEPARTED after %d positions (budget %d) — padded from the "
                            "next batch step\n", t, q.walked, q.budget);
            }
        }
    }

    }   // !score_mode (phases A/B are the decode walk)

    // ---- outputs + the self-comparison ------------------------------------------------------
    // (decode mode only: score mode wrote every group's dump inside its loop, above)
    int rc = 0;
    if (!score_mode) for (int s = 0; s < N; ++s) {
        Seq & q = seqs[(size_t) s];
        if (q.st == Seq::ST_UNJOINED && q.rows.empty()) {   // --solo-slot's never-joined pads
            std::printf("# slot %d: never joined (fixed-width solo control) — no output\n", s);
            continue;
        }
        const std::string p = out + "-s" + std::to_string(s) + ".logits";
        if (!rig_write_logits(p.c_str(), q.rows, q.gen)) { m1_session_close(S); return 1; }
        std::printf("# slot %d: %d rows, first ids [", s, (int) q.gen.size());
        for (size_t k = 0; k < q.gen.size() && k < 10; ++k) std::printf(" %d", (int) q.gen[k]);
        std::printf(" ]\n");
        if (q.expect.empty()) continue;

        std::vector<std::vector<float>> xrows;
        std::vector<int32_t> xtoks;
        if (!rig_read_logits(q.expect.c_str(), xrows, xtoks)) { m1_session_close(S); return 1; }
        bool bad = false;
        if (xtoks.size() != q.gen.size()) {
            std::printf("FAIL: slot %d vs %s: id COUNT %zu vs %zu\n", s, q.expect.c_str(),
                        q.gen.size(), xtoks.size());
            bad = true;
        }
        for (size_t k = 0; !bad && k < q.gen.size() && k < xtoks.size(); ++k)
            if (q.gen[k] != xtoks[k]) {
                std::printf("FAIL: slot %d vs %s: FIRST DIVERGENT ID at position %zu: batch %d, "
                            "solo %d\n", s, q.expect.c_str(), k, (int) q.gen[k], (int) xtoks[k]);
                bad = true;
            }
        if (!bad) {
            if (xrows.size() != q.rows.size() || xrows[0].size() != q.rows[0].size()) {
                std::printf("FAIL: slot %d vs %s: logits SHAPE %zux%zu vs %zux%zu\n", s,
                            q.expect.c_str(), q.rows.size(), q.rows[0].size(), xrows.size(),
                            xrows[0].size());
                bad = true;
            }
            for (size_t r = 0; !bad && r < q.rows.size(); ++r)
                for (size_t k = 0; k < q.rows[r].size(); ++k)
                    if (std::memcmp(&q.rows[r][k], &xrows[r][k], sizeof(float)) != 0) {
                        std::printf("FAIL: slot %d vs %s: ids IDENTICAL, first differing LOGITS "
                                    "float at row %zu offset %zu: batch %.9e (0x%08x), solo "
                                    "%.9e (0x%08x)\n", s, q.expect.c_str(), r, k,
                                    (double) q.rows[r][k],
                                    *(const unsigned *) &q.rows[r][k],
                                    (double) xrows[r][k],
                                    *(const unsigned *) &xrows[r][k]);
                        bad = true;
                        break;
                    }
        }
        if (bad) rc = 1;
        else std::printf("COMPARE slot %d vs %s: ids and logits BYTE-IDENTICAL (%zu rows)\n", s,
                         q.expect.c_str(), q.rows.size());
    }
    m1_session_close(S);
    std::printf("BATCH RIG: %s\n", rc == 0 ? "PASS" : "FAIL");
    return rc;
}
