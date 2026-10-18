// engine/m1_engine.cpp — M1: the whole-token engine.
//
// U1 landed the LOADER (the full per-layer surface + the checklist check in-binary).
// U2 adds the ONE-LAYER FORWARD: engine/m1_layer.h/.cpp builds one layer (attention + MLA
// absorption + KV + router + MoE split + shared expert) from ggml ops in the engine's own
// cgraphs, and this file gains --layer-check, which drives it for a fixed token sequence and
// checks it against the reference's own per-layer checksums (tier A) and against the
// monolithic-op-sequence arm (tier B, the rig control).
// U3 adds --loop: THE WHOLE-TOKEN ENGINE — embedding -> 36 x the U2 layer with its own KV ->
// output_norm -> output head -> greedy argmax -> next token, the reference's logits format out,
// the per-layer tier-A table as the localisation instrument, and --falsify, the loop's paired
// concurrent/serialised statistic with its --serialize-both control.
//
// usage: builds/engine/m1 [--model <gguf>] [--quiet]                    # U1 loader check
//        builds/engine/m1 --layer-check --tokens <file> [...]           # U2 layer check
//        builds/engine/m1 --loop --tokens <file> --out <logits> [...]   # U3 the token loop
//        builds/engine/m1 --falsify [--serialize-both] [...]            # U3 the falsifier
//
// EXIT CODES: 0 = the check that was asked for passed; 1 = it failed (the FAIL lines say how).

#include "gguf_model.h"
#define STORE_INSTR 1     // U7: the get/combine split inside Store::drain (see engine/store.h)
#include "m1_layer.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"   // the ROCm0 backend behind ggml-cuda's API (U4a: the split's GPU half)

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <map>
#include <memory>
#include <numeric>
#include <string>
#include <sys/resource.h>
#include <vector>

static const char * DEFAULT_MODEL = "models/Mistral-Small-4-119B-2603-UD-IQ2_XXS.gguf";

// ---------------------------------------------------------------------------
// the three checksums, printed in the reference's own format so a tier-A comparison is
// line-for-line (the reference prints "sum=%.9e sumsq=%.9e first4= %.4f ...").
// ---------------------------------------------------------------------------
static void print_sums(const char * tag, const Sums & s) {
    std::printf("  %-14s sum=%+.9e sumsq=%.9e first4=%+.4f %+.4f %+.4f %+.4f\n",
                tag, s.sum, s.sumsq, s.f4[0], s.f4[1], s.f4[2], s.f4[3]);
}

// ---------------------------------------------------------------------------
// token_embd -> the layer-0 activation, exactly as the reference builds it
// (llama-graph.cpp:build_inp_embd: ggml_get_rows(tok_embd, tokens); f_embedding_scale is 0
//  for mistral4 and n_embd_inp == n_embd, so there is NO scaling and NO padding)
// ---------------------------------------------------------------------------
struct EmbGraph {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_tensor * w = nullptr, * tin = nullptr, * out = nullptr;
    ggml_cgraph * gf = nullptr;
    ~EmbGraph() { if (buf) ggml_backend_buffer_free(buf); if (ctx) ggml_free(ctx); }

    void build(GGUFModel & m, ggml_backend_t backend) {
        ggml_init_params ip = { 16u << 20, nullptr, true };
        ctx = ggml_init(ip);
        CHECK(ctx, "ggml_init failed for the embedding graph");
        LogicalTensor lt = m.logical("token_embd.weight");
        // SEAM 3 GUARD: this graph uploads ONE segment (segs[0]).  A multi-segment GLOBAL tensor
        // would load 1/G of the file silently, so refuse instead.  (The per-layer graph carries
        // the same guard at m1_layer.cpp:127; found by the U4e review — the doc claimed every
        // non-expert multi-segment was refused, but only the per-layer path checked.)
        CHECK(lt.segs.size() == 1, "token_embd.weight is multi-segment (%zu) — this graph loads one"
              " segment only", lt.segs.size());
        CHECK(lt.segs[0].type == GGML_TYPE_Q4_K, "token_embd is %s, expected q4_K",
              ggml_type_name(lt.segs[0].type));
        w   = ggml_new_tensor_2d(ctx, lt.segs[0].type, lt.ne[0], lt.ne[1]);
        tin = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
        ggml_set_input(tin);
        out = ggml_get_rows(ctx, w, tin);   // F32 [H, 1]
        buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        CHECK(buf, "alloc failed for the embedding graph");
        ggml_backend_tensor_set(w, lt.segs[0].bytes, 0, lt.segs[0].nbytes);
        gf = ggml_new_graph_custom(ctx, 64, false);
        ggml_build_forward_expand(gf, out);
    }
    void run(ggml_backend_t backend, int32_t tok, float * x) {
        ggml_backend_tensor_set(tin, &tok, 0, sizeof(tok));
        ggml_backend_graph_compute(backend, gf);
        ggml_backend_tensor_get(out, x, 0, sizeof(float) * (size_t) H_DIM);
    }
};

// ---------------------------------------------------------------------------
// the graph tail: output_norm (RMS, F32) -> output head (Q4_K [4096,131072] x 4096) -> logits
//
// MEASURED, transcribed from src/llama-fork/src/models/deepseek2.cpp:768-786 (the main graph's
// tail — NOT the MTP branch at :430-455):
//     cur = build_norm(cur, model.output_norm, NULL, LLM_NORM_RMS, -1);   // == "result_norm"
//     if (crop_after_nextn(inp_out_ids)) cur = ggml_get_rows(ctx0, cur, inp_out_ids);
//     cur = ggml_mul_mat(ctx0, model.output, cur);                        // == "result_output"
//     res->t_logits = cur;
// crop_after_nextn() is FALSE for this model: it needs cparams.embeddings_nextn and the model
// carries no nextn/MTP tensors at all (579 = 36*16+3, U1's checklist).  With one token per
// ubatch (-ub 1) inp_out_ids selects that row alone, so the get_rows is the identity too.
// The reference therefore dumps exactly mul_mat(output.weight, rms_norm(x)*output_norm).
// ---------------------------------------------------------------------------
struct HeadGraph {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_tensor * w_norm = nullptr, * w_out = nullptr;
    ggml_tensor * g_h = nullptr, * g_norm = nullptr, * g_logits = nullptr;
    ggml_cgraph * gf = nullptr;
    int64_t n_vocab = 0;
    ~HeadGraph() { if (buf) ggml_backend_buffer_free(buf); if (ctx) ggml_free(ctx); }

    void build(GGUFModel & m, ggml_backend_t backend) {
        ggml_init_params ip = { 16u << 20, nullptr, true };
        ctx = ggml_init(ip);
        CHECK(ctx, "ggml_init failed for the output head graph");
        LogicalTensor ln = m.logical("output_norm.weight");
        LogicalTensor lo = m.logical("output.weight");
        // SEAM 3 GUARD (same reason as the embedding graph's): both are uploaded as ONE segment.
        CHECK(ln.segs.size() == 1, "output_norm.weight is multi-segment (%zu) — one segment only",
              ln.segs.size());
        CHECK(lo.segs.size() == 1, "output.weight is multi-segment (%zu) — one segment only",
              lo.segs.size());
        CHECK(ln.segs[0].type == GGML_TYPE_F32, "output_norm is %s, expected f32",
              ggml_type_name(ln.segs[0].type));
        CHECK(lo.segs[0].type == GGML_TYPE_Q4_K, "output.weight is %s, expected q4_K",
              ggml_type_name(lo.segs[0].type));
        w_norm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ln.ne[0]);
        w_out  = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, lo.ne[0], lo.ne[1]);
        n_vocab = lo.ne[1];
        ggml_set_name(w_norm, "output_norm.weight");
        ggml_set_name(w_out,  "output.weight");

        g_h = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H_DIM, 1);
        ggml_set_input(g_h);
        ggml_set_name(g_h, "m1_head_in");
        g_norm = ggml_mul(ctx, ggml_rms_norm(ctx, g_h, L_RMS_EPS), w_norm);
        ggml_set_name(g_norm, "result_norm");       // the reference's own cb() name
        g_logits = ggml_mul_mat(ctx, w_out, g_norm);
        ggml_set_name(g_logits, "result_output");

        buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        CHECK(buf, "alloc failed for the output head graph");
        // M1/Q1: the damage probe reaches the head too — `output_norm.weight` is the LAST weight
        // before the logits, so scaling it is a pure TEMPERATURE on the output distribution (every
        // logit x (1+eps)) and therefore a first-order, signed, monotone quality knob: dNLL ~
        // +eps*(E_p[z] - z_target).  That is the calibration dose the layer norms cannot supply
        // (a layer-norm scale is absorbed by the next RMS norm / is order-preserving for the
        // router, MEASURED: flat to eps=0.3 — see results/q1.md).
        if (m1_damage_match(w_norm))
            m1_damage_set(w_norm, ln.segs[0].bytes, 0, ln.segs[0].nbytes);
        else
            ggml_backend_tensor_set(w_norm, ln.segs[0].bytes, 0, ln.segs[0].nbytes);
        ggml_backend_tensor_set(w_out,  lo.segs[0].bytes, 0, lo.segs[0].nbytes);

        gf = ggml_new_graph_custom(ctx, 16, false);
        ggml_build_forward_expand(gf, g_logits);
    }
    void run(ggml_backend_t backend, const float * h, float * logits) const {
        ggml_backend_tensor_set(g_h, h, 0, sizeof(float) * (size_t) H_DIM);
        ggml_backend_graph_compute(backend, gf);
        ggml_backend_tensor_get(g_logits, logits, 0, sizeof(float) * (size_t) n_vocab);
    }
};

// ---------------------------------------------------------------------------
// the sampler — llama's greedy, transcribed
//
// MEASURED, src/llama-fork/src/llama-sampler.cpp:1053-1060: the greedy sampler walks the
// candidate array (built in vocab-id order at :945) and moves `selected` only on a STRICTLY
// greater logit, so the LOWEST id wins a tie.  Same convention here.
// ---------------------------------------------------------------------------
static int32_t greedy_argmax(const float * row, int64_t n) {
    int32_t best = 0;
    for (int64_t i = 1; i < n; ++i) if (row[i] > row[best]) best = (int32_t) i;
    return best;
}

// the reference's own logits file format (moe_logits_dump.cpp:25-46): u64 n_decode,
// u32 n_vocab, then n_decode x n_vocab f32, plus <path>.tokens = u32 n, then n i32.
static bool write_logits(const char * path, const std::vector<std::vector<float>> & rows,
                         const std::vector<int32_t> & toks) {
    if (rows.empty()) { std::printf("FAIL: no logits rows to write\n"); return false; }
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

static bool read_tokens_bin(const char * path, std::vector<int32_t> & toks) {
    FILE * f = std::fopen(path, "rb");
    if (!f) { std::printf("FAIL: cannot open %s\n", path); return false; }
    uint32_t n = 0;
    if (std::fread(&n, sizeof n, 1, f) != 1) { std::fclose(f); std::printf("FAIL: %s: no count\n", path); return false; }
    toks.resize(n);
    const size_t got = std::fread(toks.data(), sizeof(int32_t), n, f);
    std::fclose(f);
    if (got != n) { std::printf("FAIL: %s: %u tokens announced, %zu read\n", path, n, got); return false; }
    return true;
}

// ---------------------------------------------------------------------------
// tier A, second instrument: the reference's own logits fingerprints.  The harness prints
//   [moe_dump] get_logits_ith(-1) n_vocab=131072 sum=%.9e sumsq=%.9e first4=...
// once per decode step (moe_logits_dump.cpp), i.e. a double-precision checksum of the whole
// 131072-float logits row at that position.  It is independent of our layer code and it is
// what validates the position mapping (only the steps that HAVE a logits row can match).
// ---------------------------------------------------------------------------
struct RefLogits {
    std::vector<Sums> v;
    int  n_vocab = 0;
    bool parse(const char * path) {
        FILE * f = std::fopen(path, "r");
        if (!f) return false;
        char line[2048];
        while (std::fgets(line, sizeof line, f)) {
            const char * p = std::strstr(line, "[moe_dump] get_logits_ith");
            if (!p) continue;
            const char * pv = std::strstr(p, "n_vocab=");
            const char * ps = std::strstr(p, "sum=");
            const char * pq = std::strstr(p, "sumsq=");
            const char * pf = std::strstr(p, "first4=");
            if (!pv || !ps || !pq) continue;
            Sums s;
            n_vocab = std::atoi(pv + 8);
            s.sum   = std::strtod(ps + 4, nullptr);
            s.sumsq = std::strtod(pq + 6, nullptr);
            if (pf) std::sscanf(pf + 7, "%f %f %f %f", &s.f4[0], &s.f4[1], &s.f4[2], &s.f4[3]);
            v.push_back(s);
        }
        std::fclose(f);
        return !v.empty();
    }
};

// timing/load: every timing number this binary prints is preceded or followed by the load
// average it was measured under (the host is not quiet — see results/m1-u3.md).
static double now_s() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double) ts.tv_sec + 1e-9 * (double) ts.tv_nsec;
}
static const char * loadavg_str(char * buf, size_t n) {
    std::snprintf(buf, n, "unreadable");
    FILE * f = std::fopen("/proc/loadavg", "r");
    if (!f) return buf;
    if (!std::fgets(buf, (int) n, f)) buf[0] = 0;
    std::fclose(f);
    for (char * p = buf; *p; ++p) if (*p == '\n') { *p = 0; break; }
    return buf;
}
static void print_loadavg(const char * tag) {
    char b[256];
    std::printf("# loadavg %-16s %s\n", tag, loadavg_str(b, sizeof b));
}

// the reference's print resolution: it prints %.9e (10 significant digits), so a relative
// delta <= 5e-10 is indistinguishable from zero at the resolution the reference can express.
// The LOCALISATION TRIGGER is 1e-6 — three orders of magnitude above that floor, i.e. a real
// divergence, not the float floor.  Neither is a "picked tolerance": the token line is the
// verdict (plan §Q1), these two numbers only decide when the bisection starts.
static constexpr double REL_PRINT_FLOOR = 5e-10;
static constexpr double REL_TRIGGER     = 1e-6;

// ---------------------------------------------------------------------------
// H1 — THE RESIDENT SET (--resident-set <path.json>)
// ---------------------------------------------------------------------------
// The accepted format is U6's OWN, the file its tool already emits (logs/engine/m1-u6/hotset-A48.json,
// produced by scripts/analyse_route_u6.py):
//   {"n_res": S, "hot": [<N_LAYERS ORDERED lists of S expert ids>], ...}
// "ordered" is the whole point: slot i of the layer's VRAM store is hot[l][i].  The producer's other
// keys (traces, fitted_coverage, counts) are ignored — this reads THAT file's shape and does not
// invent a second one.
//
// The scan is hand-rolled, and stated as such: the engine links ggml and nothing else (no JSON
// dependency), the accepted object is fixed and small, and every field it consumes is CHECKed
// (row count, row length, id range, uniqueness).  It accepts the producer's file and refuses
// everything else — it is not a general JSON parser and does not pretend to be.
static bool read_hotset_json(const char * path, std::vector<std::vector<int>> & hot, int & n_res) {
    FILE * f = std::fopen(path, "r");
    if (!f) { std::printf("FAIL: --resident-set: cannot open %s\n", path); return false; }
    std::string s;
    { char buf[1 << 16]; size_t n; while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) s.append(buf, n); }
    std::fclose(f);

    auto after_key = [&](const char * k) -> size_t {
        const size_t p = s.find(k);
        if (p == std::string::npos) return p;
        const size_t c = s.find(':', p + std::strlen(k));
        return c == std::string::npos ? c : c + 1;
    };
    size_t p = after_key("\"n_res\"");
    if (p == std::string::npos) { std::printf("FAIL: %s: no \"n_res\" key\n", path); return false; }
    n_res = (int) std::strtol(s.c_str() + p, nullptr, 10);
    if (n_res < 1 || n_res > (int) N_EXPERT) {
        std::printf("FAIL: %s: n_res=%d out of range [1,%d]\n", path, n_res, (int) N_EXPERT);
        return false;
    }
    p = after_key("\"hot\"");
    if (p == std::string::npos) { std::printf("FAIL: %s: no \"hot\" key\n", path); return false; }
    size_t i = s.find('[', p);
    if (i == std::string::npos) { std::printf("FAIL: %s: \"hot\" is not an array\n", path); return false; }
    ++i;   // the set's outer '['
    auto skip_ws = [&]() { while (i < s.size() && (s[i] == ' ' || s[i] == '\n' || s[i] == '\r' ||
                                                    s[i] == '\t' || s[i] == ',')) ++i; };
    while (true) {
        skip_ws();
        if (i >= s.size()) { std::printf("FAIL: %s: \"hot\" array unterminated\n", path); return false; }
        if (s[i] == ']') break;                                    // end of the outer array
        if (s[i] != '[') { std::printf("FAIL: %s: \"hot\" element %zu is not an array\n", path, hot.size()); return false; }
        ++i;
        std::vector<int> row;
        while (true) {
            skip_ws();
            if (i < s.size() && s[i] == ']') { ++i; break; }
            char * end = nullptr;
            const long v = std::strtol(s.c_str() + i, &end, 10);
            if (end == s.c_str() + i) { std::printf("FAIL: %s: bad integer in \"hot\" row %zu\n", path, hot.size()); return false; }
            row.push_back((int) v);
            i = (size_t) (end - s.c_str());
        }
        hot.push_back(row);
    }
    if ((int) hot.size() != (int) N_LAYERS) {
        std::printf("FAIL: %s: %zu per-layer sets, expected %d\n", path, hot.size(), (int) N_LAYERS);
        return false;
    }
    for (int l = 0; l < (int) N_LAYERS; ++l) {
        if ((int) hot[(size_t) l].size() != n_res) {
            std::printf("FAIL: %s: layer %d has %zu members, \"n_res\" says %d\n",
                        path, l, hot[(size_t) l].size(), n_res);
            return false;
        }
        std::vector<char> seen((size_t) N_EXPERT, 0);
        for (int e : hot[(size_t) l]) {
            if (e < 0 || e >= (int) N_EXPERT) { std::printf("FAIL: %s: layer %d id %d out of range\n", path, l, e); return false; }
            if (seen[(size_t) e]) { std::printf("FAIL: %s: layer %d repeats expert %d\n", path, l, e); return false; }
            seen[(size_t) e] = 1;
        }
    }
    return true;
}

// H2/U2a — BUILD ONE STORE-LIST ENTRY.  A RAM-labelled entry built on the CPU backend is
// ZERO-COPY: it aliases the mmap's whole [0,N_EXPERT) expert block, so its membership is a
// PREDICATE over the list ("a non-member of the VRAM entry is a member of the alias") and an
// interval spec's [lo,hi) is handed over as the equivalent membership SET.  Every other entry
// keeps the pre-U2a shape.  ONE definition, used by BOTH list build sites, so the two arms of the
// engine cannot drift apart on which form they build.
static Store * make_store(const StoreSpec & sp, const LogicalTensor & gu_lt, const LogicalTensor & dn_lt,
                          ggml_backend_t be, int nt) {
    std::vector<int> interval_set;
    const std::vector<int> * setp = sp.by_set ? &sp.members : nullptr;
    const bool alias = sp.loc == Store::RAM && be &&
                       ggml_backend_dev_type(ggml_backend_get_device(be)) == GGML_BACKEND_DEVICE_TYPE_CPU;
    if (alias && !sp.by_set) {
        interval_set.reserve((size_t) std::max(0, sp.hi - sp.lo));
        for (int e = sp.lo; e < sp.hi; ++e) interval_set.push_back(e);
        setp = &interval_set;
    }
    Store * s = new Store();
    s->build(sp.loc, gu_lt, dn_lt, sp.lo, sp.hi, be, nt, setp, alias);
    return s;
}

// H1 — THE SET'S OWN AUDIT, run once per layer over the RUNNING store list (not over a copy of the
// spec).  `Store::slot()` is the seam H2's policy reads, and a seat nothing calls is
// indistinguishable from an absent seat, so the mechanism asserts its own predicate ON THE LIST:
// every expert in [0,N_EXPERT) has EXACTLY ONE home, and a SET-backed store's slot() maps each
// member to the position its weights were packed at (set order == slot order == the order the
// upload wrote).  The interval list must satisfy the same predicate, which is why this doubles as
// an M0 regression check.
static void audit_store_list(const std::vector<Store *> & stores, int layer, bool verbose,
                             const LogicalTensor * gu_lt = nullptr,
                             const LogicalTensor * d_lt = nullptr) {
    for (int e = 0; e < (int) N_EXPERT; ++e) {
        int homes = 0;
        for (const Store * s : stores) if (s->slot(layer, e) >= 0) ++homes;
        CHECK(homes == 1, "layer %d: expert %d has %d homes in the store list (want exactly 1)", layer, e, homes);
    }
    for (const Store * s : stores) {
        if (!s->is_set()) continue;
        if (s->aliased) {
            // U2a: an ALIASED store's slot is the ROW the expert already occupies in the file, not
            // its position in a packed set, so the SET predicate above does not apply.  What must
            // hold instead is (i) the tensors POINT AT THE SOURCE BYTES — the check that would
            // fire if anything re-pointed them at an anonymous allocation — and (ii) every member's
            // slot is its own file row inside its slice.
            //
            // U2b — THE THREE GAPS THE L47 REVIEW NAMED ARE CLOSED HERE, and they are load-bearing
            // because once rotation changes the partition, byte-identity (I1) is no longer the
            // oracle and THIS audit is the standing gate:
            //   (A) sl.local_of — the table `enqueue` actually reads (store.h:559), which was
            //       unchecked: a slice can carry a correct pos_of and still hand the kernel the
            //       NEIGHBOURING expert's row.  A member's row must be its own (e - sl.lo), a
            //       non-member must be -1, and the two tables must agree on membership.
            //   (B) the slice tensor's TYPE and SHAPE — a wrong-but-in-bounds tensor (one row short,
            //       or the other quantisation) reads the wrong bytes and `alias_bind` accepts it.
            //   (C) gu_base/d_base point at the RIGHT expert row of the RIGHT segment — checked
            //       against the file's own geometry (stride from the declared type, offset from the
            //       segment's lo), which is the only reconstruction the engine can make independently
            //       of the code that computed it.
            for (const Slice & sl : s->slices) {
                CHECK(sl.as_gu->data == sl.gu_base && sl.as_d->data == sl.d_base,
                      "layer %d: %s alias slice [%d,%d) does not point at the source bytes",
                      layer, s->loc_name, sl.lo, sl.hi);
                // (A) sl.local_of == the membership row table enqueue reads, and agrees with slot()
                for (int e = sl.lo; e < sl.hi; ++e) {
                    const int row = s->slot(layer, e);
                    if (row < 0) continue;   // not a member: the VRAM entry is this expert's home
                    CHECK(row == e - sl.lo,
                          "layer %d: %s alias row mismatch at expert %d (row %d, want %d)",
                          layer, s->loc_name, e, row, e - sl.lo);
                }
                for (int e = 0; e < (int) N_EXPERT; ++e) {
                    const int lof = (e >= sl.lo && e < sl.hi) ? sl.local_of[(size_t) e] : -1;
                    const int want = (e >= sl.lo && e < sl.hi && s->slot(layer, e) >= 0) ? e - sl.lo : -1;
                    CHECK(lof == want,
                          "layer %d: %s alias LOCAL_OF mismatch at expert %d (local_of %d, want %d) — "
                          "this is the table enqueue reads, so the kernel would be served the wrong row",
                          layer, s->loc_name, e, lof, want);
                }
                // (C) the base address is the RIGHT row of the RIGHT segment, recomputed here from
                // the file's geometry (the same arithmetic, but from the segment table rather than
                // from the builder's own locals).
                {
                    const size_t gu_stride = ggml_row_size(sl.gu_type, gu_lt->ne[0]) * (size_t) gu_lt->ne[1];
                    const size_t d_stride  = ggml_row_size(sl.d_type,  d_lt->ne[0])  * (size_t) d_lt->ne[1];
                    const Segment & gs = gu_lt->segs[(size_t) Store::seg_index(*gu_lt, sl.lo)];
                    const Segment & ds = d_lt->segs[(size_t)  Store::seg_index(*d_lt,  sl.lo)];
                    const uint8_t * gu_want = gs.bytes + (size_t) (sl.lo - gs.lo) * gu_stride;
                    const uint8_t * d_want  = ds.bytes + (size_t) (sl.lo - ds.lo) * d_stride;
                    CHECK(sl.gu_stride == gu_stride && sl.d_stride == d_stride,
                          "layer %d: %s alias stride mismatch (%zu/%zu want %zu/%zu)",
                          layer, s->loc_name, sl.gu_stride, sl.d_stride, gu_stride, d_stride);
                    CHECK(sl.gu_base == gu_want && sl.d_base == d_want,
                          "layer %d: %s alias slice [%d,%d) base is not its own first row "
                          "(gu %p want %p, d %p want %p)",
                          layer, s->loc_name, sl.lo, sl.hi,
                          (const void *) sl.gu_base, (const void *) gu_want,
                          (const void *) sl.d_base,  (const void *) d_want);
                    CHECK(sl.gu_nbytes == gu_stride * (size_t) (sl.hi - sl.lo) &&
                          sl.d_nbytes == d_stride  * (size_t) (sl.hi - sl.lo),
                          "layer %d: %s alias slice [%d,%d) byte count is not rows x stride",
                          layer, s->loc_name, sl.lo, sl.hi);
                }
                // (B) the slice tensor's type and shape
                CHECK(sl.as_gu->type == sl.gu_type && sl.as_d->type == sl.d_type,
                      "layer %d: %s alias slice [%d,%d) type mismatch (gu %s want %s, d %s want %s)",
                      layer, s->loc_name, sl.lo, sl.hi, ggml_type_name(sl.as_gu->type),
                      ggml_type_name(sl.gu_type), ggml_type_name(sl.as_d->type), ggml_type_name(sl.d_type));
                CHECK(sl.as_gu->ne[0] == gu_lt->ne[0] && sl.as_gu->ne[1] == gu_lt->ne[1] &&
                      sl.as_gu->ne[2] == sl.hi - sl.lo,
                      "layer %d: %s alias gate_up shape [%lld,%lld,%lld] != [%lld,%lld,%d) rows",
                      layer, s->loc_name, (long long) sl.as_gu->ne[0], (long long) sl.as_gu->ne[1],
                      (long long) sl.as_gu->ne[2], (long long) gu_lt->ne[0], (long long) gu_lt->ne[1],
                      sl.hi - sl.lo);
                CHECK(sl.as_d->ne[0] == d_lt->ne[0] && sl.as_d->ne[1] == d_lt->ne[1] &&
                      sl.as_d->ne[2] == sl.hi - sl.lo,
                      "layer %d: %s alias down shape [%lld,%lld,%lld] != [%lld,%lld,%d) rows",
                      layer, s->loc_name, (long long) sl.as_d->ne[0], (long long) sl.as_d->ne[1],
                      (long long) sl.as_d->ne[2], (long long) d_lt->ne[0], (long long) d_lt->ne[1],
                      sl.hi - sl.lo);
            }
            if (!verbose) continue;
            std::printf("# layer %d %s ALIAS[%d] zero-copy over %zu slice(s), rows = global ids\n",
                        layer, s->loc_name, s->n_e, s->slices.size());
            continue;
        }
        for (int i = 0; i < s->n_e; ++i)
            CHECK(s->slot(layer, s->members[(size_t) i]) == i,
                  "layer %d: %s SET slot mismatch at %d (expert %d)", layer, s->loc_name, i, s->members[(size_t) i]);
        // the predicate above reads pos_of.  The table the KERNEL reads is
        // Slice::local_of, which replace_slot rewrites on every refresh — so it is audited here, in
        // the same pass, on the running list.  This is the seat the set branch was missing: a wrong
        // local_of gives right bytes, right pos_of and exit 0.
        s->audit_index_table(layer);
        if (!verbose) continue;
        std::printf("# layer %d %s SET[%d] slot 0..7 =", layer, s->loc_name, s->n_e);
        for (int i = 0; i < s->n_e && i < 8; ++i) std::printf(" %d", s->members[(size_t) i]);
        std::printf("%s\n", s->n_e > 8 ? " ..." : "");
    }
}

// U10 — THE REFRESH AUDIT, SCOPED TO WHAT THE REFRESH CHANGED.
//
// L50 wired the full `audit_store_list` in after EVERY refresh, and that is the standing gate once
// rotation moves the partition (byte-identity is no longer the oracle).  The audit is O(N_EXPERT)
// per store plus O(M LOG M) sorting-free walks, and a token runs ~19 refreshes, so it sits on the
// hot path having been written as an end-of-run check.  THIS function asserts the SAME invariants
// for the ONLY entries a refresh can write: the admitted expert, the evicted expert, and their two
// tables.  Nothing else in either store is touched by Store::replace_slot / Store::set_member —
// that is the whole argument for the narrowing, and it is falsified by the mutation probe
// (logs/engine/m1-h3-u10/mutate.sh), which mutates a swapped slot and shows BOTH this check and the
// full walk fire with the same diagnosis.
//
// WHAT EACH SIDE MUST STILL HOLD after one swap of `e` INTO `slot` of the VRAM store, evicting
// `old`, with the alias store's membership mirrored (the two are complements):
//   (1) EXACTLY ONE HOME for `e` and for `old` across the pair — the partition invariant, which is
//       what "the alias filters instead of copying" makes load-bearing.
//   (2) the VRAM store's three tables name the same row: members[slot], pos_of[e], local_of[e] all
//       equal `slot`, and all three entries for `old` are cleared (-1).  local_of is the table
//       `enqueue` reads, so (2) is the one a table-only gate can miss while the bytes are right.
//   (3) the byte VERIFIED source pointer is the file row of the expert the slot claims to hold —
//       recomputed here from the segment table (the full audit's (C) check, per row instead of per
//       slice).  Without this a refresh can insert the right expert's id with the wrong source row.
//   (4) the ALIAS store's local_of for both experts, and its slice membership — the alias moves no
//       bytes, so its predicate IS its correctness.
// NOT checked here, by construction, and left to the end-of-run full walk (and to --audit-full):
// the other 126 experts' entries and the slice tensor's type/shape/base — none of which a refresh
// can write.
static void audit_refresh_swap(Store * vram, Store * ram, int layer, int slot, int e, int old,
                               const LogicalTensor * gu_lt, const LogicalTensor * d_lt) {
    CHECK(vram && ram, "layer %d: rotation refresh audit needs both stores", layer);
    CHECK(!vram->aliased && vram->is_set() && vram->slices.size() == 1,
          "layer %d: refresh audit: the VRAM store is not a flat SET store", layer);
    CHECK(ram->aliased && ram->is_set() && ram->slices.size() == 1,
          "layer %d: refresh audit: the alias store is not a flat SET store", layer);
    const Slice & vs = vram->slices[0];
    const Slice & rs = ram->slices[0];
    const int rlo = rs.lo;
    CHECK(e >= rlo && e < rs.hi && old >= rlo && old < rs.hi,
          "layer %d: refresh audit: experts %d/%d outside the alias run [%d,%d)", layer, e, old, rlo, rs.hi);

    // (1) exactly one home, both directions, for both experts
    CHECK(vram->slot(layer, e) == slot,
          "layer %d: refresh audit: routed store says admitted expert %d is at %d, the swap put it "
          "at %d", layer, e, vram->slot(layer, e), slot);
    CHECK(ram->slot(layer, e) < 0,
          "layer %d: refresh audit: admitted expert %d is STILL a member of the alias (row %d) — "
          "two homes", layer, e, ram->slot(layer, e));
    CHECK(vram->slot(layer, old) < 0,
          "layer %d: refresh audit: evicted expert %d is STILL resident in the VRAM store (slot %d)",
          layer, old, vram->slot(layer, old));
    CHECK(ram->slot(layer, old) == old - rlo,
          "layer %d: refresh audit: evicted expert %d has alias row %d, want its own row %d",
          layer, old, ram->slot(layer, old), old - rlo);

    // (2) the three tables the policy and the kernel read
    CHECK((int) vram->members.size() > slot && vram->members[(size_t) slot] == e,
          "layer %d: refresh audit: VRAM members[%d] is %d, not the admitted expert %d",
          layer, slot, slot < (int) vram->members.size() ? vram->members[(size_t) slot] : -1, e);
    CHECK((int) vs.members.size() > slot && vs.members[(size_t) slot] == e,
          "layer %d: refresh audit: VRAM slice members[%d] is %d, not the admitted expert %d",
          layer, slot, slot < (int) vs.members.size() ? vs.members[(size_t) slot] : -1, e);
    CHECK(vs.local_of[(size_t) e] == slot,
          "layer %d: refresh audit: VRAM LOCAL_OF for the admitted expert %d says %d, the swap put "
          "it at %d (this is the table enqueue reads)",
          layer, e, vs.local_of[(size_t) e], slot);
    CHECK(vs.local_of[(size_t) old] == -1,
          "layer %d: refresh audit: VRAM LOCAL_OF stale entry for the evicted expert %d -> %d "
          "(a non-member's entry must be -1)", layer, old, vs.local_of[(size_t) old]);
    CHECK(vram->pos_of[(size_t) e] == slot && vram->pos_of[(size_t) old] == -1,
          "layer %d: refresh audit: VRAM POS_OF disagreement after the swap (expert %d -> %d, "
          "evicted %d -> %d)", layer, e, vram->pos_of[(size_t) e], old, vram->pos_of[(size_t) old]);

    // (3) the source row is the file's row for the admitted expert, recomputed from the segment
    // table (the same arithmetic the full audit makes for the slice base, per row)
    {
        const size_t gu_stride = ggml_row_size(gu_lt->segs[(size_t) Store::seg_index(*gu_lt, e)].type,
                                               gu_lt->ne[0]) * (size_t) gu_lt->ne[1];
        const size_t d_stride  = ggml_row_size(d_lt->segs[(size_t)  Store::seg_index(*d_lt,  e)].type,
                                               d_lt->ne[0])  * (size_t) d_lt->ne[1];
        CHECK(vs.gu_stride == gu_stride && vs.d_stride == d_stride,
              "layer %d: refresh audit: VRAM slice stride %zu/%zu does not match the file's %zu/%zu "
              "for expert %d", layer, vs.gu_stride, vs.d_stride, gu_stride, d_stride, e);
        const Segment & gs = gu_lt->segs[(size_t) Store::seg_index(*gu_lt, e)];
        const Segment & ds = d_lt->segs[(size_t)  Store::seg_index(*d_lt,  e)];
        const uint8_t * gu_want = gs.bytes + (size_t) (e - gs.lo) * gu_stride;
        const uint8_t * d_want  = ds.bytes + (size_t) (e - ds.lo) * d_stride;
        CHECK(vs.gu_src[(size_t) slot] == gu_want && vs.d_src[(size_t) slot] == d_want,
              "layer %d: refresh audit: slot %d (expert %d) uploads from %p/%p, the file's row is "
              "%p/%p — the row that landed at that offset may be another expert's",
              layer, slot, e, (const void *) vs.gu_src[(size_t) slot], (const void *) vs.d_src[(size_t) slot],
              (const void *) gu_want, (const void *) d_want);
    }

    // (4) the alias side: a predicate, so its table IS its correctness
    CHECK(rs.local_of[(size_t) e] == -1,
          "layer %d: refresh audit: alias LOCAL_OF stale entry for the ADMITTED expert %d -> %d "
          "(it must have left the alias)", layer, e, rs.local_of[(size_t) e]);
    CHECK(rs.local_of[(size_t) old] == old - rlo,
          "layer %d: refresh audit: alias LOCAL_OF for the evicted expert %d says %d, want its own "
          "row %d", layer, old, rs.local_of[(size_t) old], old - rlo);
    CHECK(ram->pos_of[(size_t) e] == -1 && ram->pos_of[(size_t) old] == old - rlo,
          "layer %d: refresh audit: alias POS_OF disagreement after the swap (admitted %d -> %d, "
          "evicted %d -> %d)", layer, e, ram->pos_of[(size_t) e], old, ram->pos_of[(size_t) old]);
    // set_member edits the alias's membership LIST as well as its predicate; both must agree
    bool in_list = false;
    for (int x : ram->members) in_list |= (x == e);
    CHECK(!in_list, "layer %d: refresh audit: admitted expert %d is still in the alias's member "
          "list (its predicate says it left)", layer, e);
    in_list = false;
    for (int x : ram->members) in_list |= (x == old);
    CHECK(in_list, "layer %d: refresh audit: evicted expert %d is not in the alias's member list",
          layer, old);
}

// ---------------------------------------------------------------------------
// --layer-check
// ---------------------------------------------------------------------------
struct LayerCheckArgs {
    std::string model   = DEFAULT_MODEL;
    std::string tokens;
    std::string ref;
    std::string dump;
    int         layers  = 1;
    int         n_res   = 64;
    int         ref_pos0 = -1;   // the position that ref step 0 is at (-1 => n_prompt)
    int         kv_max  = 0;     // the KV/mask WIDTH (0 => n_pos, U2b: 256 matches the reference)
    bool        router_margin = false;
    int         threads = 16;
    bool        do_oracle = true;
    bool        do_store  = true;
    // H1: per-layer ordered resident sets (empty => the identity interval [0,n_res), today's
    // behaviour).  Read once from --resident-set and handed to the store list.
    std::vector<std::vector<int>> res_hot;
    std::string res_path;
};

static bool read_tokens(const char * path, std::vector<int32_t> & ids, int & n_prompt) {
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
    if (ids.empty() || np < 0 || np > (int) ids.size()) {
        std::fprintf(stderr, "FAIL: %s: %zu ids, n_prompt %d\n", path, ids.size(), np);
        return false;
    }
    return true;
}

static int layer_check(const LayerCheckArgs & a) {
    std::vector<int32_t> ids;
    int n_prompt = 0;
    if (!read_tokens(a.tokens.c_str(), ids, n_prompt)) return 1;
    const int n_pos = (int) ids.size();
    // U2b/U2c: the KV/mask WIDTH.  The reference's is the KV cache's hardcoded pad
    // (max(n_pad = 1, 256u), src/llama-fork/src/llama-kv-cache.cpp:1285 — 256 while fewer than
    // 257 cells are used, 512 after), NOT its ubatch/-b and NOT the sequence length: the
    // kq/softmax shape it dumps is [256,1,32].  The engine default (n_pos rows) reproduces the
    // reference's *masked* values but through a shorter k dimension in the kq and V*P
    // reductions.  --kv-max lets that width be matched.
    const int kv_max = a.kv_max > 0 ? a.kv_max : n_pos;
    if (a.layers < 1 || a.layers > (int) N_LAYERS) { std::fprintf(stderr, "FAIL: --layers %d\n", a.layers); return 1; }

    GGUFModel * m = GGUFModel::load(a.model.c_str());

    std::printf("=== M1 / U2 LAYER CHECK ===\n");
    std::printf("# model      = %s\n", m->path.c_str());
    std::printf("# tokens     = %d (n_prompt %d, %d decoded)\n", n_pos, n_prompt, n_pos - n_prompt);
    std::printf("# layers     = %d (0..%d)\n", a.layers, a.layers - 1);
    std::printf("# store list = n_res=%d\n", a.n_res);
    std::printf("# kv width   = %d (the reference's padded KV-cache width: 256 until 257 used cells, then 512 - NOT its -b)\n", kv_max);
    std::printf("# arms       = oracle(tier B, monolithic op sequence)=%d store(engine arm)=%d\n",
                (int) a.do_oracle, (int) a.do_store);
    std::printf("# ref pos0   = %d (the position ref step 0 is at)\n", a.ref_pos0 >= 0 ? a.ref_pos0 : n_prompt);
    std::printf("# kv type    = f16 (the pin's q8_0 caches are a NAMED divergence, plan risk 2)\n");
    {
        char buf[512];
        std::printf("# attention  = %s\n", m1_attn_params_str(buf, sizeof buf));
    }

    // ---------- backends ----------
    ggml_backend_t cpu = ggml_backend_cpu_init();
    CHECK(cpu, "cpu backend init failed");
    ggml_backend_cpu_set_n_threads(cpu, a.threads);
    std::printf("# backend    = %s, threads=%d\n", ggml_backend_name(cpu), a.threads);

    EmbGraph emb;
    emb.build(*m, cpu);

    // ---------- the layers ----------
    std::vector<std::unique_ptr<M1Layer>> L;
    std::vector<std::vector<Store *>>     store_lists;   // per layer: the store LIST (M0)
    for (int l = 0; l < a.layers; ++l) {
        std::unique_ptr<M1Layer> lp(new M1Layer());
        lp->build(*m, l, cpu, kv_max, /*nt*/ 1, a.n_res,
                  /*gpu_core*/ false, /*want_oracle*/ a.do_oracle);
        std::vector<Store *> keep;
        if (a.do_store) {
            // the store list is DERIVED from the geometry (M0's derive_store_specs), never
            // hardcoded: a flat file yields [0,n_res) + [n_res,128).  The CPU arm runs every
            // entry on the CPU backend; the list itself is the product machinery.
            // H1: with --resident-set the same list is derived from the layer's ORDERED SET
            // instead of the interval cut (one VRAM entry = the set, one RAM entry = its
            // complement); the shape and the locations are unchanged.
            const std::vector<int> * res = a.res_hot.empty() ? nullptr : &a.res_hot[(size_t) l];
            std::vector<StoreSpec> specs = derive_store_specs(lp->gu_lt, lp->dn_lt, a.n_res, {}, false, res);
            CHECK(!specs.empty(), "empty store list for layer %d", l);
            std::printf("# layer %d store list:", l);
            for (const StoreSpec & sp : specs) {
                if (sp.by_set) std::printf(" SET[%d]%s", (int) sp.members.size(),
                                           sp.loc == Store::VRAM ? "VRAM" : "RAM");
                else           std::printf(" [%d,%d)%s", sp.lo, sp.hi,
                                           sp.loc == Store::VRAM ? "VRAM" : "RAM");
            }
            std::printf("\n");
            for (const StoreSpec & sp : specs) {
                // H2/U2a: a RAM-labelled entry is built ZERO-COPY over the mmap (make_store).
                Store * s = make_store(sp, lp->gu_lt, lp->dn_lt, cpu, /*nt*/ 1);
                keep.push_back(s);
            }
            audit_store_list(keep, l, l == 0, &lp->gu_lt, &lp->dn_lt);
        }
        store_lists.push_back(keep);
        L.push_back(std::move(lp));
    }

    // ---------- tier A ----------
    RefDump ref;
    bool have_ref = false;
    if (!a.ref.empty()) {
        have_ref = ref.parse(a.ref.c_str());
        if (!have_ref) { std::printf("FAIL: could not parse the reference dump %s\n", a.ref.c_str()); }
    }

    // ---------- the run ----------
    std::vector<float> x((size_t) H_DIM, 0.0f);
    std::vector<float> cur((size_t) H_DIM), ffn_inp((size_t) H_DIM), shexp((size_t) H_DIM);
    std::vector<float> moe_o((size_t) H_DIM), moe_s((size_t) H_DIM), y((size_t) H_DIM);
    std::vector<float> ffn_out_o((size_t) H_DIM), ffn_out_s((size_t) H_DIM), l_out((size_t) H_DIM);
    std::vector<float> cur_prev((size_t) H_DIM);
    std::vector<int32_t> sel((size_t) N_USED, 0);
    std::vector<float>   wts((size_t) N_USED, 0.0f);

    // per (layer, tensor) worst relative deltas against tier A, and the bitwise counters
    int n_bitwise_moe = 0, n_cmp_moe = 0, n_bitwise_out = 0, n_cmp_out = 0, n_bitwise_tap = 0, n_cmp_tap = 0;
    int n_router_disagree = 0, n_router_cmp = 0;
    int n_fail = 0;
    double worst_rel_moe = 0, worst_rel_shexp = 0, worst_rel_out = 0;
    double worst_rel_inp = 0, worst_rel_lout = 0;
    int worst_rel_moe_at[2] = {-1, -1}, worst_rel_shexp_at[2] = {-1, -1}, worst_rel_out_at[2] = {-1, -1};
    int worst_rel_inp_at[2] = {-1, -1}, worst_rel_lout_at[2] = {-1, -1};

    FILE * dumps = a.dump.empty() ? nullptr : std::fopen(a.dump.c_str(), "w");
    if (dumps) {
        std::fprintf(dumps, "# rel deltas vs tier A: layer pos | ffn_moe_out dsum dsumsq | ffn_shexp dsum dsumsq | ffn_out dsum dsumsq\n");
        std::fprintf(dumps, "# our sums: layer pos | moe sum/sumsq | shexp sum/sumsq | ffn_out sum/sumsq\n");
    }

    for (int p = 0; p < n_pos; ++p) {
        emb.run(cpu, ids[p], x.data());
        for (int l = 0; l < a.layers; ++l) {
            M1Layer & ly = *L[l];
            const int32_t pos32 = p;
            const int64_t kidx  = p;
            const int64_t nkv   = p + 1;
            ly.set_input(x.data(), &pos32, &kidx, &nkv);

            if (a.do_oracle) {
                ggml_backend_graph_compute(cpu, ly.gf_oracle);
                ly.read(ly.g_cur, cur.data());
                ly.read(ly.g_ffn_inp, ffn_inp.data());
                ly.read(ly.g_shexp, shexp.data());
                ly.read(ly.g_moe, moe_o.data());
                ly.read(ly.g_sel, sel.data());
                ly.read(ly.g_wts, wts.data());
                for (size_t i = 0; i < (size_t) H_DIM; ++i) ffn_out_o[i] = moe_o[i] + shexp[i];
                if (a.router_margin) {
                    // plan §Q4 risk 3's near-tie probe: the router is DISCONTINUOUS
                    // (argsort_top_k), so the margin between the 4th and 5th logit bounds how
                    // large a perturbation of this layer's input can be without changing the
                    // selected expert set.
                    std::vector<float> lg((size_t) N_EXPERT);
                    ly.read(ly.g_logits, lg.data());
                    std::vector<int> o((size_t) N_EXPERT);
                    for (int i = 0; i < (int) N_EXPERT; ++i) o[i] = i;
                    std::partial_sort(o.begin(), o.begin() + 6, o.end(),
                                      [&](int x, int y) { return lg[x] > lg[y]; });
                    std::printf("# router margin L%d p%d top6", l, p);
                    for (int i = 0; i < 6; ++i) std::printf(" %d:%.6f", o[i], lg[o[i]]);
                    std::printf("  | 4-5 gap=%.3e  1-2 gap=%.3e\n",
                                (double) (lg[o[3]] - lg[o[4]]), (double) (lg[o[0]] - lg[o[1]]));
                }
            }
            if (a.do_store) {
                ggml_backend_graph_compute(cpu, ly.gf_core);
                // the two cgraphs must produce identical taps: gf_core is a strict subgraph
                // of gf_oracle and is computed by the same kernels with the same inputs.
                ly.read(ly.g_cur, cur_prev.data());
                if (a.do_oracle) {
                    ++n_cmp_tap;
                    if (max_abs_diff(cur.data(), cur_prev.data(), H_DIM) == 0.0) ++n_bitwise_tap;
                    else { std::printf("FAIL: tap ffn_norm'd activation differs between the core and oracle cgraphs (layer %d pos %d)\n", l, p); ++n_fail; }
                } else {
                    cur = cur_prev;
                    ly.read(ly.g_ffn_inp, ffn_inp.data());
                    ly.read(ly.g_shexp, shexp.data());
                    ly.read(ly.g_sel, sel.data());
                    ly.read(ly.g_wts, wts.data());
                }
                std::vector<int32_t> ids4((size_t) N_USED);
                std::vector<float>   w4((size_t) N_USED);
                for (int e = 0; e < (int) N_USED; ++e) { ids4[e] = sel[e]; w4[e] = wts[e]; }

                // the host-router near-tie probe (plan §Q4 risk 3)
                {
                    size_t gn = 0;
                    const float * gate_inp = m->f32_tensor(
                        layer_tensor_name(l, LAYER_TENSOR_SPECS[LW_GATE_INP].suffix).c_str(), &gn);
                    std::vector<int32_t> ids_h; std::vector<float> w_h;
                    m1_host_router(gate_inp, cur.data(), 1, ids_h, w_h, nullptr);
                    ++n_router_cmp;
                    bool agree = true;
                    for (int e = 0; e < (int) N_USED; ++e) if (ids_h[e] != ids4[e]) agree = false;
                    if (!agree) {
                        ++n_router_disagree;
                        std::printf("# router near-tie: layer %d pos %d graph=[%d %d %d %d] host=[%d %d %d %d]\n",
                                    l, p, ids4[0], ids4[1], ids4[2], ids4[3],
                                    ids_h[0], ids_h[1], ids_h[2], ids_h[3]);
                    }
                }

                for (Store * s : store_lists[l]) { s->fill(ids4.data(), 1); s->enqueue(ids4.data(), w4.data(), cur.data(), 1); }
                std::fill(y.begin(), y.end(), 0.0f);
                for (Store * s : store_lists[l]) s->drain(y.data(), 1, false, nullptr, nullptr);
                moe_s = y;
                for (size_t i = 0; i < (size_t) H_DIM; ++i) ffn_out_s[i] = moe_s[i] + shexp[i];

                if (a.do_oracle) {
                    ++n_cmp_moe;
                    if (max_abs_diff(moe_o.data(), moe_s.data(), H_DIM) == 0.0) ++n_bitwise_moe;
                    else { std::printf("FAIL: MoE block differs between the store arm and tier B (layer %d pos %d) max|d|=%.3e\n",
                                       l, p, max_abs_diff(moe_o.data(), moe_s.data(), H_DIM)); ++n_fail; }
                    ++n_cmp_out;
                    if (max_abs_diff(ffn_out_o.data(), ffn_out_s.data(), H_DIM) == 0.0) ++n_bitwise_out;
                    else { std::printf("FAIL: ffn_out differs between the store arm and tier B (layer %d pos %d) max|d|=%.3e\n",
                                       l, p, max_abs_diff(ffn_out_o.data(), ffn_out_s.data(), H_DIM)); ++n_fail; }
                }
            }
            if (!a.do_store) { moe_s = moe_o; ffn_out_s = ffn_out_o; }
            for (size_t i = 0; i < (size_t) H_DIM; ++i) l_out[i] = ffn_out_s[i] + ffn_inp[i];

            // ---- the per-(layer,pos) fingerprints, in the reference's format -----------
            const Sums s_moe   = sum_of(moe_s.data(), H_DIM);
            const Sums s_shexp = sum_of(shexp.data(), H_DIM);
            const Sums s_out   = sum_of(ffn_out_s.data(), H_DIM);
            const Sums s_inp   = sum_of(ffn_inp.data(), H_DIM);   // == the reference's ffn_inp
            const Sums s_lout  = sum_of(l_out.data(), H_DIM);     // == the reference's l_out
            std::printf("L%d p%d sel=[%d %d %d %d]\n", l, p, sel[0], sel[1], sel[2], sel[3]);
            print_sums("ffn_moe_out", s_moe);
            print_sums("ffn_shexp",   s_shexp);
            print_sums("ffn_out",     s_out);
            print_sums("ffn_inp",     s_inp);
            print_sums("l_out",       s_lout);

            if (dumps) {
                std::fprintf(dumps, "OURS %d %d %+.9e %.9e %+.9e %.9e %+.9e %.9e\n",
                             l, p, s_moe.sum, s_moe.sumsq, s_shexp.sum, s_shexp.sumsq, s_out.sum, s_out.sumsq);
                std::fprintf(dumps, "OURS2 %d %d %+.9e %.9e %+.9e %.9e\n",
                             l, p, s_inp.sum, s_inp.sumsq, s_lout.sum, s_lout.sumsq);
                // the attention-path taps (U2b): the same tensors the reference's MLA graph
                // names q/kv_cmpr_pe/Qcur/Kcur, so a divergence can be attributed to the q path,
                // the kv path or the attention core rather than to "the attention".
                if (a.do_oracle) {
                    const ggml_tensor * taps[12] = {
                        ly.g_q, ly.g_q_nope_abs, ly.g_qcur, ly.g_kcur, ly.g_kv_cmpr,
                        ly.g_k_pe, ly.g_kq, ly.g_kqv, ly.g_attn_out, ly.g_kv_cmpr_pe,
                        ly.g_kqv_mla, ly.g_kqv_out,
                    };
                    std::fprintf(dumps, "TAPS %d %d", l, p);
                    for (int t = 0; t < 12; ++t) {
                        std::vector<float> buf((size_t) ggml_nelements(taps[t]));
                        ly.read(taps[t], buf.data());
                        const Sums st = sum_of(buf.data(), buf.size());
                        std::fprintf(dumps, " %+.9e %.9e", st.sum, st.sumsq);
                    }
                    std::fprintf(dumps, "\n");
                }
            }

            // ---- tier A ---------------------------------------------------------------
            if (have_ref) {
                const int step = p - (a.ref_pos0 >= 0 ? a.ref_pos0 : n_prompt);
                if (step >= 0 && step < ref.n_dec_steps) {
                    const RefChecksum & r_moe   = ref.v[step][(size_t) l * RefDump::N_KINDS + 0];
                    const RefChecksum & r_shexp = ref.v[step][(size_t) l * RefDump::N_KINDS + 1];
                    const RefChecksum & r_out   = ref.v[step][(size_t) l * RefDump::N_KINDS + 2];
                    const RefChecksum & r_inp   = ref.v[step][(size_t) l * RefDump::N_KINDS + 3];
                    const RefChecksum & r_lout  = ref.v[step][(size_t) l * RefDump::N_KINDS + 4];
                    auto rel = [](double a, double b) {
                        const double d = std::fabs(a - b);
                        return d / std::max(std::fabs(b), 1e-12);
                    };
                    const double dm = rel(s_moe.sum, r_moe.sum),      dms = rel(s_moe.sumsq, r_moe.sumsq);
                    const double ds = rel(s_shexp.sum, r_shexp.sum),  dss = rel(s_shexp.sumsq, r_shexp.sumsq);
                    const double do_ = rel(s_out.sum, r_out.sum),     dos = rel(s_out.sumsq, r_out.sumsq);
                    std::printf("  REF          ffn_moe_out sum=%+.9e sumsq=%.9e | rel dsum=%.3e dsumsq=%.3e\n",
                                r_moe.sum, r_moe.sumsq, dm, dms);
                    std::printf("  REF          ffn_shexp   sum=%+.9e sumsq=%.9e | rel dsum=%.3e dsumsq=%.3e\n",
                                r_shexp.sum, r_shexp.sumsq, ds, dss);
                    std::printf("  REF          ffn_out     sum=%+.9e sumsq=%.9e | rel dsum=%.3e dsumsq=%.3e\n",
                                r_out.sum, r_out.sumsq, do_, dos);
                    if (dms > worst_rel_moe)   { worst_rel_moe = dms;   worst_rel_moe_at[0] = l; worst_rel_moe_at[1] = p; }
                    if (dss > worst_rel_shexp) { worst_rel_shexp = dss; worst_rel_shexp_at[0] = l; worst_rel_shexp_at[1] = p; }
                    if (dos > worst_rel_out)   { worst_rel_out = dos;   worst_rel_out_at[0] = l; worst_rel_out_at[1] = p; }
                    if (dumps) std::fprintf(dumps, "DELTA %d %d %.6e %.6e %.6e %.6e %.6e %.6e\n",
                                            l, p, dm, dms, ds, dss, do_, dos);

                    // the residual-path taps (U2b): ffn_inp = the FFN input (inpSA + attn_out) and
                    // l_out = the layer output that feeds the next layer.  A dump predating U2b has
                    // neither, in which case these cells are simply not compared.
                    if (r_inp.have && r_lout.have) {
                        const double di = rel(s_inp.sum, r_inp.sum),  dis = rel(s_inp.sumsq, r_inp.sumsq);
                        const double dl = rel(s_lout.sum, r_lout.sum), dls = rel(s_lout.sumsq, r_lout.sumsq);
                        std::printf("  REF          ffn_inp     sum=%+.9e sumsq=%.9e | rel dsum=%.3e dsumsq=%.3e\n",
                                    r_inp.sum, r_inp.sumsq, di, dis);
                        std::printf("  REF          l_out       sum=%+.9e sumsq=%.9e | rel dsum=%.3e dsumsq=%.3e\n",
                                    r_lout.sum, r_lout.sumsq, dl, dls);
                        if (dis > worst_rel_inp)  { worst_rel_inp = dis;  worst_rel_inp_at[0] = l; worst_rel_inp_at[1] = p; }
                        if (dls > worst_rel_lout) { worst_rel_lout = dls; worst_rel_lout_at[0] = l; worst_rel_lout_at[1] = p; }
                        if (dumps) std::fprintf(dumps, "DELTA2 %d %d %.6e %.6e %.6e %.6e\n", l, p, di, dis, dl, dls);
                    }
                }
            }
            x = l_out;
        }
        std::printf("\n");
    }
    if (dumps) std::fclose(dumps);

    // ---------- the verdict -----------------------------------------------------------
    std::printf("=== VERDICT (U2) ===\n");
    if (a.do_oracle && a.do_store) {
        std::printf("RIG CONTROL (engine store arm vs tier B oracle, CPU, bitwise):\n");
        std::printf("  MoE block      : %d/%d bitwise\n", n_bitwise_moe, n_cmp_moe);
        std::printf("  ffn_out        : %d/%d bitwise\n", n_bitwise_out, n_cmp_out);
        std::printf("  core taps      : %d/%d bitwise (gf_core vs gf_oracle)\n", n_bitwise_tap, n_cmp_tap);
        if (n_bitwise_moe != n_cmp_moe || n_bitwise_out != n_cmp_out) {
            std::printf("FAIL: the rig control is NOT bitwise — the store machinery diverges from the "
                        "monolithic op sequence.  This is a RIG failure, not a model result.\n");
            ++n_fail;
        }
    }
    std::printf("ROUTER near-tie probe (host router vs ggml argsort_top_k): %d/%d selections agree\n",
                n_router_cmp - n_router_disagree, n_router_cmp);
    if (have_ref) {
        std::printf("TIER A (vs the reference's own checksums), worst relative sumsq delta:\n");
        if (worst_rel_moe_at[0] >= 0)
            std::printf("  ffn_moe_out : %.6e at layer %d pos %d\n", worst_rel_moe,
                        worst_rel_moe_at[0], worst_rel_moe_at[1]);
        if (worst_rel_shexp_at[0] >= 0)
            std::printf("  ffn_shexp   : %.6e at layer %d pos %d\n", worst_rel_shexp,
                        worst_rel_shexp_at[0], worst_rel_shexp_at[1]);
        if (worst_rel_out_at[0] >= 0)
            std::printf("  ffn_out     : %.6e at layer %d pos %d\n", worst_rel_out,
                        worst_rel_out_at[0], worst_rel_out_at[1]);
        if (worst_rel_inp_at[0] >= 0)
            std::printf("  ffn_inp     : %.6e at layer %d pos %d\n", worst_rel_inp,
                        worst_rel_inp_at[0], worst_rel_inp_at[1]);
        if (worst_rel_lout_at[0] >= 0)
            std::printf("  l_out       : %.6e at layer %d pos %d\n", worst_rel_lout,
                        worst_rel_lout_at[0], worst_rel_lout_at[1]);
    } else {
        std::printf("TIER A: not run (no --ref).  Tier B alone CANNOT certify the attention path.\n");
    }
    std::printf("VERDICT: %s (%d FAIL lines)\n", n_fail ? "FAIL" : "PASS", n_fail);

    for (auto & kv : store_lists) for (Store * s : kv) delete s;
    delete m;
    return n_fail ? 1 : 0;
}

// ---------------------------------------------------------------------------
// --loop: the 36-layer token loop (U3).  ONE prompt per invocation.
// ---------------------------------------------------------------------------
struct LoopArgs {
    std::string model  = DEFAULT_MODEL;
    std::string tokens;          // read_tokens format: "n_prompt <P>" then the ids
    std::string out;             // logits file (writes <out> and <out>.tokens)
    std::string ref;             // the reference's stderr (tier A + the KV width assertion)
    std::string ref_tokens;      // the reference's <logits>.tokens (the token line)
    std::string dump;            // per-(layer, step) delta table
    std::string route_dump;      // U6: per-(layer, pos) trace of the router's OWN ids+weights
    std::string route_perm;      // U6: the treatment's per-layer id permutation (fixture map)
    // H1 — THE RESIDENT SET: the per-layer ORDERED resident sets, read once from
    // --resident-set <path.json>.  EMPTY => the identity interval [0,n_res) per layer, which is
    // today's behaviour exactly (the store list is derived from the interval cut, unchanged).
    std::vector<std::vector<int>> res_hot;
    std::string res_path;
    int    res_n = 0;            // the JSON's own n_res (CHECKed against --n-res when both are given)
    bool   res_stats = false;    // H1: count the routed slots a VRAM store's slot() claims
    int    n_decode = 128;
    int    kv_max   = 256;       // <= 0 => the sequence length (U2's pre-fix width)
    int    threads  = 16;
    int    n_res    = 64;
    bool   do_store = false;     // store arm (the product path); default = the oracle arm
    // U4a — THE SPLIT: with this, a store whose SPEC says VRAM is built on the ROCm0 backend
    // and a RAM store stays on the CPU (M0's mechanism, engine/moe_split.cpp build_stores).
    // WITHOUT it every store is built on the CPU backend — the CPU-only configuration, which is
    // the correctness path (its token line is byte-identical to the reference, U3/U3b) and is
    // left untouched by this flag.
    bool   gpu_store = false;
    // U4c — THE CORE ON ROCm0: the layers' non-expert surface (attention/MLA/router/shexp
    // weights + KV + inputs + intermediates) and the embedding/head graphs are built and run on
    // the same ROCm0 backend the split uses, instead of the CPU.  INDEPENDENT of --gpu-store
    // (core-on-GPU with the stores all-CPU is a meaningful row).  Default OFF: the CPU-only
    // correctness path is untouched.  Requires --arm-store (the expert weights are not resident
    // in the layer buffer in this mode — they live in the stores' slices) and will not be
    // byte-identical to the CPU reference (different kernels; MEASURED and reported, never the
    // token line).
    bool   gpu_core = false;
    // the falsifier
    bool   falsify  = false;
    int    f_layers = 1;
    int    iters    = 5;
    int    warmup   = 1;
    bool   serialize_both = false;
    double thresh   = 0.950;
    // U7 — the per-position split (timing only): --instr turns on the named component counters
    // in the store drive; --extra-stores N appends N EMPTY list entries per layer, an in-situ
    // probe for the per-store-cycle overhead (numerics-neutral: an empty store enqueues no
    // graph and drains no pending).
    bool   instr = false;
    int    extra_stores = 0;
    // U7: the paired A/B arm for the output-D2H move (see engine/store.h's g_store_pull_at_drain)
    bool   store_pull_at_drain = false;
    // H2/U1 — THE FOOTPRINT PROBE (accounting only; behaviour-neutral, one print after the build).
    // Every store allocates its OWN tensors (store.h `ggml_backend_alloc_ctx_tensors`) and
    // upload_slice() memcpys the weights in, so the RAM store owns a PACKED COPY of its
    // complement — host bytes that duplicate what the GGUF mmap already carries.  This prints the
    // bytes each LOCATION class actually allocated, from the ggml buffers themselves, split
    // VRAM-labelled vs RAM-labelled, plus the process's own VmRSS/RssAnon/RssFile so the
    // anonymous share can be read off directly.  No numerics, no timing, no code path change.
    bool   footprint = false;
    // U2b: --alias-prefault (default OFF) hoists the zero-copy RAM store's file-page faults into
    // the startup slot (store.h prefault_alias_slice).  A mechanism switch, not a numerics switch:
    // the partition, the slot mapping and the combine order are untouched.
    bool   alias_prefault = false;
    // U2b — 2Q ROTATION (default OFF: with it off the engine is byte-identical to the shipped one,
    // which is GATE 3a).  --rotate-verify adds the post-loop BYTE check of every resident slot.
    bool   rotate = false;
    bool   rotate_verify = false;
    // U10 — the refresh audit's scope.  DEFAULT (false) checks what the refresh wrote (the admitted
    // expert, the evicted one, their tables in both stores); --audit-full restores L50's full
    // O(N_EXPERT) walk on the hot path.  The full walk runs once after the loop in BOTH cases.
    bool   audit_full = true;
    bool   audit_scope_set = false;   // U10: was the scope named EXPLICITLY (for the refusal above)
    // U10 — the 2Q window's threshold: the k-th reference while non-resident admits.  DEFAULT 4,
    // not the shipped 2: the frontier sweep (results/m1-h3-u10.md) measured k=4 at 6.98 uploads/token
    // and served 0.6797 against 2Q's 19.32 and 0.6771 — the same coverage for a third of the bytes,
    // and 4-8 ms/pos less loop wall.  `--rot-admit-k 2` restores the shipped policy bit for bit.
    int    rot_admit_k = 4;
    bool   rot_admit_k_set = false;
    // U8 — THE RESIDENT FOLD (§10.2 item 3): fold the RESIDENT slice's MoE into the core graph
    // so the per-layer VRAM-store sync boundary disappears.  Gated to --gpu-core + --gpu-store
    // (the core and the resident store must be on the SAME backend) and to FLAT expert tensors.
    bool   fold_resident = false;
    // U9 — the fold's slot-count A/B (TIMING ONLY): 0..4 overrides the resident block's slot
    // count (0 = no block, the "empty" variant; <4 drops top-k slots, so its numerics are WRONG
    // by construction).  -1 (default) keeps the landed 4-slot fold.  Sizes the amplification
    // bounded without touching the product numerics.
    int    fold_slots = -1;
    // H3/U1 — THE CORE PROBE (timing only, default OFF): after the loop has produced every number
    // the run is judged on, measure the core graph on an idle stream under several launch/sync
    // structures and at several nt values, so the core bracket can be decomposed into named
    // components (launch, per-layer wait, tap enqueue, GPU exec of each phase).  It reads the
    // layer the engine builds and changes nothing on the default path.
    bool   core_probe = false;
    int    probe_reps = 20;
    // H3/U2 — the interleaved-idle probe (L54 MAJOR-2).  The product runs the core interleaved
    // with ~440 us/layer of CPU-side store work, so the card sits idle-then-bursty between core
    // launches; a probe that launches back to back measures full boost and overstates what the
    // fusion would bank in situ.  --probe-gap-us inserts that host-busy idle between launches.
    int    probe_gap_us = 440;
    // H3/U2 — the attention chain's expression (see M1Layer::build): 0 = the landed reference
    // expansion, 1 = the same math without the redundant head-recombination copy.  -1 (default)
    // = 1 on the GPU-core path, i.e. the measured winner SHIPS: the two expressions are
    // byte-identical on the product invocation (results/m1-h3-u2.md §6) and the fused one is
    // faster in every paired row.  --core-fuse 0 is the A/B arm and keeps the reference
    // expansion; the CPU-only path is always 0 (it is the numerics oracle).
    int    core_fuse = -1;
    // H3/U2 — the per-node census print (--instr only): every node of gf_core, classified.
    bool   core_nodes = false;
    // H3/U4 — the wk_b custom-op hatch (default OFF): replace the MLA-absorption mul_mat with
    // the specialized GGML_OP_M4_WKB kernel inside the captured graph.  GPU-core only; the
    // CPU-only arm (the gate's numerics oracle) keeps the reference mul_mat by construction.
    int    m4_wkb = -1;      // H3/U4 + L58/L59 — TRI-STATE: -1 = AUTO (the wk_b custom-op is ON wherever
                             // it applies, i.e. --gpu-core; quality EQUIVALENT at delta=0.01 over 39,984
                             // paired positions, core -18.7 us/layer), 0 = --no-m4-wkb (the stock MMVQ
                             // path, byte-identical to pre-H3-U4), 1 = --m4-wkb (explicit; needs --gpu-core).
    int    m4_kqkv = -1;     // H3/U5 — TRI-STATE, same shape as m4_wkb: -1 = AUTO (the fused attention
                             // core is ON wherever it applies, i.e. --gpu-core), 0 = --no-m4-kqkv (the
                             // reference kq/softmax/kqv sequence), 1 = --m4-kqkv (explicit; needs --gpu-core).
    // H3/U11 — the two remaining attention phases, TRI-STATE with the same shape as m4_wkb /
    // m4_kqkv: -1 = AUTO (ON wherever it applies, i.e. --gpu-core), 0 = --no-<x> (the reference
    // op sequence, byte-identical to the pre-H3-U11 binary), 1 = explicit (needs --gpu-core).
    int    m4_kvwrite = -1;  // kvwrite: wkv_a gemv + kv_a_norm + k_pe rope + the cache scatter
    int    m4_wvb     = -1;  // wvb: the MLA value absorption gemv (`kqv_mla`)
    // ---- M1/Q1 — THE QUALITY INSTRUMENT's two engine-side surfaces (both default OFF) -----
    // --score <tokens file>: SCORING mode.  Prefill the file's tokens and then feed the file's
    // OWN next token instead of the argmax (teacher forcing), dumping ONE logits row per
    // position that has a target (<out>.tokens then carries the TARGET ids, not generated
    // ones).  Same forward path, same kernels, same flags as a decode run — a score run
    // measures exactly the computation the product does.  The decode loop's position mapping
    // (ref_pos0 = n_prompt-1) does NOT apply here; --ref/--ref-tokens are refused.
    std::string score_tokens;
    // --damage <tensor suffix> <eps>: CALIBRATION ONLY — scale one named F32 layer tensor by
    // (1+eps) as it is uploaded (see m1_set_damage).  Empty name => off.  The dose ladder in
    // results/q1.md is the instrument's proof that it CAN see a real degradation.
    std::string damage_tensor;
    double      damage_eps = 0.0;
};

// U4b — the serial-tail decomposition of the token loop (plan §Q6 open question 5): where one
// token's time goes.  Timing ONLY — the op order and the numerics are untouched, so the
// correctness path is unchanged (the gate's byte-identity control re-proves it).  "stage" is
// everything after the head on a sampled position: argmax, row copy, the sums, the prints.
static double t_emb_acc = 0, t_core_acc = 0, t_store_acc = 0, t_head_acc = 0, t_tail_acc = 0;
static double t_setin_acc = 0, t_reads_acc = 0;
// H3/U1 — inside the core bracket: the HOST side of one layer (the graph launch) and the tap
// enqueues, both of which the bracket has always included without naming them.  The remainder of
// the bracket is the event wait.  --instr only (see LoopLayer::forward).
static double t_core_launch_acc = 0, t_core_taps_acc = 0;
// ...and the same three again over the STEADY-STATE positions (>= 3): the backend captures a CUDA
// graph per layer on its first calls, and that capture cost lands in whichever bracket issues it.
// Quoting the launch cost from position 0 would price the capture as if it were the launch.
static double t_core_launch_ss = 0, t_core_taps_ss = 0, t_core_acc_ss = 0;
static long   n_core_ss = 0;

// U7 — THE PER-POSITION SPLIT (plan §10.2 item 1).  The drive bracket (t_store_acc) is one
// number; the U6 review's 3-term fit puts a PER-POSITION term of 5.81 ms/pos in it and its
// attribution is not established (the control rows are collinear in (GiB, positions), so a
// bytes-only slope absorbs the per-position work).  These counters split the bracket into
// NAMED components, keyed by the store's OWN backend class (RAM = the CPU store, which
// enqueues BLOCKING — its enqueue IS the MoE compute and the bytes — vs VRAM = the HIP store,
// whose enqueue is a launch and whose work is waited for at the sync).  Timing only, gated on
// --instr so the falsifier/correctness paths pay one branch and nothing else.
static bool   g_instr = false;
static bool   g_core_nodes = false;   // H3/U2 --core-nodes: print the per-node classification
static bool   g_res_stats = false;   // H1 --resident-stats: what the residency actually served
static bool   g_rotate_verify = false;  // U2b --rotate-verify: read back each swapped slot and memcmp
static double t_fill_acc = 0, t_enq_ram_acc = 0, t_enq_vram_acc = 0;
static double t_sync_ram_acc = 0, t_sync_vram_acc = 0;
static double t_drain_ram_acc = 0, t_drain_vram_acc = 0;
static double t_combine_acc = 0;          // the y += inside Store::drain (store.h, STORE_INSTR)
static long   n_store_cycles = 0, n_store_active = 0, n_store_empty = 0;
static long   n_store_groups = 0;         // pending groups seen (the graphs actually computed)
// U9 — THE IN-SITU SHAPE OF THE RAM STORE, so the isolated rig's t(c) can be applied to the product
// instead of to a modelled c.  Per store: how many graphs, how many experts they carried, and the
// histogram of c (a graph's expert count).  Timing-neutral (a loop over a vector already built).
static long   n_ram_graphs = 0, n_ram_experts = 0, n_ram_hist[N_USED + 1] = {};
static long   n_vram_graphs = 0, n_vram_experts = 0, n_vram_hist[N_USED + 1] = {};
// U6 — the routing trace (timing-neutral unless the flag is passed, and then identical in
// control and treatment so the A/B stays clean) and the treatment's id permutation.  The
// permutation maps the router's ORIGINAL expert id to the PERMUTED fixture id; the fixture
// carries the hot experts in [0, S_l) per layer, so the store list's fixed [0,n_res) cut
// keeps its meaning and its VRAM size while the COMPOSITION of the resident set changes.
static FILE * g_rdump = nullptr;
static const int32_t * g_perm = nullptr;   // 36 x 128, row l = new fixture id of expert e
// U4c — the GPU-core staging cost (timing only): the per-layer input H2D (set_input) and the
// per-layer result D2H (the five graph taps the store arm reads back).  On the CPU backend these
// are memcpy-scale; on ROCm0 each blocking transfer carries its sync latency, and the loop's
// wall clock exceeds the sum of the staged components by exactly these — measured, not assumed.
static int    n_head_calls = 0;

// U10 — ROTATION'S OWN COST, WHERE IT IS SPENT.  L69 named the loop's largest unnamed term: the
// residual (wall − named) of 9.67 ms/pos, which contains everything the store drive's bracket and
// the core bracket do NOT cover.  rotation's refresh and policy both run OUTSIDE every existing
// bracket (rot_apply_pending at the top of forward, rot_policy after the taps), so they land there
// unmeasured.  These counters split that part into named pieces:
//   upload : Store::replace_slot — the two ggml_backend_tensor_set_async H2D calls, their CHECKs
//            and the three table updates.  HOST-SIDE only: the copy is async, so the transfer's
//            own time lands wherever its stream is next waited for (drain_all's sync_vram), NOT
//            here.  The two are therefore reported separately and neither is assumed.
//   alias  : the two Store::set_member calls on the zero-copy store (a predicate flip, no bytes).
//   verify : Store::verify_slot — only when --rotate-verify is on (synchronous readback per swap).
//   audit  : the post-refresh structural audit (full walk, or U10's incremental one).
//   policy : rot_policy — the 2Q decision over N_EXPERT ghosts and n_e stamps.
//   apply_x: rot_apply_pending minus the four terms above (its own loop and clear).
// Timing only, every bracket gated on --instr: with it off the loop executes the SAME calls in the
// SAME order and the binary is byte-identical to the shipped one (proved by identity.sh).
static double t_rot_upload_acc = 0, t_rot_alias_acc = 0, t_rot_verify_acc = 0;
static double t_rot_audit_acc = 0, t_rot_policy_acc = 0, t_rot_apply_acc = 0;
static long   n_rot_swap = 0;        // replace_slot calls == refreshes == audits
static long   n_rot_apply = 0;       // rot_apply_pending calls that had work
// U10 — WHICH AUDIT THE REFRESH RUNS.  DEFAULT = L50's FULL walk, because U10 measured it at
// 0.058 ms/pos (0.6 % of the 9.5 ms/pos residual) and a gate is not narrowed for 0.05 ms: the
// measured benefit does not clear the unit's own 1 ms/token integration bar.  --audit-incremental
// selects the refresh-scoped check U10 built and PROVED equivalent under mutation (13x cheaper per
// swap: 0.22 us against 2.98); either way the full walk now runs ONCE after the loop, which the
// pre-unit code did NOT do — so this unit's net effect on the gate is a strengthening at both
// settings, and a measured price tag on the cheaper arm.
static bool   g_audit_full = true;
// U10 — the 2Q window's threshold (--rot-admit-k): the k-th reference while non-resident admits.
// DEFAULT 2 = exactly the bit-window the shipped policy has (proved by identity.sh), 1 = admit on
// every first miss, 3/4 = admit only habitual misses.  The coverage-vs-uploads frontier is traced
// by moving THIS and nothing else.
static int    g_rot_admit_k = 4;
static long   n_rot_audit_inc = 0, n_rot_audit_full = 0;

// one layer of the loop: its forward + the per-token buffers it hands back
struct LoopLayer {
    M1Layer * ly = nullptr;
    ggml_backend_t be_core = nullptr;          // U4c: non-null => the core graph runs here (ROCm0)
    std::vector<Store *> stores;               // empty => ARM_MOE_ORACLE (the transcription)
    std::vector<float> cur, ffn_inp, shexp, moe, moe_res, fout, y, l_out;
    std::vector<int32_t> sel;
    std::vector<float>   wts;
    // H1 — --resident-stats: routed slots and the subset a VRAM store's slot() claims (see forward).
    long routed_n = 0, served_n = 0;

    // H2/U2b — 2Q ROTATION STATE (all host-side; nothing here exists when --rotate is off, and the
    // run is then byte-identical to the shipped binary — GATE 3a).
    //
    // THE POLICY (2Q admission control, minimal): a per-layer GHOST WINDOW — one bit per expert,
    // set on an expert's first reference while it is non-resident, cleared on the second — and a
    // CLOCK/LRU order over the resident slots (`stamp[s]` = the position slot s was last used).
    // A second reference in the window ADMITS the expert; the victim is the least-recently-used
    // slot that THIS position does not route to.  That is the whole mechanism: a 128-bit window and
    // 48 stamps is the entire cache.  LRU's own (larger) traffic is what the plan priced at 21 ms
    // against a 12.9 ms window; admission control is the only form whose bill fits (L47).
    struct RotState {
        bool on = false;
        Store * vram = nullptr, * ram = nullptr;
        // U10: the ghost window's THRESHOLD, `--rot-admit-k` (default 2 = 2Q = today's policy
        // exactly).  K=1 admits on the first reference (LRU-like: most uploads, most coverage),
        // K=3/4 admit only habitual references (fewer uploads, less coverage).  It is the knob the
        // coverage-vs-uploads frontier is traced with: same mechanism, one constant moved.
        int admit_k = 2;
        std::vector<unsigned char> ghost;        // N_EXPERT: references since it was last resident
        std::vector<long> stamp;                 // per slot: the position it was last served at
        std::vector<int>  cand;                  // this position's admission candidates
        std::vector<std::pair<int,int>> pending; // (slot, expert) decided last position, applied now
        long pos = 0, admitted = 0, evicted = 0, refreshes = 0, audits = 0, vbad = 0;
    } rot;

    void init(M1Layer * l) {
        ly = l;
        cur.assign(H_DIM, 0.0f);     ffn_inp.assign(H_DIM, 0.0f); shexp.assign(H_DIM, 0.0f);
        moe.assign(H_DIM, 0.0f);     moe_res.assign(H_DIM, 0.0f); fout.assign(H_DIM, 0.0f);
        y.assign(H_DIM, 0.0f);
        l_out.assign(H_DIM, 0.0f);
        sel.assign(N_USED, 0);       wts.assign(N_USED, 0.0f);
    }
    bool store_arm() const { return !stores.empty(); }

    void rot_setup(Store * vram, Store * ram) {
        rot.on = true; rot.vram = vram; rot.ram = ram;
        rot.ghost.assign((size_t) N_EXPERT, 0);
        rot.admit_k = g_rot_admit_k;
        rot.stamp.assign((size_t) vram->n_e, 0);
        // the initial order is the SET's own order (slot 0 first to go), so the bootstrap does not
        // invent an order the resident set file did not express
        for (int s = 0; s < vram->n_e; ++s) rot.stamp[(size_t) s] = s - vram->n_e;
    }

    // THE REFRESH, and WHERE it happens is the correctness-critical choice.  It is called at the TOP
    // of forward(), i.e. (a) after this layer's PREVIOUS position completed its drain_all (which
    // synchronized this store's backend, so the stream is IDLE and the async row copies cannot
    // interleave with in-flight graph work), and (b) before THIS position's enqueue, so the block
    // that is about to run sees the new membership.  It is never called between an enqueue and its
    // drain.  The routed ids for position t are only known after the layer's core has run, so the
    // declaration made at t is applied at t+1 — the one-position lag is the price of not
    // re-dispatching a layer's block, and it costs the policy exactly one position of staleness.
    void rot_apply_pending() {
        if (!rot.on || rot.pending.empty()) return;
        // U10: the whole refresh, bracketed (--instr only; with it off this is today's code path
        // exactly, which identity.sh proves byte-identical).
        const double ta0 = g_instr ? now_s() : 0.0;
        ++n_rot_apply;
        for (const auto & pr : rot.pending) {
            const int slot = pr.first, e = pr.second;
            // U10 SPLIT — the upload (host side of replace_slot: two async H2D row copies + the
            // three table updates + their CHECKs).
            const double tu0 = g_instr ? now_s() : 0.0;
            const int old = rot.vram->replace_slot(slot, e);
            if (g_instr) t_rot_upload_acc += now_s() - tu0;
            // the two stores are COMPLEMENTS, so the swap is mirrored: the admitted expert leaves
            // the alias, the evicted one joins it.  The alias moves no bytes — it is a predicate.
            const double tm0 = g_instr ? now_s() : 0.0;
            rot.ram->set_member(e, false);
            rot.ram->set_member(old, true);
            if (g_instr) t_rot_alias_acc += now_s() - tm0;
            if (g_rotate_verify) {
                const double tv0 = g_instr ? now_s() : 0.0;
                rot.vbad += rot.vram->verify_slot(slot);
                if (g_instr) t_rot_verify_acc += now_s() - tv0;
            }
            ++rot.admitted; ++rot.evicted; ++rot.refreshes;
            // THE STANDING GATE: re-audit the partition and BOTH stores' tables after EVERY
            // refresh, on the running list.  U10: the DEFAULT check is now scoped to what the
            // refresh wrote (audit_refresh_swap: the admitted expert, the evicted one, and their
            // tables in both stores) and --audit-full restores the full O(N_EXPERT) walk; the full
            // walk still runs once after the loop either way (audit_store_list, below).
            const double td0 = g_instr ? now_s() : 0.0;
            if (g_audit_full) {
                audit_store_list(stores, ly->layer, false, &ly->gu_lt, &ly->dn_lt);
                ++n_rot_audit_full;
            } else {
                audit_refresh_swap(rot.vram, rot.ram, ly->layer, slot, e, old, &ly->gu_lt, &ly->dn_lt);
                ++n_rot_audit_inc;
            }
            if (g_instr) t_rot_audit_acc += now_s() - td0;
            ++rot.audits;
            ++n_rot_swap;
        }
        rot.pending.clear();
        if (g_instr) t_rot_apply_acc += now_s() - ta0;
    }

    // the admission policy, run on the router's OWN ids (sel[]) after the layer's core has produced
    // them.  It only DECIDES; the move happens at the next position's rot_apply_pending().
    void rot_policy() {
        if (!rot.on) return;
        ++rot.pos;
        const int ns = rot.vram->n_e;
        bool ref[N_EXPERT] = {};
        for (int i = 0; i < (int) N_USED; ++i) {
            const int e = sel[i];
            CHECK(e >= 0 && e < (int) N_EXPERT, "rotate: router id %d out of range", e);
            ref[e] = true;
            const int s = rot.vram->slot(ly->layer, e);
            if (s >= 0) { rot.stamp[(size_t) s] = rot.pos; rot.ghost[(size_t) e] = 0; continue; }
            // U10: `--rot-admit-k` generalises the 2Q rule to "the k-th reference while
            // non-resident admits".  K=2 reproduces the bit-window exactly (record on the first,
            // admit on the second, clear on a resident reference).
            if ((int) rot.ghost[(size_t) e] + 1 >= rot.admit_k) {
                rot.ghost[(size_t) e] = 0;
                rot.cand.push_back(e);
            } else {
                ++rot.ghost[(size_t) e];                      // record the reference and wait
            }
        }
        for (int e : rot.cand) {
            int best = -1; long bs = 0;
            for (int s = 0; s < ns; ++s) {
                if (ref[rot.vram->members[(size_t) s]]) continue;   // never evict this position's routes
                if (best < 0 || rot.stamp[(size_t) s] < bs) { best = s; bs = rot.stamp[(size_t) s]; }
            }
            if (best < 0) break;
            rot.pending.push_back({ best, e });
            rot.stamp[(size_t) best] = rot.pos;   // reserve, so two admissions cannot pick one slot
            ref[e] = true;
        }
        rot.cand.clear();
    }

    void enqueue_all(ggml_backend_t cpu) {
        for (Store * s : stores) {
            if (!g_instr) { s->fill(sel.data(), 1); s->enqueue(sel.data(), wts.data(), cur.data(), 1); continue; }
            const double t0 = now_s();
            s->fill(sel.data(), 1);
            const double t1 = now_s();
            s->enqueue(sel.data(), wts.data(), cur.data(), 1);
            const double t2 = now_s();
            t_fill_acc += t1 - t0;
            // the RAM (CPU) store's enqueue BLOCKS: it is the MoE compute and the weight bytes.
            // The VRAM store's is a launch; its work is waited for at the sync below.
            if (s->backend == cpu) t_enq_ram_acc  += t2 - t1;
            else                   t_enq_vram_acc += t2 - t1;
            ++n_store_cycles;
            if (s->n_e > 0) ++n_store_active; else ++n_store_empty;
            n_store_groups  += (long) s->pending.size();
            // U9: the group shape, per store (the RAM store's groups ARE the rig's c-graphs)
            if (s->backend == cpu) {
                n_ram_graphs += (long) s->pending.size();
                for (const auto & pg : s->pending) {
                    n_ram_experts += pg.c;
                    if (pg.c >= 0 && pg.c <= (int) N_USED) ++n_ram_hist[pg.c];
                }
            } else {
                n_vram_graphs += (long) s->pending.size();
                for (const auto & pg : s->pending) {
                    n_vram_experts += pg.c;
                    if (pg.c >= 0 && pg.c <= (int) N_USED) ++n_vram_hist[pg.c];
                }
            }
        }
    }
    void drain_all(ggml_backend_t cpu) {
        std::fill(y.begin(), y.end(), 0.0f);
        // U5 (the batched drain): one synchronize per DISTINCT backend for the whole list,
        // then every store drains without its own sync.  The old shape — Store::drain's
        // ggml_backend_synchronize before each list entry — paid two full stream waits per
        // VRAM store (the explicit sync, then the sync inside tensor_get), and the M0 review
        // had already flagged the per-store drain as dead weight for an inert store.
        // Semantics: tensor_get is stream-ordered after everything enqueued on that backend,
        // so one wait before the gets covers all of them.
        ggml_backend_t bes[4] = { nullptr, nullptr, nullptr, nullptr };
        int n_be = 0;
        for (Store * s : stores) {
            bool seen = false;
            for (int i = 0; i < n_be; ++i) seen |= (bes[i] == s->backend);
            if (!seen && n_be < 4) bes[n_be++] = s->backend;
        }
        for (int i = 0; i < n_be; ++i) {
            if (!g_instr) { ggml_backend_synchronize(bes[i]); continue; }
            const double t0 = now_s();
            ggml_backend_synchronize(bes[i]);
            const double t1 = now_s();
            if (bes[i] == cpu) t_sync_ram_acc  += t1 - t0;
            else               t_sync_vram_acc += t1 - t0;
        }
        for (Store * s : stores) {
            if (!g_instr) { s->drain(y.data(), 1, false, nullptr, nullptr, /*sync_backend*/ false); continue; }
            const double t0 = now_s();
            s->drain(y.data(), 1, false, nullptr, nullptr, /*sync_backend*/ false);
            const double t1 = now_s();
            if (s->backend == cpu) t_drain_ram_acc  += t1 - t0;
            else                   t_drain_vram_acc += t1 - t0;
        }
    }
    void enqueue_one(Store * s) { s->fill(sel.data(), 1); s->enqueue(sel.data(), wts.data(), cur.data(), 1); }
    void drain_one(Store * s)   { s->drain(y.data(), 1, false, nullptr, nullptr); }

    // one token at `pos` with `nkv` live cells; x = the layer input, l_out = the layer output
    // (= the next layer's input).  The two arms differ ONLY in the expert block, exactly as in
    // engine/m1_layer.cpp: the oracle arm runs the monolithic ggml_mul_mat_id sequence, the
    // store arm runs the product store list (M0's machinery) over the same router output.
    void forward(ggml_backend_t cpu, int32_t pos, int64_t nkv, const float * x) {
        const int64_t kidx = pos;
        // U2b: the layer's pending admissions land HERE — the stream is idle (last position's
        // drain_all synchronized this backend) and this position's enqueue has not been issued.
        if (rot.on) rot_apply_pending();
        const double tsi0 = now_s();
        // U5: with the core on ROCm0 the inputs stage through pinned buffers and queue as
        // async H2D on the backend's stream (one stream wait per layer happens in read_taps);
        // the CPU path keeps the exact pre-U5 blocking calls (byte-identity by construction).
        if (be_core && ly->hstage) ly->set_input_async(x, &pos, &kidx, &nkv, be_core);
        else                       ly->set_input(x, &pos, &kidx, &nkv);
        t_setin_acc += now_s() - tsi0;
        ggml_backend_t be = be_core ? be_core : cpu;
        if (!store_arm()) {
            const double tc0 = now_s();
            ggml_backend_graph_compute(be, ly->gf_oracle);
            t_core_acc += now_s() - tc0;
            ly->read(ly->g_ffn_inp, ffn_inp.data());
            ly->read(ly->g_shexp, shexp.data());
            ly->read(ly->g_moe, moe.data());
        } else {
            const double tc0 = now_s();
            // async queue for ROCm0 (the stream is idle here; identical semantics to the
            // blocking compute, whose sync happens at the taps read below)
            if (be_core) {
                ggml_backend_graph_compute_async(be, ly->gf_core);
                // the timer split: the event marks the graph's end, so the wait below is the
                // CORE's execution exactly as the old blocking compute measured it; the tap
                // copies queue past the event and are waited for in the staging bracket.
                if (ly->ev_core) ggml_backend_event_record(ly->ev_core, be);
                // H3/U1 — the bracket's own split: everything up to the event wait is HOST work
                // (the graph launch + the six tap enqueues); the wait is where the process sits
                // until the GPU reaches the event.  --instr only: two extra clock reads per
                // layer, no call reordered (the launches and the wait keep today's order).
                const double tl = g_instr ? now_s() : 0.0;
                ly->queue_taps(be);
                const double tq = g_instr ? now_s() : 0.0;
                if (ly->ev_core) ggml_backend_event_synchronize(ly->ev_core);
                // H3/U4 debug: dump layer-0 pos-0 g_q_nope_abs (flag-ON vs flag-OFF comparison)
                if (getenv("M4_DUMP_QNA") && ly->layer == 0 && pos == 0) {
                    static bool dumped = false;
                    if (!dumped) {
                        dumped = true;
                        std::vector<float> q(256 * 32);
                        ly->read(ly->g_q_nope_abs, q.data());
                        FILE * f = std::fopen(getenv("M4_DUMP_QNA"), "w");
                        for (int i = 0; i < 256 * 32; ++i) std::fprintf(f, "%.9g\n", q[i]);
                        std::fclose(f);
                    }
                }
                // H3/U5 debug: dump the attention output (g_attn_out [4096]) per layer per
                // position as raw f32, for the per-position correctness report (masking/length
                // check).  Binary: u32 pos, u32 layer, then 4096 f32 — one record per (pos,layer).
                if (const char * p = getenv("M4_DUMP_ATTN"); p && *p) {
                    static FILE * fa = nullptr;
                    if (!fa) fa = std::fopen(p, "wb");
                    const uint32_t u32pos = (uint32_t) pos, u32l = (uint32_t) ly->layer;
                    // g_attn_out [4096]
                    std::vector<float> ao((size_t) ggml_nelements(ly->g_attn_out));
                    ly->read(ly->g_attn_out, ao.data());
                    std::fwrite(&u32pos, 4, 1, fa); std::fwrite(&u32l, 4, 1, fa);
                    std::fwrite(ao.data(), 4, ao.size(), fa);
                    // g_kqv [256*32] (null in the flag-OFF arm: no softmax/intermediate)
                    if (ly->g_kqv) {
                        std::vector<float> kq((size_t) ggml_nelements(ly->g_kqv));
                        ly->read(ly->g_kqv, kq.data());
                        std::fwrite(&u32pos, 4, 1, fa); std::fwrite(&u32l, 4, 1, fa);
                        std::fwrite(kq.data(), 4, kq.size(), fa);
                    }
                    // g_cur [4096] (the layer output, feeds the next layer and the router)
                    std::vector<float> cur((size_t) ggml_nelements(ly->g_cur));
                    ly->read(ly->g_cur, cur.data());
                    std::fwrite(&u32pos, 4, 1, fa); std::fwrite(&u32l, 4, 1, fa);
                    std::fwrite(cur.data(), 4, cur.size(), fa);
                    // g_ffn_inp [4096] (attn_out + residual) and g_x [4096] (the layer input)
                    std::vector<float> fi((size_t) ggml_nelements(ly->g_ffn_inp));
                    ly->read(ly->g_ffn_inp, fi.data());
                    std::fwrite(&u32pos, 4, 1, fa); std::fwrite(&u32l, 4, 1, fa);
                    std::fwrite(fi.data(), 4, fi.size(), fa);
                    std::vector<float> gx((size_t) ggml_nelements(ly->g_x));
                    ly->read(ly->g_x, gx.data());
                    std::fwrite(&u32pos, 4, 1, fa); std::fwrite(&u32l, 4, 1, fa);
                    std::fwrite(gx.data(), 4, gx.size(), fa);
                }
                // H3/U11 debug: dump each fused PHASE's own output tensor per (pos, layer).
                // The brief's correctness check is at the position level on the phase's output
                // (not only the logits), so a mis-scattered KV row or a truncated gemv cannot
                // hide behind a plausible logits drift.  Binary records: u32 pos, u32 layer,
                // u32 n, then n f32 — g_kcur [320] (kvwrite) and g_kqv_mla [128*32] (wvb); both
                // tensors exist in both arms, so the record stream is symmetric.
                if (const char * pp = getenv("M4_DUMP_PHASE"); pp && *pp) {
                    static FILE * fp = nullptr;
                    if (!fp) fp = std::fopen(pp, "wb");
                    const uint32_t u32pos = (uint32_t) pos, u32l = (uint32_t) ly->layer;
                    ggml_tensor * ts[2] = { ly->g_kcur, ly->g_kqv_mla };
                    for (int t = 0; t < 2; ++t) {
                        const uint32_t n = (uint32_t) ggml_nelements(ts[t]);
                        std::vector<float> b((size_t) n);
                        ly->read(ts[t], b.data());
                        std::fwrite(&u32pos, 4, 1, fp); std::fwrite(&u32l, 4, 1, fp);
                        std::fwrite(&n, 4, 1, fp); std::fwrite(b.data(), 4, b.size(), fp);
                    }
                }
                if (g_instr) {
                    t_core_launch_acc += tl - tc0; t_core_taps_acc += tq - tl;
                    if (pos >= 3) { t_core_launch_ss += tl - tc0; t_core_taps_ss += tq - tl; }
                }
            } else {
                ggml_backend_graph_compute(be, ly->gf_core);
            }
            t_core_acc += now_s() - tc0;
            if (g_instr && pos >= 3) { t_core_acc_ss += now_s() - tc0; ++n_core_ss; }
            const double trd0 = now_s();
            if (be_core && ly->hstage) {
                // U8: in fold mode moe_res (the resident MoE, now in-core) is a sixth tap; in
                // the non-fold gpu_core path g_moe_res is null and the buffer stays unused.
                ly->taps_out(cur.data(), ffn_inp.data(), shexp.data(), moe_res.data(),
                             sel.data(), wts.data(), be);
            } else {
                ly->read(ly->g_cur, cur.data());
                ly->read(ly->g_ffn_inp, ffn_inp.data());
                ly->read(ly->g_shexp, shexp.data());
                ly->read(ly->g_sel, sel.data());
                ly->read(ly->g_wts, wts.data());
            }
            t_reads_acc += now_s() - trd0;
            // U6 — the routing trace FIRST (the router's OWN output, PRE-permutation: this is
            // the identity instrument between the arms), then the treatment's permutation.
            // Both are pointer-guarded, so the flags-off path (every speed row without them,
            // and the whole CPU-only correctness path) is unchanged.
            if (g_rdump || g_perm) {
                const int32_t * P = g_perm ? g_perm + (size_t) ly->layer * (size_t) N_EXPERT : nullptr;
                if (g_rdump)
                    std::fprintf(g_rdump, "R %d %d %d %d %d %d %.9e %.9e %.9e %.9e\n",
                                 ly->layer, (int) pos, sel[0], sel[1], sel[2], sel[3],
                                 wts[0], wts[1], wts[2], wts[3]);
                if (P) {
                    for (int i = 0; i < (int) N_USED; ++i) {
                        CHECK(sel[i] >= 0 && sel[i] < (int32_t) N_EXPERT,
                              "router id %d out of range at layer %d", (int) sel[i], ly->layer);
                        sel[i] = P[sel[i]];
                    }
                }
            }
            // U2b — THE ROTATION DECISION, on the router's own ids, before this position's enqueue
            // (it only decides; the move is applied at the NEXT position, see rot_apply_pending).
            // U10: bracketed, because it also lands in the loop's residual (--instr only).
            if (rot.on) {
                const double tp0 = g_instr ? now_s() : 0.0;
                rot_policy();
                if (g_instr) t_rot_policy_acc += now_s() - tp0;
            }
            // H1 — THE SERVED-SLOT COUNTER (--resident-stats; timing-neutral, one branch per layer).
            // Of THIS layer's routed slots, how many are in the store list's VRAM entry — i.e. what
            // the residency ACTUALLY served.  This is the ENGINE's own answer, computed on the
            // deciding path (the same sel[] the stores are handed, after any permutation), to be
            // checked against the analysis tool's independent re-computation from the route dump:
            // for a SET-backed store this must equal "the hotset's coverage" to the slot.
            if (g_res_stats) {
                for (int i = 0; i < (int) N_USED; ++i) ++routed_n;   // per slot, not per (slot x VRAM store)
                for (Store * s : stores) {
                    if (s->loc != Store::VRAM) continue;
                    for (int i = 0; i < (int) N_USED; ++i)
                        if (s->slot(ly->layer, sel[i]) >= 0) ++served_n;   // safe: the audit enforces one home
                }
            }
            const double ts0 = now_s();
            enqueue_all(cpu);
            drain_all(cpu);
            t_store_acc += now_s() - ts0;
        }
        // ffn_out = the FFN block output BEFORE the residual add (the reference's own "ffn_out"
        // = ffn_moe_out + ffn_shexp), l_out = ffn_out + ffn_inp (the reference's "l_out", the
        // tensor that feeds the next layer).  U7: the store arm accumulates straight into `y`
        // (zeroed by drain_all), so the old `moe = y` copy of a whole H_DIM block per layer is
        // gone — the same values, one fewer pass.
        // U8: in fold mode the resident MoE came out of the core graph (moe_res), the misses out
        // of the CPU store (y) — the accumulation order changes, so this is NOT byte-identical
        // to the store arm (U4e: measured, never claimed).
        const bool fold_arm = store_arm() && ly->g_moe_res != nullptr;
        const std::vector<float> & moe_acc = store_arm() ? y : moe;
        for (size_t i = 0; i < (size_t) H_DIM; ++i) {
            const float moe_i = fold_arm ? (moe_res[i] + y[i]) : moe_acc[i];
            fout[i]  = moe_i + shexp[i];
            l_out[i] = fout[i] + ffn_inp[i];
        }
    }
};

// the reference's own dumped kq_soft_max shape — U2c: `kq_soft_max-<il> shape=[256,1,32]`.
// The engine's KV/mask width must equal it; it is READ, never assumed.
static int ref_kv_width(const char * path) {
    FILE * f = std::fopen(path, "r");
    if (!f) return -1;
    char line[2048];
    int w = -1;
    while (std::fgets(line, sizeof line, f)) {
        const char * p = std::strstr(line, "kq_soft_max-");
        if (!p) continue;
        const char * s = std::strstr(p, "shape=[");
        if (!s) continue;
        w = std::atoi(s + 7);
        break;
    }
    std::fclose(f);
    return w;
}

// relative delta of two doubles, guarded
static double rel_delta(double a, double b) {
    return std::fabs(a - b) / std::max(std::fabs(b), 1e-12);
}

struct TierAStats {
    int    n_cells = 0, n_floor = 0, n_trigger = 0;
    double worst = 0; int worst_step = -1;
    int    first_trigger = -1; double first_trigger_val = 0;
    void add(double r, int step) {
        ++n_cells;
        if (r > REL_PRINT_FLOOR) ++n_floor;
        if (r > worst) { worst = r; worst_step = step; }
        if (r > REL_TRIGGER) {
            ++n_trigger;
            if (first_trigger < 0) { first_trigger = step; first_trigger_val = r; }
        }
    }
};

// H3/U1 — THE CORE GRAPH'S CENSUS (--instr only).  Reads the graph the engine ACTUALLY built
// (layer 0's gf_core) and reports, per ggml op type, the node count and the bytes the op reads
// from THIS LAYER'S resident weight tensors (matched by POINTER against w[]), so the per-layer
// byte floor is measured from the artifact rather than from a type/geometry table.  A print:
// no allocation, no call, no numerics.
static void print_core_census(const M1Layer & ly, double wsum, double bsum, int n_layers) {
    struct Row { int count = 0; double wbytes = 0, act = 0; };
    std::map<std::string, Row> rows;
    double w_tot = 0, act_tot = 0;
    ggml_cgraph * g = ly.gf_core;
    const int n_nodes = ggml_graph_n_nodes(g);
    for (int i = 0; i < n_nodes; ++i) {
        const ggml_tensor * n = ggml_graph_node(g, i);
        Row & r = rows[ggml_op_name(n->op)];
        ++r.count;
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            const ggml_tensor * t = n->src[s];
            if (!t) continue;
            bool is_w = false;
            for (int k = 0; k < LW_N; ++k) if (ly.w[k] == t) { is_w = true; break; }
            if (is_w) { r.wbytes += (double) ggml_nbytes(t); w_tot  += (double) ggml_nbytes(t); }
            else      { r.act    += (double) ggml_nbytes(t); act_tot += (double) ggml_nbytes(t); }
        }
    }
    std::printf("# core census (layer 0, gf_core as built): nodes=%d weight-src=%.3f MiB "
                "activation-src=%.3f MiB\n",
                n_nodes, w_tot / (1024.0*1024.0), act_tot / (1024.0*1024.0));
    for (const auto & kv : rows)
        std::printf("#   ops %-16s n=%2d w=%7.3f MiB act=%7.3f MiB\n", kv.first.c_str(),
                    kv.second.count, kv.second.wbytes / (1024.0*1024.0),
                    kv.second.act / (1024.0*1024.0));
    // the budget side: what the layer buffer holds (weights) vs what was allocated for it
    // (weights + KV cache + every intermediate the graph writes).
    std::printf("# core layer bytes: weights %.3f MiB/layer (x%d = %.3f MiB), allocated "
                "%.3f MiB/layer (x%d = %.3f MiB)\n",
                wsum / n_layers / (1024.0*1024.0), n_layers, wsum / (1024.0*1024.0),
                bsum / n_layers / (1024.0*1024.0), n_layers, bsum / (1024.0*1024.0));
}

// H3/U2 — THE PER-NODE CLASSIFICATION (--instr --core-nodes).  The census above counts nodes
// per op; this prints the chain itself, one line per node, CLASSIFIED BY THE RULES THE CUDA
// BACKEND ITSELF APPLIES rather than by what the op is called:
//
//   NOOP    the backend's compute loop does nothing for it (ggml-cuda.cu:5687-5691:
//           GGML_OP_NONE / RESHAPE / VIEW / PERMUTE / TRANSPOSE return immediately).  Under the
//           CUDA-graph capture the product runs at positions >= 3 these cost no kernel AND no
//           host time at all.
//   FOLDED  a node a fusion consumes: the MUL right after an RMS_NORM it multiplies, which
//           ggml_cuda.cu:4360 turns into ONE ggml_cuda_op_rms_norm_fused kernel.  The pair is
//           detected with the backend's own predicate, re-implemented here (ggml-impl.h:681-712
//           ggml_can_fuse: consecutive indices, both GGML_TENSOR_FLAG_COMPUTE, the first with
//           exactly ONE use, node->src[0] or src[1] == the previous node, same shape).
//   KERNEL  everything else launches at least one kernel.
//
// The count of KERNEL nodes is the serial chain the bracket is paying for; the count of nodes
// is not.  Anything quoted about "the node chain" should say which of the two it means.
static bool cu_noop_op(ggml_op op) {
    return op == GGML_OP_NONE || op == GGML_OP_RESHAPE || op == GGML_OP_VIEW ||
           op == GGML_OP_PERMUTE || op == GGML_OP_TRANSPOSE;
}

static int node_n_uses(ggml_cgraph * g, int idx) {
    const ggml_tensor * t = ggml_graph_node(g, idx);
    int n = 0;
    for (int i = 0; i < ggml_graph_n_nodes(g); ++i) {
        const ggml_tensor * o = ggml_graph_node(g, i);
        for (int s = 0; s < GGML_MAX_SRC; ++s) if (o->src[s] == t) ++n;
    }
    return n;
}

// the backend's ggml_can_fuse() for the two-node {RMS_NORM, MUL} pair (ggml-impl.h:681)
static bool can_fuse_norm_mul(ggml_cgraph * g, int idx) {
    const int n = ggml_graph_n_nodes(g);
    if (idx + 1 >= n) return false;
    const ggml_tensor * a = ggml_graph_node(g, idx);       // RMS_NORM
    const ggml_tensor * b = ggml_graph_node(g, idx + 1);   // MUL
    if (a->op != GGML_OP_RMS_NORM || b->op != GGML_OP_MUL) return false;
    if ((a->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) return false;
    if ((b->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) return false;
    if (node_n_uses(g, idx) != 1) return false;
    if (b->src[0] != a && b->src[1] != a) return false;
    return a->ne[0] == b->ne[0] && a->ne[1] == b->ne[1] && a->ne[2] == b->ne[2] && a->ne[3] == b->ne[3];
}

enum { NODE_KERNEL = 0, NODE_NOOP = 1, NODE_FOLDED = 2, NODE_FUSED_PAIR = 3 };

static void print_core_nodes(const M1Layer & ly, const char * tag) {
    ggml_cgraph * g = ly.gf_core;
    const int n = ggml_graph_n_nodes(g);
    int n_noop = 0, n_folded = 0, n_kernel = 0;
    double kbytes = 0;   // the bytes the KERNEL nodes write (their outputs)
    for (int i = 0; i < n; ++i) {
        const ggml_tensor * t = ggml_graph_node(g, i);
        int cls = NODE_KERNEL;
        if (cu_noop_op(t->op)) cls = NODE_NOOP;
        else if (t->op == GGML_OP_MUL && i > 0 && can_fuse_norm_mul(g, i - 1)) cls = NODE_FOLDED;
        else if (t->op == GGML_OP_RMS_NORM && can_fuse_norm_mul(g, i)) cls = NODE_FUSED_PAIR;
        const char * cn = cls == NODE_NOOP ? "NOOP  " : cls == NODE_FOLDED ? "FOLDED"
                        : cls == NODE_FUSED_PAIR ? "FUSE->" : "KERNEL";
        if (cls == NODE_NOOP) ++n_noop;
        else if (cls == NODE_FOLDED) ++n_folded;
        else { ++n_kernel; kbytes += (double) ggml_nbytes(t); }
        std::printf("#   node %2d %-16s %-8s out=%9zu B  src0=%s\n", i, ggml_op_name(t->op), cn,
                    ggml_nbytes(t), t->src[0] ? ggml_op_name(t->src[0]->op) : "-");
    }
    std::printf("# core nodes %s: %d total = %d NOOP + %d FOLDED(norm+mul) + %d KERNEL nodes "
                "(kernel outputs %.2f MiB)\n", tag, n, n_noop, n_folded, n_kernel,
                kbytes / (1024.0*1024.0));
}

// ---------------------------------------------------------------------------
// H3/U1 — THE CORE PROBE (--core-probe).  A TIMING-ONLY decomposition of the core bracket, run
// AFTER the loop has produced and printed every number the run is judged on, so it can perturb
// nothing that gets reported.
//
// It reproduces the product's per-layer launch structure — ggml_backend_graph_compute_async on the
// core backend, then either an event wait per layer (today's shape) or one wait per token — over
// 36 launches per token, and prices the difference.  Three axes:
//   * STRUCTURE: sync per layer vs once per token, with or without the six tap enqueues.  This
//     bounds the per-layer event wait and the tap host cost (lever "one graph per token").
//   * nt: the same graph launched with 2 and 4 columns.  The weight BYTES per launch are
//     identical, so a flat per-layer time while the useful columns double says the core is not
//     byte-bound (it is per-byte ALU/latency bound), and a proportional rise says it is.
//   * PHASE: graphs cut at attention / +router / +shared-expert, priced by subtraction.
//
// THE L2 CONFOUND, and how it is controlled: a probe that launches ONE layer's graph 36 times in a
// row re-reads 40.5 MiB that then sits in the card's cache, i.e. it measures a WARM regime and
// overstates bandwidth (the §8.7 correction's trap).  Each set therefore CYCLES k distinct layer
// builds (round-robin, so consecutive launches of one graph are k apart) with k = 6: a 246 MiB
// footprint per token, and the k = 1 variant is kept as the WARM CONTROL so the size of that
// effect is measured rather than argued.
// The phase stops are separate LAYER BUILDS (separate contexts) because the backend's CUDA-graph
// capture is keyed on the graph's first node pointer (ggml-cuda.cu:2691) — rebuilding the SAME
// context with a different frontier would re-capture on every alternation and measure the
// un-captured path.
// the two numbers a phase graph is priced by: how many nodes it must execute in series, and how
// many weight bytes it reads (the same pointer match print_core_census uses).
static void graph_shape(const M1Layer & ly, int * n_nodes, double * w_mib) {
    ggml_cgraph * g = ly.gf_core;
    *n_nodes = ggml_graph_n_nodes(g);
    double w = 0;
    for (int i = 0; i < *n_nodes; ++i) {
        const ggml_tensor * n = ggml_graph_node(g, i);
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            const ggml_tensor * t = n->src[s];
            if (!t) continue;
            for (int k = 0; k < LW_N; ++k) if (ly.w[k] == t) { w += (double) ggml_nbytes(t); break; }
        }
    }
    *w_mib = w / (1024.0 * 1024.0);
}

static void core_probe_run(GGUFModel & m, ggml_backend_t gpu, int64_t kv_max, int n_res,
                           const std::vector<float> & x0, int pos0, int reps, int gap_us,
                           bool m4_wkb = false, bool m4_kqkv = false,
                           bool m4_kvwrite = false, bool m4_wvb = false) {
    const int N_LAUNCH = (int) N_LAYERS;   // the product issues one core launch per layer
    const int K = 6;                       // layers per set: 6 x 41.2 MiB = 247 MiB > any L2 here
    // H3/U2: `expr` is the attention chain's expression (see M1Layer::build) and `gap_us` the
    // host-busy idle inserted between launches — the interleaved regime the product runs in
    // (L54 MAJOR-2).  A gap row is only meaningful with sync/layer, because with one wait per
    // token the host is ahead of the GPU and a host stall does not idle the card.
    struct V { const char * name; int stop; int expr; int nt; int k; bool taps; bool sync_each; int gap; };
    static const V vs[] = {
        { "full nt1 k6 sync+taps",  M1Layer::CORE_STOP_FULL, 0, 1, K, true,  true,  0 },  // the product
        { "full nt1 k6 sync",       M1Layer::CORE_STOP_FULL, 0, 1, K, false, true,  0 },
        { "full nt1 k6 once",       M1Layer::CORE_STOP_FULL, 0, 1, K, false, false, 0 },  // the base
        { "full nt1 k1 once",       M1Layer::CORE_STOP_FULL, 0, 1, 1, false, false, 0 },  // WARM control
        { "full nt4 k6 once",       M1Layer::CORE_STOP_FULL, 0, 4, K, false, false, 0 },
        // the attention chain at three expressions, same frontier, same math
        { "attn     nt1 k6 once",   M1Layer::CORE_STOP_ATTN, 0, 1, K, false, false, 0 },
        { "attn     nt1 k6 sync",   M1Layer::CORE_STOP_ATTN, 0, 1, K, false, true,  0 },
        { "attnf1   nt1 k6 once",   M1Layer::CORE_STOP_ATTN, 1, 1, K, false, false, 0 },
        { "attnf1   nt1 k6 sync",   M1Layer::CORE_STOP_ATTN, 1, 1, K, false, true,  0 },
        // ...and the same four with the product's interleaved idle between launches
        { "full nt1 k6 sync GAP",   M1Layer::CORE_STOP_FULL, 0, 1, K, false, true,  0 },  // gap set below
        { "attn     nt1 k6 sync GAP", M1Layer::CORE_STOP_ATTN, 0, 1, K, false, true, 0 },
        { "attnf1   nt1 k6 sync GAP", M1Layer::CORE_STOP_ATTN, 1, 1, K, false, true, 0 },
        { "full nt1 k6 sync+taps GAP", M1Layer::CORE_STOP_FULL, 0, 1, K, true, true, 0 },
    };
    const int N_GAP_FROM = 9;              // the rows from here take gap_us instead of 0
    const int n_v = (int) (sizeof vs / sizeof vs[0]);

    struct Set { int stop, expr, nt, k; std::vector<M1Layer *> ly; double mib; };
    std::vector<Set> sets;
    auto get = [&](int stop, int expr, int nt, int k) -> Set * {
        for (Set & s : sets) if (s.stop == stop && s.expr == expr && s.nt == nt && s.k == k) return &s;
        // VRAM guard: the product's stores + core are already resident, so refuse rather than die.
        size_t vfree = 0, vtotal = 0;
        ggml_backend_dev_t dev = ggml_backend_get_device(gpu);
        if (dev) ggml_backend_dev_memory(dev, &vfree, &vtotal);
        const double need_mib = 44.0 * k;      // 41.2 MiB/layer allocated + slack
        if (dev && (double) vfree / (1024.0*1024.0) < need_mib * 1.25) {
            std::printf("# core probe: SKIPPED from here: %.0f MiB free, need %.0f MiB for %d probe "
                        "layer(s) — the product's residency is untouched\n",
                        (double) vfree / (1024.0*1024.0), need_mib * 1.25, k);
            return nullptr;
        }
        sets.push_back({ stop, expr, nt, k, {}, 0.0 });
        Set & s = sets.back();
        for (int j = 0; j < k; ++j) {
            M1Layer * ly = new M1Layer();
            ly->build(m, j, gpu, kv_max, nt, n_res, /*gpu_core*/ true, /*want_oracle*/ false,
                      /*fold_resident*/ false, /*fold_slots*/ -1, stop, expr, m4_wkb, m4_kqkv,
                      m4_kvwrite, m4_wvb);
            s.ly.push_back(ly);
            s.mib += (double) ggml_backend_buffer_get_size(ly->buf) / (1024.0*1024.0);
            if (g_core_nodes && j == 0) {   // H3/U2: the chain, as built, per frontier
                char tag[96];
                std::snprintf(tag, sizeof tag, "(probe set stop=%d expr=%d nt=%d k=%d)", stop, expr, nt, k);
                print_core_nodes(*ly, tag);
            }
        }
        return &s;
    };

    std::printf("# core probe: %d timed token(s) + 3 warmup per variant, %d launches/token, k "
                "distinct graphs cycled per set; %d us host-busy idle between launches on the GAP "
                "rows; run AFTER the loop (nothing reported above depends on it)\n",
                reps, N_LAUNCH, gap_us);
    double base = 0.0;   // the full nt1 k6 once variant, the ratio's denominator
    for (int v = 0; v < n_v; ++v) {
        const V & vv = vs[v];
        const int gap = v >= N_GAP_FROM ? gap_us : 0;
        if (gap && !vv.sync_each) continue;   // a host stall only idles the card under sync/layer
        Set * sp = get(vv.stop, vv.expr, vv.nt, vv.k);
        if (!sp) break;
        const int nt = vv.nt;
        for (M1Layer * ly : sp->ly) {
            std::vector<float> xin((size_t) H_DIM * nt);
            for (int t = 0; t < nt; ++t)
                std::memcpy(&xin[(size_t) H_DIM * t], x0.data(), sizeof(float) * (size_t) H_DIM);
            std::vector<int32_t> pos((size_t) nt);
            std::vector<int64_t> kidx((size_t) nt), nkv((size_t) nt);
            for (int t = 0; t < nt; ++t) {
                CHECK(pos0 + t < kv_max, "core probe: position %d past the KV width %lld",
                      pos0 + t, (long long) kv_max);
                pos[t] = pos0 + t; kidx[t] = pos0 + t; nkv[t] = pos0 + t + 1;
            }
            ly->set_input_async(xin.data(), pos.data(), kidx.data(), nkv.data(), gpu);
        }
        // the store phase's idle: CPU-BUSY host time of the same length the product spends in
        // its drive bracket, so the card's clock behaviour matches the product's regime
        auto spin_us = [&](int us) {
            if (us <= 0) return;
            const double t_end = now_s() + 1e-6 * (double) us;
            while (now_s() < t_end) { }
        };
        auto one_token = [&]() {
            for (int j = 0; j < N_LAUNCH; ++j) {
                M1Layer * ly = sp->ly[(size_t) (j % vv.k)];
                ggml_backend_graph_compute_async(gpu, ly->gf_core);
                if (vv.sync_each) {
                    ggml_backend_event_record(ly->ev_core, gpu);
                    if (vv.taps) ly->queue_taps(gpu);
                    ggml_backend_event_synchronize(ly->ev_core);
                    spin_us(gap);
                } else if (vv.taps) {
                    ly->queue_taps(gpu);
                }
            }
            if (!vv.sync_each) {   // ONE wait for the whole token
                M1Layer * last = sp->ly[(size_t) ((N_LAUNCH - 1) % vv.k)];
                ggml_backend_event_record(last->ev_core, gpu);
                ggml_backend_event_synchronize(last->ev_core);
            }
        };
        for (int w = 0; w < 3; ++w) one_token();   // capture warmup (ggml needs 2 stable calls/graph)
        const double t0 = now_s();
        for (int r = 0; r < reps; ++r) one_token();
        const double dt = now_s() - t0;
        // the GAP rows carry the idle INSIDE the measurement, so their per-layer time includes
        // it; strip it for the exec-only column and report both.
        const double per_layer = dt / (double) reps / (double) N_LAUNCH;
        const double exec = per_layer - 1e-6 * (double) gap;
        if (vv.stop == M1Layer::CORE_STOP_FULL && nt == 1 && vv.k == K && !vv.taps && !vv.sync_each)
            base = per_layer;
        int nn = 0; double wmib = 0;
        graph_shape(*sp->ly[0], &nn, &wmib);
        std::printf("#   probe %-24s stop=%d expr=%d nt=%d k=%d %s %-10s nodes=%2d w=%6.2f MiB "
                    "ms/layer=%7.4f us/node=%5.2f us/layer_exec=%7.4f gap=%d ms/token=%7.4f "
                    "ratio=%.3f GiB/s=%6.1f (footprint %.0f MiB)\n",
                    vv.name, vv.stop, vv.expr, nt, vv.k, vv.taps ? "taps" : "    ",
                    vv.sync_each ? "sync/layer" : "once/token", nn, wmib,
                    per_layer * 1000.0, per_layer * 1e6 / (double) nn, exec * 1000.0, gap,
                    per_layer * 1000.0 * N_LAUNCH, base > 0 ? per_layer / base : 0.0,
                    per_layer > 0 ? wmib / 1024.0 / per_layer : 0.0, sp->mib);
    }
    for (Set & s : sets) for (M1Layer * ly : s.ly) delete ly;
    std::printf("# core probe done (%zu set(s); the product's own graphs are untouched)\n",
                sets.size());
}

// ---------------------------------------------------------------------------
// H3/U3 — THE ATTENTION-CHAIN BISECTION (--core-probe).  Prices each frontier of the
// attention chain by subtraction: one layer set per CORE_STOP_* prefix, timed under BOTH
// regimes the product spans (back-to-back = full boost, "once"; interleaved = the +~44us/layer
// in-situ tax, "GAP").  The incremental us/layer of a kernel group is the difference of two
// adjacent frontiers, so the dominant 2-3 kernels are read straight off the table without a
// profiler.  Built one frontier at a time (6 cycled layers ~126-264 MiB) and freed before the
// next, so the peak is one set regardless of how many frontiers are measured.
// ---------------------------------------------------------------------------
static void attn_bisection_run(GGUFModel & m, ggml_backend_t gpu, int64_t kv_max, int n_res,
                               const std::vector<float> & x0, int pos0, int reps, int gap_us,
                               bool m4_wkb = false, bool m4_kqkv = false,
                               bool m4_kvwrite = false, bool m4_wvb = false) {
    static const struct { const char * name; int stop; } FS[] = {
        { "kvwrite   attn_norm + wkv_a + kcur + set_rows", M1Layer::CORE_STOP_KVWRITE },
        { "q_a       wq_a (Q5_K 2.75 MiB)",                 M1Layer::CORE_STOP_Q_A },
        { "q_norm    q_a_norm (rms+mul)",                   M1Layer::CORE_STOP_Q_NORM },
        { "q_b       wq_b (Q8_0 4.46 MiB)",                 M1Layer::CORE_STOP_Q_B },
        { "q_nope_abs wk_b absorption (Q8_0 0.56 MiB)",     M1Layer::CORE_STOP_Q_NOPE },
        { "qcur      q_pe rope + concat",                   M1Layer::CORE_STOP_QCUR },
        { "kq        scores K^T.Q + softmax",               M1Layer::CORE_STOP_KQ },
        { "kqv       vT copy + attention contraction",      M1Layer::CORE_STOP_KQV },
        { "kqv_mla   wv_b (Q8_0 1.11 MiB)",                 M1Layer::CORE_STOP_KQV_MLA },
        { "kqv_out   head recombination",                   M1Layer::CORE_STOP_KQV_OUT },
        { "attn_out  wo (Q5_K 11.53 MiB)",                  M1Layer::CORE_STOP_ATTN },
    };
    const int n_f = (int) (sizeof FS / sizeof FS[0]);
    const int N_LAUNCH = (int) N_LAYERS;
    const int K = 6;

    std::printf("# attn bisection: %d frontier(s), %d timed token(s) + 3 warmup per regime, "
                "%d launches/token, k=%d cycled layers/set, gap=%d us on the interleaved rows\n",
                n_f, reps, N_LAUNCH, K, gap_us);
    std::printf("#   frontier is a PREFIX of the attention chain (a superset of the row above it); "
                "the delta column is the incremental us/layer of the kernel group named in the row\n");
    std::printf("#   %-42s %4s %5s %6s | %9s %9s | %9s %9s %9s\n",
                "frontier", "stop", "nodes", "w MiB",
                "once us/L", "GAP us/L", "once d", "GAP d", "GAPd-44");
    std::vector<double> once_us(n_f), gap_us_arr(n_f);
    for (int f = 0; f < n_f; ++f) {
        // VRAM guard, same shape as the phase probe's: build one set at a time and refuse loudly.
        size_t vfree = 0, vtotal = 0;
        ggml_backend_dev_t dev = ggml_backend_get_device(gpu);
        if (dev) ggml_backend_dev_memory(dev, &vfree, &vtotal);
        const double need_mib = 44.0 * K;
        if (dev && (double) vfree / (1024.0*1024.0) < need_mib * 1.25) {
            std::printf("# attn bisection: STOPPED at frontier %d (%s): %.0f MiB free < %.0f MiB "
                        "needed — the product's residency is untouched\n",
                        f, FS[f].name, (double) vfree / (1024.0*1024.0), need_mib * 1.25);
            break;
        }
        std::vector<M1Layer *> ly((size_t) K);
        double mib = 0;
        for (int j = 0; j < K; ++j) {
            ly[(size_t) j] = new M1Layer();
            ly[(size_t) j]->build(m, j, gpu, kv_max, /*nt*/ 1, n_res, /*gpu_core*/ true,
                                  /*want_oracle*/ false, /*fold_resident*/ false, /*fold_slots*/ -1,
                                  FS[f].stop, /*attn_expr*/ 0, m4_wkb, m4_kqkv,
                                  m4_kvwrite, m4_wvb);
            mib += (double) ggml_backend_buffer_get_size(ly[(size_t) j]->buf) / (1024.0*1024.0);
        }
        // stage the inputs once (same position for every layer, as the phase probe does)
        for (int j = 0; j < K; ++j) {
            std::vector<int32_t> pos(1); std::vector<int64_t> kidx(1), nkv(1);
            pos[0] = pos0; kidx[0] = pos0; nkv[0] = pos0 + 1;
            ly[(size_t) j]->set_input_async(x0.data(), pos.data(), kidx.data(), nkv.data(), gpu);
        }
        auto spin_us = [](int us) {
            if (us <= 0) return;
            const double t_end = now_s() + 1e-6 * (double) us;
            while (now_s() < t_end) { }
        };
        auto run_regime = [&](bool interleaved) -> double {
            const int gap = interleaved ? gap_us : 0;
            for (int w = 0; w < 3; ++w) {
                for (int j = 0; j < N_LAUNCH; ++j) {
                    M1Layer * lp = ly[(size_t) (j % K)];
                    ggml_backend_graph_compute_async(gpu, lp->gf_core);
                    if (interleaved) {
                        ggml_backend_event_record(lp->ev_core, gpu);
                        ggml_backend_event_synchronize(lp->ev_core);
                        spin_us(gap);
                    }
                }
                if (!interleaved) {
                    ggml_backend_event_record(ly[(size_t) ((N_LAUNCH - 1) % K)]->ev_core, gpu);
                    ggml_backend_event_synchronize(ly[(size_t) ((N_LAUNCH - 1) % K)]->ev_core);
                }
            }
            const double t0 = now_s();
            for (int r = 0; r < reps; ++r) {
                for (int j = 0; j < N_LAUNCH; ++j) {
                    M1Layer * lp = ly[(size_t) (j % K)];
                    ggml_backend_graph_compute_async(gpu, lp->gf_core);
                    if (interleaved) {
                        ggml_backend_event_record(lp->ev_core, gpu);
                        ggml_backend_event_synchronize(lp->ev_core);
                        spin_us(gap);
                    }
                }
                if (!interleaved) {
                    ggml_backend_event_record(ly[(size_t) ((N_LAUNCH - 1) % K)]->ev_core, gpu);
                    ggml_backend_event_synchronize(ly[(size_t) ((N_LAUNCH - 1) % K)]->ev_core);
                }
            }
            const double dt = now_s() - t0;
            return dt / (double) reps / (double) N_LAUNCH;   // s/layer
        };
        const double o = run_regime(false);
        const double g = run_regime(true);
        once_us[(size_t) f] = o * 1e6;
        gap_us_arr[(size_t) f] = g * 1e6;
        int nn = 0; double wmib = 0;
        graph_shape(*ly[0], &nn, &wmib);
        const double once_d  = f ? once_us[(size_t) f] - once_us[(size_t) f - 1] : once_us[(size_t) f];
        const double gap_d   = f ? gap_us_arr[(size_t) f] - gap_us_arr[(size_t) f - 1] : gap_us_arr[(size_t) f];
        std::printf("#   %-42s %4d %5d %6.2f | %9.2f %9.2f | %9.2f %9.2f %9.2f\n",
                    FS[f].name, FS[f].stop, nn, wmib, once_us[(size_t) f], gap_us_arr[(size_t) f],
                    once_d, gap_d, gap_d - 44.0);
        for (M1Layer * lp : ly) delete lp;
    }
    std::printf("# attn bisection done.  'once' = back-to-back (full boost); 'GAP' = interleaved "
                "(+~44us/layer in-situ tax); the last column strips the tax so the GAP delta is "
                "comparable to the once delta.\n");
}

static int loop_run(const LoopArgs & a) {
    const bool scoring = !a.score_tokens.empty();   // M1/Q1: scoring mode (teacher forcing)
    std::vector<int32_t> ids;
    int n_prompt = 0;
    if (!read_tokens((scoring ? a.score_tokens : a.tokens).c_str(), ids, n_prompt)) return 1;
    if (scoring) {
        // The scoring corpus is the WHOLE file, and its header's count must agree with the ids
        // it carries: a corpus file whose header disagrees with its body is exactly the kind of
        // silent drift the manifest exists to prevent.
        if (n_prompt != (int) ids.size()) {
            std::printf("FAIL: --score %s: header says n_prompt=%d but the file carries %zu ids; "
                        "the scoring mode feeds the WHOLE file and needs them equal\n",
                        a.score_tokens.c_str(), n_prompt, ids.size());
            return 1;
        }
        if (ids.size() < 2) {
            std::printf("FAIL: --score %s: %zu token(s) — needs >= 2 to score one position\n",
                        a.score_tokens.c_str(), ids.size());
            return 1;
        }
        // The reference dumps are indexed by the DECODE mapping (ref_pos0 = n_prompt-1); the
        // score loop's rows are indexed from 0, so a --ref here would compare the wrong rows.
        // Refuse rather than silently compare across two mappings.
        if (!a.ref.empty() || !a.ref_tokens.empty()) {
            std::printf("FAIL: --score with --ref/--ref-tokens: the score loop's row indices are "
                        "NOT the decode mapping (ref_pos0 = n_prompt-1), so a reference "
                        "comparison would be off by n_prompt-1 rows — run the two separately\n");
            return 1;
        }
    }
    if (!a.damage_tensor.empty()) {
        m1_set_damage(a.damage_tensor.c_str(), a.damage_eps);
        std::printf("# DAMAGE PROBE = '%s' x (1 + %.17g) at load — CALIBRATION ONLY; every number "
                    "below is a damaged model, not a product number\n",
                    a.damage_tensor.c_str(), a.damage_eps);
    }
    if (a.n_decode < 1 && !scoring) { std::printf("FAIL: --n-decode %d\n", a.n_decode); return 1; }
    if (n_prompt < 1 && !scoring)   { std::printf("FAIL: %s has an empty prompt (n_prompt %d)\n", a.tokens.c_str(), n_prompt); return 1; }
    const int n_total = scoring ? (int) ids.size() : n_prompt + a.n_decode;
    const int kv_max  = a.kv_max > 0 ? a.kv_max : n_total;
    if (kv_max < n_total) {
        std::printf("FAIL: --kv-max %d < %d positions — the KV cache cannot hold the sequence\n", kv_max, n_total);
        return 1;
    }
    const int ref_pos0 = n_prompt - 1;   // the -ub 1 chunk mapping, asserted below

    // ---- U6: the routing trace + the treatment's id permutation -----------------------
    // --route-perm: 36 lines "<l> <128 ints>"; entry e of line l = the PERMUTED FIXTURE id
    // that carries original expert e's weights.  Applied to the store arm's read-back ids
    // (post-taps, pre-enqueue) only; the router itself is untouched, so its own selection
    // is bit-identical to the control's by construction.
    static std::vector<int32_t> perm_storage;   // backs g_perm for the whole run
    if (!a.route_perm.empty()) {
        CHECK(a.do_store, "--route-perm maps the STORE arm's read-back ids; it needs --arm-store");
        CHECK(!a.falsify, "--route-perm is not wired into the falsifier path (and must not be)");
        // FINDING-1 (H1's review): --route-perm maps routed ids to PERMUTED FIXTURE ids while
        // --resident-set names ORIGINAL ids, so the set store's membership test would miss almost
        // every routed expert and serve it from RAM — silently wrong bytes, exit 0.  Refuse.
        CHECK(a.res_hot.empty(),
              "--route-perm maps routed ids to PERMUTED FIXTURE ids and --resident-set names ORIGINAL ids: "
              "combined they silently serve the wrong experts (exit 0) — refusing (U6's idiom was "
              "fixture+perm; H1's is the set)");
        FILE * f = std::fopen(a.route_perm.c_str(), "r");
        CHECK(f, "cannot open --route-perm %s", a.route_perm.c_str());
        perm_storage.assign((size_t) N_LAYERS * (size_t) N_EXPERT, -1);
        char line[4096];
        int rows = 0;
        while (std::fgets(line, sizeof line, f)) {
            if (line[0] == '#' || line[0] == '\n') continue;
            int l;
            char * p = line;
            int got = std::sscanf(p, "%d", &l);
            CHECK(got == 1 && l == rows && l >= 0 && l < (int) N_LAYERS,
                  "route-perm row %d malformed (expected layer %d)", rows, rows);
            p = std::strchr(line, ' ');
            for (int e = 0; e < (int) N_EXPERT; ++e) {
                CHECK(p != nullptr, "route-perm row %d: only %d entries", rows, e);
                perm_storage[(size_t) l * N_EXPERT + e] = std::strtol(p, &p, 10);
                p = std::strchr(p, ' ');
            }
            ++rows;
        }
        std::fclose(f);
        CHECK(rows == (int) N_LAYERS, "route-perm has %d rows, expected %d", rows, (int) N_LAYERS);
        for (int l = 0; l < (int) N_LAYERS; ++l) {
            uint64_t seen[2] = { 0, 0 };   // 128-bit membership: ids run to 127, so a 32-bit
            for (int e = 0; e < (int) N_EXPERT; ++e) {   // mask would collide (measured: 1u<<50 wraps to bit 18)
                const int32_t v = perm_storage[(size_t) l * N_EXPERT + e];
                CHECK(v >= 0 && v < (int) N_EXPERT, "route-perm row %d entry %d = %d out of range", l, e, v);
                CHECK(!((seen[v >> 6] >> (v & 63)) & 1), "route-perm row %d: id %d appears twice", l, v);
                seen[v >> 6] |= 1ull << (v & 63);
            }
            CHECK(seen[0] || seen[1], "route-perm row %d empty", l);   // (the dup+range checks above
                                                                       // already force a full permutation)
        }
        g_perm = perm_storage.data();
    }
    if (!a.route_dump.empty()) {
        CHECK(!a.falsify, "--route-dump is not wired into the falsifier path");
        g_rdump = std::fopen(a.route_dump.c_str(), "w");
        CHECK(g_rdump, "cannot open --route-dump %s", a.route_dump.c_str());
        static char rdbuf[1 << 20];   // setvbuf holds the pointer for the whole run
        std::setvbuf(g_rdump, rdbuf, _IOFBF, sizeof rdbuf);
        std::fprintf(g_rdump, "# M1/U6 route trace: R <layer> <pos> <sel0..3 = the router's OWN ids, PRE-permutation> <w0..w3>\n");
    }

    // U7: the per-position split (timing only) + its empty-store probe.  Both are instrument
    // flags, never the correctness path; the split cannot be combined with the falsifier (whose
    // verdict is a paired timing statistic this instrumentation would perturb).
    g_instr = a.instr;
    g_core_nodes = a.core_nodes;   // H3/U2
    g_res_stats = a.res_stats;   // H1: the served-slot counter (one branch per layer; timing-neutral)
    g_store_pull_at_drain = a.store_pull_at_drain;   // U7 A/B arm (numerics-neutral: same bytes)
    if (a.instr) {
        CHECK(a.do_store, "--instr splits the STORE DRIVE; it needs --arm-store");
        CHECK(!a.falsify, "--instr is not wired into the falsifier path (and must not be)");
    }
    if (a.extra_stores) {
        CHECK(a.do_store, "--extra-stores needs --arm-store (it extends the store list)");
        CHECK(!a.falsify, "--extra-stores is not wired into the falsifier path");
        CHECK(a.extra_stores > 0 && a.extra_stores <= 64, "--extra-stores %d out of range", a.extra_stores);
    }
    // H3/U1: the core probe times the core GRAPH on the core backend; it needs the gpu-core
    // configuration it is meant to explain, and it must not sit in the falsifier's timing path.
    if (a.core_probe) {
        CHECK(a.gpu_core, "--core-probe needs --gpu-core (the core graph runs on ROCm0)");
        CHECK(!a.falsify, "--core-probe is not wired into the falsifier path");
        CHECK(a.probe_reps > 0 && a.probe_reps <= 2000, "--probe-reps %d out of range", a.probe_reps);
        CHECK(a.probe_gap_us >= 0 && a.probe_gap_us <= 100000, "--probe-gap-us %d out of range",
              a.probe_gap_us);
    }
    // H3/U2: the fused attention expression is asserted only for the GPU-core arm, so the
    // CPU-only correctness path (gate1's oracle) keeps the reference expansion BY CONSTRUCTION
    // — the flag cannot reach it.
    if (a.core_fuse >= 0) {
        CHECK(a.core_fuse >= 0 && a.core_fuse <= 1, "--core-fuse %d out of range [0,1]", a.core_fuse);
        CHECK(a.gpu_core, "--core-fuse is GPU-core only: the CPU-only arm must keep the reference "
              "expansion (it is the numerics oracle)");
        CHECK(!a.falsify, "--core-fuse is not wired into the falsifier path");
    }
    // H3/U4: the custom-op hatch is GPU-core-only (the CPU arm is the numerics oracle) and is
    // NOT wired into the falsifier path (the falsifier times the STORE list's overlap, a
    // configuration where the core's absorption kernel is not the quantity under test).
    const bool m4_on = (a.m4_wkb < 0) ? a.gpu_core : (a.m4_wkb == 1);   // AUTO = on iff the core is on GPU
    if (a.m4_wkb == 1) {
        CHECK(a.gpu_core, "--m4-wkb needs --gpu-core (the CPU-only arm must keep the reference "
              "mul_mat — it is the numerics oracle)");
    }
    if (m4_on) CHECK(!a.falsify, "--m4-wkb is not wired into the falsifier path");
    // H3/U5: the fused attention-core hatch follows the SAME rule as m4_wkb — GPU-core-only
    // (the CPU arm is the numerics oracle) and not wired into the falsifier path.
    const bool kqkv_on = (a.m4_kqkv < 0) ? a.gpu_core : (a.m4_kqkv == 1);   // AUTO = on iff the core is on GPU
    if (a.m4_kqkv == 1) {
        CHECK(a.gpu_core, "--m4-kqkv needs --gpu-core (the CPU-only arm must keep the reference "
              "kq/softmax/kqv sequence — it is the numerics oracle)");
    }
    if (kqkv_on) CHECK(!a.falsify, "--m4-kqkv is not wired into the falsifier path");
    // H3/U11: the two remaining phase fusions follow the SAME rule (GPU-core-only, not wired
    // into the falsifier path).  TRI-STATE rather than bool for the L59 reason: a default that
    // is only legal with --gpu-core must not fire its own guard on the legal CPU-only run.
    const bool kvw_on = (a.m4_kvwrite < 0) ? a.gpu_core : (a.m4_kvwrite == 1);
    if (a.m4_kvwrite == 1) {
        CHECK(a.gpu_core, "--m4-kvwrite needs --gpu-core (the CPU-only arm must keep the reference "
              "KV write — it is the numerics oracle)");
    }
    if (kvw_on) CHECK(!a.falsify, "--m4-kvwrite is not wired into the falsifier path");
    const bool wvb_on = (a.m4_wvb < 0) ? a.gpu_core : (a.m4_wvb == 1);
    if (a.m4_wvb == 1) {
        CHECK(a.gpu_core, "--m4-wvb needs --gpu-core (the CPU-only arm must keep the reference "
              "wv_b mul_mat — it is the numerics oracle)");
    }
    if (wvb_on) CHECK(!a.falsify, "--m4-wvb is not wired into the falsifier path");
    if (a.core_nodes) CHECK(a.instr, "--core-nodes needs --instr");

    // U8: the fold is only meaningful when the core graph and the resident store share ONE
    // backend (else the resident weights are not on the core backend and the mul_mat_id cannot
    // reference them).  The falsifier times the STORE list's overlap — a configuration the fold
    // deletes — so the two are never combined.
    if (a.fold_resident) {
        CHECK(a.do_store, "--fold-resident needs --arm-store (it replaces the VRAM store's slice)");
        CHECK(a.gpu_core && a.gpu_store,
              "--fold-resident needs --gpu-core AND --gpu-store: the fold runs the resident MoE "
              "inside the core graph, so the core and the resident experts must share the ROCm0 backend");
        CHECK(!a.falsify, "--fold-resident is not wired into the falsifier path (the falsifier times "
              "the store list the fold deletes)");
    }
    // U9: the slot-count A/B is a fold-mode timing knob only.
    if (a.fold_slots >= 0) {
        CHECK(a.fold_resident, "--fold-slots overrides the fold's resident block; it needs --fold-resident");
        CHECK(!a.falsify, "--fold-slots is a timing A/B; it is not wired into the falsifier path");
        CHECK(a.fold_slots <= (int) N_USED, "--fold-slots %d out of range [0,%d]", a.fold_slots, (int) N_USED);
    }
    // H1 — the resident SET replaces the store list's INTERVAL membership, so it needs the store
    // arm.  It is NOT wired into the retired resident fold: that fold deletes the VRAM
    // store, and a set with no store to hold it would be a silent no-op — refuse instead.
    if (!a.res_hot.empty()) {
        CHECK(a.do_store, "--resident-set sets the STORE LIST's membership; it needs --arm-store");
        CHECK(!a.fold_resident, "--resident-set is not wired into the retired resident fold — "
                               "the fold has no VRAM store to hold the set");
    }

    GGUFModel * m = GGUFModel::load(a.model.c_str());

    std::printf("=== M1 / U3 — THE 36-LAYER TOKEN LOOP ===\n");
    std::printf("# model      = %s\n", m->path.c_str());
    if (scoring)
        std::printf("# SCORE      = %s : %d tokens -> %d scored positions (teacher forcing: position"
                    " p feeds ids[p], its row predicts ids[p+1])\n",
                    a.score_tokens.c_str(), n_total, n_total - 1);
    else
        std::printf("# tokens     = %s : n_prompt=%d + n_decode=%d = %d positions\n",
                    a.tokens.c_str(), n_prompt, a.n_decode, n_total);
    std::printf("# ids        = [%d", ids[0]);
    for (int i = 1; i < (int) ids.size() && i < 12; ++i) std::printf(" %d", ids[i]);
    std::printf("%s]\n", ids.size() > 12 ? " ..." : "");
    std::printf("# layers     = %d (all) x (attention+MLA+KV+router+MoE+shexp)\n", (int) N_LAYERS);
    std::printf("# kv width   = %d%s\n", kv_max,
                a.kv_max > 0 ? "  (pinned: the reference's KV-cache pad, llama-kv-cache.cpp:1285)"
                             : "  (= the sequence length: U2's PRE-FIX width, NOT the reference's)");
    std::printf("# arm        = %s%s\n",
                a.do_store ? "store (the product store list)" : "oracle (the monolithic op sequence)",
                a.do_store ? "" : " — the strict transcription of the reference graph");
    std::printf("# n_res      = %d\n", a.n_res);
    std::printf("# kv type    = f16 (the pin's q8_0 caches are a NAMED divergence, plan risk 2)\n");
    {
        char buf[512];
        std::printf("# attention  = %s\n", m1_attn_params_str(buf, sizeof buf));
    }

    ggml_backend_t cpu = ggml_backend_cpu_init();
    CHECK(cpu, "cpu backend init failed");
    ggml_backend_cpu_set_n_threads(cpu, a.threads);
    g_store_cpu_be = cpu;    // U7: so the store-side split can name the RAM (CPU) store
    std::printf("# backend    = %s, threads=%d\n", ggml_backend_name(cpu), a.threads);

    // U4a — the split's GPU half; U4c — the layer core's home.  ROCm0 only (the P40 is never
    // part of the product).  Created for EITHER flag: --gpu-store puts the spec's VRAM store
    // ranges on it, --gpu-core builds/runs the layer core + embedding + head on it.  The two
    // are INDEPENDENT (core-on-GPU with every store on the CPU is a meaningful configuration
    // and is measured); the CPU-only configuration (both off) is the correctness path.
    ggml_backend_t gpu = nullptr;
    if (a.gpu_store || a.gpu_core) {
        gpu = ggml_backend_cuda_init(0);
        CHECK(gpu, "gpu (ROCm0) backend init failed");
    }
    ggml_backend_t core_be = a.gpu_core ? gpu : cpu;   // where the layer core + emb + head run
    if (a.gpu_core) {
        std::printf("# gpu core   = ON — the layers' non-expert surface (attention/MLA/router/shexp\n"
                    "#               weights + KV + inputs + intermediates) and embedding+head live and\n"
                    "#               run on %s; the expert weights are NOT resident here (they live in\n"
                    "#               the stores' slices).  NOT byte-identical to the CPU reference\n"
                    "#               (different kernels): measured and reported, never the token line\n",
                    ggml_backend_name(core_be));
    } else {
        std::printf("# gpu core   = OFF (the layer core runs on %s — the byte-identical correctness path)\n",
                    ggml_backend_name(cpu));
    }
    if (a.gpu_store) {
        CHECK(gpu, "gpu_store without a gpu backend");
        std::printf("# store split = ON ([0,n_res) stores on %s, the rest on %s) — M0's mechanism;\n"
                    "#               the CPU-only configuration is --gpu-store OFF\n",
                    ggml_backend_name(gpu), ggml_backend_name(cpu));
    } else {
        std::printf("# store split = OFF (every store on the CPU backend%s)\n",
                    a.gpu_core ? " — the U4c core-on-GPU/stores-all-CPU configuration" :
                                 "; the byte-identical correctness path");
    }
    if (g_perm) std::printf("# route-perm = %s (36 x 128, applied to the store arm's read-back ids)\n", a.route_perm.c_str());
    if (g_rdump) std::printf("# route-dump = %s (the router's OWN pre-permutation ids+weights)\n", a.route_dump.c_str());
    // H1 — the resident set, printed so the run's own log carries WHICH experts are resident (the
    // mechanism's input) rather than only its timings.  The per-layer audit below re-checks it on
    // the built list.
    if (!a.res_hot.empty()) {
        std::printf("# resident-set = %s : %zu per-layer ORDERED sets, %d experts each; slot i = the i-th\n"
                    "#   member; the VRAM store holds the set, the RAM store its complement\n",
                    a.res_path.c_str(), a.res_hot.size(), a.res_n);
        std::printf("# resident-set layer 0 =");
        for (int i = 0; i < (int) a.res_hot[0].size() && i < 8; ++i) std::printf(" %d", a.res_hot[0][(size_t) i]);
        std::printf("%s\n", a.res_hot[0].size() > 8 ? " ..." : "");
    } else {
        std::printf("# resident-set = none : the IDENTITY interval [0,%d) per layer (the pre-H1 store list)\n",
                    a.n_res);
    }
    print_loadavg("at start");

    // ---- tier A ----------------------------------------------------------------------
    RefDump   ref;
    RefLogits rlog;
    bool have_ref = false, have_rlog = false;
    if (!a.ref.empty()) {
        have_ref  = ref.parse(a.ref.c_str());
        have_rlog = rlog.parse(a.ref.c_str());
        if (!have_ref)  std::printf("FAIL: could not parse the reference per-layer dump %s\n", a.ref.c_str());
        if (!have_rlog) std::printf("FAIL: no get_logits_ith fingerprints in %s\n", a.ref.c_str());
        const int rw = ref_kv_width(a.ref.c_str());
        if (rw > 0) {
            std::printf("# reference KV width (read off its own dumped kq_soft_max shape) = %d\n", rw);
            if (rw != kv_max) {
                std::printf("FAIL: the reference reduces over width %d, this run uses %d "
                            "(llama-kv-cache.cpp:1285 — the width is the reference's padded cache, "
                            "not a flag; U2b/U2c)\n", rw, kv_max);
                delete m;
                return 1;
            }
        } else {
            std::printf("# NOTE: the reference dump carries no kq_soft_max shape line (unpatched "
                        "harness) — the KV width could not be asserted against the artifact\n");
        }
        std::printf("# reference: %d chunks parsed (%d decode steps), %d logits fingerprints\n",
                    ref.n_evals, ref.n_dec_steps, (int) rlog.v.size());
        std::printf("# position mapping: ref step k <-> position %d+k  (the prompt is %d single-token "
                    "ubatches landing in ONE chunk; last-write-wins) — controlled by the logits line below\n",
                    ref_pos0, n_prompt);
        if (rlog.n_vocab) std::printf("# reference n_vocab = %d\n", rlog.n_vocab);
    } else {
        std::printf("# TIER A: not run (no --ref).  The token line then has NO external instrument.\n");
    }

    // U4c: --gpu-core without the store arm has no expert weights anywhere (the layer buffer
    // does not carry them in this mode, and no store exists to slice them) — refuse loudly.
    CHECK(!a.gpu_core || a.do_store,
          "--gpu-core needs --arm-store: in this mode the expert weights are NOT resident in the "
          "layer buffer (they live in the stores' slices), so the oracle arm (and --arms-both) "
          "cannot run — the layer's own mul_mat_id would read a null tensor");

    // U8: the fold needs FLAT expert tensors (its mul_mat_id indexes ONE resident tensor; a
    // grouped file's mixed member types have no single such tensor).  Refuse up front, not
    // mid-build — grouped files keep the store path by dropping the flag.
    // U2b — ROTATION'S PRECONDITIONS, refused up front and loudly (never discovered mid-run):
    //  * --route-perm remaps sel[] into the PERMUTED FIXTURE's id space while the resident set (and
    //    therefore every swap) names ORIGINAL ids: the same silent-wrong combination L43 FINDING-1
    //    refused for --resident-set (the store would serve the wrong experts with no warning).
    //  * the retired fold has no VRAM store to rotate (L40/L41).
    if (a.rotate) {
        CHECK(a.route_perm.empty(),
              "--rotate + --route-perm would mix id spaces (the permutation rewrites sel[] but the "
              "rotation swaps VRAM rows by ORIGINAL id) — that is L43 FINDING-1's silent-wrong class");
        CHECK(!a.fold_resident, "--rotate needs a VRAM store; --fold-resident keeps the residents in "
              "the core graph (the fold is RETIRED, L40/L41)");
        CHECK(a.do_store, "--rotate needs --arm-store (it rotates the store list's VRAM entry)");
        CHECK(a.rot_admit_k >= 1 && a.rot_admit_k <= 8,
              "--rot-admit-k %d out of range [1,8] (4 = the measured frontier, 2 = the shipped 2Q "
              "policy)", a.rot_admit_k);
    } else {
        // U10: these two settings only mean something to a rotating run.  The refusal is on an
        // EXPLICIT flag, never on the default (a default that refuses its own absence is exactly
        // the bug this line exists to prevent — the gate runs the oracle arm, and it tripped).
        CHECK(!a.audit_scope_set, "--audit-full/--audit-incremental are rotation settings; they "
              "need --rotate");
        CHECK(!a.rot_admit_k_set, "--rot-admit-k is a rotation setting; it needs --rotate");
    }
    const bool fold_active = a.fold_resident;
    if (fold_active) {
        const LogicalTensor gu0 = m->logical(
            layer_tensor_name(0, LAYER_TENSOR_SPECS[LW_GATE_UP_EXPS].suffix).c_str());
        const LogicalTensor dn0 = m->logical(
            layer_tensor_name(0, LAYER_TENSOR_SPECS[LW_DOWN_EXPS].suffix).c_str());
        CHECK(gu0.segs.size() == 1 && dn0.segs.size() == 1,
              "--fold-resident is gated to FLAT (single-segment) expert tensors; layer 0 has "
              "%zu/%zu segments — grouped files keep the store path (drop the flag)",
              gu0.segs.size(), dn0.segs.size());
    }

    // ---- the graphs -----------------------------------------------------------------
    // U4c: --gpu-core builds the embedding and the head on the core backend too (and the
    // layers below); without the flag this line is the unchanged CPU build.
    EmbGraph emb;
    emb.build(*m, core_be);
    HeadGraph head;
    head.build(*m, core_be);
    const int n_build_layers = a.falsify ? a.f_layers : (int) N_LAYERS;
    std::printf("# head       = output_norm (f32, rms) -> output.weight (q4_K) x %lld = %lld logits\n",
                (long long) head.n_vocab, (long long) head.n_vocab);

    g_alias_prefault = a.alias_prefault;   // U2b: read by Store::build_alias_slices
    g_rotate_verify  = a.rotate_verify;    // U2b: read by the rotation's refresh
    g_audit_full     = a.audit_full;       // U10: the refresh audit's scope
    g_rot_admit_k    = a.rot_admit_k;      // U10: the 2Q window's threshold (default 2)
    const double t_build0 = now_s();   // U2b: the phase split (build | loop | tail), --instr only
    std::vector<std::unique_ptr<M1Layer>> layer_own;
    std::vector<LoopLayer> L((size_t) n_build_layers);
    size_t vram_bytes = 0;      // the split's resident expert bytes (U4a: does it fit?)
    size_t core_bytes = 0;      // U4c: the core's resident bytes (layer bufs + emb + head)
    // H2/U1: the PACKED-COPY accounting (see LoopArgs::footprint) — the bytes each location
    // class allocated, taken from the stores' own ggml buffers.
    size_t fp_vram_store_bytes = 0, fp_ram_store_bytes = 0;
    size_t fp_ram_alias_bytes = 0;   // U2a: mmap bytes the RAM store's alias COVERS (0 allocated)
    for (int l = 0; l < n_build_layers; ++l) {
        M1Layer * lp = new M1Layer();
        // U4e: the oracle graph is requested iff this run's arm reads it — the loop's store
        // arm never does, so its layers carry no layer-expert copies (a mixed-type grouped
        // file then LOADS; the oracle arm on the same file refuses loudly in
        // make_layer_tensor).  U8: in fold mode the layer DOES carry the RESIDENT experts
        // (the fold's mul_mat_id reads them from the core backend).
        lp->build(*m, l, core_be, kv_max, /*nt*/ 1, a.n_res, a.gpu_core, /*want_oracle*/ !a.do_store,
                  fold_active, a.fold_slots, M1Layer::CORE_STOP_FULL,
                  // H3/U2: the GPU path only — the CPU-only arm keeps the reference expansion
                  // (it is the numerics oracle gate1 compares byte for byte).
                  /*attn_expr*/ a.gpu_core ? (a.core_fuse < 0 ? 1 : a.core_fuse) : 0,
                  /*m4_wkb*/ m4_on, /*m4_kqkv*/ kqkv_on,
                  /*m4_kvwrite*/ kvw_on, /*m4_wvb*/ wvb_on);
        std::vector<Store *> keep;
        if (a.do_store) {
            // H1: the layer's ORDERED resident set (nullptr => the identity interval [0,n_res),
            // i.e. the pre-H1 store list, derived by the same unchanged code path below).
            const std::vector<int> * res = a.res_hot.empty() ? nullptr : &a.res_hot[(size_t) l];
            std::vector<StoreSpec> specs = derive_store_specs(lp->gu_lt, lp->dn_lt, a.n_res, {}, false, res);
            for (const StoreSpec & sp : specs) {
                // U8: the resident slice is folded into the core graph — no VRAM store is built
                // (its range is computed in-core; the CPU store still handles the misses).
                if (fold_active && sp.loc == Store::VRAM) continue;
                // U4a: the spec's LOCATION selects the backend — VRAM on ROCm0 (only with
                // --gpu-store), RAM on the CPU.  Off, the `cpu` backend for every entry keeps
                // exactly the previous (byte-identical) behaviour.
                ggml_backend_t be = (a.gpu_store && sp.loc == Store::VRAM) ? gpu : cpu;
                Store * s = make_store(sp, lp->gu_lt, lp->dn_lt, be, 1);
                keep.push_back(s);
                if (a.gpu_store && sp.loc == Store::VRAM)
                    for (const Slice & sl : s->slices) vram_bytes += sl.gu_nbytes + sl.d_nbytes;
                if (a.footprint && s->buf) {   // H2/U1: what THIS store actually allocated
                    const size_t bb = ggml_backend_buffer_get_size(s->buf);
                    if (sp.loc == Store::VRAM) fp_vram_store_bytes += bb;
                    else                       fp_ram_store_bytes  += bb;
                }
                // H2/U2a: the mmap bytes the RAM store's alias COVERS (not allocated — the rows
                // are the file's own pages, which is why this is reported separately from (i)).
                if (a.footprint && s->aliased)
                    for (const Slice & sl : s->slices) fp_ram_alias_bytes += sl.gu_nbytes + sl.d_nbytes;
                if (l == 0 && a.gpu_store) {
                    char lb[64];
                    if (sp.by_set) std::snprintf(lb, sizeof lb, "SET[%d]", (int) sp.members.size());
                    else           std::snprintf(lb, sizeof lb, "[%d,%d)", sp.lo, sp.hi);
                    std::printf("# layer 0 store %s%s on %s\n", lb,
                                sp.loc == Store::VRAM ? " (VRAM)" : " (RAM)", ggml_backend_name(be));
                }
            }
            if (l == 0) {
                std::printf("# store list = layer 0:");
                for (const StoreSpec & sp : specs) {
                    if (sp.by_set) std::printf(" SET[%d]%s", (int) sp.members.size(),
                                               sp.loc == Store::VRAM ? " (VRAM)" : " (RAM)");
                    else           std::printf(" [%d,%d)%s", sp.lo, sp.hi,
                                               sp.loc == Store::VRAM ? " (VRAM)" : " (RAM)");
                }
                std::printf("\n");
            }
            // H1: the audit asserts the list PARTITIONS the experts; the retired fold deletes the
            // VRAM entry (the resident block lives in the core graph), so its list is deliberately
            // not a partition and must not be audited (the resident fold is not part of
            // this unit's configuration and is left exactly as it was).
            if (!fold_active) audit_store_list(keep, l, l == 0, &lp->gu_lt, &lp->dn_lt);
            // U7 probe: N EMPTY (n_e == 0) list entries — a legal list entry (store.h SEAM 1)
            // that enqueues no graph and drains no pending, so it adds the per-store-cycle
            // overhead of the list loop and nothing else.  Numerics-neutral by construction.
            for (int k = 0; k < a.extra_stores; ++k) {
                Store * s = new Store();
                s->build(Store::RAM, lp->gu_lt, lp->dn_lt, 0, 0, cpu, 1);
                keep.push_back(s);
            }
        }
        L[(size_t) l].init(lp);
        L[(size_t) l].stores = keep;
        L[(size_t) l].be_core = a.gpu_core ? gpu : nullptr;
        if (a.gpu_core) core_bytes += ggml_backend_buffer_get_size(lp->buf);
        layer_own.emplace_back(lp);
    }
    if (a.gpu_core) core_bytes += ggml_backend_buffer_get_size(emb.buf)
                                + ggml_backend_buffer_get_size(head.buf);
    const double t_build1 = now_s();
    std::printf("# built      = %d layer(s) + embedding + head; store entries/layer = %d\n",
                n_build_layers, L.empty() ? 0 : (int) L[0].stores.size());
    // M1/Q1: a damage suffix that matched nothing would leave a "damaged" run that is in fact
    // the undamaged model — the ladder would read flat and look like a dead instrument.  The
    // probe is only a probe if it fired.
    if (!a.damage_tensor.empty()) {
        std::printf("# damage probe: %ld tensor(s) scaled by (1 + %.17g) for '%s'\n",
                    m1_damage_hits(), a.damage_eps, a.damage_tensor.c_str());
        if (m1_damage_hits() == 0) {
            std::printf("FAIL: --damage '%s' matched NO layer tensor — nothing was damaged, so "
                        "this run is the UNDAMAGED model wearing a damage label\n",
                        a.damage_tensor.c_str());
            return 1;
        }
    }
    if (g_instr)
        std::printf("# U2b phases (s): build=%.2f (of which the alias prefault is named below)\n",
                    t_build1 - t_build0);
    // H3/U1 — the core graph's census and its byte budget (--instr only; a print of the graph the
    // engine just built).  This is the artifact the "bytes per layer" claim must come from.
    if (g_instr && a.gpu_core && !L.empty()) {
        double wsum = 0, bsum = 0;
        for (const LoopLayer & ly : L) {
            wsum += (double) ly.ly->wbytes;
            bsum += (double) ggml_backend_buffer_get_size(ly.ly->buf);
        }
        print_core_census(*L[0].ly, wsum, bsum, (int) L.size());
        if (a.core_nodes) print_core_nodes(*L[0].ly, "(layer 0, gf_core as built)");
    }
    // U2b: the alias's page table was built at startup — say how much of the mmap that cost, so a
    // drive delta can be attributed to where the faults went rather than argued.
    if (g_alias_prefault_bytes)
        std::printf("# alias prefault = %.3f GiB walked at startup (MADV_WILLNEED %.2f s + one "
                    "touch/page %.2f s over the member rows, %d layer(s); whole calls %.2f s)\n",
                    (double) g_alias_prefault_bytes / (1024.0*1024.0*1024.0),
                    g_alias_prefault_advise_s, g_alias_prefault_touch_s, n_build_layers,
                    g_alias_prefault_call_s);
    // U2b — ARM THE ROTATION, once, on the built list of every layer.
    if (a.rotate) {
        for (int l = 0; l < n_build_layers; ++l) {
            Store * vram = nullptr, * ram = nullptr;
            for (Store * st : L[(size_t) l].stores) {
                if (st->loc == Store::VRAM && !vram) vram = st;
                else if (st->aliased && !ram)         ram  = st;
            }
            CHECK(vram && ram,
                  "--rotate needs ONE VRAM set entry and ONE zero-copy RAM entry per layer "
                  "(layer %d has %zu entries); it cannot rotate a list it cannot address", l,
                  L[(size_t) l].stores.size());
            CHECK(vram->is_set() && vram->slices.size() == 1,
                  "--rotate is FLAT-FILE and SET-backed only (layer %d: set=%d slices=%zu — a grouped "
                  "file's per-member segments are FINDING-3, unmeasured)", l, (int) vram->is_set(),
                  vram->slices.size());
            CHECK(ram->slices.size() == 1, "--rotate: layer %d's alias has %zu slices (flat only)",
                  l, ram->slices.size());
            CHECK(vram->n_e > 0 && vram->n_e < (int) N_EXPERT,
                  "--rotate: layer %d's VRAM entry holds %d of %d experts — rotation needs a proper "
                  "subset to swap into", l, vram->n_e, (int) N_EXPERT);
            L[(size_t) l].rot_setup(vram, ram);
        }
        std::printf("# rotate     = 2Q ADMISSION CONTROL: a ghost bit per expert is set on the 1st "
                    "reference of a non-resident expert and ADMITS it on the 2nd; the victim is the "
                    "LRU slot this position does not route to.\n");
        std::printf("# rotate grid = %d slots/layer, ghost window = %d experts/layer, refresh = a "
                    "membership swap (VRAM row write + alias predicate flip) applied at the NEXT "
                    "position's forward entry, before that layer's enqueue\n",
                    L[0].rot.vram ? L[0].rot.vram->n_e : 0, (int) N_EXPERT);
    }
    if (a.gpu_store || a.gpu_core)
        std::printf("# resident VRAM = %.2f GiB (stores %.2f + core %.2f: %d layer buffer(s) + emb + head, "
                    "the card has 15.92 GiB) at n_res=%d\n",
                    (double) (vram_bytes + core_bytes) / (1024.0*1024.0*1024.0),
                    (double) vram_bytes       / (1024.0*1024.0*1024.0),
                    (double) core_bytes       / (1024.0*1024.0*1024.0),
                    a.gpu_core ? n_build_layers : 0, a.n_res);
    if (fold_active)
        std::printf("# fold       = ON — the resident [0,%d) experts live in the LAYER buffer (counted in "
                    "'core' above); no VRAM store: its MoE runs inside the core graph\n", a.n_res);
    if (a.fold_slots >= 0)
        std::printf("# fold-slots = %d — TIMING A/B: the resident block computes %d slot%s%s\n",
                    a.fold_slots, a.fold_slots, a.fold_slots == 1 ? "" : "s",
                    a.fold_slots < (int) N_USED ? " (numerics WRONG by construction — timing only)" : "");

    if (a.footprint) {
        // H2/U1 — THE FOOTPRINT PROBE.  Two independent readings of the same question:
        //  (i)  the stores' own ggml buffers (what the machinery ALLOCATED, by location class);
        //  (ii) the process's VmRSS split into RssAnon / RssFile — the packed copies are
        //       anonymous (the store backend memcpys into its own allocation, store.h
        //       upload_slice), while the GGUF is a MAP_PRIVATE/PROT_READ file mapping
        //       (gguf_model.h), so its pages land in RssFile and are shared, not duplicated.
        auto proc_kb = [](const char * key, double & kib) {
            kib = -1.0;
            FILE * f = std::fopen("/proc/self/status", "r");
            if (!f) return;
            char line[512];
            const size_t kl = std::strlen(key);
            while (std::fgets(line, sizeof line, f)) {
                if (std::strncmp(line, key, kl) != 0) continue;
                kib = std::strtod(line + kl, nullptr);
                break;
            }
            std::fclose(f);
        };
        double rss = -1, anon = -1, rssfile = -1;
        proc_kb("VmRSS:", rss); proc_kb("RssAnon:", anon); proc_kb("RssFile:", rssfile);
        const double G = 1024.0*1024.0*1024.0;
        std::printf("# store footprint = VRAM-labelled stores %.3f GiB + RAM-labelled stores %.3f GiB "
                    "(%d layers, PACKED copies — the GGUF mmap is NOT counted here)\n",
                    (double) fp_vram_store_bytes / G, (double) fp_ram_store_bytes / G, n_build_layers);
        // H2/U2a: the RAM store no longer packs its experts — its weight tensors ARE the mmap, so
        // what it ALLOCATES (the line above) is the graph's activation tensors only, and the bytes
        // it COVERS are the file's own pages (RssFile, shared, not a second copy).  Printed apart
        // so "the 19.1 GiB is gone" cannot be confused with "moved".
        std::printf("# RAM store alias = %.3f GiB of the mmap COVERED zero-copy, 0 bytes allocated "
                    "(rows are the file's own pages)\n", (double) fp_ram_alias_bytes / G);
        std::printf("# process memory = VmRSS %.3f GiB (RssAnon %.3f + RssFile %.3f GiB)\n",
                    rss / (1024.0*1024.0), anon / (1024.0*1024.0), rssfile / (1024.0*1024.0));
    }

    // ---- the falsifier (its own verdict section; the token loop is not run) -----------
    if (a.falsify) {
        auto stat3 = [](std::vector<double> v, double & mn, double & md, double & mean) {
            std::sort(v.begin(), v.end());
            mn = v.front(); md = v[v.size() / 2];
            mean = std::accumulate(v.begin(), v.end(), 0.0) / (double) v.size();
        };
        std::printf("\n=== THE LOOP'S PAIRED FALSIFIER (%s) ===\n", a.serialize_both ? "CONTROL" : "REAL");
        std::printf("# unit     = ONE token through %d layer(s) with the PRODUCT store list (n_res=%d)\n",
                    a.f_layers, a.n_res);
        std::printf("# store split = %s\n",
                    a.gpu_store ? "ON — [0,n_res) on ROCm0, [n_res,128) on CPU (the GPU/CPU split)"
                                : "OFF — every store on the CPU backend (the two arms are the SAME "
                                  "instruction sequence; the ratio is ~1.00 by construction)");
        std::printf("# REAL     = per layer: all stores enqueued, then all synchronized, then all drained\n");
        std::printf("# SERIALISED = per layer: store by store, enqueue + synchronize + drain (no overlap)\n");
        std::printf("# CONTROL  = --serialize-both: the SERIALISED implementation in BOTH arms; the\n"
                    "#            ratio is ~1.00 BY CONSTRUCTION and the line must trip it, else exit 2\n");
        print_loadavg("falsifier start");

        std::vector<float> x((size_t) H_DIM, 0.0f);
        // one pass: the embedding of the first prompt token + the K layers, with the arm's
        // enqueue/drain ordering.  The timed unit is the LOOP BODY (the layers' forward with the
        // product store list); the token source is fixed, so both arms see identical inputs.
        // `store_ms` (U4a) additionally accumulates ONLY the store drive's own interval
        // (fill/enqueue/synchronize/drain) inside that body — M0's unit, and the unit the 0.950
        // line was calibrated on.  The loop body adds the attention/MLA/router/shexp core, which
        // is identical in both arms and cannot overlap: it is a constant added to BOTH terms of
        // the ratio and therefore DILUTES it toward 1.00 (the measured dilution, §results).
        auto one_pass = [&](bool serial, double * store_ms) {
            double acc = 0.0;
            emb.run(core_be, ids[0], x.data());
            for (int l = 0; l < a.f_layers; ++l) {
                LoopLayer & ly = L[(size_t) l];
                const int32_t pos = 0;
                const int64_t kidx = 0, nkv = 1;
                ly.ly->set_input(x.data(), &pos, &kidx, &nkv);
                ggml_backend_graph_compute(ly.be_core ? ly.be_core : cpu, ly.ly->gf_core);
                ly.ly->read(ly.ly->g_cur, ly.cur.data());
                ly.ly->read(ly.ly->g_sel, ly.sel.data());
                ly.ly->read(ly.ly->g_wts, ly.wts.data());
                const double u0 = now_s();
                if (serial) {
                    std::fill(ly.y.begin(), ly.y.end(), 0.0f);
                    for (Store * s : ly.stores) {
                        ly.enqueue_one(s);
                        ggml_backend_synchronize(s->backend);
                        ly.drain_one(s);
                    }
                } else {
                    ly.enqueue_all(cpu);
                    ly.drain_all(cpu);
                }
                acc += now_s() - u0;
                x = ly.y;   // the store block's output IS the compare point between the arms
            }
            if (store_ms) *store_ms = acc * 1000.0;
        };
        std::vector<double> pc, ps, pcs, pss;
        std::vector<float> yc((size_t) H_DIM, 0.0f), ys((size_t) H_DIM, 0.0f);
        double arm_diff = 0.0;
        for (int it = -a.warmup; it < a.iters; ++it) {
            // the arms alternate inside ONE timing loop, so the pairing cancels host drift.
            // CONTROL (--serialize-both): the SERIALISED implementation in BOTH arms, so the
            // ratio is ~1.00 by construction and the line must trip it (M0's semantics,
            // moe_split.cpp:628-631).  Without this the "control" is the identical real pair.
            const bool arm1_serial = a.serialize_both;
            double cs = 0, ss = 0;
            const double t0 = now_s(); one_pass(arm1_serial, &cs); const double t1 = now_s();
            yc = x;
            const double t2 = now_s(); one_pass(true, &ss);  const double t3 = now_s();
            ys = x;
            arm_diff = std::max(arm_diff, max_abs_diff(yc.data(), ys.data(), H_DIM));
            if (it >= 0) {
                pc.push_back((t1 - t0) * 1000.0); ps.push_back((t3 - t2) * 1000.0);
                pcs.push_back(cs); pss.push_back(ss);
            }
        }
        double c_min, c_med, c_mean, s_min, s_med, s_mean;
        stat3(pc, c_min, c_med, c_mean);
        stat3(ps, s_min, s_med, s_mean);
        std::vector<double> pr(pc.size());
        for (size_t i = 0; i < pc.size(); ++i) pr[i] = pc[i] / ps[i];
        double r_min, r_med, r_mean;
        stat3(pr, r_min, r_med, r_mean);
        const double r_max = *std::max_element(pr.begin(), pr.end());
        const double ratio = s_min > 0 ? c_min / s_min : 0.0;
        const bool ok = ratio <= a.thresh;
        std::printf("paired c_CONC ms (concurrent arm): min=%.4f med=%.4f mean=%.4f\n", c_min, c_med, c_mean);
        std::printf("paired c_SER  ms (serialised arm): min=%.4f med=%.4f mean=%.4f\n", s_min, s_med, s_mean);
        std::printf("paired per-iteration ratio c_conc/c_ser: min=%.3f med=%.3f mean=%.3f max=%.3f\n",
                    r_min, r_med, r_mean, r_max);
        std::printf("# paired arms compute the same thing: max|y_conc - y_ser| = %.3e\n", arm_diff);
        std::printf("PAIRED FALSIFIER[%s]: conc_min=%.4f ser_min=%.4f ratio=%.3f threshold=%.3f -> %s\n",
                    a.serialize_both ? "CONTROL" : "REAL", c_min, s_min, ratio, a.thresh,
                    ok ? "PASS" : "FAIL");
        // ---- THE STORE UNIT (U4a) — M0's unit, and the unit the 0.950 line was calibrated on.
        // The loop body's ratio above is DILUTED by the attention/MLA/router/shexp core, which is
        // identical in both arms and cannot overlap, so it is a constant added to both terms.  The
        // split's own claim is about its store drive, so that is timed on its own here (the core
        // graph still runs, untimed, between the drives — the unit is one token through the same
        // layers, same stores, same ids/weights, same order).
        double cs_min, cs_med, cs_mean, ss_min, ss_med, ss_mean;
        stat3(pcs, cs_min, cs_med, cs_mean);
        stat3(pss, ss_min, ss_med, ss_mean);
        const double store_ratio = ss_min > 0 ? cs_min / ss_min : 0.0;
        const bool store_ok = store_ratio <= a.thresh;
        std::printf("\n=== PAIRED FALSIFIER, STORE UNIT (M0's unit) (%s) ===\n",
                    a.serialize_both ? "CONTROL" : "REAL");
        std::printf("store-unit c_CONC ms (concurrent store drive): min=%.4f med=%.4f mean=%.4f\n",
                    cs_min, cs_med, cs_mean);
        std::printf("store-unit c_SER  ms (serialised store drive): min=%.4f med=%.4f mean=%.4f\n",
                    ss_min, ss_med, ss_mean);
        std::printf("STORE-UNIT PAIRED FALSIFIER[%s]: conc_min=%.4f ser_min=%.4f ratio=%.3f threshold=%.3f -> %s\n",
                    a.serialize_both ? "CONTROL" : "REAL", cs_min, ss_min, store_ratio, a.thresh,
                    store_ok ? "PASS" : "FAIL");
        std::printf("# store-unit share of the loop body: concurrent %.1f%%, serialised %.1f%% — the rest is\n"
                    "#   the attention/MLA/router/shexp core, which neither arm can overlap.\n",
                    100.0 * cs_med / (c_med > 0 ? c_med : 1.0),
                    100.0 * ss_med / (s_med > 0 ? s_med : 1.0));
        if (a.gpu_store)
            std::printf("# IDEAL OVERLAP: with the GPU half fully hidden the store-unit ratio is\n"
                        "#   max(gpu_half,cpu_half)/(gpu_half+cpu_half); the loop-body ratio is the same\n"
                        "#   effect divided by a unit C/(C+store) times larger, i.e. diluted toward 1.00\n"
                        "#   by exactly the core's share above.\n");
        print_loadavg("falsifier end");
        std::printf(a.gpu_store
                    ? "# NOTE: the split arms CAN differ: the VRAM store's enqueue is ASYNC on the HIP stream\n"
                      "#       (ggml_backend_graph_compute_async), so the CPU store's work runs underneath the\n"
                      "#       GPU's; the serialised arm waits for the GPU before it starts the CPU store.  The\n"
                      "#       loop's overlap claim (plan §Q4 risk 9) is being tested here, with a real split.\n"
                    : "# NOTE: on a CPU-only backend this ratio is the transmission of the SAME work: the\n"
                      "#       CPU backend has no async iface (ggml-cpu.cpp NULL set_tensor_async/\n"
                      "#       synchronize) and graph_compute_async == graph_compute (ggml-backend.cpp), so\n"
                      "#       the two arms cannot differ by construction.  The loop's overlap claim (plan\n"
                      "#       §Q4 risk 9) is not testable in THIS configuration; it needs --gpu-store.\n");
        int rc = 0;
        if (arm_diff != 0.0) {
            std::printf("FAIL: the two paired arms computed DIFFERENT values (max|d|=%.3e) — "
                        "the ratio is meaningless\n", arm_diff);
            rc = 1;
        }
        for (LoopLayer & ly : L) { for (Store * s : ly.stores) delete s; ly.stores.clear(); }
        layer_own.clear();          // U4c: free the (possibly CUDA) layer buffers BEFORE the backend
        if (gpu) ggml_backend_free(gpu);
        delete m;
        return rc;
    }

    // ---- the token loop --------------------------------------------------------------
    // The engine's per-token cost is load-dependent; the load average is printed around the
    // loop.  NOTHING here is a speed claim (that is U4's, on a quiet host).
    std::vector<std::vector<float>> rows;
    std::vector<int32_t> gen;
    rows.reserve((size_t) a.n_decode);
    gen.reserve((size_t) a.n_decode);

    std::vector<float> x((size_t) H_DIM, 0.0f), row((size_t) head.n_vocab, 0.0f);
    std::vector<TierAStats> t_out((size_t) N_LAYERS), t_inp((size_t) N_LAYERS), t_lout((size_t) N_LAYERS);
    TierAStats t_logits;
    int n_fail = 0;
    FILE * dumps = a.dump.empty() ? nullptr : std::fopen(a.dump.c_str(), "w");
    if (dumps) {
        std::fprintf(dumps, "# per (layer, step) relative deltas vs the reference's own ffn_out / ffn_inp / l_out sumsq\n");
        std::fprintf(dumps, "# layer step | ffn_out | ffn_inp | l_out\n");
    }

    // U2b: the IN-LOOP minor-fault count.  rusage is a process counter, so the loop's share is the
    // bracket's delta — the number the prefault is supposed to drive to ~0 (L47: ~195 K at 51
    // positions before it).  Accounting only; nothing reads it on a deciding path.
    auto self_minflt = []() -> long {
        struct rusage ru;
        return getrusage(RUSAGE_SELF, &ru) == 0 ? (long) ru.ru_minflt : -1;
    };
    const long mf_before = g_instr ? self_minflt() : 0;
    const double t_loop0 = now_s();
    for (int p = 0; p < n_total; ++p) {
        // M1/Q1 — SCORING feeds the FILE's own next token (teacher forcing); the decode path
        // feeds the argmax it just produced.  This ONE line is the whole difference, and it is
        // what makes a held-out text scorable in one O(N) pass.
        const int32_t tok = scoring ? ids[(size_t) p]
                                    : ((p < n_prompt) ? ids[p] : gen.back());
        const double te0 = now_s();
        emb.run(core_be, tok, x.data());
        t_emb_acc += now_s() - te0;
        for (int l = 0; l < (int) N_LAYERS; ++l) {
            L[(size_t) l].forward(cpu, p, p + 1, x.data());
            x = L[(size_t) l].l_out;
        }
        if (scoring) {
            // One row per position that HAS a target: position p's logits predict ids[p+1].  The
            // last position is computed (its KV row is written) and never dumped — it has no
            // successor to score.  <out>.tokens carries the TARGET ids, so a dump is
            // self-describing and the offline tool needs no second file.
            if (p + 1 >= (int) ids.size()) {
                std::printf("p=%04d tok=%d (the last position: no target to score)\n", p, tok);
                continue;
            }
            const double th0 = now_s();
            head.run(core_be, x.data(), row.data());
            t_head_acc += now_s() - th0;
            ++n_head_calls;
            rows.push_back(row);
            gen.push_back(ids[(size_t) p + 1]);
            std::printf("p=%04d tok=%d target=%d\n", p, tok, (int) ids[(size_t) p + 1]);
            continue;
        }
        if (p < ref_pos0) continue;                       // no logits row at these positions
        const int step = p - ref_pos0;                    // == the reference's chunk index
        // The reference walks n_prompt + n_decode positions and keeps n_decode logits rows: the
        // row at position n_prompt-1+n_decode is COMPUTED by its last decode and never sampled,
        // so it is never dumped (moe_logits_dump.cpp's loop dumps at the top of the body).  The
        // walk here matches it row for row: the same head pass runs, the row is discarded.
        const double th0 = now_s();
        head.run(core_be, x.data(), row.data());
        t_head_acc += now_s() - th0;
        ++n_head_calls;
        const double tt0 = now_s();
        if (step >= a.n_decode) {
            std::printf("p=%03d tok=%d step=%03d (the last walked position: the reference computes "
                        "this row and never dumps it)\n", p, tok, step);
            t_tail_acc += now_s() - tt0;
            continue;
        }
        const int32_t got = greedy_argmax(row.data(), head.n_vocab);
        rows.push_back(row);
        gen.push_back(got);

        const Sums s_log = sum_of(row.data(), head.n_vocab);
        std::printf("p=%03d tok=%d step=%03d", p, tok, step);
        if (have_rlog && step < (int) rlog.v.size()) {
            const Sums & r = rlog.v[step];
            const double dsum = rel_delta(s_log.sum, r.sum), dsq = rel_delta(s_log.sumsq, r.sumsq);
            t_logits.add(dsq, step);
            std::printf(" | ref logits rel dsum=%.3e dsumsq=%.3e", dsum, dsq);
            if (dsq > REL_TRIGGER && n_fail < 8) {
                std::printf("\nFAIL: the logits fingerprint diverges at step %d (rel dsumsq=%.3e > %.0e); "
                            "the reference's own whole-row checksum, independent of our layer code\n",
                            step, dsq, REL_TRIGGER);
                ++n_fail;
            }
        }
        std::printf("\n");

        // the per-layer tier A table — the localisation instrument.  Compare BEFORE the tokens:
        // a layer that diverges identifies where a token divergence comes from.
        if (have_ref && step < ref.n_dec_steps) {
            for (int l = 0; l < (int) N_LAYERS; ++l) {
                const LoopLayer & ly = L[(size_t) l];
                const RefChecksum & r_out = ref.v[step][(size_t) l * RefDump::N_KINDS + 2];
                if (!r_out.have) continue;
                const Sums s_out = sum_of(ly.fout.data(), H_DIM);
                const double d = rel_delta(s_out.sumsq, r_out.sumsq);
                t_out[(size_t) l].add(d, step);
                const RefChecksum & r_inp  = ref.v[step][(size_t) l * RefDump::N_KINDS + 3];
                const RefChecksum & r_lout = ref.v[step][(size_t) l * RefDump::N_KINDS + 4];
                const Sums s_inp = sum_of(ly.ffn_inp.data(), H_DIM);
                const Sums s_lout = sum_of(ly.l_out.data(), H_DIM);
                const double d_inp  = r_inp.have  ? rel_delta(s_inp.sumsq,  r_inp.sumsq)  : 0.0;
                const double d_lout = r_lout.have ? rel_delta(s_lout.sumsq, r_lout.sumsq) : 0.0;
                if (r_inp.have)  t_inp[(size_t) l].add(d_inp, step);
                if (r_lout.have) t_lout[(size_t) l].add(d_lout, step);
                if (dumps) std::fprintf(dumps, "%2d %3d %.6e %.6e %.6e\n", l, step, d,
                                        r_inp.have ? d_inp : -1.0, r_lout.have ? d_lout : -1.0);
                // every localisation column fails the engine on its own, not only ffn_out: a
                // residual-path divergence above the trigger must not need ffn_out to carry it.
                if (d > REL_TRIGGER && n_fail < 8) {
                    std::printf("FAIL: tier A: layer %d step %d ffn_out rel sumsq=%.3e > %.0e "
                                "(the reference prints sum=%.9e sumsq=%.9e)\n",
                                l, step, d, REL_TRIGGER, r_out.sum, r_out.sumsq);
                    ++n_fail;
                }
                if (r_inp.have && d_inp > REL_TRIGGER && n_fail < 8) {
                    std::printf("FAIL: tier A: layer %d step %d ffn_inp rel sumsq=%.3e > %.0e "
                                "(the reference prints sum=%.9e sumsq=%.9e)\n",
                                l, step, d_inp, REL_TRIGGER, r_inp.sum, r_inp.sumsq);
                    ++n_fail;
                }
                if (r_lout.have && d_lout > REL_TRIGGER && n_fail < 8) {
                    std::printf("FAIL: tier A: layer %d step %d l_out rel sumsq=%.3e > %.0e "
                                "(the reference prints sum=%.9e sumsq=%.9e)\n",
                                l, step, d_lout, REL_TRIGGER, r_lout.sum, r_lout.sumsq);
                    ++n_fail;
                }
            }
        }
        t_tail_acc += now_s() - tt0;
    }
    const double t_loop1 = now_s();
    // U2c — THE INDEX TABLE, after the last swap: the per-refresh audit already covers every
    // intermediate state; this asserts the state the LAST positions actually ran on, so a table
    // corrupted after the final refresh cannot leave the run with a clean exit.  Cheap (36 x ~176
    // integer compares) and outside every timing bracket.  FATALs on its own; no counter.
    if (a.rotate) {
        for (const LoopLayer & ly : L)
            if (ly.rot.vram) ly.rot.vram->audit_index_table(ly.ly->layer);
        // U10 — THE FULL WALK MOVES HERE.  The per-refresh check is now scoped to what a refresh
        // writes (audit_refresh_swap); this is the O(36 x 128) walk it was before, run ONCE on the
        // state the last positions actually ran on — which today's code did not do at all (it ran
        // only audit_index_table here).  So the end-of-run guarantee is strictly STRONGER than it
        // was: exactly-one-home over the running list, both stores' slice geometry, the alias's
        // 128-entry local_of table and every member's slot identity.  Outside every timing bracket.
        long n_full = 0;
        for (const LoopLayer & ly : L) {
            if (!ly.rot.vram) continue;
            audit_store_list(ly.stores, ly.ly->layer, false, &ly.ly->gu_lt, &ly.ly->dn_lt);
            ++n_full;
        }
        std::printf("# U10 end-of-run full audit: %ld layer(s) walked (exactly-one-home, slot "
                    "identity, local_of both directions, alias geometry) — the per-refresh check ran "
                    "incremental=%ld full=%ld mode\n", n_full, n_rot_audit_inc, n_rot_audit_full);
    }
    // U2b — SLOT IDENTITY BY BYTES, after the last swap: every resident slot is read BACK from the
    // VRAM store and memcmp'd against the file row of the expert the store claims lives there.  The
    // tables can agree with each other and still have written the row at the wrong offset; only this
    // sees it.  Runs once, outside every timing bracket, and fails the run if anything is off.
    if (a.rotate && a.rotate_verify) {
        long bad = 0, slots = 0;
        for (LoopLayer & ly : L) {
            if (!ly.rot.vram) continue;
            bad += ly.rot.vram->verify_slot_bytes();
            slots += ly.rot.vram->n_e;
        }
        long vbad = 0, vn = 0;
        for (const LoopLayer & ly : L) { vbad += ly.rot.vbad; vn += ly.rot.admitted; }
        std::printf("# rotate verify = %ld resident slot(s) read back and memcmp'd to the file row of "
                    "the expert they claim: %ld mismatch(es); PLUS %ld per-swap read-back(s), %ld "
                    "mismatch(es)\n", slots, bad, vn, vbad);
        bad += vbad;
        if (bad) { std::printf("FAIL: rotate verify: %ld resident slot(s) hold the WRONG expert's "
                               "bytes\n", bad); ++n_fail; }
    }
    const long mf_inloop = g_instr ? self_minflt() - mf_before : 0;
    if (g_instr)
        std::printf("# U2b phases (s): loop=%.2f (from build end to the last position)\n",
                    t_loop1 - t_build1);
    if (g_instr)
        std::printf("# U2b in-loop minflt = %ld over %d positions (whole process %ld at loop start; "
                    "the prefault's share is outside this bracket)\n", mf_inloop, n_total, mf_before);
    if (dumps) std::fclose(dumps);
    if (g_rdump) { std::fclose(g_rdump); g_rdump = nullptr; }   // U6: flush the route trace
    print_loadavg("after the loop");

    // ---- the write-out: the reference's own logits format ----------------------------
    if (!a.out.empty() && !write_logits(a.out.c_str(), rows, gen)) return 1;

    // ---- the token line --------------------------------------------------------------
    int n_tok_cmp = 0, n_tok_bad = 0, tok_first_div = -1;
    std::vector<int32_t> ref_tok;
    if (!a.ref_tokens.empty()) {
        if (!read_tokens_bin(a.ref_tokens.c_str(), ref_tok)) return 1;
        n_tok_cmp = (int) std::min(ref_tok.size(), gen.size());
        for (int i = 0; i < n_tok_cmp; ++i)
            if (ref_tok[i] != gen[i]) { ++n_tok_bad; if (tok_first_div < 0) tok_first_div = i; }
    }

    // ---- the verdict -----------------------------------------------------------------
    std::printf("\n=== VERDICT (U3: the 36-layer token loop) ===\n");
    std::printf("# steps compared          = %d (%d positions walked through %d layers)\n",
                (int) rows.size(), n_total, (int) N_LAYERS);
    {
        char lb[256];
        std::printf("# wall clock (LOAD-DEPENDENT, not a speed claim): %.1f s, loadavg now %s\n",
                    t_loop1 - t_loop0, loadavg_str(lb, sizeof lb));
    }
    // H1 — WHAT THE RESIDENCY SERVED (--resident-stats), the engine's own count, next to the
    // analysis tool's re-computation from the route dump.
    if (g_res_stats) {
        long rn = 0, sn = 0;
        double lmin = 1.0, lmax = 0.0;
        for (const LoopLayer & ly : L) {
            if (!ly.routed_n) continue;
            rn += ly.routed_n; sn += ly.served_n;
            const double c = (double) ly.served_n / (double) ly.routed_n;
            lmin = std::min(lmin, c); lmax = std::max(lmax, c);
        }
        std::printf("# resident stats: routed slots %ld, SERVED BY A VRAM STORE %ld (%.4f);"
                    " per-layer min/max %.4f / %.4f\n",
                    rn, sn, rn ? (double) sn / (double) rn : 0.0, lmin, lmax);
        // U2b — THE POSITIVE CONTROL for rotation: these counters can only be non-zero if the policy
        // DECIDED and the refresh APPLIED, and the served fraction above is read AFTER the refresh,
        // i.e. it is the membership that actually served each position (not the bootstrapped set).
        if (a.rotate) {
            long adm = 0, evi = 0, rfr = 0, aud = 0;
            for (const LoopLayer & ly : L) {
                adm += ly.rot.admitted; evi += ly.rot.evicted;
                rfr += ly.rot.refreshes; aud += ly.rot.audits;
            }
            double exp_mib = 0.0;
            for (const LoopLayer & ly : L)
                if (ly.rot.vram) {
                    exp_mib = (double) (ly.rot.vram->slices[0].gu_stride +
                                        ly.rot.vram->slices[0].d_stride) / (1024.0*1024.0);
                    break;
                }
            const double np = (double) n_total;
            std::printf("# rotation stats: refreshes %ld, post-refresh audits %ld, admitted %ld, "
                        "evicted %ld, uploads/token %.2f, upload %.1f MiB/token, upload %.2f ms/token "
                        "at 26.69 GiB/s | admit_k=%d audit=%s\n",
                        rfr, aud, adm, evi, np ? (double) adm / np : 0.0,
                        np ? (double) adm * exp_mib / np : 0.0,
                        np ? (double) adm / np * exp_mib * 1024.0 * 1024.0 / (26.69 * 1073741824.0) * 1000.0 : 0.0,
                        g_rot_admit_k, g_audit_full ? "full" : "incremental");
        }
    }
    // U4b — the serial-tail decomposition (plan §Q6 open question 5): measured, not asserted.
    std::printf("# loop decomposition over %d positions (s, sum ≈ the wall clock): emb %.3f |"
                " %dx layer core %.3f | store drive %.3f | head x%d %.3f (per head %.2f ms) |"
                " stage (argmax+copy+sums+print) %.3f\n",
                n_total, t_emb_acc, (int) N_LAYERS, t_core_acc, t_store_acc, n_head_calls,
                t_head_acc, n_head_calls ? 1000.0 * t_head_acc / n_head_calls : 0.0, t_tail_acc);
    std::printf("#   staging (U4c, GPU core only — inside the wall, outside the components above):"
                " layer-input H2D %.3f | layer-result D2H %.3f\n", t_setin_acc, t_reads_acc);
    // U7 — THE PER-POSITION SPLIT: the drive bracket above, decomposed into NAMED components
    // (plan §10.2 item 1).  Printed only under --instr; one `key=value` line in seconds and one
    // in ms/position so a row's components are greppable and fittable.
    if (g_instr) {
        const double np    = (double) n_total;
        const double drive = t_fill_acc + t_enq_ram_acc + t_enq_vram_acc
                           + t_sync_ram_acc + t_sync_vram_acc + t_drain_ram_acc + t_drain_vram_acc;
        const double wall  = t_loop1 - t_loop0;
        const double named = t_emb_acc + t_setin_acc + t_core_acc + t_reads_acc + drive + t_head_acc + t_tail_acc;
        std::printf("# U7 drive split (s): fill=%.5f enq_ram=%.5f enq_vram=%.5f sync_ram=%.5f sync_vram=%.5f"
                    " drain_ram=%.5f drain_vram=%.5f | drive=%.5f (bracket %.5f) get=%.5f combine=%.5f\n",
                    t_fill_acc, t_enq_ram_acc, t_enq_vram_acc, t_sync_ram_acc, t_sync_vram_acc,
                    t_drain_ram_acc, t_drain_vram_acc, drive, t_store_acc, g_store_get_s, g_store_combine_s);
        std::printf("# U7 inside the enqueue (s): pack_ram=%.5f launch_ram=%.5f pack_vram=%.5f launch_vram=%.5f"
                    " (groups=%ld)\n",
                    g_store_pack_ram_s, g_store_launch_ram_s, g_store_pack_vram_s, g_store_launch_vram_s,
                    g_store_group_n);
        std::printf("# U7 drive split (ms/pos over %d): fill=%.4f enq_ram=%.4f enq_vram=%.4f sync_ram=%.4f"
                    " sync_vram=%.4f drain_ram=%.4f drain_vram=%.4f | drive=%.4f\n",
                    n_total, 1000.0*t_fill_acc/np, 1000.0*t_enq_ram_acc/np, 1000.0*t_enq_vram_acc/np,
                    1000.0*t_sync_ram_acc/np, 1000.0*t_sync_vram_acc/np,
                    1000.0*t_drain_ram_acc/np, 1000.0*t_drain_vram_acc/np, 1000.0*drive/np);
        std::printf("# U7 cycles: store_cycles=%ld (%.2f/pos: active=%ld empty=%ld) graphs=%ld (%.2f/pos)"
                    " gets=%ld; named=%.5f wall=%.5f residual=%.5f (%.4f ms/pos)\n",
                    n_store_cycles, (double) n_store_cycles/np, n_store_active, n_store_empty,
                    n_store_groups, (double) n_store_groups/np, g_store_get_n,
                    named, wall, wall - named, 1000.0*(wall - named)/np);
        // U10 — ROTATION'S SHARE OF THE RESIDUAL, measured with brackets instead of priced from a
        // transfer rate and a refresh count.  `refresh` is the whole rot_apply_pending bracket
        // (which CONTAINS upload/alias/verify/audit); `apply_other` is what is left of it once the
        // four named pieces are subtracted.  The upload number is the HOST side only — the async
        // copies' transfer time lands wherever their stream is next waited for (sync_vram), which
        // is why it is reported here as a call cost and not as bytes.
        {
            const double sub = t_rot_upload_acc + t_rot_alias_acc + t_rot_verify_acc + t_rot_audit_acc;
            const double rot_tot = t_rot_apply_acc + t_rot_policy_acc;
            const double resid = wall - named;
            std::printf("# U10 rotation split (ms/pos over %d, %ld swaps = %.2f/pos, %ld apply calls):"
                        " upload=%.4f alias=%.4f verify=%.4f audit=%.4f apply_other=%.4f"
                        " | refresh=%.4f policy=%.4f | rot total=%.4f\n",
                        n_total, n_rot_swap, np ? (double) n_rot_swap/np : 0.0, n_rot_apply,
                        1000.0*t_rot_upload_acc/np, 1000.0*t_rot_alias_acc/np,
                        1000.0*t_rot_verify_acc/np, 1000.0*t_rot_audit_acc/np,
                        1000.0*(t_rot_apply_acc - sub)/np,
                        1000.0*t_rot_apply_acc/np, 1000.0*t_rot_policy_acc/np, 1000.0*rot_tot/np);
            std::printf("# U10 rotation split (us/swap, %ld swaps; %ld incremental + %ld full audits):"
                        " upload=%.2f alias=%.2f verify=%.2f audit=%.2f | policy=%.2f us/layer-pos\n",
                        n_rot_swap, n_rot_audit_inc, n_rot_audit_full,
                        n_rot_swap ? 1e6*t_rot_upload_acc/n_rot_swap : 0.0,
                        n_rot_swap ? 1e6*t_rot_alias_acc/n_rot_swap : 0.0,
                        n_rot_swap ? 1e6*t_rot_verify_acc/n_rot_swap : 0.0,
                        n_rot_swap ? 1e6*t_rot_audit_acc/n_rot_swap : 0.0,
                        n_total ? 1e6*t_rot_policy_acc/(np*N_LAYERS) : 0.0);
            std::printf("# U10 residual: before=%.4f ms/pos, rotation's measured share=%.4f (%.1f%%),"
                        " after=%.4f ms/pos (%.1f%% of the loop wall)\n",
                        1000.0*resid/np, 1000.0*rot_tot/np, resid > 0 ? 100.0*rot_tot/resid : 0.0,
                        1000.0*(resid - rot_tot)/np, wall > 0 ? 100.0*(resid - rot_tot)/wall : 0.0);
        }
        // U9 — THE PAGE-FAULT LEDGER of the whole run.  The alias store reads FILE-BACKED pages
        // (engine/store.h build_alias_slices), so every expert row the RAM store serves can fault;
        // a fault count is deterministic under load and immune to the box's noise, unlike a time.
        // ru_minflt = a page already in the page cache mapped in (L47: 1.2-1.7 us on this ZFS mmap),
        // ru_majflt = I/O.  Printed here so the fault share of enq_ram can be SUBTRACTED, not argued.
        {
            struct rusage ru; std::memset(&ru, 0, sizeof ru);
            if (getrusage(RUSAGE_SELF, &ru) == 0)
                std::printf("# U9 faults (whole run): minflt=%ld majflt=%ld (%.2f minor+%.2f major faults/pos)"
                            " inblock=%ld oublock=%ld | over %d positions\n",
                            (long) ru.ru_minflt, (long) ru.ru_majflt,
                            (double) ru.ru_minflt/np, (double) ru.ru_majflt/np,
                            (long) ru.ru_inblock, (long) ru.ru_oublock, n_total);
        }
        // U9 — the store SHAPE per position: what the isolated rig's t(c) has to be applied to.
        std::printf("# U9 store shape (/pos): ram_graphs=%.3f ram_experts=%.3f hist=%.3f/%.3f/%.3f/%.3f"
                    " | vram_graphs=%.3f vram_experts=%.3f hist=%.3f/%.3f/%.3f/%.3f\n",
                    (double) n_ram_graphs/np, (double) n_ram_experts/np,
                    (double) n_ram_hist[1]/np, (double) n_ram_hist[2]/np,
                    (double) n_ram_hist[3]/np, (double) n_ram_hist[4]/np,
                    (double) n_vram_graphs/np, (double) n_vram_experts/np,
                    (double) n_vram_hist[1]/np, (double) n_vram_hist[2]/np,
                    (double) n_vram_hist[3]/np, (double) n_vram_hist[4]/np);
        // H3/U1 — the core bracket's own split (see LoopLayer::forward).
        const long n_core_calls = (long) n_total * (long) N_LAYERS;
        std::printf("# U1 core split (s): bracket=%.5f = launch=%.5f + taps=%.5f + wait=%.5f\n",
                    t_core_acc, t_core_launch_acc, t_core_taps_acc,
                    t_core_acc - t_core_launch_acc - t_core_taps_acc);
        std::printf("# U1 core split (us/layer over %ld): bracket=%.2f = launch=%.2f + taps=%.2f + "
                    "wait=%.2f\n", n_core_calls, 1e6*t_core_acc/n_core_calls,
                    1e6*t_core_launch_acc/n_core_calls, 1e6*t_core_taps_acc/n_core_calls,
                    1e6*(t_core_acc - t_core_launch_acc - t_core_taps_acc)/n_core_calls);
        // the same split over the steady-state positions (the CUDA-graph captures are positions 0-2)
        const double ns = n_core_ss ? (double) n_core_ss : 1.0;
        std::printf("# U1 core split STEADY (positions>=3, us/layer over %ld): bracket=%.2f = "
                    "launch=%.2f + taps=%.2f + wait=%.2f\n", n_core_ss,
                    1e6*t_core_acc_ss/ns, 1e6*t_core_launch_ss/ns, 1e6*t_core_taps_ss/ns,
                    1e6*(t_core_acc_ss - t_core_launch_ss - t_core_taps_ss)/ns);
    }
    // H3/U1 — THE CORE PROBE, last: every number the run is judged on has been computed and
    // printed above, so the probe cannot perturb one.
    if (a.core_probe && a.gpu_core) {
        const int ppos = (int) std::min<int64_t>(n_total, kv_max - 4);
        core_probe_run(*m, gpu, kv_max, a.n_res, x, ppos, a.probe_reps, a.probe_gap_us, m4_on, kqkv_on,
                       kvw_on, wvb_on);
        // H3/U3: the attention-chain bisection (the per-kernel-group split).
        attn_bisection_run(*m, gpu, kv_max, a.n_res, x, ppos, a.probe_reps, a.probe_gap_us, m4_on, kqkv_on,
                           kvw_on, wvb_on);
    }
    std::printf("#   (U4 owns tok/s; this number exists only to say what the correctness line cost)\n");

    if (!a.ref_tokens.empty()) {
        // agreement is over the positions that could be compared: min(reference chain, ours).  The
        // denominator is that n, not ours — with --n-decode 128 against an 8-token reference the
        // old print read "126/128", which looks like a near-pass when 2 of the 8 compared differ.
        std::printf("TOKEN LINE: %d/%d identical to the reference's own greedy chain (%d of our %d tokens compared)\n",
                    n_tok_cmp - n_tok_bad, n_tok_cmp, n_tok_cmp, (int) gen.size());
        if (tok_first_div < 0) std::printf("  first divergence  : none\n");
        else                   std::printf("  first divergence  : step %d (ours=%d, reference=%d)\n",
                                            tok_first_div, gen[tok_first_div], ref_tok[tok_first_div]);
        std::printf("  ours      = [");
        for (int i = 0; i < (int) gen.size() && i < 16; ++i) std::printf("%s%d", i ? " " : "", gen[i]);
        std::printf(" ...]\n  reference = [");
        for (int i = 0; i < (int) ref_tok.size() && i < 16; ++i) std::printf("%s%d", i ? " " : "", ref_tok[i]);
        std::printf(" ...]\n");
        if (tok_first_div >= 0) {
            std::printf("FAIL: the token line diverges at step %d (ours=%d, reference=%d)\n",
                        tok_first_div, gen[tok_first_div], ref_tok[tok_first_div]);
            ++n_fail;
        }
        if ((int) ref_tok.size() < a.n_decode)
            std::printf("# INSUFFICIENT: the reference emitted only %d tokens for %d requested (EOG?) — "
                        "it cannot support the full line\n", (int) ref_tok.size(), a.n_decode);
        else if ((int) gen.size() < a.n_decode)
            std::printf("FAIL: the engine produced %d rows for %d requested\n", (int) gen.size(), a.n_decode);
    } else {
        std::printf("TOKEN LINE: not evaluated (no --ref-tokens)\n");
    }

    if (have_ref) {
        std::printf("TIER A (per-layer fingerprints vs the reference's own sums; rel delta of sumsq):\n");
        std::printf("  layer  steps  worst ffn_out     at  worst ffn_inp     worst l_out      cells>%.0e (out/inp/lout)\n", REL_PRINT_FLOOR);
        for (int l = 0; l < (int) N_LAYERS; ++l) {
            const TierAStats & o = t_out[(size_t) l];
            const TierAStats & i = t_inp[(size_t) l];
            const TierAStats & u = t_lout[(size_t) l];
            std::printf("  %5d  %5d  %.3e  %4d  %.3e  %.3e  %d/%d/%d\n", l, o.n_cells, o.worst, o.worst_step,
                        i.worst, u.worst, o.n_floor, i.n_floor, u.n_floor);
        }
        double worst_layer = 0; int worst_layer_at = -1;
        int n_layers_trigger = 0;
        for (int l = 0; l < (int) N_LAYERS; ++l) {
            if (t_out[(size_t) l].worst > worst_layer) { worst_layer = t_out[(size_t) l].worst; worst_layer_at = l; }
            if (t_out[(size_t) l].n_trigger || t_inp[(size_t) l].n_trigger || t_lout[(size_t) l].n_trigger) ++n_layers_trigger;
        }
        std::printf("TIER A worst layer: %d (ffn_out rel sumsq %.6e); layers with a cell above the %.0e "
                    "localisation trigger: %d/%d\n", worst_layer_at, worst_layer, REL_TRIGGER,
                    n_layers_trigger, (int) N_LAYERS);
        std::printf("TIER A logits fingerprint: %d steps compared, worst rel dsumsq %.6e at step %d, "
                    "cells above the %.0e print floor: %d, above the %.0e trigger: %d%s\n",
                    t_logits.n_cells, t_logits.worst, t_logits.worst_step, REL_PRINT_FLOOR,
                    t_logits.n_floor, REL_TRIGGER, t_logits.n_trigger,
                    t_logits.first_trigger >= 0 ? "" : " (no step diverged)");
        std::printf("# the reference prints %%9e: a rel delta <= %.0e is zero at the resolution it can express.\n",
                    REL_PRINT_FLOOR);
    } else {
        std::printf("TIER A: not run (no --ref) — the token line has no external instrument.\n");
    }

    std::printf("VERDICT: %s (%d FAIL lines)\n", n_fail ? "FAIL" : "PASS", n_fail);
    for (LoopLayer & ly : L) { for (Store * s : ly.stores) delete s; }
    layer_own.clear();          // frees the layers (unique_ptr)
    if (gpu) ggml_backend_free(gpu);
    delete m;
    return n_fail ? 1 : 0;
}

// ---------------------------------------------------------------------------
int main(int argc, char ** argv) {
    LayerCheckArgs lc;
    LoopArgs       lp;
    std::string model = DEFAULT_MODEL;
    bool quiet = false, do_layer_check = false, do_loop = false, n_res_given = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto need = [&](const char * opt) -> std::string {
            if (i + 1 >= argc) FATAL("%s needs a value", opt);
            return argv[++i];
        };
        if (a == "--model")            { model = need("--model"); lc.model = model; lp.model = model; }
        else if (a == "--quiet")       quiet = true;
        else if (a == "--layer-check") do_layer_check = true;
        else if (a == "--loop")        do_loop = true;
        else if (a == "--tokens")      { lc.tokens = need("--tokens"); lp.tokens = lc.tokens; }
        else if (a == "--ref")         { lc.ref = need("--ref"); lp.ref = lc.ref; }
        else if (a == "--dump")        { lc.dump = need("--dump"); lp.dump = lc.dump; }
        else if (a == "--route-dump")  lp.route_dump = need("--route-dump");
        else if (a == "--route-perm")  lp.route_perm = need("--route-perm");
        else if (a == "--resident-set") { lp.res_path = need("--resident-set");
                                          lc.res_path = lp.res_path; }
        else if (a == "--ref-pos0")    lc.ref_pos0 = std::atoi(need("--ref-pos0").c_str());
        else if (a == "--kv-max")      { lc.kv_max = std::atoi(need("--kv-max").c_str());
                                         lp.kv_max = lc.kv_max; }
        else if (a == "--router-margin") lc.router_margin = true;
        else if (a == "--layers")      lc.layers = std::atoi(need("--layers").c_str());
        else if (a == "--n-res")       { lc.n_res = std::atoi(need("--n-res").c_str());
                                         lp.n_res = lc.n_res; n_res_given = true; }
        else if (a == "--threads")     { lc.threads = std::atoi(need("--threads").c_str());
                                         lp.threads = lc.threads; }
        else if (a == "--arm-oracle")  { lc.do_oracle = true;  lc.do_store = false; lp.do_store = false; }
        else if (a == "--arm-store")   { lc.do_oracle = false; lc.do_store = true;  lp.do_store = true; }
        else if (a == "--arms-both")   { lc.do_oracle = true;  lc.do_store = true; }
        // ---- U3: the loop + the falsifier ------------------------------------------
        else if (a == "--out")         lp.out = need("--out");
        else if (a == "--ref-tokens")  lp.ref_tokens = need("--ref-tokens");
        else if (a == "--n-decode")    lp.n_decode = std::atoi(need("--n-decode").c_str());
        else if (a == "--falsify")     lp.falsify = true;
        else if (a == "--falsify-layers") lp.f_layers = std::atoi(need("--falsify-layers").c_str());
        else if (a == "--iters")       lp.iters = std::atoi(need("--iters").c_str());
        else if (a == "--warmup")      lp.warmup = std::atoi(need("--warmup").c_str());
        else if (a == "--serialize-both") lp.serialize_both = true;
        else if (a == "--gpu-store")   lp.gpu_store = true;
        else if (a == "--gpu-core")    lp.gpu_core = true;
        else if (a == "--thresh")      lp.thresh = std::atof(need("--thresh").c_str());
        else if (a == "--instr")       lp.instr = true;
        else if (a == "--resident-stats") lp.res_stats = true;
        else if (a == "--extra-stores") lp.extra_stores = std::atoi(need("--extra-stores").c_str());
        else if (a == "--store-pull-at-drain") lp.store_pull_at_drain = true;
        else if (a == "--store-footprint") lp.footprint = true;   // H2/U1: accounting print only
        else if (a == "--alias-prefault") lp.alias_prefault = true;  // U2b: hoist the fault cost
        // U9: the EXPLICIT opposite arm.  Without it an A/B names only the ON side and relies on a
        // flag's ABSENCE for the control — the ON-vs-ON error L60 recorded.  The default is
        // unchanged (OFF): U9 measured that hoisting wins the loop but costs 1.20 s of startup,
        // which is a net loss at the product configuration's 155 positions (see results/m1-h3-u9.md).
        else if (a == "--no-alias-prefault") lp.alias_prefault = false;
        else if (a == "--rotate")          lp.rotate = true;         // U2b: 2Q rotation
        else if (a == "--rotate-verify")   lp.rotate_verify = true;  // U2b: post-loop byte check
        // U10: the refresh audit's scope (default = the swap's own experts) and the 2Q window's
        // threshold (default 2 = the shipped policy).
        else if (a == "--audit-full")      { lp.audit_full = true;  lp.audit_scope_set = true; }
        else if (a == "--audit-incremental") { lp.audit_full = false; lp.audit_scope_set = true; }
        else if (a == "--rot-admit-k")   { lp.rot_admit_k = std::atoi(need("--rot-admit-k").c_str());
                                           lp.rot_admit_k_set = true; }
        else if (a == "--fold-resident") lp.fold_resident = true;
        else if (a == "--fold-slots")   lp.fold_slots = std::atoi(need("--fold-slots").c_str());
        else if (a == "--core-probe")   lp.core_probe = true;   // H3/U1: the post-loop core probe
        else if (a == "--probe-reps")   lp.probe_reps = std::atoi(need("--probe-reps").c_str());
        else if (a == "--probe-gap-us") lp.probe_gap_us = std::atoi(need("--probe-gap-us").c_str());
        else if (a == "--core-fuse")    lp.core_fuse = std::atoi(need("--core-fuse").c_str());
        else if (a == "--core-nodes")   lp.core_nodes = true;
        else if (a == "--m4-wkb")       lp.m4_wkb = 1;      // H3/U4: the wk_b custom-op hatch (explicit)
        else if (a == "--no-m4-wkb")    lp.m4_wkb = 0;      // the stock MMVQ path (pre-H3-U4 bytes)
        else if (a == "--m4-kqkv")      lp.m4_kqkv = 1;     // H3/U5: the fused attention-core hatch (explicit)
        else if (a == "--no-m4-kqkv")   lp.m4_kqkv = 0;     // the reference kq/softmax/kqv sequence
        else if (a == "--m4-kvwrite")   lp.m4_kvwrite = 1;  // H3/U11: the fused KV-write phase (explicit)
        else if (a == "--no-m4-kvwrite") lp.m4_kvwrite = 0; // the reference wkv_a/norm/rope/scatter
        else if (a == "--m4-wvb")       lp.m4_wvb = 1;      // H3/U11: the fused wv_b gemv (explicit)
        else if (a == "--no-m4-wvb")    lp.m4_wvb = 0;      // the reference wv_b mul_mat
        // ---- M1/Q1 — the quality instrument's engine side (both additive, default OFF) ----
        else if (a == "--score")       { lp.score_tokens = need("--score"); do_loop = true; }
        else if (a == "--damage")      { lp.damage_tensor = need("--damage");
                                         const std::string ev = need("--damage");
                                         char * end = nullptr;
                                         lp.damage_eps = std::strtod(ev.c_str(), &end);
                                         // a typo'd eps that parses to 0 would make the damage
                                         // probe a SILENT no-op — refuse instead
                                         if (end == ev.c_str() || *end != '\0')
                                             FATAL("--damage %s: '%s' is not a number",
                                                   lp.damage_tensor.c_str(), ev.c_str()); }
        else FATAL("unknown arg '%s'", a.c_str());
    }

    // H1 — load the resident set ONCE, for both paths, and refuse a CONFLICTING --n-res rather
    // than silently preferring one of them: the set's size IS the resident count, and a caller who
    // passes both must have them agree.
    if (!lp.res_path.empty()) {
        int rn = 0;
        if (!read_hotset_json(lp.res_path.c_str(), lp.res_hot, rn)) return 2;
        lp.res_n = rn;
        if (n_res_given && lp.n_res != rn)
            FATAL("--n-res %d conflicts with --resident-set %s (n_res=%d): the SET is the resident "
                  "count — drop --n-res or pass the matching value", lp.n_res, lp.res_path.c_str(), rn);
        lp.n_res = rn;
        lc.n_res = rn;
        lc.res_hot = lp.res_hot;
    }

    if (do_layer_check) {
        if (lc.tokens.empty()) FATAL("--layer-check needs --tokens <file>");
        return layer_check(lc);
    }
    if (do_loop) {
        if (lp.tokens.empty() && lp.score_tokens.empty())
            FATAL("--loop needs --tokens <file> (or --score <file> for the scoring mode)");
        if (lp.falsify) {
            if (!lp.score_tokens.empty()) FATAL("--score and --falsify are different runs");
            if (!lp.do_store) FATAL("--falsify needs --arm-store (the paired statistic times the "
                                    "PRODUCT store list, not the monolithic op sequence)");
            if (lp.f_layers < 1 || lp.f_layers > (int) N_LAYERS) FATAL("--falsify-layers %d", lp.f_layers);
            if (lp.iters < 1) FATAL("--iters %d would measure nothing", lp.iters);
            if (lp.warmup < 0) FATAL("--warmup %d", lp.warmup);
        } else {
            if (lp.out.empty()) FATAL("--loop needs --out <logits file>");
        }
        return loop_run(lp);
    }

    GGUFModel * m = GGUFModel::load(model.c_str());
    const ModelSurface ms = m->read_surface();
    const SurfaceCheck ck = check_surface(*m, ms);

    const int n_expect = (int) N_LAYERS * N_LAYER_TENSOR + N_GLOBAL_TENSOR;

    std::printf("# M1 skeleton (U1) — loader only: no layer forward, no attention, no KV, no sampler.\n");
    std::printf("# model=%s  bytes=%zu  tensors_in_file=%lld\n",
                m->path.c_str(), m->file_size, (long long) ms.n_tensors_file);
    std::printf("# constants: H=%lld FF=%lld N_EXPERT=%lld N_USED=%lld N_LAYERS=%lld\n",
                (long long) H_DIM, (long long) FF_DIM, (long long) N_EXPERT,
                (long long) N_USED, (long long) N_LAYERS);
    std::printf("# surface: %lld layers x %d tensors + %d globals = %d rows (checklist)\n\n",
                (long long) N_LAYERS, N_LAYER_TENSOR, N_GLOBAL_TENSOR, n_expect);

    if (!quiet) {
        std::printf("# per-layer type table AS READ (one line per layer):\n");
        for (int l = 0; l < (int) N_LAYERS; ++l) {
            std::printf("%2d ", l);
            for (int i = 0; i < N_LAYER_TENSOR; ++i) {
                const TensorInfo & ti = ms.layer[l].t[i];
                std::printf(" %s=%s%s", LAYER_TENSOR_SPECS[i].suffix,
                            ti.present ? ggml_type_name(ti.type) : "ABSENT",
                            tensor_shape_str(ti).c_str());
            }
            std::printf("\n");
        }
        std::printf("\n# globals AS READ:\n");
        for (int i = 0; i < N_GLOBAL_TENSOR; ++i) {
            const TensorInfo & ti = ms.global[i];
            std::printf("#   %-18s %s%s  %zu B\n", GLOBAL_TENSOR_SPECS[i].suffix,
                        ti.present ? ggml_type_name(ti.type) : "ABSENT",
                        tensor_shape_str(ti).c_str(), ti.nbytes);
        }
        std::printf("\n");
    }

    std::printf("# per-(tensor,type) counts over the %lld layers (type -> layers):\n", (long long) N_LAYERS);
    for (int i = 0; i < N_LAYER_TENSOR; ++i) {
        std::map<std::string, int> counts;
        for (int l = 0; l < (int) N_LAYERS; ++l) {
            const TensorInfo & ti = ms.layer[l].t[i];
            ++counts[ti.present ? ggml_type_name(ti.type) : "ABSENT"];
        }
        std::string s;
        for (const auto & kv : counts) {
            if (!s.empty()) s += ", ";
            s += kv.first + " x" + std::to_string(kv.second);
        }
        std::printf("#   %-22s %s\n", LAYER_TENSOR_SPECS[i].suffix, s.c_str());
    }
    std::printf("\n");

    // ---- the check ---------------------------------------------------------------
    std::printf("=== TYPE TABLE CHECK (U1) ===\n");
    std::printf("rows compared        = %d (expected %d)\n", ck.n_rows, n_expect);
    // SEAM 3: the count above is FILE TENSORS accounted for.  A grouped file has no flat
    // logical tensor for a grouped row — its members ARE the file tensors — so say which
    // count this is, or a grouped file reads as a broken one.
    if (ms.n_group_members > 0)
        std::printf("tensors classified   = %d of %lld in the file (%d are group members standing\n"
                    "                       in for %d grouped logical rows: the file holds no flat\n"
                    "                       tensor for those rows, so rows+members > file tensors)\n",
                    ms.n_classified, (long long) ms.n_tensors_file,
                    ms.n_group_members, ck.n_group_rows);
    else
        std::printf("tensors classified   = %d of %lld in the file\n",
                    ms.n_classified, (long long) ms.n_tensors_file);
    std::printf("missing              = %d\n", ck.n_missing);
    std::printf("type mismatch        = %d\n", ck.n_type_bad);
    std::printf("shape mismatch       = %d\n", ck.n_shape_bad);
    std::printf("unclassified tensors = %d\n", ck.n_unclassified);
    std::printf("duplicate rows       = %d\n", ck.n_duplicate);
    if (!ck.fails.empty()) {
        std::printf("--- FAIL lines (up to 64) ---\n");
        for (const std::string & f : ck.fails) std::printf("FAIL: %s\n", f.c_str());
    }
    const bool file_ok = (ms.n_classified == ms.n_tensors_file && ck.n_unclassified == 0);
    const bool ok = ck.ok() && file_ok;
    std::printf("CHECK: %s — %d checklist rows (%lld x %d + %d)%s; %d/%lld FILE TENSORS classified%s\n",
                ok ? "PASS" : "FAIL", n_expect, (long long) N_LAYERS, N_LAYER_TENSOR, N_GLOBAL_TENSOR,
                ck.ok() ? " all match" : " MISMATCH (see FAIL lines)",
                ms.n_classified, (long long) ms.n_tensors_file,
                ms.n_group_members > 0
                    ? " [grouped surface: members counted, their logical rows carry no file tensor]"
                    : " [ungrouped file: one file tensor per checklist row]");
    std::printf("# NOTE: this is a LOADER check only.  Use --layer-check for the U2 layer forward.\n");

    delete m;
    return ok ? 0 : 1;
}
