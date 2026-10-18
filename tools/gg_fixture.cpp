// tools/gg_fixture.cpp — SEAM 3 fixture builder + loader/store exerciser (loader-groups unit).
//
// taillis/layout.md §2 defines the grouped-expert format; no grouped GGUF exists anywhere
// (T2 is design-only), so this tool BUILDS the small synthetic fixtures the loader proofs
// run on, and exercises exactly the engine's own reading path: engine/gguf_model.h
// (GGUFModel::logical — the segment list) and engine/store.h (derive_store_specs + Store
// slices).  One layer (blk.0), real tensor shapes (gate_up [4096,4096,128], down
// [2048,4096,128], gate_inp [4096,128] F32), quantized per group with ggml's own quantizer,
// so the bytes are what a real writer would emit.
//
// Determinism: expert e's master F32 values are regenerated per-expert from a fixed seed
// that does not depend on the fixture, and quantization is row-independent, so experts that
// share a type across two fixtures carry BYTE-IDENTICAL bytes — which makes
// "same type ⇒ bitwise output" a control the tool can actually fail.
//
// Usage:
//   gg_fixture build <outdir> [name...]   write flat / onegroup / mixed3 / g128 / lcmixed
//                                         .gguf fixtures (naming fixtures builds only those —
//                                         g128 alone takes ~20 min of quantization, the others
//                                         need not be redone; lcmixed is mixed3 + the three
//                                         global tensors for m1 --layer-check, U4e)
//   gg_fixture show  <file.gguf>       loader view: segment list + derived store list
//   gg_fixture bytes <a.gguf> <b.gguf> per-EXPERT byte identity (type + nbytes + memcmp) of
//                                      both expert tensors; exit 1 if any expert differs.
//                                      The instrument for "same weights, partitioned N ways".
//   gg_fixture sweep <a.gguf> <b.gguf> [n_res] [nt]
//                                      every expert routed ALONE on both files: isolates the
//                                      per-expert compute from the multi-expert summation
//   gg_fixture moe   <a.gguf> <b.gguf> [n_res] [nt]
//                                      run the MoE block through the store list on both
//                                      files (CPU backend only — no HIP, no gpu.lock),
//                                      print the store/slice lists and the divergence,
//                                      plus per-sampled-expert attribution
//
// CPU-only by construction: links ggml-base + ggml-cpu, never ggml-hip.

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include "gguf_model.h"
#include "store.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// minimal GGUF v3 writer — only what the fixtures need: no KV, or ONE string-array KV
// ---------------------------------------------------------------------------
namespace {

struct W {
    FILE * f;
    void u32(uint32_t v) { std::fwrite(&v, 1, 4, f); }
    void u64(uint64_t v) { std::fwrite(&v, 1, 8, f); }
    void str(const std::string & s) { u64((uint64_t) s.size()); std::fwrite(s.data(), 1, s.size(), f); }
};

struct OutTensor {
    std::string name;
    ggml_type type;
    int64_t ne[3];
    int n_dims;
    std::vector<uint8_t> data;   // quantized bytes, file-ready
    uint64_t off = 0;            // relative to the data section
};

void put_kv_expert_groups(W & w, const std::vector<std::string> & members) {
    w.str(TAILLIS_EXPERT_GROUPS_KEY);
    w.u32((uint32_t) GGUF_TYPE_ARRAY);
    w.u32((uint32_t) GGUF_TYPE_STRING);
    w.u64((uint64_t) members.size());
    for (const std::string & m : members) w.str(m);
}

uint64_t kv_expert_groups_len(const std::vector<std::string> & members) {
    auto slen = [](const std::string & s) { return (uint64_t) s.size() + 8; };
    uint64_t n = slen(TAILLIS_EXPERT_GROUPS_KEY) + 4 + 4 + 8;
    for (const std::string & m : members) n += slen(m);
    return n;
}

// master F32 values for expert `e` of tensor `which` (1 = gate_up, 2 = down, 0 = gate_inp):
// the SAME values in every fixture that carries that expert.
void fill_master(float * dst, int64_t n, int which, int e) {
    std::mt19937 rng((uint32_t) (0x9e3779b9u ^ (uint32_t) (e * 0x85ebca6bu) ^ (uint32_t) (which * 0xc2b2ae35u)));
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    for (int64_t i = 0; i < n; ++i) dst[i] = gauss(rng);
}

struct GroupSpec { int n_exp; ggml_type type; };

struct FixtureSpec {
    const char * name;
    bool grouped;                       // write the key + ".g<k>" member names
    std::vector<GroupSpec> gu, dn;      // in expert order; sums must be N_EXPERT
    bool globals = false;               // also write token_embd/output_norm/output (U4e) so
                                        // m1 --layer-check can reach make_layer_tensor: the
                                        // refusal needs a file that builds embedding + 1 layer
};

// the three global tensors --layer-check needs (m1_engine.cpp's EmbGraph: token_embd q4_K
// [H, n_vocab]; HeadGraph is loop-only but the shapes are kept honest anyway).  n_vocab is
// deliberately small (512): layer_check never runs the head, and a real 131072-row embedding
// would make the fixture 300x bigger for no extra coverage.
//
// PLUS the 14 non-expert tensors of blk.0 (LAYER_TENSOR_SPECS's shapes and types, verbatim):
// M1Layer::build walks the checklist in order and FATALs on the first absent tensor, so a
// file without the full layer never REACHES the expert tensors — and the mixed-type refusal
// lives exactly there.  (gate_inp, idx 10, is written by build_tensors.)
void add_globals(std::vector<OutTensor> & out) {
    const int64_t n_vocab = 512;
    for (int i = 0; i < (int) N_LAYER_TENSOR; ++i) {
        const TensorSpec & sp = LAYER_TENSOR_SPECS[i];
        // the two expert rows (grouped) and gate_inp (already written by build_tensors)
        if (std::strcmp(sp.suffix, "ffn_gate_up_exps.weight") == 0 ||
            std::strcmp(sp.suffix, "ffn_down_exps.weight") == 0 ||
            std::strcmp(sp.suffix, "ffn_gate_inp.weight") == 0) continue;
        OutTensor t;
        t.name = std::string("blk.0.") + sp.suffix;
        t.type = sp.type;
        t.ne[0] = sp.ne[0];
        t.ne[1] = sp.n_ne >= 2 ? sp.ne[1] : 1;
        t.ne[2] = sp.n_ne >= 3 ? sp.ne[2] : 1;
        t.n_dims = sp.n_ne;
        const size_t qrow = ggml_row_size(t.type, (size_t) t.ne[0]);
        size_t n = qrow;
        for (int d = 1; d < t.n_dims; ++d) n *= (size_t) t.ne[d];
        t.data.resize(n);
        if (t.type == GGML_TYPE_F32) {
            fill_master((float *) t.data.data(), (int64_t) (n / 4), 6 + i, 0);
        } else {
            std::vector<float> master((size_t) t.ne[0]);
            std::vector<float> imx((size_t) t.ne[0], 1.0f);
            float * im = ggml_quantize_requires_imatrix(t.type) ? imx.data() : nullptr;
            const int64_t rows = n / qrow;
            for (int64_t r = 0; r < rows; ++r) {
                fill_master(master.data(), t.ne[0], 6 + i, (int) r);
                ggml_quantize_chunk(t.type, master.data(), t.data.data() + (size_t) r * qrow,
                                    0, 1, t.ne[0], im);
            }
        }
        out.push_back(std::move(t));
    }
    {
        OutTensor t;
        t.name = "token_embd.weight";
        t.type = GGML_TYPE_Q4_K;
        t.ne[0] = H_DIM; t.ne[1] = n_vocab; t.ne[2] = 1;
        t.n_dims = 2;
        const size_t qrow = ggml_row_size(t.type, (size_t) H_DIM);
        t.data.resize(qrow * (size_t) n_vocab);
        std::vector<float> master((size_t) H_DIM);
        std::vector<float> imx((size_t) H_DIM, 1.0f);
        for (int64_t r = 0; r < n_vocab; ++r) {
            fill_master(master.data(), H_DIM, 3, (int) r);
            ggml_quantize_chunk(t.type, master.data(), t.data.data() + (size_t) r * qrow,
                                0, 1, H_DIM, imx.data());
        }
        out.push_back(std::move(t));
    }
    {
        OutTensor t;
        t.name = "output_norm.weight";
        t.type = GGML_TYPE_F32;
        t.ne[0] = H_DIM; t.ne[1] = 1; t.ne[2] = 1;
        t.n_dims = 1;
        t.data.resize(sizeof(float) * (size_t) H_DIM);
        fill_master((float *) t.data.data(), H_DIM, 4, 0);
        out.push_back(std::move(t));
    }
    {
        OutTensor t;
        t.name = "output.weight";
        t.type = GGML_TYPE_Q4_K;
        t.ne[0] = H_DIM; t.ne[1] = n_vocab; t.ne[2] = 1;
        t.n_dims = 2;
        const size_t qrow = ggml_row_size(t.type, (size_t) H_DIM);
        t.data.resize(qrow * (size_t) n_vocab);
        std::vector<float> master((size_t) H_DIM);
        std::vector<float> imx((size_t) H_DIM, 1.0f);
        for (int64_t r = 0; r < n_vocab; ++r) {
            fill_master(master.data(), H_DIM, 5, (int) r);
            ggml_quantize_chunk(t.type, master.data(), t.data.data() + (size_t) r * qrow,
                                0, 1, H_DIM, imx.data());
        }
        out.push_back(std::move(t));
    }
}

std::vector<OutTensor> build_tensors(const FixtureSpec & fs, const char * layer_prefix,
                                     std::vector<std::string> * key_members) {
    std::vector<OutTensor> out;
    if (fs.globals) add_globals(out);
    std::vector<float> master((size_t) H_DIM * (2 * FF_DIM));   // 64 MB, refilled per expert
    std::vector<float> imx((size_t) H_DIM, 1.0f);               // neutral imatrix (per row, len ne[0])

    auto add_block = [&](const char * suffix, int64_t ne0, int64_t ne1,
                         const std::vector<GroupSpec> & groups, int which) {
        int e = 0;
        for (size_t g = 0; g < groups.size(); ++g) {
            const GroupSpec & gs = groups[g];
            OutTensor t;
            const std::string gsuf = fs.grouped ? ".g" + std::to_string(g) : "";
            t.name = std::string(layer_prefix) + suffix + gsuf + ".weight";
            t.type = gs.type;
            t.ne[0] = ne0; t.ne[1] = ne1; t.ne[2] = gs.n_exp;
            t.n_dims = 3;
            const size_t qrow = ggml_row_size(gs.type, (size_t) ne0);
            t.data.resize(qrow * (size_t) ne1 * (size_t) gs.n_exp);
            if (fs.grouped && key_members) {
                key_members->push_back(t.name.substr(0, t.name.size() - 7));   // key names members (no .weight)
            }
            float * im = ggml_quantize_requires_imatrix(gs.type) ? imx.data() : nullptr;
            for (int k = 0; k < gs.n_exp; ++k, ++e) {
                fill_master(master.data(), ne0 * ne1, which, e);
                // one call per expert: src is the expert-local matrix, start 0 is valid,
                // and row-independence makes the bytes group-placement-independent.
                ggml_quantize_chunk(gs.type, master.data(), t.data.data() + (size_t) k * qrow * (size_t) ne1,
                                    0, ne1, ne0, im);
            }
            out.push_back(std::move(t));
        }
    };

    {
        OutTensor gi;
        gi.name = std::string(layer_prefix) + "ffn_gate_inp.weight";
        gi.type = GGML_TYPE_F32;
        gi.ne[0] = H_DIM; gi.ne[1] = N_EXPERT; gi.ne[2] = 1;
        gi.n_dims = 2;
        gi.data.resize(sizeof(float) * (size_t) H_DIM * N_EXPERT);
        fill_master((float *) gi.data.data(), (int64_t) H_DIM * N_EXPERT, 0, 0);
        out.push_back(std::move(gi));
    }
    add_block("ffn_gate_up_exps", H_DIM, 2 * FF_DIM, fs.gu, 1);
    add_block("ffn_down_exps",    FF_DIM, H_DIM,      fs.dn, 2);
    return out;
}

void write_fixture(const char * outdir, const FixtureSpec & fs) {
    std::vector<std::string> key_members;
    std::vector<OutTensor> ts = build_tensors(fs, "blk.0.", fs.grouped ? &key_members : nullptr);

    char path[512];
    std::snprintf(path, sizeof path, "%s/%s.gguf", outdir, fs.name);
    FILE * f = std::fopen(path, "wb");
    if (!f) { std::fprintf(stderr, "FATAL: cannot write %s\n", path); std::exit(1); }

    // ---- info section, serialized to memory so the data start is known ---------------
    std::vector<uint8_t> info_buf;
    auto iu32 = [&](uint32_t v) { const uint8_t * p = (const uint8_t *) &v; info_buf.insert(info_buf.end(), p, p + 4); };
    auto iu64 = [&](uint64_t v) { const uint8_t * p = (const uint8_t *) &v; info_buf.insert(info_buf.end(), p, p + 8); };
    auto istr = [&](const std::string & s) { iu64((uint64_t) s.size()); const uint8_t * p = (const uint8_t *) s.data(); info_buf.insert(info_buf.end(), p, p + s.size()); };

    const uint64_t align = GGUF_DEFAULT_ALIGNMENT;
    uint64_t off = 0;
    for (OutTensor & t : ts) {
        off = (off + align - 1) / align * align;
        t.off = off;
        off += t.data.size();
        istr(t.name);
        iu32((uint32_t) t.n_dims);
        for (int d = 0; d < t.n_dims; ++d) iu64((uint64_t) t.ne[d]);
        iu32((uint32_t) t.type);
        iu64(t.off);
    }
    const uint64_t pos_after_header = 4 + 4 + 8 + 8;                       // magic, version, n_tensors, n_kv
    const uint64_t kv_len = fs.grouped ? kv_expert_groups_len(key_members) : 0;
    uint64_t data_start = pos_after_header + kv_len + info_buf.size();
    data_start = (data_start + align - 1) / align * align;

    // ---- emit ------------------------------------------------------------------------
    W w{f};
    w.u32(0x46554747);              // "GGUF" magic (LE)
    w.u32(3);                       // version
    w.u64((uint64_t) ts.size());
    w.u64(fs.grouped ? 1 : 0);      // n_kv
    if (fs.grouped) put_kv_expert_groups(w, key_members);
    std::fwrite(info_buf.data(), 1, info_buf.size(), f);
    for (long cur = ftell(f), p = cur; p < (long) data_start; ++p) fputc(0, f);
    for (const OutTensor & t : ts) {
        long cur = ftell(f);
        long want = (long) (data_start + t.off);
        for (long p = cur; p < want; ++p) fputc(0, f);
        std::fwrite(t.data.data(), 1, t.data.size(), f);
    }
    std::fclose(f);
    std::printf("wrote %s (%zu tensors, grouped=%d, %zu key members, %llu bytes)\n",
                path, ts.size(), (int) fs.grouped, key_members.size(),
                (unsigned long long) (data_start + off));
}

// ---------------------------------------------------------------------------
// fixture set — layout.md §2's example split is mixed3's (96/24/8, IQ2_XXS/IQ3_S/IQ4_XS)
// ---------------------------------------------------------------------------
const FixtureSpec FLAT_SPEC     = { "flat",     false,
                                    { { 128, GGML_TYPE_IQ2_XXS } },
                                    { { 128, GGML_TYPE_IQ3_S } } };
const FixtureSpec ONEGROUP_SPEC = { "onegroup", true,
                                    { { 128, GGML_TYPE_IQ2_XXS } },
                                    { { 128, GGML_TYPE_IQ3_S } } };
// rule (c): G=3 with DIFFERENT types per group, layout.md §2's own example split
const FixtureSpec MIXED3_SPEC   = { "mixed3",   true,
                                    { { 96, GGML_TYPE_IQ2_XXS }, { 24, GGML_TYPE_IQ3_S }, { 8, GGML_TYPE_IQ4_XS } },
                                    { { 96, GGML_TYPE_IQ3_S },  { 24, GGML_TYPE_IQ2_S }, { 8, GGML_TYPE_IQ4_XS } } };
// U4e: mixed3's expert plan PLUS the three global tensors, so `m1 --layer-check` can reach
// make_layer_tensor and the ORACLE arm's mixed-type refusal is exercisable on a real build
// path (mixed3 alone has no token_embd, so the embedding graph fails first).  The STORE arm
// on this same file must LOAD (the layer skips its expert copies; the stores slice per
// segment) — that is the deliberate asymmetry the unit exists to prove.
const FixtureSpec LCMIXED_SPEC  = { "lcmixed",  true,
                                    { { 96, GGML_TYPE_IQ2_XXS }, { 24, GGML_TYPE_IQ3_S }, { 8, GGML_TYPE_IQ4_XS } },
                                    { { 96, GGML_TYPE_IQ3_S },  { 24, GGML_TYPE_IQ2_S }, { 8, GGML_TYPE_IQ4_XS } },
                                    true };

std::vector<FixtureSpec> all_fixtures() {
    std::vector<FixtureSpec> out = { FLAT_SPEC, ONEGROUP_SPEC, MIXED3_SPEC, LCMIXED_SPEC };
    // no special-case G: 128 groups of one expert (layout.md §6's per-expert layout).
    // The types are deliberately FLAT's types: g128 is then the flat file's same
    // masters/weights partitioned 128 ways, so the flat-vs-g128 comparison is BITWISE and
    // the only thing that varies is G.  (Mixed types at G>1 are mixed3's job.)
    FixtureSpec g128 = FLAT_SPEC;
    g128.name = "g128";
    g128.grouped = true;
    std::vector<GroupSpec> gu, dn;
    for (int e = 0; e < (int) N_EXPERT; ++e) {
        gu.push_back({ 1, FLAT_SPEC.gu[0].type });
        dn.push_back({ 1, FLAT_SPEC.dn[0].type });
    }
    g128.gu = gu;
    g128.dn = dn;
    out.push_back(g128);
    return out;
}

void build_all(const char * outdir, const std::vector<std::string> & only) {
    for (const FixtureSpec & fs : all_fixtures()) {
        if (!only.empty() && std::find(only.begin(), only.end(), fs.name) == only.end()) continue;
        write_fixture(outdir, fs);
    }
}

}   // namespace

// ---------------------------------------------------------------------------
// per-EXPERT byte identity between two logical tensors (SEAM 3)
//
// The proof that "g128 is flat's weights partitioned 128 ways" is a byte claim, so it must be
// tested in bytes, not through float arithmetic: two different store PARTITIONS of the same
// bytes give different summation orders, and float addition is not associative.  This walks
// every expert of both tensors and compares the expert's own slice (type, nbytes, memcmp)
// using Store's own slice_src — the same accessor the stores use — so a segment-math bug
// cannot hide behind a matching summary.
// ---------------------------------------------------------------------------
struct ExpertBytes { int n_exp = 0, n_cmp = 0, n_type_diff = 0, n_size_diff = 0, n_byte_diff = 0;
                     int first_diff = -1; };

static ExpertBytes expert_bytes_cmp(const LogicalTensor & a, const LogicalTensor & b) {
    ExpertBytes r;
    r.n_exp = (int) N_EXPERT;
    for (int e = 0; e < r.n_exp; ++e) {
        MatSrc sa = slice_src(a, e, e + 1);
        MatSrc sb = slice_src(b, e, e + 1);
        ++r.n_cmp;
        if (sa.type != sb.type) { ++r.n_type_diff; if (r.first_diff < 0) r.first_diff = e; continue; }
        if (sa.nbytes != sb.nbytes) { ++r.n_size_diff; if (r.first_diff < 0) r.first_diff = e; continue; }
        if (std::memcmp(sa.base, sb.base, sa.nbytes) != 0) { ++r.n_byte_diff; if (r.first_diff < 0) r.first_diff = e; }
    }
    return r;
}

static void expert_bytes_print(const char * what, const ExpertBytes & r) {
    const int n_same = r.n_cmp - r.n_type_diff - r.n_size_diff - r.n_byte_diff;
    std::printf("    %-4s %d/%d expert slices byte-identical (type + nbytes + memcmp)\n",
                what, n_same, r.n_cmp);
    if (n_same != r.n_cmp)
        std::printf("         differing: %d type, %d size, %d bytes; first differing expert: %d\n",
                    r.n_type_diff, r.n_size_diff, r.n_byte_diff, r.first_diff);
    else
        std::printf("         (every expert of both tensors carries the same bytes)\n");
}

// ---------------------------------------------------------------------------
// show / moe
// ---------------------------------------------------------------------------
static void print_segments(const LogicalTensor & lt) {
    std::printf("  %-34s ne=[%lld,%lld,%lld]  %zu segment(s):\n", lt.name.c_str(),
                (long long) lt.ne[0], (long long) lt.ne[1], (long long) lt.ne[2], lt.segs.size());
    for (const Segment & s : lt.segs)
        std::printf("    [%3d,%3d)  %-8s  %10zu B\n", s.lo, s.hi, ggml_type_name(s.type), s.nbytes);
}

static int cmd_show(const char * path) {
    GGUFModel * m = GGUFModel::load(path);
    std::printf("# %s: %lld tensors, %zu grouped logical tensor(s)\n", path,
                (long long) m->n_tensors(), m->expert_groups.size());
    for (const auto & kv : m->expert_groups)
        std::printf("  group %s <- %zu member(s)\n", kv.first.c_str(), kv.second.size());
    LogicalTensor gu = m->logical("blk.0.ffn_gate_up_exps.weight");
    LogicalTensor dn = m->logical("blk.0.ffn_down_exps.weight");
    print_segments(gu);
    print_segments(dn);
    const int n_res = 64;
    std::vector<StoreSpec> specs = derive_store_specs(gu, dn, n_res, {}, false);
    std::printf("  derived store list (n_res=%d): %zu store(s)\n", n_res, specs.size());
    for (const StoreSpec & s : specs)
        std::printf("    store [%3d,%3d) %s\n", s.lo, s.hi, s.loc == Store::VRAM ? "VRAM" : "RAM");
    delete m;
    return 0;
}

// routing identical on both fixtures: gate_inp bytes are asserted equal, ids/w computed once
static void route(const float * gate_inp, const float * x, int nt,
                  std::vector<int32_t> & ids, std::vector<float> & wgt) {
    ids.resize((size_t) N_USED * nt);
    wgt.resize((size_t) N_USED * nt);
    std::vector<float> logits((size_t) N_EXPERT);
    for (int t = 0; t < nt; ++t) {
        for (int e = 0; e < (int) N_EXPERT; ++e) {
            float s = 0;
            for (int64_t j = 0; j < H_DIM; ++j) s += gate_inp[j + H_DIM * (int64_t) e] * x[j + H_DIM * (int64_t) t];
            logits[(size_t) e] = s;
        }
        float mx = *std::max_element(logits.begin(), logits.end());
        for (float & l : logits) l = std::exp(l - mx);
        std::vector<int> order((size_t) N_EXPERT);
        for (int e = 0; e < (int) N_EXPERT; ++e) order[(size_t) e] = e;
        std::stable_sort(order.begin(), order.end(),
                         [&](int a, int b) { return logits[(size_t) a] > logits[(size_t) b]; });
        float wsum = 0;
        for (int k = 0; k < (int) N_USED; ++k) wsum += std::max(logits[(size_t) order[(size_t) k]], 6.103515625e-5f);
        for (int k = 0; k < (int) N_USED; ++k) {
            ids[(size_t) k + N_USED * (size_t) t] = order[(size_t) k];
            wgt[(size_t) k + N_USED * (size_t) t] = std::max(logits[(size_t) order[(size_t) k]], 6.103515625e-5f) / wsum;
        }
    }
}

static void moe_forward(GGUFModel * m, const std::vector<int32_t> & ids,
                        const std::vector<float> & wgt, const std::vector<float> & x,
                        int nt, int n_res, std::vector<float> & y,
                        std::vector<std::string> * slice_lines) {
    LogicalTensor gu = m->logical("blk.0.ffn_gate_up_exps.weight");
    LogicalTensor dn = m->logical("blk.0.ffn_down_exps.weight");
    ggml_backend_t cpu = ggml_backend_cpu_init();
    if (!cpu) FATAL("cpu backend init failed");
    ggml_backend_cpu_set_n_threads(cpu, 8);   // polite: never grab the whole host (U4c's window)
    std::vector<Store *> stores;
    std::vector<StoreSpec> specs = derive_store_specs(gu, dn, n_res, {}, false);
    for (const StoreSpec & s : specs) {
        Store * st = new Store();
        st->build(Store::RAM, gu, dn, s.lo, s.hi, cpu, nt);   // all-RAM: CPU buffers only
        if (slice_lines)
            for (const Slice & sl : st->slices) {
                char b[256];
                std::snprintf(b, sizeof b, "store [%3d,%3d) slice [%3d,%3d) gu=%-8s d=%-8s",
                              s.lo, s.hi, sl.lo, sl.hi, ggml_type_name(sl.gu_type), ggml_type_name(sl.d_type));
                slice_lines->push_back(b);
            }
        stores.push_back(st);
    }
    for (Store * st : stores) st->enqueue(ids.data(), wgt.data(), x.data(), nt);
    y.assign((size_t) H_DIM * nt, 0.0f);
    for (Store * st : stores) st->drain(y.data(), nt, false, nullptr, nullptr);
    for (Store * st : stores) delete st;
    ggml_backend_free(cpu);
}

static int cmd_bytes(const char * pa, const char * pb) {
    GGUFModel * ma = GGUFModel::load(pa);
    GGUFModel * mb = GGUFModel::load(pb);
    std::printf("=== EXPERT-BY-EXPERT BYTES: %s vs %s\n", pa, pb);
    int n_diff = 0;
    for (int which = 0; which < 2; ++which) {
        const char * nm = which ? "blk.0.ffn_down_exps.weight" : "blk.0.ffn_gate_up_exps.weight";
        LogicalTensor la = ma->logical(nm);
        LogicalTensor lb = mb->logical(nm);
        std::printf("  %s: %zu segment(s) vs %zu\n", nm, la.segs.size(), lb.segs.size());
        ExpertBytes r = expert_bytes_cmp(la, lb);
        expert_bytes_print(which ? "dn" : "gu", r);
        n_diff += r.n_type_diff + r.n_size_diff + r.n_byte_diff;
    }
    std::printf("=== EXPERT BYTES VERDICT: %s\n",
                n_diff == 0 ? "every expert carries the same bytes in both files (type + nbytes + memcmp)"
                            : "DIFFERING EXPERT BYTES — the partition claim fails");
    delete ma;
    delete mb;
    return n_diff == 0 ? 0 : 1;
}

// every expert routed ALONE, both files: isolates the per-expert COMPUTE from the multi-expert
// summation, which is the one thing a different store partition changes.
static int cmd_sweep(const char * pa, const char * pb, int n_res, int nt) {
    GGUFModel * ma = GGUFModel::load(pa);
    GGUFModel * mb = GGUFModel::load(pb);
    std::mt19937 rng(1234);
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    std::vector<float> x((size_t) H_DIM * nt);
    for (auto & v : x) v = gauss(rng);
    int n_nonzero = 0;
    double worst = 0;
    int worst_e = -1;
    std::printf("=== SWEEP (routed ALONE, weight 0.25): %s vs %s\n", pa, pb);
    for (int e = 0; e < (int) N_EXPERT; ++e) {
        std::vector<int32_t> ids((size_t) N_USED, e);
        std::vector<float> w((size_t) N_USED, 0.25f);
        std::vector<float> y1, y2;
        moe_forward(ma, ids, w, x, nt, n_res, y1, nullptr);
        moe_forward(mb, ids, w, x, nt, n_res, y2, nullptr);
        double d = 0;
        size_t nd = 0;
        double rmax = 0, rmax_ya = 0, big_ya = 0;
        for (size_t i = 0; i < y1.size(); ++i) {
            double dd = std::fabs((double) y1[i] - (double) y2[i]);
            if (dd > d) d = dd;
            if (y1[i] != y2[i]) {
                ++nd;
                const double ay = std::fabs((double) y1[i]);
                if (ay > big_ya) big_ya = ay;
                double r = dd / (ay > 0 ? ay : 1.0);
                if (r > rmax) { rmax = r; rmax_ya = ay; }
            }
        }
        if (nd) ++n_nonzero;
        if (d > worst) { worst = d; worst_e = e; }
        if (nd)
            std::printf("  expert %3d: cells %zu/%zu differ; max|dy| = %.6e; worst relative = %.6e at\n"
                        "              |y| = %.6e; max |y| among differing cells = %.6e\n",
                        e, nd, y1.size(), d, rmax, rmax_ya, big_ya);
    }
    std::printf("=== SWEEP VERDICT: %d/%d experts bitwise when routed alone; worst max|dy| = %.6e (expert %d)\n",
                (int) N_EXPERT - n_nonzero, (int) N_EXPERT, worst, worst_e);
    delete ma;
    delete mb;
    return n_nonzero == 0 ? 0 : 1;
}

static int cmd_moe(const char * pa, const char * pb, int n_res, int nt) {
    GGUFModel * ma = GGUFModel::load(pa);
    GGUFModel * mb = GGUFModel::load(pb);
    size_t gi_n_a = 0, gi_n_b = 0;
    const float * gi_a = ma->f32_tensor("blk.0.ffn_gate_inp.weight", &gi_n_a);
    const float * gi_b = mb->f32_tensor("blk.0.ffn_gate_inp.weight", &gi_n_b);
    CHECK(gi_n_a == gi_n_b && std::memcmp(gi_a, gi_b, gi_n_a * sizeof(float)) == 0,
          "gate_inp differs between the fixtures — routing would not be comparable");

    std::mt19937 rng(1234);
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    std::vector<float> x((size_t) H_DIM * nt);
    for (auto & v : x) v = gauss(rng);

    std::vector<int32_t> ids;
    std::vector<float> wgt;
    route(gi_a, x.data(), nt, ids, wgt);
    std::printf("# routed experts (top-4, nt=%d):", nt);
    for (int t = 0; t < nt; ++t) {
        std::printf(" [");
        for (int k = 0; k < (int) N_USED; ++k)
            std::printf("%s%d", k ? "," : "", ids[(size_t) k + N_USED * (size_t) t]);
        std::printf("]");
    }
    std::printf("\n");

    std::printf("=== A = %s ===\n", pa);
    std::vector<std::string> slices_a;
    std::vector<float> ya;
    moe_forward(ma, ids, wgt, x, nt, n_res, ya, &slices_a);
    for (const std::string & s : slices_a) std::printf("  %s\n", s.c_str());

    std::printf("=== B = %s ===\n", pb);
    std::vector<std::string> slices_b;
    std::vector<float> yb;
    moe_forward(mb, ids, wgt, x, nt, n_res, yb, &slices_b);
    for (const std::string & s : slices_b) std::printf("  %s\n", s.c_str());

    double maxd = 0, maxa = 0;
    size_t n_diff = 0;
    double maxrel = 0, maxrel_ya = 0, maxdiff_ya = 0;
    for (size_t i = 0; i < ya.size(); ++i) {
        double d = std::fabs((double) ya[i] - (double) yb[i]);
        if (d > maxd) maxd = d;
        if (std::fabs((double) ya[i]) > maxa) maxa = std::fabs((double) ya[i]);
        if (ya[i] != yb[i]) {
            ++n_diff;
            if (std::fabs((double) ya[i]) > maxdiff_ya) maxdiff_ya = std::fabs((double) ya[i]);
            double r = d / (std::fabs((double) ya[i]) > 0 ? std::fabs((double) ya[i]) : 1.0);
            if (r > maxrel) { maxrel = r; maxrel_ya = std::fabs((double) ya[i]); }
        }
    }
    std::printf("=== DIVERGENCE A vs B: cells %zu/%zu differ; max|d| = %.6e; max|y| = %.6e; rel = %.6e\n",
                n_diff, ya.size(), maxd, maxa, maxd / (maxa > 0 ? maxa : 1));
    // the SHAPE of the divergence, which is what separates a rounding-order difference from a
    // wrong-weight one: where do the differing cells SIT?  A difference confined to cells whose
    // own |y| is at the small end is cancellation on near-zero outputs; a difference on the
    // LARGE cells is an error in the arithmetic itself.
    if (n_diff)
        std::printf("=== DIVERGENCE SHAPE: worst per-cell relative difference = %.6e (at |y_a| = %.6e);\n"
                    "    max |y_a| among the %zu differing cells = %.6e (overall max|y| = %.6e)\n",
                    maxrel, maxrel_ya, n_diff, maxdiff_ya, maxa);

    // loaded-bytes identity: when the two files carry the same segment list shape, memcmp
    // every pair — the rule-2 one-group fallback must be byte-identical AT REST too.  When the
    // segment lists differ in SHAPE (g128 vs flat: 128 segments vs 1), the segment-wise memcmp
    // has nothing to compare — but the claim under test is about the WEIGHTS, not the segment
    // layout, so compare PER EXPERT instead (each expert's own slice bytes, via slice_src).
    LogicalTensor gu_a = ma->logical("blk.0.ffn_gate_up_exps.weight");
    LogicalTensor dn_a = ma->logical("blk.0.ffn_down_exps.weight");
    LogicalTensor gu_b = mb->logical("blk.0.ffn_gate_up_exps.weight");
    LogicalTensor dn_b = mb->logical("blk.0.ffn_down_exps.weight");
    if (gu_a.segs.size() == gu_b.segs.size() && dn_a.segs.size() == dn_b.segs.size()) {
        bool all_eq = true;
        for (size_t i = 0; i < gu_a.segs.size() && all_eq; ++i)
            all_eq = gu_a.segs[i].type == gu_b.segs[i].type &&
                     gu_a.segs[i].nbytes == gu_b.segs[i].nbytes &&
                     std::memcmp(gu_a.segs[i].bytes, gu_b.segs[i].bytes, gu_a.segs[i].nbytes) == 0;
        for (size_t i = 0; i < dn_a.segs.size() && all_eq; ++i)
            all_eq = dn_a.segs[i].type == dn_b.segs[i].type &&
                     dn_a.segs[i].nbytes == dn_b.segs[i].nbytes &&
                     std::memcmp(dn_a.segs[i].bytes, dn_b.segs[i].bytes, dn_a.segs[i].nbytes) == 0;
        std::printf("=== LOADED BYTES: %zu gu + %zu dn segment(s) pairwise %s\n",
                    gu_a.segs.size(), dn_a.segs.size(), all_eq ? "IDENTICAL (type, nbytes, memcmp)" : "DIFFER");
    } else {
        std::printf("=== LOADED BYTES: segment lists differ in SHAPE (%zu vs %zu, %zu vs %zu) — no\n"
                    "    segment-wise memcmp is possible; compared PER EXPERT instead:\n",
                    gu_a.segs.size(), gu_b.segs.size(), dn_a.segs.size(), dn_b.segs.size());
        ExpertBytes eg = expert_bytes_cmp(gu_a, gu_b);
        ExpertBytes ed = expert_bytes_cmp(dn_a, dn_b);
        expert_bytes_print("gu", eg);
        expert_bytes_print("dn", ed);
        const int same = (eg.n_cmp - eg.n_type_diff - eg.n_size_diff - eg.n_byte_diff) +
                         (ed.n_cmp - ed.n_type_diff - ed.n_size_diff - ed.n_byte_diff);
        std::printf("=== LOADED BYTES (per expert): %d/%d expert slices byte-identical\n",
                    same, eg.n_cmp + ed.n_cmp);
    }
    auto seg_type = [](const LogicalTensor & lt, int e) {
        for (const Segment & s : lt.segs) if (s.lo <= e && e < s.hi) return ggml_type_name(s.type);
        return "?";
    };
    for (int e : { 0, 100, 124 }) {
        std::vector<int32_t> ids1((size_t) N_USED, e);
        std::vector<float> w1((size_t) N_USED, 0.25f);
        std::vector<float> y1, y2;
        moe_forward(ma, ids1, w1, x, nt, n_res, y1, nullptr);
        moe_forward(mb, ids1, w1, x, nt, n_res, y2, nullptr);
        double d = 0;
        for (size_t i = 0; i < y1.size(); ++i) d = std::max(d, (double) std::fabs(y1[i] - y2[i]));
        const bool same = std::string(seg_type(gu_a, e)) == seg_type(gu_b, e) &&
                          std::string(seg_type(dn_a, e)) == seg_type(dn_b, e);
        std::printf("  expert %3d: gate_up %s vs %s, down %s vs %s -> max|dy| = %.6e%s\n",
                    e, seg_type(gu_a, e), seg_type(gu_b, e), seg_type(dn_a, e), seg_type(dn_b, e), d,
                    same ? "  (same types: BITWISE expected)" : "");
    }
    delete ma;
    delete mb;
    return 0;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: gg_fixture build <outdir> | show <file.gguf> | "
                             "moe <a.gguf> <b.gguf> [n_res] [nt]\n");
        return 1;
    }
    const std::string cmd = argv[1];
    if (cmd == "build" && argc >= 3) {
        std::vector<std::string> only(argv + 3, argv + argc);   // optional fixture names
        build_all(argv[2], only);
        ggml_quantize_free();
        return 0;
    }
    if (cmd == "show" && argc == 3) return cmd_show(argv[2]);
    if (cmd == "bytes" && argc == 4) return cmd_bytes(argv[2], argv[3]);
    if (cmd == "sweep" && argc >= 4)
        return cmd_sweep(argv[2], argv[3], argc > 4 ? std::atoi(argv[4]) : 64,
                         argc > 5 ? std::atoi(argv[5]) : 2);
    if (cmd == "moe" && argc >= 4)
        return cmd_moe(argv[2], argv[3], argc > 4 ? std::atoi(argv[4]) : 64,
                       argc > 5 ? std::atoi(argv[5]) : 2);
    std::fprintf(stderr, "unknown invocation\n");
    return 1;
}
