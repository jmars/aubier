// engine/gguf_model.h — the GGUF reading machinery + the FULL per-layer tensor surface (M1/U1).
//
// EXTRACTED VERBATIM from engine/moe_split.cpp (M0) — Segment / LogicalTensor / GGUFModel and
// the tensor-table parse — so that the M0 split primitive and the M1 loop read the model through
// ONE reader.  Mistral Small 4 ONLY, hardcoded, no generality (user directive 2026-10-11).
//
// EXTENDED (M1/U1) to the whole per-layer surface: all 16 per-layer tensors and the 3 globals
// (token_embd, output_norm, output).  The type is carried PER TENSOR AND PER LAYER, because the
// types differ across tensors (F32 / Q5_K / Q8_0 / Q6_K / IQ2_XXS / IQ2_S / IQ3_S / Q4_K) AND
// across layers (ffn_down_exps is IQ2_S on 32 layers and IQ3_S on layers 4, 28, 32, 35, MEASURED
// by parsing the GGUF — see LAYER_TENSOR_SPECS).  No arm may assume one type or one kernel.
//
// SEAM 3 (loader/type axis, engine-plan.md §3) is PRESERVED, not closed: weights are still
// ordered lists of (expert_lo, expert_hi, TYPE, bytes) SEGMENTS populated from ggml's loader
// (one segment per tensor today).  The surface below carries the type per tensor so that a
// streamed or grouped store stays a POLICY VARIANT over the store list, not a rewrite.
//
// This header is metadata-only: it parses the tensor table from the mmap'd file and never reads
// tensor data (gguf_init_from_buffer with no_alloc = true).

#pragma once

#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

// ---------------------------------------------------------------------------
// Fixed constants — Mistral Small 4, MEASURED (docs/TARGET-ARCHITECTURE.md §12.2,
// re-parsed from the GGUF in M1/U1)
// ---------------------------------------------------------------------------
static constexpr int64_t H_DIM     = 4096;   // model dim
static constexpr int64_t FF_DIM    = 2048;   // per-expert FF width
static constexpr int64_t N_EXPERT  = 128;
static constexpr int64_t N_USED    = 4;      // routed experts per token
static constexpr int64_t N_LAYERS  = 36;

#define FATAL(...) do { \
    std::fprintf(stderr, "FATAL: " __VA_ARGS__); \
    std::fprintf(stderr, "\n"); \
    std::exit(1); \
} while (0)

#define CHECK(cond, ...) do { if (!(cond)) FATAL(__VA_ARGS__); } while (0)

// ---------------------------------------------------------------------------
// The M1 tensor surface — 36 layers x 16 tensors + 3 globals = 579 (MEASURED)
// ---------------------------------------------------------------------------
// ne[] is the GGUF/ggml dimension order (ne[0] = fastest / the quant block axis).
// type_except != GGML_TYPE_COUNT names the layers whose type differs from the default.
struct TensorSpec {
    const char * suffix;                 // per-layer: name = "blk.<l>.<suffix>"; global: full name
    ggml_type    type;
    int64_t      ne[3];
    int          n_ne;
    ggml_type    type_except;            // GGML_TYPE_COUNT => no layer exception
    const int *  except_layers;
    int          n_except;

    bool is_except_layer(int layer) const {
        for (int i = 0; i < n_except; ++i) if (except_layers[i] == layer) return true;
        return false;
    }
    ggml_type type_for_layer(int layer) const {
        return is_except_layer(layer) ? type_except : type;
    }
};

// The layers whose ffn_down_exps is IQ3_S (MEASURED by parsing the GGUF: 32 layers are IQ2_S).
static const int DOWN_EXCEPT_LAYERS[] = { 4, 28, 32, 35 };

// attn_k_b / attn_v_b are 3-D: [64,256,32] and [256,128,32] (the 32 = head_count).
static const TensorSpec LAYER_TENSOR_SPECS[] = {
    { "attn_norm.weight",     GGML_TYPE_F32,      { 4096, 0,    0 },   1, GGML_TYPE_COUNT, nullptr, 0 },
    { "attn_q_a.weight",      GGML_TYPE_Q5_K,     { 4096, 1024, 0 },   2, GGML_TYPE_COUNT, nullptr, 0 },
    { "attn_q_a_norm.weight", GGML_TYPE_F32,      { 1024, 0,    0 },   1, GGML_TYPE_COUNT, nullptr, 0 },
    { "attn_q_b.weight",      GGML_TYPE_Q8_0,     { 1024, 4096, 0 },   2, GGML_TYPE_COUNT, nullptr, 0 },
    { "attn_kv_a_mqa.weight", GGML_TYPE_Q8_0,     { 4096, 320,  0 },   2, GGML_TYPE_COUNT, nullptr, 0 },
    { "attn_kv_a_norm.weight",GGML_TYPE_F32,      { 256,  0,    0 },   1, GGML_TYPE_COUNT, nullptr, 0 },
    { "attn_k_b.weight",      GGML_TYPE_Q8_0,     { 64,   256,  32 },  3, GGML_TYPE_COUNT, nullptr, 0 },
    { "attn_v_b.weight",      GGML_TYPE_Q8_0,     { 256,  128,  32 },  3, GGML_TYPE_COUNT, nullptr, 0 },
    { "attn_output.weight",   GGML_TYPE_Q5_K,     { 4096, 4096, 0 },   2, GGML_TYPE_COUNT, nullptr, 0 },
    { "ffn_norm.weight",      GGML_TYPE_F32,      { 4096, 0,    0 },   1, GGML_TYPE_COUNT, nullptr, 0 },
    { "ffn_gate_inp.weight",  GGML_TYPE_F32,      { 4096, 128,  0 },   2, GGML_TYPE_COUNT, nullptr, 0 },
    { "ffn_gate_up_exps.weight", GGML_TYPE_IQ2_XXS, { 4096, 4096, 128 }, 3, GGML_TYPE_COUNT, nullptr, 0 },
    // THE within-model type exception: IQ3_S on exactly these 4 layers, IQ2_S on the other 32.
    { "ffn_down_exps.weight", GGML_TYPE_IQ2_S,    { 2048, 4096, 128 }, 3, GGML_TYPE_IQ3_S, DOWN_EXCEPT_LAYERS, (int)(sizeof(DOWN_EXCEPT_LAYERS) / sizeof(DOWN_EXCEPT_LAYERS[0])) },
    { "ffn_gate_shexp.weight",GGML_TYPE_Q5_K,     { 4096, 2048, 0 },   2, GGML_TYPE_COUNT, nullptr, 0 },
    { "ffn_up_shexp.weight",  GGML_TYPE_Q5_K,     { 4096, 2048, 0 },   2, GGML_TYPE_COUNT, nullptr, 0 },
    { "ffn_down_shexp.weight",GGML_TYPE_Q6_K,     { 2048, 4096, 0 },   2, GGML_TYPE_COUNT, nullptr, 0 },
};
static const int N_LAYER_TENSOR = (int) (sizeof(LAYER_TENSOR_SPECS) / sizeof(LAYER_TENSOR_SPECS[0]));

static const TensorSpec GLOBAL_TENSOR_SPECS[] = {
    { "token_embd.weight",  GGML_TYPE_Q4_K, { 4096, 131072, 0 }, 2, GGML_TYPE_COUNT, nullptr, 0 },
    { "output_norm.weight", GGML_TYPE_F32,  { 4096, 0,      0 }, 1, GGML_TYPE_COUNT, nullptr, 0 },
    { "output.weight",      GGML_TYPE_Q4_K, { 4096, 131072, 0 }, 2, GGML_TYPE_COUNT, nullptr, 0 },
};
static const int N_GLOBAL_TENSOR = (int) (sizeof(GLOBAL_TENSOR_SPECS) / sizeof(GLOBAL_TENSOR_SPECS[0]));

// ---------------------------------------------------------------------------
// The surface as READ FROM THE FILE (one row per spec) + its check
// ---------------------------------------------------------------------------
struct TensorInfo {
    std::string name;
    bool        present = false;
    ggml_type   type    = GGML_TYPE_COUNT;
    int64_t     ne[4]   = {0, 0, 0, 0};
    int         n_ne    = 0;
    size_t      nbytes  = 0;
    size_t      offset  = 0;   // offset into the tensor-data blob
};

struct LayerSurface {
    TensorInfo t[(int) (sizeof(LAYER_TENSOR_SPECS) / sizeof(LAYER_TENSOR_SPECS[0]))];
};

struct ModelSurface {
    LayerSurface layer[N_LAYERS];
    TensorInfo   global[(int) (sizeof(GLOBAL_TENSOR_SPECS) / sizeof(GLOBAL_TENSOR_SPECS[0]))];
    int64_t      n_tensors_file = 0;   // tensors in the file
    int          n_classified   = 0;   // tensors the surface accounts for
    int          n_group_members = 0;  // of those, members of grouped logical tensors (SEAM 3)
};

struct SurfaceCheck {
    int n_rows         = 0;   // rows compared: 36*16 + 3
    int n_missing      = 0;   // absent from the file
    int n_type_bad     = 0;
    int n_shape_bad    = 0;
    int n_unclassified = 0;   // tensors in the file no row accounts for
    int n_duplicate    = 0;   // names the surface claims twice
    int n_group_rows   = 0;   // rows covered through group members instead of a flat tensor
    std::vector<std::string> fails;

    bool ok() const {
        return n_missing == 0 && n_type_bad == 0 && n_shape_bad == 0 &&
               n_unclassified == 0 && n_duplicate == 0;
    }
};

static inline std::string tensor_shape_str(const TensorInfo & t) {
    std::string s = "[";
    for (int i = 0; i < t.n_ne; ++i) {
        if (i) s += ",";
        s += std::to_string((long long) t.ne[i]);
    }
    return s + "]";
}

static inline std::string layer_tensor_name(int layer, const char * suffix) {
    char buf[256];
    std::snprintf(buf, sizeof buf, "blk.%d.%s", layer, suffix);
    return std::string(buf);
}

// ---------------------------------------------------------------------------
// SEAM 3 — the grouped-expert key (taillis/layout.md §2, PROPOSAL implemented here)
//
//   taillis.expert_groups = [ "blk.3.ffn_gate_up_exps.g0", "blk.3.ffn_gate_up_exps.g1", ... ]
//
// ONE string array for the whole file; each entry names a member tensor WITHOUT its
// ".weight" suffix, and the member's logical tensor is the entry minus its trailing
// ".g<k>".  The reader derives each logical tensor's segments as the CUMULATIVE ne[2]
// of its members in ARRAY ORDER (rule 1), each segment with its own type and bytes.
// Key absent => the map stays empty => logical() is exactly today's single-segment
// read (rule 2, byte-identical fallback).  A layer may be grouped while another is
// not, and gate_up/down are grouped independently: both fall out of the per-logical
// lookup, there is no per-file or per-layer switch anywhere.
//
// R2 (taillis/layout.md §8): with taillis.format_version = 1 the SAME map is filled
// from the member tensor names instead of the key (ascending k is the expert order,
// §8.2.3) and the key must be ABSENT — its presence in a v1 file is malformed.
// ---------------------------------------------------------------------------
#define TAILLIS_EXPERT_GROUPS_KEY "taillis.expert_groups"

// R2 (taillis/layout.md §8): the format version.  Absent or 0 = v0 (the key IS the
// declaration); 1 = v1 (membership and order are derived from the member tensor
// names, and the key must be ABSENT).
#define TAILLIS_FORMAT_VERSION_KEY "taillis.format_version"

// "blk.3.ffn_gate_up_exps.g0" -> logical "blk.3.ffn_gate_up_exps"; false unless the
// entry ends in ".g<digits>".
static inline bool group_member_logical(const std::string & member, std::string * logical) {
    const size_t dot = member.rfind('.');
    if (dot == std::string::npos || dot + 2 >= member.size() || member[dot + 1] != 'g') return false;
    for (size_t i = dot + 2; i < member.size(); ++i)
        if (!std::isdigit((unsigned char) member[i])) return false;
    *logical = member.substr(0, dot);
    return true;
}

// R2 v1: is `name` a member tensor of `logical` — "<logical>.g<digits>.weight" — and
// what is k?  k is purely an ordering token (layout.md §8.2.4): contiguity is NOT
// required, duplicates are rejected by the caller.
static inline bool parse_member_k(const std::string & name, const std::string & logical,
                                  uint32_t * k_out) {
    const std::string prefix = logical + ".g";
    if (name.compare(0, prefix.size(), prefix) != 0) return false;
    size_t d = prefix.size();
    uint64_t k = 0;
    while (d < name.size() && std::isdigit((unsigned char) name[d])) {
        // guard BEFORE the multiply: k*10 must not wrap uint64 on a long digit
        // run in a hand-built file.  With k <= (0xffffffff-9)/10 here,
        // k*10+digit <= 0xffffffff by arithmetic — no wrap is possible.
        if (k > (0xffffffffull - 9) / 10) return false;   // not a member k; unclassified
        k = k * 10 + (uint64_t) (name[d] - '0');
        ++d;
    }
    if (d == prefix.size() || d + 7 != name.size() || name.compare(d, 7, ".weight") != 0)
        return false;
    *k_out = (uint32_t) k;
    return true;
}

// ---------------------------------------------------------------------------
// GGUF model + SEAM 3 weight representation (M0, extracted)
// ---------------------------------------------------------------------------
struct Segment {
    int lo;                // expert_lo (inclusive), along ne[2]
    int hi;                // expert_hi (exclusive)
    ggml_type type;
    const uint8_t * bytes; // points into the mmap'd file
    size_t nbytes;
};

struct LogicalTensor {
    std::string name;
    int64_t ne[4] = {0,0,0,0};
    int n_expert = 0;
    std::vector<Segment> segs;   // ordered, contiguous; ranges = cumulative ne[2]
};

struct GGUFModel {
    std::string path;
    int fd = -1;
    size_t file_size = 0;
    uint8_t * map = nullptr;
    gguf_context * gctx = nullptr;
    // SEAM 3 (R2): logical expert tensor name (WITHOUT .weight) -> ordered member
    // names (WITHOUT .weight).  Source per taillis/layout.md §8.1: v0 (format_version
    // absent or 0) parses taillis.expert_groups; v1 derives the lists from the member
    // tensor names.  Empty => every expert logical is read flat (today's v0 file; a
    // v1 file with no members fails the flat lookup exactly the same way).
    std::map<std::string, std::vector<std::string>> expert_groups;

    ~GGUFModel() {
        if (gctx) gguf_free(gctx);
        if (map)  munmap(map, file_size);
        if (fd >= 0) close(fd);
    }

    const std::vector<std::string> * group_of(const char * logical) const {
        auto it = expert_groups.find(logical);
        return it == expert_groups.end() ? nullptr : &it->second;
    }

    void read_expert_groups() {
        // R2 version dispatch (taillis/layout.md §8.1): absent or 0 = v0 (the key IS
        // the declaration, today's path below); 1 = v1 (derived from member names).
        uint32_t version = 0;
        const int64_t vid = gguf_find_key(gctx, TAILLIS_FORMAT_VERSION_KEY);
        if (vid >= 0) {
            CHECK(gguf_get_kv_type(gctx, vid) == GGUF_TYPE_UINT32,
                  "%s is not a u32", TAILLIS_FORMAT_VERSION_KEY);
            version = gguf_get_val_u32(gctx, vid);
        }
        // §7.1's promise ("distinguished at read time") is only honest if an
        // UNKNOWN version is refused, not silently read as v1 by `version < 1`.
        CHECK(version <= 1, "unknown %s = %u (this reader knows 0 and 1)",
              TAILLIS_FORMAT_VERSION_KEY, version);
        const int64_t kid = gguf_find_key(gctx, TAILLIS_EXPERT_GROUPS_KEY);

        if (version < 1) {
            // ---- v0: today's loop, unchanged -------------------------------
            if (kid < 0) return;   // key absent => today's file (SEAM 3 fallback, rule 2)
            CHECK(gguf_get_kv_type(gctx, kid) == GGUF_TYPE_ARRAY,
                  "%s is not a string array", TAILLIS_EXPERT_GROUPS_KEY);
            const size_t n = gguf_get_arr_n(gctx, kid);
            for (size_t i = 0; i < n; ++i) {
                const char * s = gguf_get_arr_str(gctx, kid, i);
                CHECK(s != nullptr, "%s[%zu] is null", TAILLIS_EXPERT_GROUPS_KEY, i);
                std::string member(s), logical;
                CHECK(group_member_logical(member, &logical),
                      "%s[%zu]='%s' is not <logical>.g<k>", TAILLIS_EXPERT_GROUPS_KEY, i, s);
                CHECK(gguf_find_tensor(gctx, (member + ".weight").c_str()) >= 0,
                      "%s[%zu]: member tensor '%s.weight' not in file",
                      TAILLIS_EXPERT_GROUPS_KEY, i, member.c_str());
                CHECK(gguf_find_tensor(gctx, (logical + ".weight").c_str()) < 0,
                      "'%s.weight' is grouped by %s but ALSO present flat — malformed file",
                      logical.c_str(), TAILLIS_EXPERT_GROUPS_KEY);
                std::vector<std::string> & v = expert_groups[logical];
                CHECK(std::find(v.begin(), v.end(), member) == v.end(),
                      "%s names '%s' twice", TAILLIS_EXPERT_GROUPS_KEY, member.c_str());
                v.push_back(member);   // array order IS the expert order (rule 1)
            }
            return;
        }

        // ---- v1: the key MUST NOT be present (layout.md §8.1) -------------
        CHECK(kid < 0,
              "%s present in a format_version=1 file — v1 derives the grouping from the "
              "member tensor names (taillis/layout.md §8); the key is malformed",
              TAILLIS_EXPERT_GROUPS_KEY);
        // The derivation surface is the reader's OWN fixed checklist (gguf_model.h:97-99):
        // blk.[0, N_LAYERS) x {ffn_gate_up_exps, ffn_down_exps} = 72 logicals.  §8.2.1.
        static const char * const EXPERT_SUFFIXES[] = { "ffn_gate_up_exps", "ffn_down_exps" };
        for (int l = 0; l < (int) N_LAYERS; ++l)
        for (const char * suf : EXPERT_SUFFIXES) {
            const std::string logical = layer_tensor_name(l, suf);
            std::vector<std::pair<uint32_t, std::string>> members;   // (k, member name)
            for (int64_t i = 0; i < gguf_get_n_tensors(gctx); ++i) {
                const std::string name = gguf_get_tensor_name(gctx, i);
                uint32_t k;
                if (parse_member_k(name, logical, &k))
                    members.emplace_back(k, name.substr(0, name.size() - 7));  // strip ".weight"
            }
            if (members.empty()) continue;   // §8.2.2: no members => this logical is FLAT
            std::sort(members.begin(), members.end());
            // §8.2.3: k unique per logical; ascending k IS the expert order (§8.2.4:
            // contiguity is NOT required — k is an ordering token).
            for (size_t i = 1; i < members.size(); ++i)
                CHECK(members[i-1].first != members[i].first,
                      "'%s': duplicate member index g%u — k must be unique per logical "
                      "(taillis/layout.md §8.2.3)", logical.c_str(), members[i-1].first);
            // §8.2.5: the ONE surviving key-free consistency check.
            CHECK(gguf_find_tensor(gctx, (logical + ".weight").c_str()) < 0,
                  "'%s.weight' is grouped by its members but ALSO present flat — "
                  "malformed file (taillis/layout.md §8.2.5)", logical.c_str());
            std::vector<std::string> & v = expert_groups[logical];
            for (const auto & mk : members)
                v.push_back(mk.second);   // ascending k IS the expert order (§8.2.3)
        }
    }

    static GGUFModel * load(const char * path) {
        GGUFModel * m = new GGUFModel();
        m->path = path;
        m->fd = open(path, O_RDONLY);
        CHECK(m->fd >= 0, "open %s failed", path);
        struct stat st;
        CHECK(fstat(m->fd, &st) == 0, "fstat failed");
        m->file_size = (size_t) st.st_size;
        m->map = (uint8_t *) mmap(nullptr, m->file_size, PROT_READ, MAP_PRIVATE, m->fd, 0);
        CHECK(m->map != MAP_FAILED, "mmap failed (%zu bytes)", m->file_size);
        gguf_init_params p = { /* no_alloc */ true, /* ctx */ nullptr };
        m->gctx = gguf_init_from_buffer(m->map, m->file_size, p);
        CHECK(m->gctx != nullptr, "gguf_init_from_buffer failed");
        m->read_expert_groups();
        return m;
    }

    // one logical expert tensor as a list of (expert_lo, expert_hi, type, bytes) segments.
    // Not in the expert_groups map (v0 without the key, or v1 with no members) => ONE
    // segment covering all experts.  In the map (filled from the v0 key or the v1 name
    // derivation) => G segments built from the members (cumulative ne[2], per-member
    // type), layout.md §2 / §8.2.3.
    // The list is G-generic: G=1 is the trivial partition, G=128 (per-expert, layout.md §6)
    // is the same loop.
    LogicalTensor logical(const char * name) {
        LogicalTensor lt;
        lt.name = name;
        // the group map is keyed by the LOGICAL name (no ".weight") — strip it if present
        std::string lname = name;
        if (lname.size() > 7 && lname.compare(lname.size() - 7, 7, ".weight") == 0)
            lname.resize(lname.size() - 7);
        const std::vector<std::string> * grp = group_of(lname.c_str());
        if (grp && !grp->empty()) {
            CHECK(gguf_find_tensor(gctx, name) < 0,
                  "'%s' is grouped but also present flat", name);   // re-checked: cheap
            int lo = 0;
            for (size_t gi = 0; gi < grp->size(); ++gi) {
                const std::string mname = (*grp)[gi] + ".weight";
                const int64_t tid = gguf_find_tensor(gctx, mname.c_str());
                CHECK(tid >= 0, "group member '%s' of '%s' not found", mname.c_str(), name);
                const int64_t * ne = gguf_get_tensor_ne(gctx, tid);
                if (gi == 0) {
                    lt.ne[0] = ne[0]; lt.ne[1] = ne[1];
                } else {
                    CHECK(ne[0] == lt.ne[0] && ne[1] == lt.ne[1],
                          "group member '%s' is [%lld,%lld,*], expected [%lld,%lld,*]",
                          mname.c_str(), (long long) ne[0], (long long) ne[1],
                          (long long) lt.ne[0], (long long) lt.ne[1]);
                }
                CHECK(ne[2] > 0, "group member '%s' has empty expert range", mname.c_str());
                Segment s;
                s.lo = lo;
                s.hi = lo + (int) ne[2];
                s.type = gguf_get_tensor_type(gctx, tid);
                s.bytes = map + gguf_get_data_offset(gctx) + gguf_get_tensor_offset(gctx, tid);
                s.nbytes = gguf_get_tensor_size(gctx, tid);
                lt.segs.push_back(s);
                lo = s.hi;
            }
            CHECK(lo == (int) N_EXPERT,
                  "'%s': sum(member ne[2]) = %d != %d (layout.md §2 rule 1)",
                  name, lo, (int) N_EXPERT);
            lt.ne[2] = lo;
            lt.ne[3] = 1;
            lt.n_expert = lo;
            return lt;
        }
        CHECK(gguf_find_tensor(gctx, name) >= 0, "tensor '%s' not found", name);
        const int64_t tid = gguf_find_tensor(gctx, name);
        const int64_t * ne = gguf_get_tensor_ne(gctx, tid);
        for (int i = 0; i < 4; ++i) lt.ne[i] = ne[i];
        lt.n_expert = (int) ne[2];
        ggml_type type = gguf_get_tensor_type(gctx, tid);
        size_t tsize = gguf_get_tensor_size(gctx, tid);
        size_t off = gguf_get_data_offset(gctx) + gguf_get_tensor_offset(gctx, tid);
        Segment s;
        s.lo = 0; s.hi = lt.n_expert; s.type = type;
        s.bytes = map + off; s.nbytes = tsize;
        lt.segs.push_back(s);
        return lt;
    }

    // host pointer to a whole F32 tensor (used for the router gate_inp)
    const float * f32_tensor(const char * name, size_t * n_floats) {
        int64_t tid = gguf_find_tensor(gctx, name);
        CHECK(tid >= 0, "tensor '%s' not found", name);
        CHECK(gguf_get_tensor_type(gctx, tid) == GGML_TYPE_F32, "'%s' not F32", name);
        size_t off = gguf_get_data_offset(gctx) + gguf_get_tensor_offset(gctx, tid);
        *n_floats = gguf_get_tensor_size(gctx, tid) / sizeof(float);
        return (const float *) (map + off);
    }

    // ---- M1: the full per-layer surface ----------------------------------------------
    int64_t n_tensors() const { return gguf_get_n_tensors(gctx); }
    const char * tensor_name(int64_t i) const { return gguf_get_tensor_name(gctx, i); }

    TensorInfo info(const char * name) const {
        TensorInfo ti;
        ti.name = name;
        int64_t tid = gguf_find_tensor(gctx, name);
        if (tid < 0) return ti;                       // present = false
        ti.present = true;
        ti.type = gguf_get_tensor_type(gctx, tid);
        const int64_t * ne = gguf_get_tensor_ne(gctx, tid);
        ti.n_ne = 0;
        for (int i = 0; i < 4; ++i) {
            ti.ne[i] = ne[i];
            if (ne[i] > 1) ti.n_ne = i + 1;           // gguf reports ne[i] = 1 for absent dims
        }
        ti.nbytes = gguf_get_tensor_size(gctx, tid);
        ti.offset = gguf_get_tensor_offset(gctx, tid);
        return ti;
    }

    ModelSurface read_surface() const {
        ModelSurface ms;
        ms.n_tensors_file = n_tensors();
        // n_classified counts FILE TENSORS the checklist accounts for, which is what the
        // "%d of %lld in the file" comparison means — so a row is worth 1 only when a file
        // tensor carries it.  A grouped logical tensor is ABSENT flat by design (SEAM 3):
        // its members are the file tensors, the logical row is not, so the row is worth
        // the member count.  Counting the flat row AND its members counted the same file
        // tensor twice (MEASURED: the one-group full fixture has 579 file tensors and the
        // old code reported 651 = 579 rows + 72 members).
        for (int l = 0; l < (int) N_LAYERS; ++l) {
            for (int i = 0; i < N_LAYER_TENSOR; ++i) {
                ms.layer[l].t[i] = info(layer_tensor_name(l, LAYER_TENSOR_SPECS[i].suffix).c_str());
                if (ms.layer[l].t[i].present) { ++ms.n_classified; continue; }
                std::string logical = layer_tensor_name(l, LAYER_TENSOR_SPECS[i].suffix);
                logical.resize(logical.size() - 7);   // strip ".weight"
                const std::vector<std::string> * grp = group_of(logical.c_str());
                if (grp && !grp->empty()) {
                    ms.n_group_members += (int) grp->size();
                    ms.n_classified    += (int) grp->size();
                }
                // else: absent and ungrouped — no file tensor carries it (a real gap, and
                // the row check counts it as missing); it is worth 0 file tensors here.
            }
        }
        for (int i = 0; i < N_GLOBAL_TENSOR; ++i) {
            ms.global[i] = info(GLOBAL_TENSOR_SPECS[i].suffix);
            if (ms.global[i].present) ++ms.n_classified;
        }
        return ms;
    }

    void list_layers() {
        std::printf("layer  ffn_gate_up_exps   ffn_down_exps       ffn_gate_inp\n");
        for (int l = 0; l < N_LAYERS; ++l) {
            char gu[256], dn[256], gi[256];
            std::snprintf(gu, sizeof gu, "blk.%d.ffn_gate_up_exps.weight", l);
            std::snprintf(dn, sizeof dn, "blk.%d.ffn_down_exps.weight", l);
            std::snprintf(gi, sizeof gi, "blk.%d.ffn_gate_inp.weight", l);
            auto f = [&](const char * n) {
                int64_t t = gguf_find_tensor(gctx, n);
                return t >= 0 ? ggml_type_name(gguf_get_tensor_type(gctx, t)) : "-";
            };
            std::printf("%5d  %-18s %-18s %s\n", l, f(gu), f(dn), f(gi));
        }
    }
};

// ---------------------------------------------------------------------------
// The surface check — every row against its spec, and every file tensor against the surface.
// This is the U1 type-table verification step, not a print: a mismatch is a FAIL line and a
// non-zero exit in the skeleton.
// ---------------------------------------------------------------------------
static inline void surface_check_row(const TensorInfo & ti, const TensorSpec & spec, ggml_type want_type,
                                     const char * what, SurfaceCheck & ck) {
    ++ck.n_rows;
    if (!ti.present) {
        ++ck.n_missing;
        if (ck.fails.size() < 64) ck.fails.push_back(std::string("MISSING ") + what);
        return;
    }
    if (ti.type != want_type) {
        ++ck.n_type_bad;
        char b[512];
        std::snprintf(b, sizeof b, "TYPE %s: read %s, checklist %s", what, ggml_type_name(ti.type),
                      ggml_type_name(want_type));
        if (ck.fails.size() < 64) ck.fails.push_back(b);
    }
    bool shape_ok = (ti.n_ne == spec.n_ne);
    for (int i = 0; shape_ok && i < spec.n_ne; ++i) shape_ok = (ti.ne[i] == spec.ne[i]);
    if (!shape_ok) {
        ++ck.n_shape_bad;
        char b[512];
        std::string want = "[";
        for (int i = 0; i < spec.n_ne; ++i) { if (i) want += ","; want += std::to_string((long long) spec.ne[i]); }
        want += "]";
        std::snprintf(b, sizeof b, "SHAPE %s: read %s, checklist %s", what, tensor_shape_str(ti).c_str(), want.c_str());
        if (ck.fails.size() < 64) ck.fails.push_back(b);
    }
}

// SEAM 3: a grouped row is compared THROUGH its members: each member must carry the row's
// ne[0], ne[1] and a type whose block size divides ne[0] (layout.md §1), and the members'
// ne[2] must sum to the row's ne[2] (§2 rule 1).  Member TYPES are deliberately NOT checked
// against the row's spec type — different types per group are the feature.  Returns false
// if the row is not grouped, so the caller falls through to the flat check.
static inline bool surface_check_grouped_row(const GGUFModel & m, const std::string & logical,
                                             const TensorSpec & spec, SurfaceCheck & ck) {
    const std::vector<std::string> * grp = m.group_of(logical.c_str());
    if (!grp || grp->empty()) return false;
    ++ck.n_rows;
    ++ck.n_group_rows;
    int64_t sum_ne2 = 0;
    for (const std::string & mem : *grp) {
        const std::string mname = mem + ".weight";
        TensorInfo mi = m.info(mname.c_str());
        if (!mi.present) {
            ++ck.n_missing;
            if (ck.fails.size() < 64) ck.fails.push_back("MISSING group member " + mname);
            continue;
        }
        if (mi.ne[0] != spec.ne[0] || mi.ne[1] != spec.ne[1]) {
            ++ck.n_shape_bad;
            if (ck.fails.size() < 64)
                ck.fails.push_back("SHAPE group member " + mname + ": read " +
                                   tensor_shape_str(mi) + ", checklist row " + logical + " disagrees");
        }
        if (mi.ne[0] % ggml_blck_size(mi.type) != 0) {
            ++ck.n_type_bad;
            if (ck.fails.size() < 64)
                ck.fails.push_back("TYPE group member " + mname + ": block size " +
                                   std::to_string(ggml_blck_size(mi.type)) + " does not divide ne[0]");
        }
        sum_ne2 += mi.ne[2];
    }
    if (sum_ne2 != spec.ne[2]) {
        ++ck.n_shape_bad;
        if (ck.fails.size() < 64)
            ck.fails.push_back("SHAPE " + logical + ": group members sum to ne[2]=" +
                               std::to_string((long long) sum_ne2) + ", checklist " +
                               std::to_string((long long) spec.ne[2]));
    }
    return true;
}

static inline SurfaceCheck check_surface(const GGUFModel & m, const ModelSurface & ms) {
    SurfaceCheck ck;
    std::vector<std::string> names;
    for (int l = 0; l < (int) N_LAYERS; ++l) {
        for (int i = 0; i < N_LAYER_TENSOR; ++i) {
            const TensorSpec & sp = LAYER_TENSOR_SPECS[i];
            const TensorInfo & ti = ms.layer[l].t[i];
            bool grouped = false;
            if (!ti.present && ti.name.size() > 7) {
                grouped = surface_check_grouped_row(m, ti.name.substr(0, ti.name.size() - 7), sp, ck);
                if (grouped) {
                    // the members are accounted-for file tensors: add them to the name set
                    // or the unclassified pass flags the very tensors the group resolved
                    const std::vector<std::string> * grp =
                        m.group_of(ti.name.substr(0, ti.name.size() - 7).c_str());
                    for (const std::string & mem : *grp) names.push_back(mem + ".weight");
                }
            }
            if (!grouped) surface_check_row(ti, sp, sp.type_for_layer(l), ti.name.c_str(), ck);
            names.push_back(ti.name);
        }
    }
    for (int i = 0; i < N_GLOBAL_TENSOR; ++i) {
        const TensorSpec & sp = GLOBAL_TENSOR_SPECS[i];
        const TensorInfo & ti = ms.global[i];
        surface_check_row(ti, sp, sp.type, ti.name.c_str(), ck);
        names.push_back(ti.name);
    }
    std::sort(names.begin(), names.end());
    for (size_t i = 1; i < names.size(); ++i) {
        if (names[i] == names[i-1]) {
            ++ck.n_duplicate;
            if (ck.fails.size() < 64) ck.fails.push_back(std::string("DUPLICATE row ") + names[i]);
        }
    }
    for (int64_t i = 0; i < ms.n_tensors_file; ++i) {
        const char * nm = m.tensor_name(i);
        if (!std::binary_search(names.begin(), names.end(), std::string(nm))) {
            ++ck.n_unclassified;
            if (ck.fails.size() < 64) ck.fails.push_back(std::string("UNCLASSIFIED file tensor ") + nm);
        }
    }
    return ck;
}

