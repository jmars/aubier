// engine/store.h — the store machinery + the store list (SEAM 1), extracted from
// engine/moe_split.cpp (M0) so the M0 split primitive and the M1 loop share ONE definition.
//
// A STORE is one entry in the store list: (expert range x TYPE x location).  The store list is
// what makes the split a policy instead of a hardcoded pair — M0's review found the first
// version hardcoded a pair, and the fix is preserved here.
//
// SEAM 1 IS PRESERVED, NOT CLOSED (engine-plan.md §3; user directive 2026-10-11: "if we can't
// fit all experts in VRAM as well as ctx then we need the streaming expert cache").  fill() is
// a REAL virtual step with a no-op VRAM/RAM implementation, so a third store whose fill() is an
// async H2D from host RAM drops in as a POLICY VARIANT over this list — it is not implemented in
// M1/U1 and this header does not close the seam.
//
// HONEST CAVEAT carried from the brief (MEASURED, results/p-m4pre.md): on THIS host a streamed
// store SUBTRACTS (+0.6 GiB/s = 2.3% net against the CPU path) because its memory path is faster
// than its link.  The directive is about CAPACITY on a machine where that ratio inverts
// (engine-plan.md §3: "the crossover is a per-machine ratio, not a constant").  A streamed store
// is not a speed claim on this host.
//
// The type is carried PER SLICE (per (expert range x segment)), because a store's range may span
// several type segments (SEAM 3) and the kernels are selected per slice.
//
// H1 — THE RESIDENT SET IS FIRST-CLASS.  Until H1 a store's membership was an INTERVAL: slot() was
// the interval test `expert in [lo,hi)` and derive_store_specs() hardcoded the [0,n_res) cut, so a
// policy could only be expressed by PERMUTING a fixture (U6 did exactly that).  H1 adds the other
// membership axis — an ORDERED SET of global expert ids, slot i = the i-th member — and both forms
// coexist so that the interval path stays byte-for-byte what it always was:
//   * `members` empty  => INTERVAL-backed: today's M0 shape, unchanged code path;
//   * `members` non-empty => SET-backed: the store's tensors are a PACKED copy of exactly those
//     experts, in that order (a scattered set is ~24 runs/layer, which by U6's measurement cannot
//     be expressed as a small number of intervals — results/m1-u6.md §1).
// The membership test and the slot mapping both go through Slice::local_of (global expert ->
// slice-local index | -1), which for the interval case is exactly `e - lo`, so the identity set is
// the old behaviour by construction.

#pragma once

#include "gguf_model.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <map>
#include <memory>
#include <utility>
#include <vector>

// ---------------------------------------------------------------------------
// Store (SEAM 1) — one entry in the store list, three axes (range x type x location)
// ---------------------------------------------------------------------------
//
// U7 instrumentation (the per-position split, plan §10.2 item 1).  COMPILE-TIME gated: the
// macro is defined ONLY in m1_engine.cpp's translation unit, so M0's gated artifact
// (moe_split.cpp, which includes this header WITHOUT it) compiles byte-for-byte the drain it
// always did — the split below exists in exactly one object file of exactly one binary.
// Timing only: no numerics anywhere near it.
#ifdef STORE_INSTR
#include <chrono>
static double g_store_get_s = 0, g_store_combine_s = 0;
// packed by the driver: the CPU (RAM) store's enqueue BLOCKS and computes, the HIP (VRAM)
// store's only launches — so the host-packing and the graph-compute call are split AND named
// per backend class.
static ggml_backend_t g_store_cpu_be = nullptr;
static double g_store_pack_ram_s = 0, g_store_launch_ram_s = 0;
static double g_store_pack_vram_s = 0, g_store_launch_vram_s = 0;
static long   g_store_get_n = 0, g_store_group_n = 0;
static inline double store_now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
#endif

// U7 EXPERIMENT SWITCH — the paired A/B for the output-D2H move, so ONE binary can run both
// arms INTERLEAVED in one session (the box drifts ~10 % row to row, which is the same size as
// the effect being measured, so an unpaired before/after session cannot decide it).
//   false (default) = U7: the D2H is queued at ENQUEUE (it runs under the other store's compute)
//   true            = the pre-U7 shape: a blocking tensor_get pulled at DRAIN time
static bool g_store_pull_at_drain = false;

// U2b — bytes the alias's startup prefault walked, and how long it took (reported by the loop's
// build line; 0 for every store that is not aliased, so the M0 split primitive pays nothing and
// prints nothing).  Two separate accumulators so the WILLNEED readahead and the touch pass can be
// told apart: they are different costs and only one of them is a page-table build.
static size_t g_alias_prefault_bytes = 0;
static double g_alias_prefault_advise_s = 0, g_alias_prefault_touch_s = 0;
static double g_alias_prefault_call_s = 0;
// --alias-prefault (default OFF — MEASURED a net loss at 51 positions, see results/m1-h2-u2b.md):
// the mechanism works (in-loop minflt 11 K -> ~0.3 K) but building 19.1 GiB of page table costs
// ~1.25 s warm (10 s cold) to save ~0.35 s of drive.  Off by default, so the default path is the
// shipped one and GATE 3a's byte-identity is the shipped binary's.
static bool g_alias_prefault = false;
static double alias_now_s() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double) ts.tv_sec + 1e-9 * (double) ts.tv_nsec;
}

struct StoreGraph {
    int c = 0, n = 0;
    ggml_cgraph * gf = nullptr;
    ggml_tensor * ids = nullptr;   // [c, n] i32, store-local expert indices
    ggml_tensor * w   = nullptr;   // [1, c, n] f32 routing weights
    ggml_tensor * xb  = nullptr;   // [H, 1, n] gathered activation columns
    ggml_tensor * gu  = nullptr;   // [2*FF, c, n] gate_up product (per-(tensor,type) table)
    ggml_tensor * d   = nullptr;   // [H, c, n]      down product
    ggml_tensor * out = nullptr;   // [H, n] summed weighted output
    // persistent host staging (async H2D must read stable memory until the copy drains)
    std::vector<int32_t> ids_buf;
    std::vector<float> w_buf, xb_buf;
};

// A SLICE is one (expert sub-range x TYPE) piece of a store: the intersection of the
// store's expert range with ONE segment of each logical tensor (SEAM 3).  A flat file
// yields exactly ONE slice per store — the plain byte-range slice of the parent tensor
// that M0 has always used; a grouped file yields one slice per group, which is where
// per-store kernel selection by type lives (taillis/layout.md §4).  Slices are what make a
// store whose range SPANS two segments REPRESENTABLE instead of a CHECK failure.
struct Slice {
    int lo = 0, hi = 0;                 // global expert range, subset of the store's [lo,hi)
    ggml_type gu_type = GGML_TYPE_F32, d_type = GGML_TYPE_F32;
    const uint8_t * gu_base = nullptr; size_t gu_nbytes = 0;
    const uint8_t * d_base  = nullptr; size_t d_nbytes  = 0;
    // H1: the slice's RESIDENT SET.  EMPTY => the slice IS the contiguous file range [lo,hi) (the
    // M0 shape, and the only shape an interval store produces).  NON-EMPTY => the slice's tensors
    // are a PACKED copy of exactly these global experts in this ORDER: slot i holds members[i]'s
    // rows, read from the (range x type) segment that covers that expert.
    //
    // U2a: a slice of an ALIASED store (Store::aliased) is the third case — [lo,hi) is a run of
    // the file's own expert axis and the tensors POINT AT THE MMAP, so row k is expert lo+k and
    // there is nothing packed.  `members` stays EMPTY there: the store's set lives in
    // Store::members/pos_of, and the slice's own membership/row table is local_of (see
    // build_alias_slices).
    std::vector<int> members;
    std::vector<const uint8_t *> gu_src, d_src;   // per-member source row (parallel to members)
    size_t gu_stride = 0, d_stride = 0;           // bytes per expert in each source tensor
    // (global expert) -> slice-local slot | -1.  THE membership test and THE slot mapping the
    // enqueue loop reads; it replaces `ex >= sl.lo && ex < sl.hi` and `ex - sl.lo` with one table
    // lookup.  For the interval case local_of[e] == e - sl.lo exactly, so the identity set is
    // byte-identical to the pre-H1 code path by construction (I1).
    std::vector<int> local_of;
    ggml_tensor * as_gu = nullptr;      // [H, 2*FF, n_e] this slice's experts, gate_up type
    ggml_tensor * as_d  = nullptr;      // [FF, H,  n_e] this slice's experts, down type
    std::map<std::pair<int,int>, StoreGraph> graphs;
};

// bytes of the expert sub-range [a,b) inside a logical tensor, and the segment type there.
// SEAM 3: the store asks for a byte range, not for "the tensor's type" — a range covered by
// two segments is split by the caller at the segment boundaries (see Store::build).
struct MatSrc { ggml_type type = GGML_TYPE_F32; const uint8_t * base = nullptr; size_t nbytes = 0; };
static MatSrc slice_src(const LogicalTensor & lt, int a, int b) {
    MatSrc ms;
    for (const Segment & s : lt.segs) {
        if (s.lo <= a && b <= s.hi) {
            size_t row    = ggml_row_size(s.type, lt.ne[0]);
            size_t stride = row * (size_t) lt.ne[1];        // bytes per expert (nb[2])
            ms.type   = s.type;
            ms.base   = s.bytes + (size_t) (a - s.lo) * stride;
            ms.nbytes = stride * (size_t) (b - a);
            return ms;
        }
    }
    FATAL("no single segment covers experts [%d,%d) in %s", a, b, lt.name.c_str());
    return ms;   // unreachable
}

struct Store {
    enum Location { RAM, VRAM } loc;
    const char * loc_name = "";
    ggml_backend_t backend = nullptr;
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    int lo = 0, hi = 0, n_e = 0;
    // H1 — THE RESIDENT SET (ordered): slot i of this store holds global expert members[i].
    // EMPTY => the store is INTERVAL-backed and its membership is [lo,hi) (M0's shape, and the
    // only shape the pre-change binary could express).
    std::vector<int> members;
    std::vector<int> pos_of;     // (global expert) -> slot | -1; only meaningful when !members.empty()
    std::vector<Slice> slices;   // empty iff n_e == 0: an EMPTY store is a legal list entry

    // U2a — THE ZERO-COPY ALIAS.  An ALIASED store holds NO copy of its experts: its weight
    // tensors are bound to the mmap'd file itself (build_alias_slices), so the anonymous bytes it
    // allocates are the graph's small activation tensors and nothing else.  Its membership is
    // still `members` (so slot()/audit_store_list are unchanged), but the tensor's expert axis is
    // the file's own [0,N_EXPERT) block and a member's slot is the ROW IT ALREADY OCCUPIES
    // (`e - run_lo`, i.e. `e` for a flat file), not its position in `members`.
    bool aliased = false;
    ggml_context * alias_ctx = nullptr;                  // holds ONLY the aliased weight tensors
    std::vector<ggml_backend_buffer_t> alias_bufs;       // free_buffer == NULL: they own nothing

    // U2b — ROTATION'S TWO MEMBERSHIP MOVES.  A store must be able to change WHO it holds BETWEEN
    // positions: the VRAM (packed set) store by writing the admitted expert's row into the evicted
    // slot, the aliased RAM store by flipping its predicate (no bytes move at all).  replace_slot
    // needs the file's own geometry to find the admitted expert's row, so the build keeps the
    // logical tensors it was built from (they belong to the model and outlive every store).
    const LogicalTensor * gu_lt_p = nullptr, * d_lt_p = nullptr;

    // pending state written by enqueue(), consumed by drain()
    struct Pending {
        int c, n;
        StoreGraph * g;
        int out_off = 0;                                    // U7: offset of g.out in out_host
        std::vector<int> toks;                              // global token indices (length n)
        std::vector<std::vector<int>> tslots;               // per token, its global slots (each length c)
    };
    std::vector<Pending> pending;

    // U7 — THE OUTPUT STAGING.  Each pending's `out` D2H is QUEUED on the backend's stream at
    // ENQUEUE time (ggml_backend_tensor_get_async) instead of being pulled at drain time, so the
    // copy runs underneath the OTHER store's blocking compute (the CPU store's mul_mat_id IS the
    // drive) rather than being exposed after it.  Same bytes, same combine order, same values.
    // The buffer is PINNED where the backend offers a host buffer type: a pageable D2H pays a
    // staging copy (MEASURED: 41 us for 16 KiB per layer on this box, ~1.5 ms/position over 36).
    ggml_backend_buffer_t out_buf = nullptr;   // pinned host buffer (null => the malloc fallback)
    float * out_host = nullptr;
    int out_host_n = 0, out_used = 0;

    // U7 — persistent packing scratch (cleared, not freed, per enqueue): the per-store
    // per-layer per-position vector allocations were pure host churn on the critical path.
    std::vector<int> toks_scratch[N_USED + 1];
    std::vector<std::vector<int>> tslots_scratch;

    // (layer, expert) -> slot | -1 : the Strata-shaped entry point.  H1 made the membership axis
    // explicit: a SET-backed store answers from its ordered resident set (slot = position in the
    // set, which is what H2's per-layer policy allocates), an INTERVAL-backed store keeps M0's
    // test exactly.  `layer` is ignored here — a Store is one layer's list entry (the signature is
    // the seam's, not this store's).
    bool is_set() const { return !members.empty(); }
    int slot(int layer, int expert) const {
        (void) layer;
        if (is_set())
            return (expert >= 0 && expert < (int) pos_of.size()) ? pos_of[(size_t) expert] : -1;
        return (expert >= lo && expert < hi) ? (expert - lo) : -1;
    }

    // a store owns ggml buffers: copying one would double-free them.
    Store() = default;
    Store(const Store &) = delete;
    Store & operator=(const Store &) = delete;

    virtual ~Store() {
        // NOTE: the backend is owned by main and SHARED across stores/oracle — not freed here.
        if (buf) ggml_backend_buffer_free(buf);
        if (ctx) ggml_free(ctx);
        // U2a: the alias buffers free NOTHING (free_buffer == NULL — the base is the mmap), so
        // this releases only the buffer objects; alias_ctx owns the tensors that point into them.
        for (ggml_backend_buffer_t b : alias_bufs) ggml_backend_buffer_free(b);
        if (alias_ctx) ggml_free(alias_ctx);
        if (out_buf) ggml_backend_buffer_free(out_buf);
        else if (out_host) std::free(out_host);
    }

    // SEAM 1: a streamed store issues an async H2D of missing experts HERE, ahead of use.
    // A VRAM/RAM store has nothing to stage, so this is a real no-op virtual step (M4's
    // streamed store overrides it — "a policy variant over the store list, not a rewrite").
    //
    // H1 ASKED WHETHER A SET-BACKED STORE GIVES fill() REAL WORK, AND THE ANSWER IS NO — stated
    // rather than invented: H1's set is STATIC for the whole run (the set is an input, not a
    // function of the routed ids), and both stores of the list keep their experts resident (the
    // VRAM set in the VRAM store's slices, the complement in the RAM store's), so there is no
    // miss to stage ahead of use and nothing for a set to change mid-run.  What a streamed
    // store's fill() would stage is exactly what is ABSENT from it — and here nothing is.  A
    // rotating set (H2's policy) changes WHICH experts are resident between decisions, but that
    // is a rebuild/re-upload step (Store::upload_slice), not a per-token fill: H1 deliberately
    // does not fold the two, so this stays a no-op and the seam stays open.
    virtual void fill(const int32_t * ids, int nt) { (void) ids; (void) nt; }

    // H1: `set` (optional, last so every pre-H1 call site — moe_split.cpp included — compiles
    // unchanged and keeps the interval semantics).  Non-null and non-empty => this store is
    // SET-backed: its membership is exactly *set, in that order, and lo/hi become informational.
    //
    // U2a: `alias` (last, defaulted) makes the store ZERO-COPY: the whole [0,N_EXPERT) expert
    // block of the mmap is aliased instead of packing the members, so a member's slot is the row
    // it already occupies.  It requires a membership set (the complement, for a RAM store) and the
    // CPU backend, because the tensors point at host memory.
    void build(Location loc_, const LogicalTensor & gu_lt, const LogicalTensor & d_lt,
               int lo_, int hi_, ggml_backend_t backend_, int nt,
               const std::vector<int> * set = nullptr, bool alias = false) {
        loc = loc_;
        loc_name = (loc == VRAM) ? "VRAM" : "RAM";
        backend = backend_;
        aliased = alias;
        gu_lt_p = &gu_lt; d_lt_p = &d_lt;
        if (aliased) {
            CHECK(loc == RAM, "U2a: only a RAM store may alias the mmap (this one is %s)", loc_name);
            CHECK(set && !set->empty(), "U2a: an aliased store needs its membership set");
        }
        if (set && !set->empty()) {
            members = *set;
            n_e = (int) members.size();
            lo = *std::min_element(members.begin(), members.end());
            hi = *std::max_element(members.begin(), members.end()) + 1;
            pos_of.assign((size_t) N_EXPERT, -1);
            for (size_t i = 0; i < members.size(); ++i) {
                const int e = members[i];
                CHECK(e >= 0 && e < (int) N_EXPERT, "resident set id %d out of range", e);
                CHECK(pos_of[(size_t) e] < 0, "resident set repeats expert %d", e);
                pos_of[(size_t) e] = (int) i;
            }
            CHECK(lo <= hi, "resident set range inverted");
        } else {
            lo = lo_; hi = hi_; n_e = hi - lo;
        }
        CHECK(n_e >= 0 && n_e <= N_EXPERT, "store range [%d,%d) invalid", lo, hi);

        ggml_init_params p = { 16*1024*1024, nullptr, true };
        ctx = ggml_init(p);
        CHECK(ctx, "ggml_init failed");
        if (n_e == 0) return;   // SEAM 1: an empty store is a legal (inert) list entry

        if (is_set()) {
            if (aliased) {
                pos_of.assign((size_t) N_EXPERT, -1);   // the alias fills ROWS here, not set positions
                build_alias_slices(gu_lt, d_lt);
            } else {
                build_set_slices(gu_lt, d_lt);
            }
        } else {
        // SEAM 3: cut [lo,hi) at every SEGMENT boundary of EITHER tensor, so each slice is
        // covered by exactly one segment per tensor and a store spanning several segments is
        // representable.  Flat file: one segment per tensor => one slice, unchanged.
        std::vector<int> cuts = { lo, hi };
        for (const LogicalTensor * lt : { &gu_lt, &d_lt })
            for (const Segment & s : lt->segs) {
                for (int c : { s.lo, s.hi })
                    if (c > lo && c < hi) cuts.push_back(c);
            }
        std::sort(cuts.begin(), cuts.end());
        cuts.erase(std::unique(cuts.begin(), cuts.end()), cuts.end());

        for (size_t k = 0; k + 1 < cuts.size(); ++k) {
            Slice sl;
            sl.lo = cuts[k]; sl.hi = cuts[k+1];
            MatSrc gs = slice_src(gu_lt, sl.lo, sl.hi);
            MatSrc ds = slice_src(d_lt,  sl.lo, sl.hi);
            sl.gu_type = gs.type; sl.gu_base = gs.base; sl.gu_nbytes = gs.nbytes;
            sl.d_type  = ds.type; sl.d_base  = ds.base; sl.d_nbytes  = ds.nbytes;
            sl.as_gu = ggml_new_tensor_3d(ctx, gs.type, gu_lt.ne[0], gu_lt.ne[1], sl.hi - sl.lo);
            sl.as_d  = ggml_new_tensor_3d(ctx, ds.type, d_lt.ne[0],  d_lt.ne[1],  sl.hi - sl.lo);
            slices.push_back(sl);
        }
        }
        // U2a: an ALIASED store's slices already carry the (expert -> ROW) table the enqueue reads,
        // and it is not slice_index's shape (the alias leaves a non-member at -1 but numbers a
        // member by the row it occupies, not by its position in a packed set).
        if (!aliased)
        for (Slice & sl : slices) slice_index(sl);

        for (Slice & sl : slices)
            for (int c = 1; c <= N_USED; ++c)
                for (int n = 1; n <= nt; ++n)
                    build_graph(sl, c, n);

        buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        CHECK(buf, "backend alloc failed for %s store", loc_name);

        // U7: the output staging (see the member comments).  nt tokens => at most nt*H_DIM
        // floats of pending output, since the pendings partition the tokens.
        out_host_n = (int) ((size_t) H_DIM * (size_t) nt);
        {
            ggml_backend_dev_t dev = ggml_backend_get_device(backend);
            ggml_backend_buffer_type_t hbuft = dev ? ggml_backend_dev_host_buffer_type(dev) : nullptr;
            out_buf = hbuft ? ggml_backend_buft_alloc_buffer(hbuft, sizeof(float) * (size_t) out_host_n) : nullptr;
            // fall back to plain malloc when the backend offers no pinned host buffer type:
            // the async call stays valid (pageable host memory costs latency, never correctness).
            out_host = out_buf ? (float *) ggml_backend_buffer_get_base(out_buf)
                               : (float *) std::malloc(sizeof(float) * (size_t) out_host_n);
        }
        CHECK(out_host, "U7 output staging alloc failed for %s store", loc_name);

        // weights must be set AFTER the backend buffers are allocated
        for (Slice & sl : slices) upload_slice(sl);

        for (Slice & sl : slices)
            for (auto & kv : sl.graphs)
                ggml_build_forward_expand(kv.second.gf, kv.second.out);
    }

    // H1 — the slice-level (global expert -> slice-local slot) table.  ONE definition of the
    // membership test for both backing forms; the interval case is exactly `e - lo`, which is what
    // the pre-H1 enqueue loop computed inline.
    void slice_index(Slice & sl) const {
        sl.local_of.assign((size_t) N_EXPERT, -1);
        if (sl.members.empty()) {
            for (int e = sl.lo; e < sl.hi; ++e) sl.local_of[(size_t) e] = e - sl.lo;
            return;
        }
        for (size_t i = 0; i < sl.members.size(); ++i) sl.local_of[(size_t) sl.members[i]] = (int) i;
    }

    // H1 — pack an ARBITRARY ORDERED SET into tensor slices.  Members are grouped into maximal
    // RUNS that share the same (gate_up segment, down segment) pair — for a flat file that is ONE
    // slice, for a grouped file one per (type pair) — and each slice's tensor holds its run in SET
    // order.  This is why a scattered set needs no new store entries: the set is packed, so one
    // VRAM store serves ~24 scattered runs, which is what U6's cost model rejected as 24 stores
    // (results/m1-u6.md §1).
    void build_set_slices(const LogicalTensor & gu_lt, const LogicalTensor & d_lt) {
        size_t i = 0;
        while (i < members.size()) {
            const int gi = seg_index(gu_lt, members[i]);
            const int di = seg_index(d_lt,  members[i]);
            size_t j = i + 1;
            while (j < members.size() && seg_index(gu_lt, members[j]) == gi &&
                                        seg_index(d_lt,  members[j]) == di) ++j;
            const Segment & gs = gu_lt.segs[(size_t) gi];
            const Segment & ds = d_lt.segs[(size_t) di];
            Slice sl;
            sl.members.assign(members.begin() + (long) i, members.begin() + (long) j);
            sl.lo = *std::min_element(sl.members.begin(), sl.members.end());
            sl.hi = *std::max_element(sl.members.begin(), sl.members.end()) + 1;
            sl.gu_type = gs.type; sl.d_type = ds.type;
            sl.gu_stride = ggml_row_size(gs.type, gu_lt.ne[0]) * (size_t) gu_lt.ne[1];
            sl.d_stride  = ggml_row_size(ds.type, d_lt.ne[0])  * (size_t) d_lt.ne[1];
            for (int e : sl.members) {
                sl.gu_src.push_back(gs.bytes + (size_t) (e - gs.lo) * sl.gu_stride);
                sl.d_src.push_back( ds.bytes + (size_t) (e - ds.lo) * sl.d_stride);
            }
            sl.as_gu = ggml_new_tensor_3d(ctx, gs.type, gu_lt.ne[0], gu_lt.ne[1], sl.members.size());
            sl.as_d  = ggml_new_tensor_3d(ctx, ds.type, d_lt.ne[0],  d_lt.ne[1],  sl.members.size());
            slices.push_back(sl);
            i = j;
        }
    }

    static int seg_index(const LogicalTensor & lt, int e) {
        for (size_t k = 0; k < lt.segs.size(); ++k)
            if (e >= lt.segs[k].lo && e < lt.segs[k].hi) return (int) k;
        FATAL("no segment covers expert %d in %s", e, lt.name.c_str());
        return -1;   // unreachable
    }

    // U2a — THE ZERO-COPY ALIAS BUILDER.  Instead of packing the store's members into a tensor of
    // its own (build_set_slices — n_e experts of anonymous memory per segment pair, MEASURED at
    // 6.80 MiB/expert = 19.107 GiB over the whole model), the store takes the FILE'S OWN expert
    // block: for each RUN [lo,hi) of experts whose gate_up and down tensors are each covered by ONE
    // segment, a slice whose tensors span the run's FULL length and whose `data` IS the mmap
    // address of the run's first expert.  Row k is therefore expert (lo+k) — so
    //     local_of[e] = e - lo   for a member,  -1 otherwise,
    // which keeps local_of as the ONE membership test and slot mapping the enqueue reads, and
    // makes pos_of[e] the row expert e already occupies.  No expert bytes are allocated or copied.
    //
    // WHY A SEPARATE ggml_context: build() allocates every tensor in `ctx` through
    // ggml_backend_alloc_ctx_tensors (store.h:321), which collects ALL of a context's tensors —
    // already-allocated ones included (ggml-alloc.c:1120-1155) — and would re-point the aliased
    // tensors at a fresh anonymous allocation.  They live in alias_ctx instead, where that call
    // cannot see them, and are bound individually by alias_bind.
    void build_alias_slices(const LogicalTensor & gu_lt, const LogicalTensor & d_lt) {
        alias_ctx = ggml_init(ggml_init_params{ 1024*1024, nullptr, true });
        CHECK(alias_ctx, "U2a: ggml_init failed for the alias context");
        std::vector<char> mine((size_t) N_EXPERT, 0);
        for (int e : members) mine[(size_t) e] = 1;

        int e0 = 0;
        while (e0 < (int) N_EXPERT) {
            const int gi = seg_index(gu_lt, e0);
            const int di = seg_index(d_lt,  e0);
            int e1 = e0 + 1;
            while (e1 < (int) N_EXPERT && seg_index(gu_lt, e1) == gi &&
                                        seg_index(d_lt,  e1) == di) ++e1;
            const Segment & gs = gu_lt.segs[(size_t) gi];
            const Segment & ds = d_lt.segs[(size_t) di];
            // a run lies inside both segments by construction; the clamp is the SEAM 3 statement
            // that a slice is covered by exactly one segment per tensor.
            const int a = std::max(e0, std::max(gs.lo, ds.lo));
            const int b = std::min(e1, std::min(gs.hi, ds.hi));
            CHECK(a < b, "U2a: empty alias run [%d,%d) in %s", e0, e1, gu_lt.name.c_str());

            Slice sl;
            sl.lo = a; sl.hi = b;
            sl.gu_type = gs.type; sl.d_type = ds.type;
            sl.gu_stride = ggml_row_size(gs.type, gu_lt.ne[0]) * (size_t) gu_lt.ne[1];
            sl.d_stride  = ggml_row_size(ds.type,  d_lt.ne[0])  * (size_t) d_lt.ne[1];
            sl.gu_base = gs.bytes + (size_t) (a - gs.lo) * sl.gu_stride;
            sl.d_base  = ds.bytes + (size_t) (a - ds.lo) * sl.d_stride;
            sl.gu_nbytes = sl.gu_stride * (size_t) (b - a);
            sl.d_nbytes  = sl.d_stride  * (size_t) (b - a);
            sl.local_of.assign((size_t) N_EXPERT, -1);
            for (int e = a; e < b; ++e)
                if (mine[(size_t) e]) { sl.local_of[(size_t) e] = e - a; pos_of[(size_t) e] = e - a; }
            sl.as_gu = ggml_new_tensor_3d(alias_ctx, gs.type, gu_lt.ne[0], gu_lt.ne[1], b - a);
            sl.as_d  = ggml_new_tensor_3d(alias_ctx, ds.type,  d_lt.ne[0],  d_lt.ne[1],  b - a);
            alias_bind(sl.as_gu, gs.bytes, gs.nbytes, sl.gu_base);
            alias_bind(sl.as_d,  ds.bytes, ds.nbytes, sl.d_base);
            if (g_alias_prefault) {
                const double pc0 = alias_now_s();
                prefault_alias_slice(sl);   // U2b: the page table is built HERE, not in the loop
                g_alias_prefault_call_s += alias_now_s() - pc0;
            }
            slices.push_back(sl);
            e0 = e1;
        }
    }

    // U2b — THE ALIAS'S FAULT COST MOVES INTO THE STARTUP SLOT.  Measured: the
    // zero-copy store pays ~195 K (51 pos) / ~303 K (155 pos) in-loop minor faults on the ZFS-backed
    // mmap at ~1.2-1.7 us each = a FIXED +0.32 s per run, because the packed store used to read the
    // very same file bytes during its build memcpy and the alias defers them into the loop.  Walking
    // the member rows here — one byte per page into a volatile sink, MADV_WILLNEED first so the
    // readahead starts ahead of the touch — builds the page table before the first token, where the
    // cost is startup rather than drive.  Only MEMBERS are touched: those are exactly the rows the
    // enqueue can serve (a non-member belongs to the VRAM entry), and touching all of
    // [0,N_EXPERT) would read the resident rows a second time and pull ~30.5 GiB instead of ~19 GiB.
    void prefault_alias_slice(const Slice & sl) {
        static volatile unsigned char sink = 0;   // the touch must not be optimised away
        static const size_t PG = 4096;
        for (int e = sl.lo; e < sl.hi; ++e) {
            const int row = sl.local_of[(size_t) e];
            if (row < 0) continue;
            for (int k = 0; k < 2; ++k) {
                const uint8_t * base = k ? sl.d_base  + (size_t) row * sl.d_stride
                                         : sl.gu_base + (size_t) row * sl.gu_stride;
                const size_t nb = k ? sl.d_stride : sl.gu_stride;
                const double a0 = alias_now_s();
                madvise((void *) base, nb, MADV_WILLNEED);
                const double a1 = alias_now_s();
                g_alias_prefault_advise_s += a1 - a0;
                g_alias_prefault_bytes += nb;
                for (size_t off = 0; off < nb; off += PG) sink = (unsigned char) (sink + base[off]);
                g_alias_prefault_touch_s += alias_now_s() - a1;
            }
        }
    }

    // bind `t` to the mmap at `row0`, inside the whole segment [seg_base, seg_base+seg_nbytes).
    // ggml_backend_cpu_buffer_from_ptr's buffer does not own the pointer (free_buffer == NULL,
    // ggml-backend.cpp:2518) and the size is the segment's, so the tensor-allocation CHECK in
    // ggml_backend_tensor_alloc (:2229) is satisfied by the run's own bytes.
    void alias_bind(ggml_tensor * t, const uint8_t * seg_base, size_t seg_nbytes,
                    const uint8_t * row0) {
        CHECK((uintptr_t) seg_base % 32 == 0 && (uintptr_t) row0 % 32 == 0,
              "U2a: alias needs 32-byte alignment (segment %p row %p)",
              (const void *) seg_base, (const void *) row0);
        CHECK((size_t) (row0 - seg_base) + ggml_nbytes(t) <= seg_nbytes,
              "U2a: alias rows %zu..%zu exceed the segment's %zu bytes",
              (size_t) (row0 - seg_base), (size_t) (row0 - seg_base) + ggml_nbytes(t), seg_nbytes);
        ggml_backend_buffer_t b = ggml_backend_cpu_buffer_from_ptr((void *) seg_base, seg_nbytes);
        CHECK(b, "U2a: cpu_buffer_from_ptr failed");
        alias_bufs.push_back(b);
        CHECK(ggml_backend_tensor_alloc(b, t, (void *) row0) == GGML_STATUS_SUCCESS,
              "U2a: tensor_alloc failed for an aliased weight tensor");
        CHECK(t->data == row0 && t->buffer == b, "U2a: alias tensor does not point at the mmap");
    }

    // H1 — the weight upload.  Interval slices are ONE contiguous copy (the pre-H1 call, byte for
    // byte); set-backed slices copy one expert-sized row per member from ITS OWN segment, at the
    // member's slot offset.  Same bytes either way for a contiguous increasing set.
    //
    // U2a: an ALIASED store has nothing to upload — its tensors already point at the source bytes
    // (build_alias_slices), which is the whole point of the change: admission/eviction stops
    // costing a host memcpy because no copy exists to update.
    void upload_slice(Slice & sl) {
        if (aliased) return;
        if (sl.members.empty()) {
            ggml_backend_tensor_set(sl.as_gu, sl.gu_base, 0, sl.gu_nbytes);
            ggml_backend_tensor_set(sl.as_d,  sl.d_base,  0, sl.d_nbytes);
            return;
        }
        for (size_t i = 0; i < sl.members.size(); ++i) {
            ggml_backend_tensor_set(sl.as_gu, sl.gu_src[i], i * sl.gu_stride, sl.gu_stride);
            ggml_backend_tensor_set(sl.as_d,  sl.d_src[i],  i * sl.d_stride,  sl.d_stride);
        }
    }

    // U2b — ADMIT expert `e` INTO SLOT `slot`, returning the expert that was evicted.
    //
    // This is the VRAM side of a rotation swap, and it is a MEMBERSHIP SWAP, not a rebuild: the
    // admitted expert's gate_up and down rows are queued onto the store's own backend stream at
    // `slot`'s offset (`ggml_backend_tensor_set_async`, the same call class enqueue already uses for
    // its ids/xb H2D), and then the three tables that describe the slot are updated in the same
    // order the enqueue reads them: the slice's (expert -> slot) table, the store's `members` and
    // `pos_of`.  Nothing is re-packed, no tensor is re-created, no context is touched.
    //
    // THE ORDERING THE CALLER MUST GUARANTEE, stated because it is the whole correctness argument:
    // the backend's stream must be IDLE when this is called — the loop calls it at the top of
    // forward(), which is after the previous position's drain_all synchronized this store's backend,
    // and before this position's enqueue.  It is never called between a layer's enqueue and its
    // drain.  The source bytes are the mmap (read-only file pages, alive for the whole run), so the
    // async copy's source stays valid until it completes.
    //
    // FLAT-FILE ONLY, refused loudly: a grouped file's set store has one slice per (type pair) and
    // that path has zero test coverage (L43 FINDING-3) — rotation must not extend an unmeasured claim.
    int replace_slot(int slot, int e) {
        CHECK(!aliased && is_set(), "U2b rotation: replace_slot needs a PACKED SET store (%s)", loc_name);
        CHECK(slices.size() == 1,
              "U2b rotation: replace_slot is FLAT-FILE only — %s has %zu slices (grouped file: "
              "FINDING-3, the grouped SET path is unmeasured)", loc_name, slices.size());
        CHECK(slot >= 0 && slot < n_e, "U2b rotation: slot %d outside [0,%d)", slot, n_e);
        CHECK(e >= 0 && (size_t) e < pos_of.size(), "U2b rotation: expert %d out of range", e);
        CHECK(pos_of[(size_t) e] < 0, "U2b rotation: expert %d is already resident at slot %d",
              e, pos_of[(size_t) e]);
        Slice & sl = slices[0];
        CHECK((int) sl.members.size() == n_e, "U2b rotation: slice/set size mismatch (%zu vs %d)",
              sl.members.size(), n_e);
        const int old = members[(size_t) slot];
        CHECK(old != e, "U2b rotation: expert %d already at slot %d", e, slot);

        const int gi = seg_index(*gu_lt_p, e), di = seg_index(*d_lt_p, e);
        const int gi0 = seg_index(*gu_lt_p, old), di0 = seg_index(*d_lt_p, old);
        CHECK(gi == gi0 && di == di0,
              "U2b rotation: expert %d and slot %d's expert %d live in different segments — a swap "
              "across a segment boundary needs a different slice (grouped file, refused)", e, slot, old);
        const Segment & gs = gu_lt_p->segs[(size_t) gi];
        const Segment & ds = d_lt_p->segs[(size_t) di];
        const size_t gu_stride = ggml_row_size(gs.type, gu_lt_p->ne[0]) * (size_t) gu_lt_p->ne[1];
        const size_t d_stride  = ggml_row_size(ds.type,  d_lt_p->ne[0])  * (size_t) d_lt_p->ne[1];
        CHECK(gu_stride == sl.gu_stride && d_stride == sl.d_stride,
              "U2b rotation: %s slice stride %zu/%zu does not match the source %zu/%zu",
              loc_name, sl.gu_stride, sl.d_stride, gu_stride, d_stride);
        const uint8_t * gsrc = gs.bytes + (size_t) (e - gs.lo) * gu_stride;
        const uint8_t * dsrc = ds.bytes + (size_t) (e - ds.lo) * d_stride;

        CHECK(sl.as_gu->ne[2] == n_e && sl.as_d->ne[2] == n_e,
              "U2b rotation: %s set tensor holds %lld/%lld rows, not %d",
              loc_name, (long long) sl.as_gu->ne[2], (long long) sl.as_d->ne[2], n_e);
        ggml_backend_tensor_set_async(backend, sl.as_gu, gsrc, (size_t) slot * gu_stride, gu_stride);
        ggml_backend_tensor_set_async(backend, sl.as_d,  dsrc, (size_t) slot * d_stride,  d_stride);

        members[(size_t) slot] = e;
        pos_of[(size_t) old] = -1;
        pos_of[(size_t) e]   = slot;
        sl.members[(size_t) slot] = e;
        sl.gu_src[(size_t) slot] = gsrc;
        sl.d_src[(size_t) slot]  = dsrc;
        sl.local_of[(size_t) old] = -1;
        sl.local_of[(size_t) e]   = slot;
        return old;
    }

    // U2b — the ALIAS side of the same swap: flip the membership PREDICATE.  No bytes move, because
    // an aliased store's tensor already spans the file's whole expert block — the row it serves is
    // the row the expert occupies.  Admitting an expert into the VRAM store means REMOVING it here
    // (the two stores are complements, so the exactly-one-home audit keeps holding), and evicting
    // means adding it back.
    void set_member(int e, bool member) {
        CHECK(aliased && is_set(), "U2b rotation: set_member needs an ALIASED store (%s)", loc_name);
        CHECK(slices.size() == 1,
              "U2b rotation: set_member is FLAT-FILE only — %s has %zu slices", loc_name, slices.size());
        Slice & sl = slices[0];
        CHECK(sl.lo <= e && e < sl.hi, "U2b rotation: expert %d outside the alias run [%d,%d)", e, sl.lo, sl.hi);
        if (member) {
            CHECK(sl.local_of[(size_t) e] < 0, "U2b rotation: expert %d is already an alias member", e);
            sl.local_of[(size_t) e] = e - sl.lo;
            pos_of[(size_t) e]      = e - sl.lo;
            members.push_back(e);              // keep the LIST honest too, not only the predicate
        } else {
            CHECK(sl.local_of[(size_t) e] >= 0, "U2b rotation: expert %d is not an alias member", e);
            sl.local_of[(size_t) e] = -1;
            pos_of[(size_t) e]      = -1;
            for (size_t i = 0; i < members.size(); ++i)
                if (members[i] == e) { members.erase(members.begin() + (long) i); break; }
        }
        n_e = (int) members.size();
    }

    // U2b — SLOT IDENTITY BY BYTES: read every resident slot back and memcmp it against the file row
    // of the expert the store CLAIMS holds it.  The audit proves the tables agree with each other;
    // this proves the tables agree with the WEIGHTS, which is the one property a table-only gate
    // cannot see (a copy landed at the wrong offset is invisible to every pointer check).
    // Synchronous on purpose: it runs once, after the loop, outside any timing bracket.
    // U2b — the same check for ONE slot: called right after a swap when --rotate-verify is on, so a
    // row that landed at the wrong offset is caught at the swap that wrote it, not only at run end.
    long verify_slot(int slot) const {
        CHECK(!aliased && is_set() && slices.size() == 1, "U2b verify: needs a flat SET-backed store");
        const Slice & sl = slices[0];
        ggml_backend_synchronize(backend);
        std::vector<uint8_t> got(sl.gu_stride > sl.d_stride ? sl.gu_stride : sl.d_stride);
        long bad = 0;
        const int e = members[(size_t) slot];
        ggml_backend_tensor_get(sl.as_gu, got.data(), (size_t) slot * sl.gu_stride, sl.gu_stride);
        if (std::memcmp(got.data(), sl.gu_src[(size_t) slot], sl.gu_stride) != 0) ++bad;
        ggml_backend_tensor_get(sl.as_d, got.data(), (size_t) slot * sl.d_stride, sl.d_stride);
        if (std::memcmp(got.data(), sl.d_src[(size_t) slot], sl.d_stride) != 0) ++bad;
        const Segment & gs = gu_lt_p->segs[(size_t) seg_index(*gu_lt_p, e)];
        const Segment & ds = d_lt_p->segs[(size_t)  seg_index(*d_lt_p,  e)];
        if (sl.gu_src[(size_t) slot] != gs.bytes + (size_t) (e - gs.lo) * sl.gu_stride) ++bad;
        if (sl.d_src[(size_t) slot]  != ds.bytes + (size_t) (e - ds.lo) * sl.d_stride)  ++bad;
        return bad;
    }

    // U2c — THE INDEX TABLE'S OWN AUDIT.  `sl.local_of` is WRITTEN by
    // replace_slot and READ by enqueue's membership test and slot mapping, and nothing checked it
    // after a refresh: the audit's set branch checked pos_of only, and --rotate-verify checks BYTES
    // and source pointers, so a wrong local_of entry serves the WRONG EXPERT'S ROW while the bytes,
    // the pos_of table and the exit code all stay correct.  Both directions are asserted here over
    // the RUNNING table (a rotated store's current membership):
    //   (i)  a member's entry is its SLICE slot — and for a single-slice (flat-file) store that is
    //        also its store slot pos_of, because the policy reads pos_of and the kernel reads
    //        local_of and the two must name the same row;
    //   (ii) a NON-member's entry is -1.  Checked as "an entry that is >= 0 must point back at its
    //        own expert" (local_of[x] == k  =>  members[k] == x), which is exactly what a STALE
    //        entry left behind by an eviction violates, and it is O(1) per expert.
    // COST: O(n_e + N_EXPERT) integer compares per store per call (48 + 128 here, one slice) —
    // called once per layer per REFRESH and once after the loop, never per token per layer.
    void audit_index_table(int layer) const {
        if (!is_set() || aliased) return;
        for (const Slice & sl : slices) {
            for (size_t k = 0; k < sl.members.size(); ++k) {
                const int e   = sl.members[k];
                const int lof = sl.local_of[(size_t) e];
                CHECK(lof == (int) k,
                      "layer %d: %s SET LOCAL_OF mismatch at expert %d — slot %zu holds it, its "
                      "local_of entry says %d (the table enqueue READS: the kernel would be served "
                      "another expert's row with the bytes and pos_of all correct)",
                      layer, loc_name, e, k, lof);
                if (slices.size() == 1)
                    CHECK(pos_of[(size_t) e] == lof,
                          "layer %d: %s SET LOCAL_OF/POS_OF disagreement at expert %d (local_of %d, "
                          "pos_of %d) — the kernel and the policy would name different rows",
                          layer, loc_name, e, lof, pos_of[(size_t) e]);
            }
            for (int x = 0; x < (int) N_EXPERT; ++x) {
                const int lof = sl.local_of[(size_t) x];
                if (lof < 0) continue;
                const int held = lof < (int) sl.members.size() ? sl.members[(size_t) lof] : -1;
                CHECK(held == x,
                      "layer %d: %s SET LOCAL_OF stale entry at expert %d -> slot %d, which holds "
                      "expert %d (a non-member's entry must be -1)",
                      layer, loc_name, x, lof, held);
            }
        }
    }

    long verify_slot_bytes() const {
        CHECK(!aliased && is_set() && slices.size() == 1, "U2b verify: needs a flat SET-backed store");
        const Slice & sl = slices[0];
        ggml_backend_synchronize(backend);
        std::vector<uint8_t> got(sl.gu_stride > sl.d_stride ? sl.gu_stride : sl.d_stride);
        long bad = 0;
        for (int slot = 0; slot < n_e; ++slot) {
            const int e = members[(size_t) slot];
            ggml_backend_tensor_get(sl.as_gu, got.data(), (size_t) slot * sl.gu_stride, sl.gu_stride);
            if (std::memcmp(got.data(), sl.gu_src[(size_t) slot], sl.gu_stride) != 0) ++bad;
            ggml_backend_tensor_get(sl.as_d, got.data(), (size_t) slot * sl.d_stride, sl.d_stride);
            if (std::memcmp(got.data(), sl.d_src[(size_t) slot], sl.d_stride) != 0) ++bad;
            // and the source row must be the file's row for that expert
            const Segment & gs = gu_lt_p->segs[(size_t) seg_index(*gu_lt_p, e)];
            const Segment & ds = d_lt_p->segs[(size_t)  seg_index(*d_lt_p,  e)];
            if (sl.gu_src[(size_t) slot] != gs.bytes + (size_t) (e - gs.lo) * sl.gu_stride) ++bad;
            if (sl.d_src[(size_t) slot]  != ds.bytes + (size_t) (e - ds.lo) * sl.d_stride)  ++bad;
        }
        return bad;
    }

    void build_graph(Slice & sl, int c, int n) {
        StoreGraph g;
        g.c = c; g.n = n;
        g.ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, c, n);
        g.w   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, c, n);
        g.xb  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, H_DIM, 1, n);

        g.gu = ggml_mul_mat_id(ctx, sl.as_gu, g.xb, g.ids);              // [2*FF, c, n]
        ggml_tensor * gate = ggml_view_3d(ctx, g.gu, FF_DIM, c, n,
                                          g.gu->nb[1], g.gu->nb[2], 0);
        ggml_tensor * up   = ggml_view_3d(ctx, g.gu, FF_DIM, c, n,
                                          g.gu->nb[1], g.gu->nb[2], FF_DIM * g.gu->nb[0]);
        ggml_tensor * h    = ggml_swiglu_split(ctx, gate, up);           // [FF, c, n]
        g.d  = ggml_mul_mat_id(ctx, sl.as_d, h, g.ids);                  // [H, c, n]
        CHECK(ggml_prec_set_src(g.d, GGML_PREC_F32, 1), "prec_set_src failed"); // MISTRAL4 F32 src1
        ggml_tensor * dm = ggml_mul(ctx, g.d, g.w);                      // [H, c, n]
        ggml_tensor * acc = ggml_view_2d(ctx, dm, H_DIM, n, dm->nb[2], 0);
        for (int e = 1; e < c; ++e) {
            ggml_tensor * col = ggml_view_2d(ctx, dm, H_DIM, n, dm->nb[2], e * dm->nb[1]);
            acc = ggml_add(ctx, acc, col);
        }
        g.out = acc;
        g.gf = ggml_new_graph_custom(ctx, 64, false);
        g.ids_buf.resize((size_t) c * n);
        g.w_buf.resize((size_t) c * n);
        g.xb_buf.resize((size_t) H_DIM * n);
        sl.graphs[{c, n}] = g;
    }

    // enqueue this store's groups for the given ids/x/w.  One pass per SLICE: a token's
    // routed experts are partitioned across the store's slices, and each slice computes the
    // subset that falls inside it (its own tensor, its own uniform type).
    //   ids4  [N_USED x nt] i32   (global expert indices, top-k order)
    //   w4    [N_USED x nt] f32   (routing weights)
    //   x     [H x nt]      f32   (layer input activations)
    void enqueue(const int32_t * ids4, const float * w4, const float * x, int nt) {
        pending.clear();
        out_used = 0;
        for (Slice & sl : slices) {
            // per-token slice-local slots, then group tokens by count (U7: the two scratch
            // vectors are members now, cleared per call so their capacity survives the loop)
            std::vector<int> (&toks)[N_USED + 1] = toks_scratch;
            for (int c = 0; c <= N_USED; ++c) toks[c].clear();
            tslots_scratch.resize((size_t) nt);
            for (int t = 0; t < nt; ++t) tslots_scratch[(size_t) t].clear();
            for (int t = 0; t < nt; ++t) {
                int c = 0;
                for (int e = 0; e < N_USED; ++e) {
                    const int ex = ids4[e + N_USED*t];
                    // H1: THE MEMBERSHIP TEST IS THE SLICE'S OWN TABLE (built by slice_index from
                    // the slice's set, or from its interval).  Pre-H1 this was `ex >= sl.lo &&
                    // ex < sl.hi` and the slot was `ex - sl.lo`; local_of[e] == e - sl.lo for an
                    // interval slice, so the identity set is unchanged by construction.
                    CHECK(ex >= 0 && ex < (int) N_EXPERT, "router id %d out of range", ex);
                    if (sl.local_of[(size_t) ex] >= 0) { tslots_scratch[(size_t) t].push_back(e); ++c; }
                }
                toks[c].push_back(t);
            }

            for (int c = 1; c <= N_USED; ++c) {
                if (toks[c].empty()) continue;
                int n = (int) toks[c].size();
                StoreGraph & g = sl.graphs.at({c, n});
#ifdef STORE_INSTR
                const double tp0 = store_now_s();
#endif
                for (int i = 0; i < n; ++i) {
                    int t = toks[c][i];
                    for (int s = 0; s < c; ++s) {
                        int gs = tslots_scratch[(size_t) t][s];
                        g.ids_buf[s + c*i] = sl.local_of[(size_t) ids4[gs + N_USED*t]];   // slice-local slot
                        g.w_buf[s + c*i]   = w4[gs + N_USED*t];
                    }
                    for (int64_t j = 0; j < H_DIM; ++j)
                        g.xb_buf[j + H_DIM*i] = x[j + H_DIM*t];
                }
                // async set: queues the input H2D on the store's own stream and returns, so
                // the RAM store's compute can overlap it (SEAM 1: a streamed store's fill() is
                // an async H2D issued ahead of use — this is the VRAM store doing exactly that).
                ggml_backend_tensor_set_async(backend, g.ids, g.ids_buf.data(), 0, (size_t) c*n*sizeof(int32_t));
                ggml_backend_tensor_set_async(backend, g.w,   g.w_buf.data(),   0, (size_t) c*n*sizeof(float));
                ggml_backend_tensor_set_async(backend, g.xb,  g.xb_buf.data(),  0, (size_t) H_DIM*n*sizeof(float));
#ifdef STORE_INSTR
                const double tp1 = store_now_s();
#endif
                // async for VRAM (queues on the HIP stream, returns); blocking for RAM.
                ggml_backend_graph_compute_async(backend, g.gf);
#ifdef STORE_INSTR
                const double tp2 = store_now_s();
                if (backend == g_store_cpu_be) { g_store_pack_ram_s  += tp1 - tp0; g_store_launch_ram_s  += tp2 - tp1; }
                else                           { g_store_pack_vram_s += tp1 - tp0; g_store_launch_vram_s += tp2 - tp1; }
                ++g_store_group_n;
#endif

                Pending p{};
                p.c = c; p.n = n; p.g = &g; p.toks = toks[c];
                p.out_off = out_used;
                out_used += (int) (H_DIM * n);
                CHECK(out_used <= out_host_n, "U7 output staging overflow in %s store", loc_name);
                for (int i = 0; i < n; ++i) p.tslots.push_back(tslots_scratch[(size_t) toks[c][i]]);
                // U7: THE OUTPUT D2H, QUEUED NOW (not pulled at drain) so it runs beneath the
                // other store's blocking compute.  Stream-ordered after this store's graph on
                // this backend's stream; the host read happens after the caller's synchronize.
#ifdef STORE_INSTR
                const double tg0 = store_now_s();
#endif
                if (!g_store_pull_at_drain)
                    ggml_backend_tensor_get_async(backend, g.out, out_host + p.out_off, 0,
                                                  (size_t) H_DIM*n*sizeof(float));
#ifdef STORE_INSTR
                g_store_get_s += store_now_s() - tg0;
#endif
                pending.push_back(p);
            }
        }
    }

    // wait for the store's work and pull results to host.
    //   y      [H x nt] zero-initialised by caller — this store's columns are WRITTEN here
    //   if export_intermediates: gu_out [2*FF x N_USED x nt], d_out [H x N_USED x nt]
    //   sync_backend (U5): the default keeps the pre-U5 shape (synchronize, then read) for
    //   every existing caller; a LIST-level driver that already synchronized each distinct
    //   backend once passes false so the per-store wait is not paid twice.
    //   U7: the output bytes were ALREADY copied to out_host by the get_async in enqueue(), so
    //   this is a pure host combine (the wait is the caller's / the one sync_backend does).
    void drain(float * y, int nt, bool export_intermediates,
               float * gu_out, float * d_out, bool sync_backend = true) {
        if (sync_backend) ggml_backend_synchronize(backend);
        for (const Pending & p : pending) {
            int c = p.c, n = p.n;
            StoreGraph & g = *p.g;
            // U7: normally the bytes are ALREADY in out_host (queued at enqueue); the legacy
            // arm (g_store_pull_at_drain) pulls them here with a blocking get, as before.
            std::vector<float> legacy;
            const float * out = out_host + p.out_off;
            if (g_store_pull_at_drain) {
                legacy.resize((size_t) H_DIM * n);
#ifdef STORE_INSTR
                const double tl0 = store_now_s();
#endif
                ggml_backend_tensor_get(g.out, legacy.data(), 0, (size_t) H_DIM*n*sizeof(float));
#ifdef STORE_INSTR
                g_store_get_s += store_now_s() - tl0;
#endif
                out = legacy.data();
            }
#ifdef STORE_INSTR
            const double tc0 = store_now_s();
#endif
            for (int i = 0; i < n; ++i) {
                int t = p.toks[i];
                for (int64_t j = 0; j < H_DIM; ++j)
                    y[j + H_DIM*t] += out[j + H_DIM*i];   // ADD: stores contribute disjoint expert sets
            }
#ifdef STORE_INSTR
            g_store_combine_s += store_now_s() - tc0;
            ++g_store_get_n;
#endif
            if (export_intermediates) {
                std::vector<float> gu((2*FF_DIM)*c*n), d(H_DIM*c*n);
                ggml_backend_tensor_get(g.gu, gu.data(), 0, (size_t)(2*FF_DIM)*c*n*sizeof(float));
                ggml_backend_tensor_get(g.d,  d.data(),  0, (size_t) H_DIM*c*n*sizeof(float));
                for (int i = 0; i < n; ++i) {
                    int t = p.toks[i];
                    for (int s = 0; s < c; ++s) {
                        int gs = p.tslots[i][s];
                        for (int64_t r = 0; r < 2*FF_DIM; ++r)
                            gu_out[r + (2*FF_DIM)*gs + (2*FF_DIM)*N_USED*t] = gu[r + (2*FF_DIM)*s + (2*FF_DIM)*c*i];
                        for (int64_t r = 0; r < H_DIM; ++r)
                            d_out[r + H_DIM*gs + H_DIM*N_USED*t] = d[r + H_DIM*s + H_DIM*c*i];
                    }
                }
            }
        }
    }
};

// ---------------------------------------------------------------------------
// The store list (SEAM 1 + SEAM 3): ranges DERIVED from the geometry, never hardcoded.
// ---------------------------------------------------------------------------
struct StoreSpec {
    Store::Location loc;
    int lo, hi;
    // H1: the membership axis.  by_set = true => the entry's membership is `members` (an ORDERED
    // set, slot i = members[i]); false => the interval [lo,hi), today's shape.  A set entry with
    // an EMPTY member list is a legal INERT entry (n_e = 0), the same as an empty interval.
    bool by_set = false;
    std::vector<int> members;
};

// Every segment boundary of either logical tensor becomes a store boundary, plus the requested
// resident-expert cut and any extra cuts.  A flat file yields exactly the two stores M0 has
// always measured ([0,n_res) VRAM / [n_res,128) RAM); a grouped file yields one boundary per
// group, and a store whose range spans groups is representable (Store cuts it into one SLICE
// per (range x type)).  An EMPTY store (n_e = 0) is a legal list entry.
//
// H1: `res_set` (optional, last so every pre-H1 call site compiles unchanged) replaces the CUT
// with a SET and keeps the list's SHAPE — one VRAM entry holding the ordered resident set, one
// RAM entry holding its complement (increasing) — so M0's semantics and the falsifier's store
// list are the same object they always were, with a policy-driven membership.  `n_res` is then
// not consulted: the set IS the resident count (the caller CHECKs they agree).
static std::vector<StoreSpec> derive_store_specs(const LogicalTensor & gu_lt, const LogicalTensor & d_lt,
                                                 int n_res, const std::vector<int> & extra_cuts,
                                                 bool add_empty_store,
                                                 const std::vector<int> * res_set = nullptr) {
    std::vector<StoreSpec> specs;
    if (res_set && !res_set->empty()) {
        std::vector<char> in((size_t) N_EXPERT, 0);
        StoreSpec vram;
        vram.loc = Store::VRAM; vram.by_set = true; vram.members = *res_set;
        for (int e : vram.members) {
            CHECK(e >= 0 && e < (int) N_EXPERT, "resident set id %d out of range", e);
            CHECK(!in[(size_t) e], "resident set repeats expert %d", e);
            in[(size_t) e] = 1;
        }
        vram.lo = *std::min_element(vram.members.begin(), vram.members.end());
        vram.hi = *std::max_element(vram.members.begin(), vram.members.end()) + 1;
        StoreSpec ram;
        ram.loc = Store::RAM; ram.by_set = true;
        for (int e = 0; e < (int) N_EXPERT; ++e) if (!in[(size_t) e]) ram.members.push_back(e);
        ram.lo = ram.members.empty() ? 0 : ram.members.front();
        ram.hi = ram.members.empty() ? 0 : ram.members.back() + 1;
        specs.push_back(vram);
        specs.push_back(ram);   // possibly EMPTY (a set that covers every expert): a legal entry
        if (add_empty_store) specs.push_back({ Store::RAM, 0, 0, false, {} });
        return specs;
    }
    std::vector<int> cuts = { 0, n_res, (int) N_EXPERT };
    for (int c : extra_cuts) cuts.push_back(c);
    for (const LogicalTensor * lt : { &gu_lt, &d_lt })
        for (const Segment & s : lt->segs) { cuts.push_back(s.lo); cuts.push_back(s.hi); }
    std::sort(cuts.begin(), cuts.end());
    cuts.erase(std::unique(cuts.begin(), cuts.end()), cuts.end());
    for (size_t k = 0; k + 1 < cuts.size(); ++k) {
        int lo = std::max(0, cuts[k]), hi = std::min((int) N_EXPERT, cuts[k+1]);
        if (hi <= lo) continue;                       // a degenerate cut adds no store
        specs.push_back({ lo < n_res ? Store::VRAM : Store::RAM, lo, hi, false, {} });
    }
    if (add_empty_store) specs.push_back({ Store::RAM, 0, 0, false, {} });   // EMPTY entry (n_e = 0)
    CHECK(!specs.empty(), "store list is empty");
    return specs;
}
