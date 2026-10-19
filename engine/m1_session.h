// engine/m1_session.h — THE REUSABLE SESSION SEAM (aubier server plan §A, unit U4).
//
// ONE engine, TWO drivers:
//   builds/engine/m1        engine/m1_main.cpp parses the flags and calls loop_run(), which is
//                           engine/m1_engine.cpp's whole single-shot run (validation, the
//                           session, the token loop, the audits, the verdict);
//   builds/engine/m1_server (U5) its own HTTP loop over the M1Session API below — the SAME
//                           M1Session::open/step, so the server cannot drift from the binary.
//
// Why this and not a second copy of the loop (plan §A): a parity test between two copies measures
// the copy, not the product.  There is ONE LoopArgs (the server fills the same product flag set —
// no mapping layer) and ONE per-position body (M1Session::step).
//
// WHAT MOVED, verbatim, out of m1_engine.cpp (only the two noted edits differ):
//   * struct LoopArgs      (was m1_engine.cpp:995-1131)  + the ONE added field seq_len, below;
//   * struct LayerCheckArgs (was m1_engine.cpp:620-637)  — shared only because main() moved to
//     m1_main.cpp and constructs it; it is the --layer-check path's argument set.
// DEFAULT_MODEL was `static const char *` in m1_engine.cpp; it is `inline constexpr` here so the
// two translation units share one definition instead of each carrying its own copy.
#ifndef M1_SESSION_H
#define M1_SESSION_H

#include <cstdint>
#include <string>
#include <vector>

struct gguf_context;

// engine/m1_engine.cpp defines STORE_INSTR 1 before it includes engine/store.h, which turns on the
// U7 get/combine TIMING counters. MEASURED (store.h:62 and :810-901, every occurrence): the macro
// guards only FILE-STATIC accumulators and the brackets around them — no member of Store is
// conditionally declared — so it does NOT change the class's shape, and a TU that includes store.h
// WITHOUT it (engine/m1_layer.cpp does exactly that) is behaviourally identical, merely
// uninstrumented. A header comment here once claimed the opposite ("an ODR mismatch, not just
// missing instrumentation"); that was wrong and is corrected rather than left to mislead the next
// unit. This header still deliberately does not include m1_layer.h, so a caller that only drives
// the session never has to know either way.

inline constexpr const char * DEFAULT_MODEL = "models/Mistral-Small-4-119B-2603-UD-IQ2_XXS.gguf";

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

    // ---- U4 (the session carve) — THE ONE ADDED FIELD, and why it is here ------------------
    // The number of positions this run walks (n_prompt + n_decode, or the whole file in the
    // scoring mode).  M1Session::open() RESOLVES `--kv-max <= 0` (U2's PRE-FIX width = the sequence
    // length) from, in this order: `kv_max` itself, this field, and the boot prompt published
    // through m1_session_boot_prompt().  A caller that has no boot prompt (the server: its prompt
    // is per request) sets this ONE field and needs no global — it is the per-session half of the
    // resolution.  Every other field above is the verbatim m1_engine.cpp struct.
    int    seq_len = 0;
};

// ---------------------------------------------------------------------------
// --layer-check: the one-layer forward's arguments (U2).
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
// ---------------------------------------------------------------------------
// THE SESSION (opaque).  open() builds every engine object the run REUSES — the GGUF mmap, the
// CPU and ROCm0 backends, the embedding and head graphs, the 36 layer graphs and their weights,
// the store lists and their VRAM contents — and prints the configuration block ("# kv width /
// # arm / # n_res / # backend / # store split / # resident VRAM / # rotate" and the rest), so both
// binaries print the SAME block from the SAME code.  step() is the per-position body.
// ---------------------------------------------------------------------------
struct M1Session;

// Build the engine state for `a`: the GGUF mmap, both backends, the embedding/head graphs, the 36
// layer graphs and their weights, the store lists and their VRAM contents — and the ONE resolution
// of the tri-state core-op hatches (--m4-*) and the in-loop instrumentation gates, so every caller
// gets the same op sequence from the same code (a resolution done by a CALLER would leave this
// entry point building the stock sequence while still printing "# gpu core = ON").
//
// FAILURE MODEL, stated exactly (U4FIX item 3): open() prints a FAIL line and returns false (with
// *out = NULL) only where the engine has an answer for the caller — the reference KV-width mismatch
// and the --damage probe that matched nothing.  Every other refusal is a CHECK, i.e. FATAL/exit 1,
// because the engine's failure model is "an engine invariant that does not hold kills the process"
// (plan R7).  A server must therefore pre-check everything CLIENT-caused; it cannot catch these.
// In particular a caller that passes `kv_max <= 0` with no seq_len and no published boot length is
// a FATAL, not a false return.
//
// The caller's own prompt is NOT read here: the m1 binary reads it in loop_run, the server reads a
// request's ids per request.  The KV width comes from `a.kv_max`, else `a.seq_len`, else the boot
// length published through m1_session_boot_prompt().  ONE SESSION PER PROCESS is the supported
// envelope (the CPU backend is not torn down; a second session would allocate a second one).
bool m1_session_open(const LoopArgs & a, M1Session ** out);

// The boot prompt's VIEW, published by the caller immediately before m1_session_open(): open()
// prints the configuration block's three prompt lines ("# SCORE"/"# tokens", "# ids") from it, so
// both binaries print that block from one place, and resolves a `kv_max <= 0` from n_total.  The
// ids are NOT copied and must outlive the open() call — open() CLEARS the view before returning,
// so it can never outlive its usefulness.  A caller that publishes nothing (the server — its
// prompt is per request) gets the block without those lines.
//
// WHAT IS *NOT* PLUMBED THROUGH THE SESSION, deliberately (U4FIX item 7): `LoopArgs::route_dump`
// (U6's per-(layer,pos) trace) — the trace file's lifetime is the whole run, not a session, and a
// per-request dump needs a naming policy that U5/U6 owns; loop_run opens and closes it.  Everything
// else in LoopArgs that changes what the engine DOES (the m4_* hatches, instr/core_nodes/
// res_stats/store_pull_at_drain, rotate/admit-k/audit scope, alias-prefault, fold, core-fuse) is
// resolved inside open() and therefore works for any caller.
void m1_session_boot_prompt(const std::vector<int32_t> * ids, int n_prompt, int n_total, bool scoring);

// ONE position: feed `tok` at position `pos` (which writes KV cell `pos`), producing position
// `pos`'s residual for the head.  When `logits_row` is non-null the head runs and writes its
// n_vocab logits there; when it is null the head is SKIPPED — that is the caller's decision and it
// is the m1 binary's existing behaviour (the decode path has no logits row before ref_pos0, and the
// scoring path has no row for its last position).
//
// RETURNS FALSE, with the reason on stderr, when `pos` is outside [0, m1_session_kv_max()) — the
// session REFUSES an out-of-range position instead of writing past the KV cache and attending a
// zero mask (the m1 binary's structural `kv_max >= n_total` guard is a CALLER contract here), and
// when the session was never opened.  A caller must therefore budget its positions against
// m1_session_kv_max().
bool m1_session_step(M1Session * s, int32_t tok, int pos, float * logits_row);

// Put the rotation policy back to its BOOT state so a second run of positions equals a fresh-boot
// one: the 2Q state (ghost window, slot stamps, pending/candidate queues, the per-layer counters)
// is rebuilt, and the VRAM MEMBERSHIP is replayed back to the boot permutation (the store split
// changes the float accumulation grouping, so a different membership is a different rounding and
// the chain is chaotic past ~100 positions — plan §E).  No-op when the session was not opened with
// --rotate.  The KV cache is NOT touched: forward() masks every cell >= the live length and every
// cell below it was written by this run, so a stale cell is unreadable by construction.
// COST: up to three replace_slot calls per DRIFTED slot (each = 2 async H2D row copies), i.e.
// bounded by a full resident-set upload; U5 measures it on the product path and prints it.
// It runs the store's own standing audit on the restored state, and drains the store's stream
// before returning, so a close() straight afterwards is safe.
// NOTE: U4 ships the seam; U5 (the server) is its caller and owns its behavioural proof (plan §H
// U5).  U4 exercised it with a throwaway rig only, and on the CPU-only arm over 35 positions the
// membership drift moved no float at all — so "second request == fresh boot" is NOT discriminating
// in that configuration.  Compare on the product path, over a long chain.
void m1_session_reset_rotation(M1Session * s);

int  m1_session_kv_max(const M1Session * s);      // the resolved KV width (`--kv-max`, or seq_len)
int  m1_session_n_vocab(const M1Session * s);     // the head's vocab (a logits row's length)
void m1_session_close(M1Session * s);             // frees everything open() built; NULL is a no-op

// THE MODEL'S OPEN GGUF CONTEXT (U5). The session owns the ONE parse of the 61 GB file, and the
// tokenizer (engine/m1_tok.cpp) and the chat-template pin (engine/m1_chat.cpp) both read the vocab
// and the tokenizer.chat_template KV out of that same context — so a caller that needs them does not
// mmap and re-parse the file a second time. The pointer is valid until m1_session_close().
gguf_context * m1_session_gctx(const M1Session * s);

// IS THE ROTATION AT ITS BOOT STATE? (U5's per-request entry invariant, plan §G1.) True when no
// layer is rotating, and otherwise when EVERY rotating layer is exactly where open() left it: the
// VRAM membership == the boot membership, the ghost window all zero, the slot stamps == the boot
// derivation (s - n_e), the pending/candidate queues empty and every counter (pos, admitted,
// evicted, refreshes, audits, vbad) zero. `why` receives the first violation found, naming the
// layer and the field — a server answers 500 with it rather than serving a drifted state.
//
// This is the observable form of m1_session_reset_rotation(), i.e. the check that can DISCRIMINATE
// where a float comparison could not (U4: on the CPU-only arm the membership drift moved no float
// at all, so "request B == fresh boot" passing was not evidence the reset worked).
bool m1_session_rotation_is_boot(const M1Session * s, std::string * why);

// The single-shot binary's whole engine run (validation -> the session -> the loop -> the audits ->
// the verdict), defined in m1_engine.cpp and called from main() in m1_main.cpp.
int loop_run(const LoopArgs & a);

// The --layer-check / loader path (m1_engine.cpp: it owns read_tokens and the surface check).
int layer_check(const LayerCheckArgs & a);

// --resident-set / --hotset: the per-layer ORDERED resident set, read from a JSON file of ids
// (m1_engine.cpp).  Shared for the same reason as LayerCheckArgs: both binaries' flag parsers
// accept the flag.
bool read_hotset_json(const char * path, std::vector<std::vector<int>> & hot, int & n_res);

#endif // M1_SESSION_H
