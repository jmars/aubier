// engine/m1_main.cpp — the SINGLE-SHOT binary's entry point (aubier server plan §A, unit U4).
//
// This is main() moved verbatim out of engine/m1_engine.cpp (it was m1_engine.cpp:3264-3484): the
// argument parse and the dispatch.  It moved for one structural reason — the ENGINE TU must be
// linkable by a second binary (the server, U5), and a `main` symbol in it would collide.  The
// engine run itself stays in m1_engine.cpp: `--loop`/`--score` dispatch to loop_run(), which is
// validation -> M1Session::open() -> the token loop (M1Session::step) -> the audits -> the verdict,
// and the loader/--layer-check path stays with the code that owns it.
//
// usage: builds/engine/m1 [--model <gguf>] [--quiet]                    # U1 loader check
//        builds/engine/m1 --layer-check --tokens <file> [...]           # U2 layer check
//        builds/engine/m1 --loop --tokens <file> --out <logits> [...]   # U3 the token loop
//        builds/engine/m1 --falsify [--serialize-both] [...]            # U3 the falsifier

#include "gguf_model.h"
#include "m1_session.h"    // LoopArgs, LayerCheckArgs, DEFAULT_MODEL, loop_run, layer_check

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

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
