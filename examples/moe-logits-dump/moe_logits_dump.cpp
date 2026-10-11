// moe_logits_dump.cpp — P-T correctness harness.
// Greedy-decode a prompt one token at a time (batch 1, the decode path that a
// fused MoE kernel targets) and dump the FULL next-token logits at every decode
// position to a binary file, plus the argmax token id sequence. Two arms
// (baseline vs fused) run through the SAME code path, so KL / top-token
// agreement / perplexity can be computed from the dumped logits afterwards.
//
// Uses the common layer so every config flag (-m, -ngl, -ncmoe, -ot,
// --moe-cache-mib, --no-op-offload, -ctk/-ctv, ...) behaves exactly as llama-cli.
//
// Binary layout of <out>.logits :
//   uint64 n_decode
//   uint32 n_vocab
//   for each decode position: n_vocab float32 logits (little-endian)
// <out>.tokens : uint32 n_decode, then n_decode int32 token ids.
#include "common.h"
#include "arg.h"
#include "llama.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static bool dump_logits(const char * path, const std::vector<std::vector<float>> & logits,
                        const std::vector<llama_token> & tokens, int n_vocab) {
    FILE * f = fopen(path, "wb");
    if (!f) return false;
    uint64_t n = (uint64_t) logits.size();
    uint32_t nv = (uint32_t) n_vocab;
    fwrite(&n, sizeof(n), 1, f);
    fwrite(&nv, sizeof(nv), 1, f);
    for (size_t i = 0; i < logits.size(); ++i) {
        fwrite(logits[i].data(), sizeof(float), n_vocab, f);
    }
    fclose(f);

    std::string tpath = std::string(path) + ".tokens";
    f = fopen(tpath.c_str(), "wb");
    if (!f) return false;
    uint32_t nt = (uint32_t) tokens.size();
    fwrite(&nt, sizeof(nt), 1, f);
    fwrite(tokens.data(), sizeof(llama_token), tokens.size(), f);
    fclose(f);
    return true;
}

// DEBUG (P-M3): dump the first few values of any ffn_moe_* tensor at eval time,
// gated by LLAMA_MOE_SPLIT_DUMP. Used to bisect the M2 flag-ON wrongness.
static bool dump_cb_eval(ggml_tensor * t, bool ask, void * user_data) {
    (void) user_data;
    if (ask || getenv("LLAMA_MOE_SPLIT_DUMP") == nullptr) {
        return true;
    }
    const char * n = t->name;
    if (n == nullptr) {
        return true;
    }
    // decode-step (ne[1]==1) — dump moe_out + shexp + ffn_out + result_norm + result_output
    if (strstr(n, "ffn_moe_out") == nullptr && strstr(n, "ffn_shexp") == nullptr &&
        strstr(n, "ffn_out") == nullptr && strstr(n, "result_norm") == nullptr &&
        strstr(n, "result_output") == nullptr) {
        return true;
    }
    if (t->ne[1] != 1 && strstr(n, "result_norm") == nullptr && strstr(n, "result_output") == nullptr) {
        return true;
    }
    if (t->type != GGML_TYPE_F32) {
        return true;
    }
    // full-tensor checksum (double sum + sum-of-squares) to detect ANY divergence
    int64_t ne = ggml_nelements(t);
    std::vector<float> v(ne);
    ggml_backend_tensor_get(t, v.data(), 0, sizeof(float) * ne);
    double s = 0.0, s2 = 0.0;
    for (int64_t i = 0; i < ne; i++) { s += v[i]; s2 += (double) v[i] * v[i]; }
    fprintf(stderr, "[moe_dump] %s shape=[%lld,%lld,%lld] sum=%.9e sumsq=%.9e first4=", n,
            (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], s, s2);
    for (int i = 0; i < 4 && i < ne; i++) fprintf(stderr, " %.4f", v[i]);
    fprintf(stderr, "\n");
    return true;
}

int main(int argc, char ** argv) {
    common_params params;
    params.n_ctx = 2048;
    common_init();
    if (getenv("LLAMA_MOE_SPLIT_DUMP")) {
        params.cb_eval = dump_cb_eval;
    }

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    if (params.prompt.empty() && params.prompt_file.empty()) {
        fprintf(stderr, "moe_logits_dump: need -p or -f prompt\n");
        return 1;
    }
    // P-FORK: the output path comes from MOE_LOGITS_FILE (this fork's common layer
    // has no --logits-file arg; P-T's fork added one to common_params).
    const char * logits_file_env = getenv("MOE_LOGITS_FILE");
    if (logits_file_env == nullptr || logits_file_env[0] == '\0') {
        fprintf(stderr, "moe_logits_dump: need MOE_LOGITS_FILE env var\n");
        return 1;
    }
    const std::string logits_file = logits_file_env;

    // greedy, no penalties: the distribution we compare is the raw model output.
    params.sampling.temp = 0.0f;
    params.sampling.top_k = 1;

    llama_backend_init();

    auto init = common_init_from_params(params);
    llama_model * model = init->model();
    llama_context * ctx  = init->context();
    if (!model || !ctx) {
        fprintf(stderr, "moe_logits_dump: model/ctx init failed\n");
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    std::string prompt = params.prompt;
    if (!params.prompt_file.empty()) {
        std::string contents;
        FILE * pf = fopen(params.prompt_file.c_str(), "rb");
        if (!pf) { fprintf(stderr, "moe_logits_dump: cannot open %s\n", params.prompt_file.c_str()); return 1; }
        char buf[4096];
        size_t r;
        while ((r = fread(buf, 1, sizeof(buf), pf)) > 0) contents.append(buf, r);
        fclose(pf);
        prompt = contents;
    }

    std::vector<llama_token> prompt_tokens = common_tokenize(vocab, prompt, true, true);
    if (prompt_tokens.empty()) { fprintf(stderr, "moe_logits_dump: empty prompt\n"); return 1; }

    auto sparams = llama_sampler_chain_default_params();
    sparams.no_perf = false;
    llama_sampler * smpl = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    llama_batch_ext * batch = llama_batch_ext_init(ctx);
    const int n_prompt = (int) prompt_tokens.size();
    llama_batch_ext_clear(batch);
    for (int32_t i = 0; i < n_prompt; ++i) {
        const int32_t idx = llama_batch_ext_add_token(batch, 0, prompt_tokens[i]);
        llama_pos pos = i;
        llama_batch_ext_set_pos(batch, idx, &pos);
    }
    llama_batch_ext_set_output_logits(batch, n_prompt - 1, true);

    if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch)) {
        fprintf(stderr, "moe_logits_dump: prefill failed\n");
        return 1;
    }

    std::vector<std::vector<float>> logits;
    std::vector<llama_token> gen_tokens;
    logits.reserve(params.n_predict);

    for (int n_pos = n_prompt; n_pos < n_prompt + params.n_predict; ++n_pos) {
        llama_token new_token_id = llama_sampler_sample(smpl, ctx, -1);
        if (llama_vocab_is_eog(vocab, new_token_id)) break;

        // -1 = the last (and only) output token this step; index 0 fails on the
        // prefill batch because output_ids[0] is not the row this token lives in.
        const float * cur_logits = llama_get_logits_ith(ctx, -1);
        if (getenv("LLAMA_MOE_SPLIT_DUMP")) {
            double s = 0.0, s2 = 0.0;
            for (int j = 0; j < n_vocab; j++) { s += cur_logits[j]; s2 += (double) cur_logits[j] * cur_logits[j]; }
            fprintf(stderr, "[moe_dump] get_logits_ith(-1) n_vocab=%d sum=%.9e sumsq=%.9e first4=%.4f %.4f %.4f %.4f\n",
                    n_vocab, s, s2, cur_logits[0], cur_logits[1], cur_logits[2], cur_logits[3]);
        }
        std::vector<float> row(cur_logits, cur_logits + n_vocab);
        logits.push_back(std::move(row));
        gen_tokens.push_back(new_token_id);

        llama_batch_ext_clear(batch);
        const int32_t idx = llama_batch_ext_add_token(batch, 0, new_token_id);
        llama_pos pos = n_pos;
        llama_batch_ext_set_pos(batch, idx, &pos);
        llama_batch_ext_set_output_logits(batch, 0, true);

        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch)) {
            fprintf(stderr, "moe_logits_dump: decode failed at pos %d\n", n_pos);
            return 1;
        }
    }

    if (!dump_logits(logits_file.c_str(), logits, gen_tokens, n_vocab)) {
        fprintf(stderr, "moe_logits_dump: dump failed\n");
        return 1;
    }

    fprintf(stderr, "moe_logits_dump: %zu decode positions dumped to %s\n", logits.size(), logits_file.c_str());
    return 0;
}
