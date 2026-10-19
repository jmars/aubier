// engine/m1_tok.h — the Mistral Small 4 tokenizer + detokenizer (server plan §B, unit U1).
//
// The model's vocab is byte-level BPE (tokenizer.ggml.model == "gpt2") with the "pixtral"
// pre-tokenizer, which llama.cpp maps to LLAMA_VOCAB_PRE_TYPE_LLAMA3 with ignore_merges=true
// and add_bos=true (vendor/llama.cpp/src/llama-vocab.cpp:2290-2299).  This is a TRANSCRIPTION
// of llama.cpp's own four stages, not an invention:
//
//   1. special-token partition     tokenizer_st_partition, llama-vocab.cpp:3385; the special
//                                  set is every id whose attr has CONTROL|USER_DEFINED|UNKNOWN
//                                  (:3163-3167), sorted by DESCENDING text length (:3169-3173)
//   2. pre-tokenize                unicode_regex_split + the llama3 regex (:284-291).  The
//                                  vendored src/unicode.cpp + src/unicode-data.cpp are COMPILED
//                                  INTO this binary (scripts/engine_build_tok.sh), so the split
//                                  is the same code llama.cpp runs, not a copy of its tables.
//   3. BPE merges                  llm_tokenizer_bpe_session::tokenize (:631-760): the
//                                  ignore_merges whole-word short-circuit (:669-672), per
//                                  UTF-8-char symbols with a rank priority queue (:697-728),
//                                  and the per-byte fallback for unknown words (:736-760)
//   4. detokenize                  llama_decode_text (:3528-3545) + token_to_piece (:3865-3885)
//
// Every read of the vocab goes through the model's ALREADY-OPEN gguf_context (GGUFModel::gctx,
// engine/gguf_model.h:251) via libggml's exported gguf_find_key / gguf_get_arr_n /
// gguf_get_arr_str / gguf_get_arr_data — no new I/O code, no second reader.
//
// A violated invariant is a BOOT FATAL, never a silent-wrong id: U1's parity gate
// (scripts/engine_gate_tok.sh) is sensitive to any single mis-mapped id, but it can only
// catch what it is run on, so the boot CHECKs are the second line.
//
// Mistral Small 4 ONLY, hardcoded, no generality (user directive 2026-10-11).

#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

struct gguf_context;   // gguf.h — passed in already open by GGUFModel

// llama_token_attr bits (vendor/llama.cpp/include/llama.h:104-112), transcribed as values so
// this unit compiles against ggml/gguf only and never links libllama.
enum M1TokAttr : uint32_t {
    M1_TOK_ATTR_UNDEFINED    = 0,
    M1_TOK_ATTR_UNKNOWN      = 1u << 0,
    M1_TOK_ATTR_UNUSED       = 1u << 1,
    M1_TOK_ATTR_NORMAL       = 1u << 2,
    M1_TOK_ATTR_CONTROL      = 1u << 3,
    M1_TOK_ATTR_USER_DEFINED = 1u << 4,
    M1_TOK_ATTR_BYTE         = 1u << 5,
};

struct M1Tok {
    struct Token {
        std::string text;     // the GGUF string, verbatim (byte-encoded for NORMAL tokens)
        uint32_t    attr = 0; // M1_TOK_ATTR_*
    };

    // ---------------------------------------------------------------------
    // loaded surface
    // ---------------------------------------------------------------------
    std::vector<Token> id_to_text;
    // text -> id.  LAST occurrence wins, because llama-vocab.cpp:2638 assigns
    // token_to_id[word] = i in ascending id order (a later duplicate overwrites) — NOT the first
    // occurrence, which is what server plan §B says.  The duplicate count is printed at boot so
    // the choice is visible rather than assumed (MEASURED: 0 for this vocab).
    std::unordered_map<std::string, int32_t> text_to_id;
    // cache_special_tokens: CONTROL|USER_DEFINED|UNKNOWN ids, DESCENDING text length
    // (llama-vocab.cpp:3163-3173).  Ties are settled by the same std::sort call llama.cpp uses,
    // on the same ascending-id input, so the permutation is identical for the same libstdc++.
    std::vector<int32_t> special_ids;
    // bpe_ranks: ("left","right") -> merge rank = index into the merges array
    // (llama-vocab.cpp:2126-2141; split at find(' ', 1); duplicates keep the LOWEST rank).
    std::map<std::pair<std::string, std::string>, int32_t> bpe_ranks;
    // the single llama3 pre-tokenizer regex (llama-vocab.cpp:284-291)
    std::vector<std::string> regex_exprs;

    bool ignore_merges = true;   // pixtral => true (:2297)
    bool add_bos       = true;   // pixtral => true (:2298), overridable by tokenizer.ggml.add_bos_token

    // FAIL-ABILITY HOOK, off by default: M1_TOK_MUTANT={merge-ranks-reversed|ignore-merges-off|
    // detok-no-byte-decode} turns this tokenizer deliberately wrong so that
    // scripts/engine_gate_tok.sh can be shown to exit 1 ON EACH ARM — mutant 1 on the id arm,
    // mutant 3 on the detokenizer arm (mutant 2 turned out to be inert on this vocab: see the
    // MEASURED note in m1_tok.cpp).  A test that cannot fail is not a test; the mutant is the
    // asymmetry proof.  Read ONCE at load(), prints a loud banner.
    int mutant = 0;              // 0 none, 1 merge ranks reversed, 2 ignore_merges off,
                                 // 3 detokenizer skips the byte-level decode

    int32_t id_bos = -1;
    int32_t id_eos = -1;
    int32_t id_unk = -1;
    int32_t id_pad = -1;

    // ---------------------------------------------------------------------
    // boot: read + CHECK the whole vocab out of an open gguf_context
    // ---------------------------------------------------------------------
    static M1Tok * load(gguf_context * gctx);

    // one line for the boot log: counts + the CHECKs' observed values
    std::string summary() const;

    // ---------------------------------------------------------------------
    // stage 1+2+3
    // ---------------------------------------------------------------------
    // add_bos == false is what the chat path uses (the template already emits '<s>', and the
    // reference tokenizer would prepend a second one — server plan §C).
    std::vector<int32_t> encode(const std::string & text, bool add_bos, bool parse_special) const;

    // ---------------------------------------------------------------------
    // stage 4
    // ---------------------------------------------------------------------
    // the exact bytes token `id` renders to ("" for an attr llama renders as nothing)
    std::string piece(int32_t id) const;
    std::string decode(const std::vector<int32_t> & ids) const;

    bool is_eog(int32_t id) const { return id == id_eos; }

    // ---------------------------------------------------------------------
    // internals, exposed only because C++17 has no module-private
    // ---------------------------------------------------------------------
    struct Frag { bool special; int32_t id; size_t off, len; };
    std::vector<Frag> partition(const std::string & text, bool parse_special) const;
    void encode_word(const std::string & word, std::vector<int32_t> & out) const;
};

// ---------------------------------------------------------------------------
// streaming detokenizer: never emits a partial UTF-8 codepoint
// ---------------------------------------------------------------------------
// A token's bytes can end mid-codepoint (byte-level BPE splits UTF-8 anywhere), so the SSE /
// incremental path must hold back an incomplete trailing sequence.  The tail is bounded by 3
// bytes (the longest incomplete prefix of a UTF-8 sequence).
struct M1TokDecoder {
    const M1Tok * tok = nullptr;
    std::string   tail;

    // append one token's bytes; returns only the COMPLETE UTF-8 prefix that is now safe to emit
    std::string push(int32_t id);

    // append raw bytes (already-detokenized text); same contract
    std::string push_bytes(const std::string & bytes);

    // end of generation: a dangling partial sequence cannot be completed -> DROP it, log it,
    // and return it through `dropped` (never silently emit an invalid codepoint)
    std::string finish(size_t * dropped);
};
