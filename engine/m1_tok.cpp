// engine/m1_tok.cpp — see engine/m1_tok.h for what is transcribed from where.
//
// MEASURED (this unit, on models/Mistral-Small-4-119B-2603-UD-IQ2_XXS.gguf):
//   tokenizer.ggml.model     = "gpt2"
//   tokenizer.ggml.pre       = "pixtral"
//   tokenizer.ggml.tokens       arr_n = 131072
//   tokenizer.ggml.token_type   arr_n = 131072   (130072 NORMAL + 1000 CONTROL)
//   tokenizer.ggml.merges       arr_n = 269443
//   tokens[0..3]             = "<unk>" "<s>" "</s>" "[INST]"
// ... and every one of those is a boot CHECK below, because a vocab read wrong is a silent-wrong
// id, which is the one failure mode a parity gate over a fixed corpus can miss.

#include "m1_tok.h"

#include "gguf.h"
#include "unicode.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <queue>
#include <stdexcept>

#ifndef FATAL
#define FATAL(...) do { \
    std::fprintf(stderr, "FATAL: " __VA_ARGS__); \
    std::fprintf(stderr, "\n"); \
    std::exit(1); \
} while (0)
#endif
#ifndef CHECK
#define CHECK(cond, ...) do { if (!(cond)) FATAL(__VA_ARGS__); } while (0)
#endif

// ---------------------------------------------------------------------------
// the one pre-tokenizer regex (vendor/llama.cpp/src/llama-vocab.cpp:284-291, LLAMA3 case)
// ---------------------------------------------------------------------------
// Copied byte-for-byte: the escaping is C++-level (\\r -> backslash-r for std::regex).
static const char * LLAMA3_REGEX =
    "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+"
    "|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

// expected values — MEASURED, Mistral Small 4 only
static constexpr size_t EXP_N_TOKENS  = 131072;
static constexpr size_t EXP_N_MERGES  = 269443;
static constexpr size_t EXP_N_SPECIAL = 1000;

namespace {

const char * tok_type_name(int32_t t) {
    switch (t) {
        case 0: return "UNDEFINED";
        case 1: return "NORMAL";
        case 2: return "UNKNOWN";
        case 3: return "CONTROL";
        case 4: return "USER_DEFINED";
        case 5: return "UNUSED";
        case 6: return "BYTE";
        default: return "?";
    }
}

// token_type -> attr, verbatim from llama-vocab.cpp:2653-2664 (GGUF enum -> llama_token_attr)
uint32_t attr_of_toktype(int32_t t) {
    switch (t) {
        case 0:  return M1_TOK_ATTR_UNDEFINED;      // LLAMA_TOKEN_TYPE_UNDEFINED
        case 1:  return M1_TOK_ATTR_NORMAL;
        case 2:  return M1_TOK_ATTR_UNKNOWN;
        case 3:  return M1_TOK_ATTR_CONTROL;
        case 4:  return M1_TOK_ATTR_USER_DEFINED;
        case 5:  return M1_TOK_ATTR_UNUSED;
        case 6:  return M1_TOK_ATTR_BYTE;
        default: return M1_TOK_ATTR_UNDEFINED;
    }
}

}  // namespace

// ---------------------------------------------------------------------------
// boot
// ---------------------------------------------------------------------------
M1Tok * M1Tok::load(gguf_context * gctx) {
    CHECK(gctx != nullptr, "m1_tok: null gguf_context (the model was not opened)");

    auto key = [&](const char * k) -> int64_t {
        const int64_t i = gguf_find_key(gctx, k);
        CHECK(i >= 0, "m1_tok: GGUF key '%s' is missing", k);
        return i;
    };
    auto str_arr = [&](const char * k) -> int64_t {
        const int64_t i = key(k);
        CHECK(gguf_get_kv_type(gctx, i) == GGUF_TYPE_ARRAY,
              "m1_tok: '%s' is not an array kv", k);
        CHECK(gguf_get_arr_type(gctx, i) == GGUF_TYPE_STRING,
              "m1_tok: '%s' is not a string array", k);
        return i;
    };
    auto int_arr = [&](const char * k) -> int64_t {
        const int64_t i = key(k);
        CHECK(gguf_get_kv_type(gctx, i) == GGUF_TYPE_ARRAY,
              "m1_tok: '%s' is not an array kv", k);
        CHECK(gguf_get_arr_type(gctx, i) == GGUF_TYPE_INT32,
              "m1_tok: '%s' is not an int32 array", k);
        return i;
    };

    M1Tok * T = new M1Tok();

    // --- the pipelined facts the transcription ASSUMES: check, do not hope ---------------
    const char * model = gguf_get_val_str(gctx, key("tokenizer.ggml.model"));
    CHECK(std::strcmp(model, "gpt2") == 0,
          "m1_tok: tokenizer.ggml.model == '%s', expected 'gpt2' (byte-level BPE)", model);
    const char * pre = gguf_get_val_str(gctx, key("tokenizer.ggml.pre"));
    CHECK(std::strcmp(pre, "pixtral") == 0,
          "m1_tok: tokenizer.ggml.pre == '%s', expected 'pixtral' (=> LLAMA3 splitter + "
          "ignore_merges + add_bos, llama-vocab.cpp:2290-2299)", pre);

    // --- tokens ---------------------------------------------------------------------------
    const int64_t k_tokens = str_arr("tokenizer.ggml.tokens");
    const size_t  n_tokens = gguf_get_arr_n(gctx, k_tokens);
    CHECK(n_tokens == EXP_N_TOKENS, "m1_tok: tokens arr_n == %zu, expected %zu",
          n_tokens, EXP_N_TOKENS);

    // --- token_type -----------------------------------------------------------------------
    const int64_t k_types  = int_arr("tokenizer.ggml.token_type");
    const size_t  n_types  = gguf_get_arr_n(gctx, k_types);
    CHECK(n_types == EXP_N_TOKENS, "m1_tok: token_type arr_n == %zu, expected %zu",
          n_types, EXP_N_TOKENS);

    // --- merges ---------------------------------------------------------------------------
    const int64_t k_merges  = str_arr("tokenizer.ggml.merges");
    const size_t  n_merges  = gguf_get_arr_n(gctx, k_merges);
    CHECK(n_merges == EXP_N_MERGES, "m1_tok: merges arr_n == %zu, expected %zu",
          n_merges, EXP_N_MERGES);

    // --- vocab_size (the <arch>.vocab_size key) must agree with the array ------------------
    {
        const int64_t n_kv = gguf_get_n_kv(gctx);
        int64_t found = -1;
        for (int64_t i = 0; i < n_kv; ++i) {
            const char * k = gguf_get_key(gctx, i);
            const size_t kl = std::strlen(k);
            if (kl > 11 && std::strcmp(k + kl - 11, ".vocab_size") == 0) { found = i; break; }
        }
        CHECK(found >= 0, "m1_tok: no '<arch>.vocab_size' key in the GGUF");
        CHECK(gguf_get_kv_type(gctx, found) == GGUF_TYPE_UINT32,
              "m1_tok: '%s' is not uint32", gguf_get_key(gctx, found));
        const uint32_t vs = gguf_get_val_u32(gctx, found);
        CHECK((size_t) vs == n_tokens, "m1_tok: %s == %u but the tokens array is %zu",
              gguf_get_key(gctx, found), vs, n_tokens);
    }

    // --- build the surfaces ---------------------------------------------------------------
    const int32_t * toktypes = (const int32_t *) gguf_get_arr_data(gctx, k_types);
    T->id_to_text.resize(n_tokens);
    T->text_to_id.reserve(n_tokens * 2);

    size_t n_dup = 0, n_normal = 0, n_control = 0, n_unknown = 0, n_userdef = 0,
           n_unused = 0, n_byte = 0, n_undef = 0;

    for (size_t i = 0; i < n_tokens; ++i) {
        const char * s = gguf_get_arr_str(gctx, k_tokens, i);
        Token & t = T->id_to_text[i];
        t.text = s;                       // empty token text is legal; llama renames it,
                                          // but this vocab has none -> checked below
        CHECK(!t.text.empty(), "m1_tok: empty token text at id %zu (llama.cpp renames these "
              "to [EMPTY_<i>] — a vocab this code does not handle)", i);
        t.attr = attr_of_toktype(toktypes[i]);

        // llama-vocab.cpp:2638 — `token_to_id[word] = i` in ASCENDING id order, so a duplicate
        // text maps to the LAST id.  Count them so the summary makes the choice visible.
        auto ins = T->text_to_id.emplace(t.text, (int32_t) i);
        if (!ins.second) {
            ++n_dup;
            ins.first->second = (int32_t) i;
        }

        switch (toktypes[i]) {
            case 1: ++n_normal;  break;
            case 3: ++n_control; break;
            case 2: ++n_unknown; break;
            case 4: ++n_userdef; break;
            case 5: ++n_unused;  break;
            case 6: ++n_byte;    break;
            default: ++n_undef;  break;
        }
    }

    // NOTE (corrects the plan's §B reading): `token_to_id[word] = i` at llama-vocab.cpp:2638 is
    // an ASSIGNMENT in ascending id order, so a duplicate text maps to the LAST id — not the
    // first.  MEASURED: this vocab has dup_texts=0, so the distinction is moot HERE; the loader
    // below does last-wins anyway and the boot line prints the count so it stays visible.
    //
    // NOTE: the reference's own log confirms the token_type classification used below IS
    // llama.cpp's: 'load: special tokens cache size = 1000' with zero 'control-looking token'
    // warnings, i.e. no NORMAL token was force-promoted to CONTROL by the text-based override
    // at :2833-2984.

    // --- the boot CHECKs the server plan names (§H U1, §G5) -------------------------------
    CHECK(T->id_to_text[1].text == "<s>",
          "m1_tok: tokens[1] == '%s', expected '<s>'", T->id_to_text[1].text.c_str());
    CHECK(T->id_to_text[2].text == "</s>",
          "m1_tok: tokens[2] == '%s', expected '</s>'", T->id_to_text[2].text.c_str());
    CHECK(T->id_to_text[3].text == "[INST]",
          "m1_tok: tokens[3] == '%s', expected '[INST]'", T->id_to_text[3].text.c_str());

    const size_t n_special_raw = n_control + n_userdef + n_unknown;
    CHECK(n_special_raw == EXP_N_SPECIAL,
          "m1_tok: control(%zu)+user_defined(%zu)+unknown(%zu) == %zu, expected %zu — the "
          "special-token cache is the wrong SIZE, so the partition would be wrong",
          n_control, n_userdef, n_unknown, n_special_raw, EXP_N_SPECIAL);

    // --- ids the caller needs -------------------------------------------------------------
    auto id_of = [&](const char * k, int32_t * out) {
        const int64_t i = gguf_find_key(gctx, k);
        if (i < 0) return false;
        *out = (int32_t) gguf_get_val_u32(gctx, i);
        return true;
    };
    if (!id_of("tokenizer.ggml.bos_token_id", &T->id_bos)) T->id_bos = -1;
    if (!id_of("tokenizer.ggml.eos_token_id", &T->id_eos)) T->id_eos = -1;
    if (!id_of("tokenizer.ggml.unknown_token_id", &T->id_unk)) T->id_unk = -1;
    if (!id_of("tokenizer.ggml.padding_token_id", &T->id_pad)) T->id_pad = -1;

    // ignore_merges / add_bos: the pixtral pre-type sets both (llama-vocab.cpp:2297-2298); the
    // GGUF may override add_bos, and llama.cpp applies the KV AFTER the pre-type branch
    // (:2763), so the KV wins here too.
    T->ignore_merges = true;
    T->add_bos       = true;
    {
        const int64_t i = gguf_find_key(gctx, "tokenizer.ggml.add_bos_token");
        if (i >= 0) T->add_bos = gguf_get_val_bool(gctx, i);
    }
    CHECK(T->id_bos >= 0 && (size_t) T->id_bos < n_tokens, "m1_tok: bad bos_token_id %d", T->id_bos);
    CHECK(T->id_eos >= 0 && (size_t) T->id_eos < n_tokens, "m1_tok: bad eos_token_id %d", T->id_eos);

    // --- merges: ("A","B") -> rank, split at find(' ', 1) (llama-vocab.cpp:2133) -------------
    for (size_t i = 0; i < n_merges; ++i) {
        const char * m = gguf_get_arr_str(gctx, k_merges, i);
        const size_t len = std::strlen(m);
        std::string first, second;
        const size_t pos = std::string(m, len).find(' ', 1);
        if (pos != std::string::npos) {
            first.assign(m, pos);
            second.assign(m + pos + 1, len - pos - 1);
        } else {
            CHECK(false, "m1_tok: merge %zu ('%s') has no separator space", i, m);
        }
        // emplace: a duplicate pair keeps the LOWEST rank (first insertion), like llama.cpp
        T->bpe_ranks.emplace(std::make_pair(std::move(first), std::move(second)), (int32_t) i);
    }

    // --- the special-token cache: DESCENDING text length (llama-vocab.cpp:3166-3173) --------
    for (size_t id = 0; id < n_tokens; ++id) {
        if (T->id_to_text[id].attr & (M1_TOK_ATTR_CONTROL | M1_TOK_ATTR_USER_DEFINED |
                                     M1_TOK_ATTR_UNKNOWN)) {
            T->special_ids.push_back((int32_t) id);
        }
    }
    // EXACTLY the reference's sort call on EXACTLY its input order (ascending id), so the
    // resulting permutation — which only matters for same-position overlaps between two
    // equal-length specials — is the reference's.
    std::sort(T->special_ids.begin(), T->special_ids.end(),
        [&] (int32_t a, int32_t b) {
            return T->id_to_text[a].text.size() > T->id_to_text[b].text.size();
        });
    CHECK(T->special_ids.size() == EXP_N_SPECIAL,
          "m1_tok: special cache size %zu, expected %zu",
          T->special_ids.size(), EXP_N_SPECIAL);

    T->regex_exprs.push_back(LLAMA3_REGEX);

    // --- the fail-ability hook (see m1_tok.h) ----------------------------------------------
    // MEASURED, and it refutes a claim in the server plan (§B): forcing ignore_merges OFF is
    // BEHAVIOURALLY INERT on this vocab.  Encoding all 130072 NORMAL token texts (one per line,
    // 1.39 MB) with the flag on and off gives BYTE-IDENTICAL id arrays (5484385 bytes each), and
    // the 78-case corpus does the same over ~40k tokens of real text.  So the whole-word
    // short-circuit below cannot change a result here: every pre-tokenized word that IS a vocab
    // entry is also reached exactly by the merges.  It is still transcribed (llama.cpp has it, a
    // different vocab would need it, and its cost is one hash lookup), but it is NOT a
    // fail-ability lever for the gate.  Scope: every NORMAL token text as its own line plus real
    // text — not a proof over all strings the regex can emit.
    if (const char * mut = std::getenv("M1_TOK_MUTANT"); mut && *mut) {
        if (std::strcmp(mut, "merge-ranks-reversed") == 0) {
            T->mutant = 1;
        } else if (std::strcmp(mut, "ignore-merges-off") == 0) {
            T->mutant = 2;
            T->ignore_merges = false;
        } else if (std::strcmp(mut, "detok-no-byte-decode") == 0) {
            T->mutant = 3;
        } else {
            FATAL("m1_tok: M1_TOK_MUTANT='%s' is not a known mutant "
                  "(merge-ranks-reversed|ignore-merges-off|detok-no-byte-decode)", mut);
        }
        std::fprintf(stderr,
            "\n*** M1_TOK_MUTANT=%s ACTIVE — THIS TOKENIZER IS DELIBERATELY WRONG.  It exists "
            "only to prove scripts/engine_gate_tok.sh can FAIL. ***\n\n", mut);
    }

    std::fprintf(stderr,
        "# tok: model=%s pre=%s toktypes[normal=%zu control=%zu unknown=%zu user_defined=%zu "
        "unused=%zu byte=%zu undefined=%zu] dup_texts=%zu specials=%zu merges=%zu\n",
        model, pre, n_normal, n_control, n_unknown, n_userdef, n_unused, n_byte, n_undef,
        n_dup, T->special_ids.size(), T->bpe_ranks.size());

    return T;
}

std::string M1Tok::summary() const {
    char buf[512];
    const size_t n_control = (size_t) std::count_if(id_to_text.begin(), id_to_text.end(),
        [] (const Token & t) { return (t.attr & M1_TOK_ATTR_CONTROL) != 0; });
    std::snprintf(buf, sizeof buf,
        "n_tokens=%zu control=%zu specials=%zu merges=%zu ignore_merges=%d add_bos=%d "
        "bos=%d eos=%d unk=%d pad=%d longest_special=%zu",
        id_to_text.size(), n_control, special_ids.size(), bpe_ranks.size(),
        (int) ignore_merges, (int) add_bos, id_bos, id_eos, id_unk, id_pad,
        special_ids.empty() ? 0 : id_to_text[special_ids[0]].text.size());
    return std::string(buf);
}

// ---------------------------------------------------------------------------
// stage 1 — special-token partition (llama-vocab.cpp:3385)
// ---------------------------------------------------------------------------
// The reference walks the fragment list once PER SPECIAL in the sorted order and splits each
// still-raw fragment at every occurrence of that special.  A special processed earlier keeps
// the regions it claimed, so the effective rule is "special priority order, then left to
// right" — NOT "longest match anywhere".  Because the list is sorted by DESCENDING length, the
// only observable order-dependence is between two specials of equal length that overlap; the
// sort call above reproduces the reference's tie permutation.
std::vector<M1Tok::Frag> M1Tok::partition(const std::string & text, bool parse_special) const {
    std::vector<Frag> frags;
    frags.push_back(Frag{false, -1, 0, text.size()});

    for (const int32_t sid : special_ids) {
        const Token & dt = id_to_text[sid];
        // llama-vocab.cpp:3393-3399: control/unknown are skipped when parse_special == false;
        // user_defined is still split (its comment cites the HF tokenizers reference)
        if (!parse_special && (dt.attr & (M1_TOK_ATTR_CONTROL | M1_TOK_ATTR_UNKNOWN))) continue;
        if (dt.text.empty()) continue;   // no such token in this vocab (checked at boot)

        std::vector<Frag> next;
        next.reserve(frags.size());
        for (const Frag & f : frags) {
            if (f.special) { next.push_back(f); continue; }
            const size_t end = f.off + f.len;
            size_t pos = f.off;
            while (pos < end) {
                const size_t m = text.find(dt.text, pos);
                // a match must START and END inside this fragment
                if (m == std::string::npos || m + dt.text.size() > end) break;
                if (m > pos) next.push_back(Frag{false, -1, pos, m - pos});
                next.push_back(Frag{true, sid, m, dt.text.size()});
                pos = m + dt.text.size();
            }
            if (pos < end) next.push_back(Frag{false, -1, pos, end - pos});
        }
        frags.swap(next);
    }
    return frags;
}

// ---------------------------------------------------------------------------
// stages 2+3 — pre-tokenize one raw-text fragment, then BPE its words
// ---------------------------------------------------------------------------
void M1Tok::encode_word(const std::string & word, std::vector<int32_t> & out) const {
    struct Sym { size_t off, n; int prev, next; };
    struct Bigram { int left, right; std::string text; int rank; };
    struct Cmp {
        bool operator()(const Bigram & l, const Bigram & r) const {
            // llama-vocab.cpp:266-268, verbatim: min-heap on (rank, left)
            return l.rank > r.rank || (l.rank == r.rank && l.left > r.left);
        }
    };
    std::priority_queue<Bigram, std::vector<Bigram>, Cmp> q;

    std::vector<Sym> symbols;
    symbols.reserve(word.size());

    auto add_new_bigram = [&] (int left, int right) {
        if (left == -1 || right == -1) return;
        const std::string lt = word.substr(symbols[left].off,  symbols[left].n);
        const std::string rt = word.substr(symbols[right].off, symbols[right].n);
        auto it = bpe_ranks.find(std::make_pair(lt, rt));
        if (it == bpe_ranks.end()) return;                     // find_bpe_rank < 0
        // mutant 1: pop the HIGHEST rank first instead of the lowest (the whole merge order
        // inverts, so every multi-char word's ids change — the gate must notice)
        const int rank = mutant == 1 ? -it->second : it->second;
        q.push(Bigram{left, right, lt + rt, rank});
    };

    // the ignore_merges whole-word short-circuit (llama-vocab.cpp:669-672).  Transcribed because
    // llama.cpp has it (pixtral sets the flag, :2297) — but MEASURED it is INERT on this vocab:
    // see the note at the fail-ability hook below, where turning it off leaves the ids identical
    // over all 130072 NORMAL token texts.
    size_t offset = 0;
    int    index  = 0;
    if (ignore_merges && text_to_id.find(word) != text_to_id.end()) {
        symbols.push_back(Sym{0, word.size(), -1, -1});
        offset = word.size();
    }

    // per-UTF-8-char symbols (llama-vocab.cpp:678-690)
    while (offset < word.size()) {
        const size_t char_len = std::min(word.size() - offset,
                                        (size_t) unicode_len_utf8(word[offset]));
        Sym s;
        s.off  = offset;
        s.n    = char_len;
        offset += char_len;
        s.prev = index - 1;
        s.next = offset == word.size() ? -1 : index + 1;
        ++index;
        symbols.push_back(s);
    }
    for (int i = 1; i < (int) symbols.size(); ++i) add_new_bigram(i - 1, i);

    // the merge loop (llama-vocab.cpp:697-728)
    while (!q.empty()) {
        const Bigram b = q.top();
        q.pop();

        Sym & L = symbols[b.left];
        Sym & R = symbols[b.right];
        if (L.n == 0 || R.n == 0) continue;
        // "skip this bigram if it's outdated" — the pair must still BE that text
        if (word.substr(L.off, L.n) + word.substr(R.off, R.n) != b.text) continue;

        L.n += R.n;
        R.n  = 0;
        L.next = R.next;
        if (R.next >= 0) symbols[R.next].prev = b.left;

        add_new_bigram(L.prev, b.left);
        add_new_bigram(b.left, L.next);
    }

    // map each surviving symbol to an id, with the per-byte fallback (llama-vocab.cpp:736-760)
    for (size_t i = 0; i < symbols.size(); ++i) {
        if (symbols[i].n == 0) continue;
        const std::string str = word.substr(symbols[i].off, symbols[i].n);
        auto it = text_to_id.find(str);
        if (it != text_to_id.end()) {
            out.push_back(it->second);
            continue;
        }
        // FAITHFUL QUIRK: the reference iterates the RAW BYTES OF THE BYTE-ENCODED STRING
        // (`for (auto j = str.begin(); ...)`, llama-vocab.cpp:751-757), not the original
        // bytes, so a symbol it cannot find is emitted as single-byte lookups or DROPPED.
        // With a complete byte-encoded vocab this branch is unreachable, but a divergence
        // here is exactly what a parity gate would flag — so it is reproduced, not "fixed".
        for (const char c : str) {
            auto it2 = text_to_id.find(std::string(1, c));
            if (it2 != text_to_id.end()) out.push_back(it2->second);
        }
    }
}

std::vector<int32_t> M1Tok::encode(const std::string & text, bool add_bos_,
                                   bool parse_special) const {
    std::vector<int32_t> out;
    if (add_bos_) {
        CHECK(id_bos >= 0, "m1_tok: add_bos requested but the vocab has no bos id");
        out.push_back(id_bos);          // llama-vocab.cpp:599 — BEFORE the fragments
    }
    if (text.empty()) return out;       // unicode_regex_split("") returns {} (:1369)

    for (const Frag & f : partition(text, parse_special)) {
        if (f.special) { out.push_back(f.id); continue; }
        // escape_whitespaces is FALSE for pixtral (llama-vocab.cpp:2272), byte_encode TRUE
        const std::string frag = text.substr(f.off, f.len);
        for (const std::string & w : unicode_regex_split(frag, regex_exprs, /*byte_encode=*/true)) {
            encode_word(w, out);
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// stage 4 — detokenize
// ---------------------------------------------------------------------------
namespace {

// llama_decode_text (llama-vocab.cpp:3528-3545), verbatim including the [UNK_BYTE_0x..] shape
std::string decode_text(const std::string & text) {
    std::string out;
    for (const uint32_t cpt : unicode_cpts_from_utf8(text)) {
        const std::string utf8 = unicode_cpt_to_utf8(cpt);
        try {
            out += (char) unicode_utf8_to_byte(utf8);
        } catch (const std::out_of_range &) {
            char buf[64];
            out += "[UNK_BYTE_0x";
            for (const char c : utf8) {
                std::snprintf(buf, sizeof buf, "%02x", (uint8_t) c);
                out += buf;
            }
            out += text + "]";
        }
    }
    return out;
}

}  // namespace

std::string M1Tok::piece(int32_t id) const {
    CHECK(id >= 0 && (size_t) id < id_to_text.size(),
          "m1_tok: token id %d out of range [0,%zu)", id, id_to_text.size());
    const Token & t = id_to_text[id];

    // token_to_piece, LLAMA_VOCAB_TYPE_BPE arm (llama-vocab.cpp:3865-3885).  `special` is TRUE
    // on the decode path (the cache is built with special=true, :3177), so control/unknown/
    // user-defined tokens render their LITERAL text ('<s>', '[INST]') rather than nothing.
    if (t.attr & (M1_TOK_ATTR_UNKNOWN | M1_TOK_ATTR_CONTROL | M1_TOK_ATTR_USER_DEFINED))
        return t.text;
    if (t.attr & M1_TOK_ATTR_NORMAL) {
        // mutant 3: emit the byte-ENCODED text instead of decoding it back to bytes.  The ids
        // arm cannot see this — which is exactly why the gate compares detokenized bytes too.
        if (mutant == 3) return t.text;
        return decode_text(t.text);
    }
    if (t.attr & M1_TOK_ATTR_BYTE) {
        // token_to_byte, BPE arm (:3329-3333): text.substr(3,2) as hex
        CHECK(t.text.size() >= 5, "m1_tok: byte token %d text '%s' too short", id, t.text.c_str());
        const std::string hex = t.text.substr(3, 2);
        return std::string(1, (char) std::strtol(hex.c_str(), nullptr, 16));
    }
    return std::string();   // UNDEFINED / UNUSED: llama's token_to_piece falls through to 0
}

std::string M1Tok::decode(const std::vector<int32_t> & ids) const {
    std::string out;
    for (const int32_t id : ids) out += piece(id);
    return out;
}

// ---------------------------------------------------------------------------
// streaming detokenizer
// ---------------------------------------------------------------------------
namespace {

// Length of the longest prefix of `s` that is a whole number of UTF-8 sequences.
// A lone lead/continuation byte at the end is held back (it may be completed by the next
// token); bytes that cannot begin a sequence are emitted as-is so the stream never stalls.
size_t utf8_complete_prefix(const std::string & s) {
    size_t i = 0;
    while (i < s.size()) {
        const uint8_t c = (uint8_t) s[i];
        size_t len;
        if      (c < 0x80)          len = 1;
        else if ((c & 0xE0) == 0xC0) len = 2;
        else if ((c & 0xF0) == 0xE0) len = 3;
        else if ((c & 0xF8) == 0xF0) len = 4;
        else                        len = 1;   // continuation or invalid: never wait on it
        if (i + len > s.size()) return i;      // incomplete tail
        i += len;
    }
    return i;
}

}  // namespace

std::string M1TokDecoder::push_bytes(const std::string & bytes) {
    tail += bytes;
    const size_t safe = utf8_complete_prefix(tail);
    std::string out = tail.substr(0, safe);
    tail.erase(0, safe);
    return out;
}

std::string M1TokDecoder::push(int32_t id) {
    CHECK(tok != nullptr, "m1_tok: decoder used without a tokenizer");
    return push_bytes(tok->piece(id));
}

std::string M1TokDecoder::finish(size_t * dropped) {
    const size_t n = tail.size();
    if (dropped) *dropped = n;
    if (n) {
        // loud, not silent: a partial sequence at end-of-generation is a dropped codepoint
        std::fprintf(stderr, "WARN: m1_tok: dropping %zu dangling UTF-8 byte(s) at end of "
                             "generation: ", n);
        for (const char c : tail) std::fprintf(stderr, "%02x", (uint8_t) c);
        std::fprintf(stderr, "\n");
    }
    tail.clear();
    return std::string();
}
