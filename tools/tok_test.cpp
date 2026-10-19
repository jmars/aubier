// tools/tok_test.cpp — CPU-only driver for engine/m1_tok (server plan §H U1).
//
// No GPU, no ggml backend, no gpu.lock: it mmaps the GGUF for its METADATA only and links
// ggml-base + ggml-cpu exactly like tools/gg_fixture.cpp does (scripts/engine_build_tok.sh).
//
// Subcommands (all take -m <model.gguf>, default the one model):
//   --encode <file>          print the id array of the file's exact bytes
//                            (--no-bos / --no-parse-special to vary the flags)
//   --decode <ids>           ids as a comma/space list or a python-ish "[1, 2, 3]"
//   --pieces <file>          per token: "<id>\t<hex of the detokenized bytes>" — the
//                            detokenizer parity arm (unambiguous, unlike the reference's
//                            quoted lines, which a piece containing a newline can break)
//   --roundtrip <file>       CHECK decode(encode(bytes)) == bytes, exit 1 on mismatch
//   --stream <file>          decode through M1TokDecoder and CHECK the concatenation equals
//                            the whole-sequence decode and that no chunk ends mid-codepoint
//   --special-list           print the 1000 special-token texts, one per line, hex-escaped
//                            (\n, \\, \\xNN) — the gate feeds these back as literal text
//   --quote <file>           print the file bytes as a C-style double-quoted string (so the
//                            corpus can be handed to the reference's -p without shell loss)
//   --info                   print summary() + the loaded counts
//
// Exit code is the claim: 0 = the subcommand's own CHECKs passed.

#include "ggml.h"
#include "gguf.h"

#include "m1_tok.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// gguf_model.h supplies FATAL/CHECK and the GGUF reader the engine uses; include it only for
// GGUFModel::load so the driver reads the vocab through the SAME open context the engine and
// the server will.
#include "gguf_model.h"

static const char * DEFAULT_MODEL =
    "models/Mistral-Small-4-119B-2603-UD-IQ2_XXS.gguf";

static std::string read_file(const char * path) {
    FILE * f = std::fopen(path, "rb");
    CHECK(f != nullptr, "cannot open '%s'", path);
    std::string out;
    char buf[65536];
    size_t n;
    while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) out.append(buf, n);
    CHECK(!std::ferror(f), "read error on '%s'", path);
    std::fclose(f);
    return out;
}

static void print_ids(const std::vector<int32_t> & ids) {
    std::printf("[");
    for (size_t i = 0; i < ids.size(); ++i) {
        std::printf(i ? ", %d" : "%d", ids[i]);
    }
    std::printf("]\n");
}

static void print_hex(const std::string & s) {
    for (const char c : s) std::printf("%02x", (uint8_t) c);
}

// parse "1,2,3" / "1 2 3" / "[1, 2, 3]"
static std::vector<int32_t> parse_ids(const char * s) {
    std::vector<int32_t> ids;
    const char * p = s;
    while (*p) {
        while (*p && !(*p >= '0' && *p <= '9') && *p != '-') ++p;
        if (!*p) break;
        char * end = nullptr;
        const long v = std::strtol(p, &end, 10);
        CHECK(end != p, "bad id list at '%s'", p);
        ids.push_back((int32_t) v);
        p = end;
    }
    return ids;
}

// C-style double-quoted rendering of raw bytes; the reference applies string_process_escapes to
// -p/-f by default, so --no-escape is used with the raw form and this form is for -p arms.
static std::string quote(const std::string & s) {
    std::string out = "\"";
    for (const char c : s) {
        const uint8_t b = (uint8_t) c;
        switch (c) {
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            case '\\': out += "\\\\"; break;
            case '"':  out += "\\\""; break;
            default:
                if (b < 0x20 || b >= 0x7f) {
                    char buf[8];
                    std::snprintf(buf, sizeof buf, "\\x%02x", b);
                    out += buf;
                } else {
                    out += c;
                }
        }
    }
    out += '"';
    return out;
}

int main(int argc, char ** argv) {
    const char * model_path = DEFAULT_MODEL;
    const char * cmd = nullptr;
    const char * arg = nullptr;

    const char * encode_file = nullptr;
    const char * pieces_file = nullptr;
    const char * roundtrip_file = nullptr;
    const char * stream_file = nullptr;
    const char * quote_file = nullptr;
    bool no_bos = false;
    bool no_parse_special = false;

    for (int i = 1; i < argc; ++i) {
        const char * a = argv[i];
        if (!std::strcmp(a, "-m") || !std::strcmp(a, "--model")) {
            CHECK(i + 1 < argc, "-m needs a path");
            model_path = argv[++i];
        } else if (!std::strcmp(a, "--encode")) {
            CHECK(i + 1 < argc, "--encode needs a file");
            encode_file = argv[++i];
        } else if (!std::strcmp(a, "--decode")) {
            CHECK(i + 1 < argc, "--decode needs an id list");
            cmd = "decode"; arg = argv[++i];
        } else if (!std::strcmp(a, "--pieces")) {
            CHECK(i + 1 < argc, "--pieces needs a file");
            pieces_file = argv[++i];
        } else if (!std::strcmp(a, "--roundtrip")) {
            CHECK(i + 1 < argc, "--roundtrip needs a file");
            roundtrip_file = argv[++i];
        } else if (!std::strcmp(a, "--stream")) {
            CHECK(i + 1 < argc, "--stream needs a file");
            stream_file = argv[++i];
        } else if (!std::strcmp(a, "--quote")) {
            CHECK(i + 1 < argc, "--quote needs a file");
            quote_file = argv[++i];
        } else if (!std::strcmp(a, "--no-bos")) {
            no_bos = true;
        } else if (!std::strcmp(a, "--no-parse-special")) {
            no_parse_special = true;
        } else if (!std::strcmp(a, "--special-list")) {
            cmd = "special-list";
        } else if (!std::strcmp(a, "--info")) {
            cmd = "info";
        } else if (!std::strcmp(a, "-h") || !std::strcmp(a, "--help")) {
            std::printf("usage: tok_test [-m model.gguf] {--encode F|--decode IDS|--pieces F|"
                        "--roundtrip F|--stream F|--quote F|--special-list|--info} "
                        "[--no-bos] [--no-parse-special]\n");
            return 0;
        } else {
            FATAL("unknown argument '%s'", a);
        }
    }

    GGUFModel * m = GGUFModel::load(model_path);
    M1Tok * tok = M1Tok::load(m->gctx);

    const bool add_bos = !no_bos && tok->add_bos;
    const bool parse_special = !no_parse_special;

    if (encode_file) {
        print_ids(tok->encode(read_file(encode_file), add_bos, parse_special));
    } else if (cmd && !std::strcmp(cmd, "decode")) {
        std::fputs(tok->decode(parse_ids(arg)).c_str(), stdout);
    } else if (pieces_file) {
        const std::string text = read_file(pieces_file);
        for (const int32_t id : tok->encode(text, add_bos, parse_special)) {
            std::printf("%d\t", id);
            print_hex(tok->piece(id));
            std::printf("\n");
        }
    } else if (roundtrip_file) {
        const std::string text = read_file(roundtrip_file);
        // round-trip is only meaningful when nothing in the text was a special token: a
        // special token detokenizes to its literal text, so it round-trips too — but a
        // BYTE-encoded vocab is not injective for invalid UTF-8 input, which is why the
        // gate's corpus is valid UTF-8.
        const std::vector<int32_t> ids = tok->encode(text, /*add_bos=*/false, parse_special);
        const std::string back = tok->decode(ids);
        if (back != text) {
            std::fprintf(stderr, "ROUNDTRIP FAIL: %zu bytes in, %zu bytes out\n",
                         text.size(), back.size());
            size_t i = 0;
            while (i < text.size() && i < back.size() && text[i] == back[i]) ++i;
            std::fprintf(stderr, "  first difference at byte %zu\n", i);
            std::fprintf(stderr, "  in : ");
            print_hex(text.substr(std::min(i, text.size()), 32));
            std::fprintf(stderr, "\n  out: ");
            print_hex(back.substr(std::min(i, back.size()), 32));
            std::fprintf(stderr, "\n");
            return 1;
        }
        std::fprintf(stderr, "roundtrip OK: %zu bytes, %zu ids\n", text.size(), ids.size());
    } else if (stream_file) {
        const std::string text = read_file(stream_file);
        const std::vector<int32_t> ids = tok->encode(text, /*add_bos=*/false, parse_special);
        M1TokDecoder dec;
        dec.tok = tok;
        std::string streamed;
        size_t chunks = 0;
        for (const int32_t id : ids) {
            const std::string chunk = dec.push(id);
            if (!chunk.empty()) ++chunks;
            streamed += chunk;
        }
        size_t dropped = 0;
        streamed += dec.finish(&dropped);
        const std::string whole = tok->decode(ids);
        if (streamed != whole) {
            std::fprintf(stderr, "STREAM FAIL: streamed %zu bytes != whole %zu bytes "
                                 "(dropped=%zu)\n", streamed.size(), whole.size(), dropped);
            return 1;
        }
        // and the stream must equal the input: a special token renders its literal text, so this
        // holds in BOTH parse_special arms (the round-trip arm asserts the same thing)
        if (streamed != text) {
            std::fprintf(stderr, "STREAM FAIL: streamed %zu bytes != input %zu bytes\n",
                         streamed.size(), text.size());
            return 1;
        }
        std::fprintf(stderr, "stream OK: %zu bytes in %zu non-empty chunks, dropped=%zu\n",
                     streamed.size(), chunks, dropped);
    } else if (quote_file) {
        std::printf("%s\n", quote(read_file(quote_file)).c_str());
    } else if (cmd && !std::strcmp(cmd, "special-list")) {
        for (const int32_t id : tok->special_ids) {
            std::printf("%d\t", id);
            print_hex(tok->id_to_text[id].text);
            std::printf("\n");
        }
    } else if (cmd && !std::strcmp(cmd, "info")) {
        std::printf("%s\n", tok->summary().c_str());
    } else {
        FATAL("no subcommand given (try --info)");
    }

    delete tok;
    delete m;
    return 0;
}
