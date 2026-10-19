// engine/m1_chat.cpp — the Mistral Small 4 chat render, hash-pinned. See m1_chat.h for the
// contract and for the MEASURED render it must reproduce.
//
// Two independent boot checks (both FATAL):
//   1. sha256 of the GGUF's raw tokenizer.chat_template bytes == the pinned constant.
//   2. the default_system_message literal EXTRACTED from those same template bytes equals the
//      compiled-in transcription in m1_chat_dsm.inc, byte for byte, and renders to the pinned
//      length. (The hash alone would catch a template change; #2 also catches a transcription
//      mistake in OUR literal, which the hash cannot see.)
//
// Compiled twice: as the renderer the server links, and (with -DM1_CHAT_DRIVER) as the test
// driver builds/engine/m1_chat_ours that scripts/engine_gate_chat.sh drives. The driver lives
// here because the unit owns exactly one .cpp for the renderer; it is test code and is never
// linked into a product binary.

#include "m1_chat.h"

#include <cctype>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>

#include "m1_chat_dsm.inc"

// The pinned sha256 as this build sees it. M1_CHAT_MUTANT_SHA (scripts/engine_build_chat.sh
// MUTANT=sha) shifts its last byte by one so the gate's FAIL path can be exercised; the
// product constant is the one in m1_chat.h.
#if defined(M1_CHAT_MUTANT_SHA)
static constexpr const char * M1_CHAT_PIN_IN_USE =
    "d579fedfe0a78b11376089da957bbd4299ffb3f95cf827ab00c9cdc77298f953";  // MUTANT: last byte -1
#else
static constexpr const char * M1_CHAT_PIN_IN_USE = M1_CHAT_TEMPLATE_SHA256;
#endif

// ---------------------------------------------------------------------------
// FATAL — same shape and exit code as engine/gguf_model.h's FATAL (1), so an operator sees
// one style. This TU does not include ggml headers and must not.
// ---------------------------------------------------------------------------
#if defined(__GNUC__)
__attribute__((format(printf, 1, 2), noreturn))
#endif
static void m1_chat_fatal(const char * fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    std::fprintf(stderr, "FATAL: ");
    std::vfprintf(stderr, fmt, ap);
    std::fprintf(stderr, "\n");
    va_end(ap);
    std::exit(1);
}

// ---------------------------------------------------------------------------
// SHA-256 — FIPS 180-4, the real thing (~60 lines). Not FNV, not a checksum.
// ---------------------------------------------------------------------------
namespace {

inline uint32_t rotr32(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

const uint32_t SHA256_K[64] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
    0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
    0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
    0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u, 0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u,
};

void sha256_block(uint32_t h[8], const uint8_t * p) {
    uint32_t w[64];
    for (int i = 0; i < 16; i++) {
        w[i] = (uint32_t)p[4 * i] << 24 | (uint32_t)p[4 * i + 1] << 16 | (uint32_t)p[4 * i + 2] << 8 | (uint32_t)p[4 * i + 3];
    }
    for (int i = 16; i < 64; i++) {
        uint32_t s0 = rotr32(w[i - 15], 7) ^ rotr32(w[i - 15], 18) ^ (w[i - 15] >> 3);
        uint32_t s1 = rotr32(w[i - 2], 17) ^ rotr32(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i]        = w[i - 16] + s0 + w[i - 7] + s1;
    }
    uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
    for (int i = 0; i < 64; i++) {
        uint32_t S1  = rotr32(e, 6) ^ rotr32(e, 11) ^ rotr32(e, 25);
        uint32_t ch  = (e & f) ^ (~e & g);
        uint32_t t1  = hh + S1 + ch + SHA256_K[i] + w[i];
        uint32_t S0  = rotr32(a, 2) ^ rotr32(a, 13) ^ rotr32(a, 22);
        uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t t2  = S0 + maj;
        hh = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
    }
    h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
}

void sha256(const void * data, std::size_t n, uint8_t out[32]) {
    uint32_t h[8] = { 0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
                      0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u };
    const uint8_t * p = static_cast<const uint8_t *>(data);
    std::size_t      i = 0;
    for (; i + 64 <= n; i += 64) sha256_block(h, p + i);
    uint8_t tail[128];
    std::size_t r = n - i;
    std::memcpy(tail, p + i, r);
    tail[r++] = 0x80;
    std::size_t total = (r <= 56) ? 64 : 128;
    std::memset(tail + r, 0, total - r);
    const uint64_t bits = (uint64_t)n * 8u;
    for (int k = 0; k < 8; k++) tail[total - 1 - k] = (uint8_t)(bits >> (8 * k));
    for (std::size_t o = 0; o < total; o += 64) sha256_block(h, tail + o);
    for (int k = 0; k < 8; k++) {
        out[4 * k]     = (uint8_t)(h[k] >> 24);
        out[4 * k + 1] = (uint8_t)(h[k] >> 16);
        out[4 * k + 2] = (uint8_t)(h[k] >> 8);
        out[4 * k + 3] = (uint8_t)h[k];
    }
}

bool starts_with(const std::string & s, const char * p) { return s.compare(0, std::strlen(p), p) == 0; }

// The template's single-quoted Jinja literal escapes, unescaped.
std::string jinja_unescape(const std::string & raw) {
    std::string o;
    o.reserve(raw.size());
    for (std::size_t i = 0; i < raw.size(); i++) {
        if (raw[i] != '\\') { o += raw[i]; continue; }
        i++;
        if (i >= raw.size()) m1_chat_fatal("jinja_unescape: trailing backslash");
        switch (raw[i]) {
            case 'n': o += '\n'; break;
            case 't': o += '\t'; break;
            case 'r': o += '\r'; break;
            case '\\': o += '\\'; break;
            case '\'': o += '\''; break;
            case '"': o += '"'; break;
            default:
                m1_chat_fatal("jinja_unescape: unsupported escape \\%c in the pinned template", raw[i]);
        }
    }
    return o;
}

}  // namespace

std::string m1_chat_sha256_hex(const void * data, std::size_t n) {
    uint8_t d[32];
    sha256(data, n, d);
    static const char * hex = "0123456789abcdef";
    std::string        o;
    o.resize(64);
    for (int i = 0; i < 32; i++) {
        o[2 * i]     = hex[d[i] >> 4];
        o[2 * i + 1] = hex[d[i] & 0xf];
    }
    return o;
}

// ---------------------------------------------------------------------------
// boot check #1 — the pin
// ---------------------------------------------------------------------------
void m1_chat_verify_template(const void * tmpl_bytes, std::size_t len) {
    if (tmpl_bytes == nullptr || len == 0) m1_chat_fatal("chat template: no bytes to verify (the GGUF has no tokenizer.chat_template?)");
    const std::string got = m1_chat_sha256_hex(tmpl_bytes, len);
    if (len != M1_CHAT_TEMPLATE_LEN) {
        m1_chat_fatal("chat template: length %zu, pinned %zu — the model's template changed; the render "
                      "in this file was written for the pinned template only",
                      len, M1_CHAT_TEMPLATE_LEN);
    }
    if (got != M1_CHAT_PIN_IN_USE) {
        m1_chat_fatal("chat template: sha256 %s, pinned %s — REFUSING to render with a template this "
                      "code was not written against",
                      got.c_str(), M1_CHAT_PIN_IN_USE);
    }
}

// ---------------------------------------------------------------------------
// boot check #2 — the literal transcribed in m1_chat_dsm.inc against the template's own bytes
//
// The template computes it as  'p0' + today_date + 'p1' + yesterday_date + 'p2'  (template line
// 145). Parse that expression out of the pinned bytes and compare each literal piece with the
// transcription. Independent of the hash: it fails if OUR copy has a typo.
// ---------------------------------------------------------------------------
namespace {

struct DsmPieces {
    std::vector<std::string> lit;
    std::vector<std::string> id;  // identifiers, in order
};

DsmPieces extract_dsm_pieces(const std::string & tmpl) {
    const std::string open = "{%- set default_system_message = ";
    const std::size_t at  = tmpl.find(open);
    if (at == std::string::npos) m1_chat_fatal("chat template: no '%s' — the pinned template no longer defines it", open.c_str());
    const std::size_t begin = at + open.size();
    const std::size_t end   = tmpl.find(" %}", begin);
    if (end == std::string::npos) m1_chat_fatal("chat template: unterminated default_system_message expression");
    const std::string expr = tmpl.substr(begin, end - begin);

    DsmPieces out;
    std::size_t i = 0;
    while (i < expr.size()) {
        const char c = expr[i];
        if (c == ' ' || c == '+' || c == '\n' || c == '\r' || c == '\t') { i++; continue; }
        if (c == '\'') {
            std::string raw;
            i++;
            for (;;) {
                if (i >= expr.size()) m1_chat_fatal("chat template: unterminated single-quoted literal");
                if (expr[i] == '\\') { raw += expr[i]; raw += expr[i + 1]; i += 2; continue; }
                if (expr[i] == '\'') { i++; break; }
                raw += expr[i++];
            }
            out.lit.push_back(jinja_unescape(raw));
        } else if (std::isalpha((unsigned char)c) || c == '_') {
            std::string id;
            while (i < expr.size() && (std::isalnum((unsigned char)expr[i]) || expr[i] == '_')) id += expr[i++];
            out.id.push_back(id);
        } else {
            m1_chat_fatal("chat template: unexpected character '%c' in the default_system_message expression", c);
        }
    }
    if (out.lit.size() != 3 || out.id.size() != 2 || out.id[0] != "today_date" || out.id[1] != "yesterday_date") {
        m1_chat_fatal("chat template: the default_system_message expression is not p0+today_date+p1+"
                      "yesterday_date+p2 (%zu literals, %zu identifiers) — the renderer's assumption is void",
                      out.lit.size(), out.id.size());
    }
    return out;
}

}  // namespace

void m1_chat_verify_default_system_message(const void * tmpl_bytes, std::size_t len) {
    const std::string tmpl(static_cast<const char *>(tmpl_bytes), len);
    const DsmPieces   p = extract_dsm_pieces(tmpl);
    const char * const mine[3] = { kDsmPiece0, kDsmPiece1, kDsmPiece2 };
    for (int k = 0; k < 3; k++) {
        if (p.lit[k] == mine[k]) continue;
        // first differing byte, for the operator
        std::size_t d = 0;
        while (d < p.lit[k].size() && d < std::strlen(mine[k]) && p.lit[k][d] == mine[k][d]) d++;
        m1_chat_fatal("default_system_message piece %d differs from the pinned template at byte %zu "
                      "(template %zu bytes, ours %zu): template '%s' vs ours '%s'",
                      k, d, p.lit[k].size(), std::strlen(mine[k]),
                      p.lit[k].substr(d, 40).c_str(), std::string(mine[k]).substr(d, 40).c_str());
    }
    // The rendered length does not depend on the date (each date is exactly 10 bytes).
    const std::size_t rendered = std::strlen(kDsmPiece0) + 10 + std::strlen(kDsmPiece1) + 10 + std::strlen(kDsmPiece2);
    if (rendered != M1_CHAT_DEFAULT_SYSTEM_MESSAGE_LEN) {
        m1_chat_fatal("default_system_message: renders to %zu bytes, pinned %zu", rendered, M1_CHAT_DEFAULT_SYSTEM_MESSAGE_LEN);
    }
}

const char * m1_chat_bos_name(M1ChatBos bos) {
    switch (bos) {
        case M1_CHAT_BOS_TEMPLATE: return "template";
        case M1_CHAT_BOS_TOKENIZER: return "tokenizer";
    }
    return "?";
}

// ---------------------------------------------------------------------------
// dates
// ---------------------------------------------------------------------------
M1ChatDate m1_chat_today() {
    const std::time_t t = std::time(nullptr);
    std::tm           tm{};
    if (localtime_r(&t, &tm) == nullptr) m1_chat_fatal("localtime_r failed");
    return M1ChatDate{ tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday };
}

std::string m1_chat_format_ymd(const M1ChatDate & d) {
    char buf[16];
    std::snprintf(buf, sizeof buf, "%04d-%02d-%02d", d.year, d.month, d.day);
    return buf;
}

bool m1_chat_parse_ymd(const char * s, M1ChatDate * out) {
    if (s == nullptr) return false;
    int y = 0, m = 0, d = 0;
    if (std::sscanf(s, "%d-%d-%d", &y, &m, &d) != 3) return false;
    if (y < 1970 || y > 9999 || m < 1 || m > 12 || d < 1 || d > 31) return false;
    if (out) *out = M1ChatDate{ y, m, d };
    return true;
}

// The template's ladder, transcribed branch for branch from lines 6-142 of the pinned template.
bool m1_chat_prev_day(M1ChatDate in, M1ChatDate * out) {
    const int y = in.year, m = in.month, d = in.day;
    if (m < 1 || m > 12 || d < 1 || d > 31) return false;
    M1ChatDate o = in;
    if (d != 1) {
        o.day = d - 1;
        if (out) *out = o;
        return true;
    }
    auto year_in = [&](std::initializer_list<int> ys) {
        for (int v : ys) if (v == y) return true;
        return false;
    };
    switch (m) {
        case 1:  // '01' -> 31 Dec of the previous year, for the template's exact year list
            if (y == 1970) { o = M1ChatDate{ 1970, 12, 31 }; break; }
            if (y < 2024 || y > 2032) return false;  // the template raises here
            o = M1ChatDate{ y - 1, 12, 31 };
            break;
        case 2:  o = M1ChatDate{ y, 1, 31 }; break;
        case 3:  // '03' -> 28 Feb, or 29 Feb for the template's exact (buggy) leap-year list
            if (year_in({ 2024, 2028, 2032, 1970 })) o = M1ChatDate{ y, 2, 29 };
            else return false;  // the template raises for every other year
            break;
        case 4:  o = M1ChatDate{ y, 3, 31 }; break;
        case 5:  o = M1ChatDate{ y, 4, 30 }; break;
        case 6:  o = M1ChatDate{ y, 5, 31 }; break;
        case 7:  o = M1ChatDate{ y, 6, 30 }; break;
        case 8:  o = M1ChatDate{ y, 7, 31 }; break;
        case 9:  o = M1ChatDate{ y, 8, 31 }; break;
        case 10: o = M1ChatDate{ y, 9, 30 }; break;
        case 11: o = M1ChatDate{ y, 10, 31 }; break;
        case 12: o = M1ChatDate{ y, 11, 30 }; break;
        default: return false;
    }
    if (out) *out = o;
    return true;
}

std::string m1_chat_default_system_message(const M1ChatDate & now) {
    M1ChatDate       prev;
    if (!m1_chat_prev_day(now, &prev)) {
        m1_chat_fatal("the pinned chat template's date ladder REFUSES %s: 1 March is only handled for "
                      "the template's hardcoded leap years {2024,2028,2032,1970}, and 1 January only for "
                      "{1970,2024..2032}. llama.cpp's Jinja raises on exactly these dates too (gate arm "
                      "'refuse' proves it) — this is a bug in the model's template, reproduced, not widened",
                      m1_chat_format_ymd(now).c_str());
    }
    std::string s;
    s.reserve(M1_CHAT_DEFAULT_SYSTEM_MESSAGE_LEN);
    s += kDsmPiece0;
    s += m1_chat_format_ymd(now);
    s += kDsmPiece1;
    s += m1_chat_format_ymd(prev);
    s += kDsmPiece2;
#if defined(M1_CHAT_MUTANT_DSM)
    // MUTANT (scripts/engine_build_chat.sh MUTANT=dsm): one byte of the literal drifts from the
    // pinned template. The boot checks cannot see it (they compare the TEMPLATE's pieces against
    // the compiled-in ones, and this changes the RENDER); the gate's byte-diff against the oracle
    // is what must catch it.
    const std::size_t at = s.find("Paris");
    if (at != std::string::npos) s[at] = 'Q';
#endif
    return s;
}

bool m1_chat_can_render(const M1ChatOpts & opts, std::string * why) {
    M1ChatDate now;
    if (opts.now_ymd && *opts.now_ymd) {
        if (!m1_chat_parse_ymd(opts.now_ymd, &now)) {
            if (why) *why = std::string("bad now_ymd '") + opts.now_ymd + "'";
            return false;
        }
    } else {
        now = m1_chat_today();
    }
    M1ChatDate prev;
    if (!m1_chat_prev_day(now, &prev)) {
        if (why) {
            *why = "the pinned template's date ladder refuses " + m1_chat_format_ymd(now) +
                   " (1 March outside {2024,2028,2032,1970} / 1 January outside {1970,2024..2032})";
        }
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// the render
// ---------------------------------------------------------------------------
std::string render_chat(const std::vector<M1ChatMessage> & messages, const M1ChatOpts & opts) {
    if (messages.empty()) m1_chat_fatal("render_chat: no messages (the template indexes messages[0])");

    M1ChatDate now;
    if (opts.now_ymd && *opts.now_ymd) {
        if (!m1_chat_parse_ymd(opts.now_ymd, &now)) m1_chat_fatal("render_chat: bad now_ymd '%s'", opts.now_ymd);
    } else {
        now = m1_chat_today();
    }
    // The template computes today_date/yesterday_date UNCONDITIONALLY (line 144, before it looks
    // at the messages), so its ladder can refuse even when an explicit system message is
    // supplied. Reproduced: m1_chat_default_system_message() FATALs on those dates, and we call
    // it here regardless of where the system text comes from — same refusal as the oracle.
    const std::string dsm = m1_chat_default_system_message(now);

    std::string out;
    out.reserve(256 + dsm.size() + messages.size() * 32);
    out += "<s>";

    std::size_t first = 0;
    if (messages[0].role == "system") {
        out += "[SYSTEM_PROMPT]";
        out += messages[0].content;
        out += "[/SYSTEM_PROMPT]";
        first = 1;
    } else if (opts.default_system_message == nullptr) {
        out += "[SYSTEM_PROMPT]";
        out += dsm;
        out += "[/SYSTEM_PROMPT]";
    } else if (*opts.default_system_message != '\0') {
        out += "[SYSTEM_PROMPT]";
        out += opts.default_system_message;
        out += "[/SYSTEM_PROMPT]";
    }  // an empty explicit default_system_message omits the block (the template's if != '')

    out += M1_CHAT_MODEL_SETTINGS;

    // The template's role-alternation check (lines 195-204), before it renders anything: an
    // exception, not a warning. We have no tool_calls, so every user/assistant message counts.
    int idx = 0;
    for (std::size_t i = first; i < messages.size(); i++) {
        const std::string & role = messages[i].role;
        if (role == "user" || role == "assistant") {
            const bool want_user = (idx % 2 == 0);
            if ((role == "user") != want_user) {
                m1_chat_fatal("render_chat: after the optional system message, conversation roles must "
                              "alternate user and assistant (message %zu is '%s' where '%s' was expected)",
                              i, role.c_str(), want_user ? "user" : "assistant");
            }
            idx++;
        }
    }

    for (std::size_t i = first; i < messages.size(); i++) {
        const M1ChatMessage & m = messages[i];
        if (m.role == "user") {
            out += "[INST]";
            out += m.content;
            out += "[/INST]";
        } else if (m.role == "assistant") {
            // Empty content is NOT an error here: the raw template raises, but llama.cpp patches
            // that exact condition out (common/chat.cpp's "[TOOL_CALLS]"/"is none or" fixup) and
            // the oracle is llama.cpp. The oracle renders '</s>' alone; so do we.
            out += m.content;
#if !defined(M1_CHAT_MUTANT_EOS)
            out += "</s>";
#endif
        } else if (m.role == "tool") {
            out += "[TOOL_RESULTS]";
            out += m.content;
            out += "[/TOOL_RESULTS]";
        } else {
            m1_chat_fatal("render_chat: only user, assistant and tool roles are supported after the "
                          "system message, got '%s' (message %zu)", m.role.c_str(), i);
        }
    }
    return out;
}

// ===========================================================================
// TEST DRIVER — compiled only with -DM1_CHAT_DRIVER (builds/engine/m1_chat_ours).
// Not part of any product binary.
// ===========================================================================
#ifdef M1_CHAT_DRIVER

#include <cctype>
#include <sys/wait.h>
#include <unistd.h>

namespace {

std::string read_file(const char * path) {
    std::FILE * f = std::fopen(path, "rb");
    if (!f) m1_chat_fatal("cannot open %s", path);
    std::string s;
    char        buf[1 << 16];
    std::size_t n;
    while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) s.append(buf, n);
    std::fclose(f);
    return s;
}

void write_file(const std::string & path, const std::string & data) {
    std::FILE * f = std::fopen(path.c_str(), "wb");
    if (!f) m1_chat_fatal("cannot write %s", path.c_str());
    if (!data.empty() && std::fwrite(data.data(), 1, data.size(), f) != data.size()) m1_chat_fatal("short write %s", path.c_str());
    std::fclose(f);
}

// \s (space) exists so that leading/trailing spaces in a case are VISIBLE in the case file.
std::string unescape_case(const std::string & raw) {
    std::string o;
    for (std::size_t i = 0; i < raw.size(); i++) {
        if (raw[i] != '\\') { o += raw[i]; continue; }
        i++;
        if (i >= raw.size()) m1_chat_fatal("case file: trailing backslash");
        switch (raw[i]) {
            case 'n': o += '\n'; break;
            case 't': o += '\t'; break;
            case 's': o += ' '; break;
            case '\\': o += '\\'; break;
            default: m1_chat_fatal("case file: unsupported escape \\%c", raw[i]);
        }
    }
    return o;
}

// case file format (identical in tools/m1_chat_ref.cpp, written independently there):
//   # comment
//   case <name>
//   <role>\t<content>
//   ...
//   (a blank line or the next "case " line ends the case)
struct Case {
    std::string              name;
    std::vector<M1ChatMessage> msgs;
};

std::vector<Case> read_cases(const char * path) {
    const std::string text = read_file(path);
    std::vector<Case> cases;
    std::size_t       i = 0;
    while (i < text.size()) {
        std::size_t e = text.find('\n', i);
        if (e == std::string::npos) e = text.size();
        std::string line = text.substr(i, e - i);
        i = e + 1;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || line[0] == '#') continue;
        if (starts_with(line, "case ")) {
            cases.push_back(Case{ line.substr(5), {} });
            if (cases.back().name.empty()) m1_chat_fatal("case file: a case with no name");
            continue;
        }
        if (cases.empty()) m1_chat_fatal("case file: message line before any 'case <name>'");
        const std::size_t tab = line.find('\t');
        if (tab == std::string::npos) m1_chat_fatal("case file: message line without a TAB separator: '%s'", line.c_str());
        cases.back().msgs.push_back(M1ChatMessage{ line.substr(0, tab), unescape_case(line.substr(tab + 1)) });
    }
    if (cases.empty()) m1_chat_fatal("case file: no cases");
    return cases;
}

int cmd_verify(const char * tmpl_path) {
    const std::string tmpl = read_file(tmpl_path);
    m1_chat_verify_template(tmpl.data(), tmpl.size());
    m1_chat_verify_default_system_message(tmpl.data(), tmpl.size());
    std::printf("PIN OK sha256=%s len=%zu dsm_len=%zu\n", m1_chat_sha256_hex(tmpl.data(), tmpl.size()).c_str(),
                tmpl.size(), m1_chat_default_system_message(m1_chat_today()).size());
    return 0;
}

int cmd_selftest() {
    struct { const char * in; const char * hex; } vec[] = {
        { "", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855" },
        { "abc", "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad" },
        { "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
          "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1" },
    };
    int bad = 0;
    for (auto & v : vec) {
        const std::string got = m1_chat_sha256_hex(v.in, std::strlen(v.in));
        if (got != v.hex) { std::printf("FAIL sha256(\"%s\") = %s, want %s\n", v.in, got.c_str(), v.hex); bad = 1; }
    }
    // a multi-block input: 1,000,000 'a'
    std::string big(1000000, 'a');
    const std::string bighex = m1_chat_sha256_hex(big.data(), big.size());
    if (bighex != "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0") {
        std::printf("FAIL sha256(1e6 x 'a') = %s\n", bighex.c_str());
        bad = 1;
    }
    // the date ladder's non-refusing and refusing branches, spot-checked
    struct { M1ChatDate in, want; bool ok; } dv[] = {
        { { 2026, 10, 19 }, { 2026, 10, 18 }, true },
        { { 2026, 10, 1 },  { 2026, 9, 30 },  true },
        { { 2026, 1, 1 },   { 2025, 12, 31 }, true },
        { { 2028, 3, 1 },   { 2028, 2, 29 },  true },
        { { 2026, 3, 1 },   { 0, 0, 0 },      false },
        { { 2033, 1, 1 },   { 0, 0, 0 },      false },
    };
    for (auto & v : dv) {
        M1ChatDate got;
        const bool ok = m1_chat_prev_day(v.in, &got);
        if (ok != v.ok || (ok && (got.year != v.want.year || got.month != v.want.month || got.day != v.want.day))) {
            std::printf("FAIL prev_day(%s) -> %s %s\n", m1_chat_format_ymd(v.in).c_str(), ok ? "ok" : "refuse",
                        ok ? m1_chat_format_ymd(got).c_str() : "");
            bad = 1;
        }
    }
    std::printf("%s\n", bad ? "SELFTEST FAIL" : "SELFTEST OK");
    return bad;
}

int cmd_render(const char * cases_path, const char * outdir, const char * now_ymd, const char * dsm_mode) {
    for (const Case & c : read_cases(cases_path)) {
        M1ChatOpts o;
        o.now_ymd = now_ymd;
        std::string explicit_dsm = dsm_mode ? dsm_mode : "";
        if (dsm_mode != nullptr) o.default_system_message = explicit_dsm.c_str();  // "" == omit the block
        const std::string s = render_chat(c.msgs, o);
        const std::string p = std::string(outdir) + "/" + c.name + ".ours.bin";
        write_file(p, s);
        std::printf("%-20s %5zu bytes  head=%.40s\n", c.name.c_str(), s.size(), s.c_str());
    }
    return 0;
}

int cmd_refuse(const char * cases_path) {
    // every case must be REFUSED (the template raises). A case that renders here is a FAIL.
    // The refusal is a FATAL (exit), so each case runs in a forked child.
    int bad = 0;
    for (const Case & c : read_cases(cases_path)) {
        std::printf("REFUSE-CHECK %s\n", c.name.c_str());
        std::fflush(stdout);
        std::fflush(stderr);
        const pid_t pid = ::fork();
        if (pid < 0) m1_chat_fatal("fork failed");
        if (pid == 0) {
            std::freopen("/dev/null", "w", stderr);
            (void)render_chat(c.msgs, M1ChatOpts{});
            std::_Exit(0);  // rendered -> the parent reports the failure below
        }
        int status = 0;
        ::waitpid(pid, &status, 0);
        const bool refused = !(WIFEXITED(status) && WEXITSTATUS(status) == 0);
        std::printf("%-20s %s\n", c.name.c_str(), refused ? "refused (ok)" : "RENDERED (bad)");
        if (!refused) bad = 1;
    }
    return bad;
}

int cmd_date(const char * now_ymd) {
    M1ChatDate in;
    if (!m1_chat_parse_ymd(now_ymd, &in)) m1_chat_fatal("bad date %s", now_ymd);
    M1ChatDate prev;
    if (!m1_chat_prev_day(in, &prev)) {
        std::printf("%s REFUSE\n", now_ymd);
        return 2;
    }
    std::printf("%s %s %s\n", now_ymd, m1_chat_format_ymd(prev).c_str(),
                m1_chat_default_system_message(in).find(m1_chat_format_ymd(prev)) != std::string::npos ? "in-dsm" : "MISSING");
    return 0;
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s verify|selftest|render|refuse|date ...\n", argv[0]);
        return 2;
    }
    const std::string cmd = argv[1];
    if (cmd == "selftest") return cmd_selftest();
    if (cmd == "verify") {
        if (argc != 3) m1_chat_fatal("verify <template-file>");
        return cmd_verify(argv[2]);
    }
    if (cmd == "render") {
        if (argc < 4) m1_chat_fatal("render <cases-file> <outdir> [--now YYYY-MM-DD] [--dsm TEXT]");
        const char * now = nullptr;
        const char * dsm = nullptr;
        for (int i = 4; i < argc; i++) {
            const std::string a = argv[i];
            if (a == "--now" && i + 1 < argc) now = argv[++i];
            else if (a == "--dsm" && i + 1 < argc) dsm = argv[++i];
            else m1_chat_fatal("render: unknown arg %s", a.c_str());
        }
        return cmd_render(argv[2], argv[3], now, dsm);
    }
    if (cmd == "refuse") {
        if (argc != 3) m1_chat_fatal("refuse <cases-file>");
        return cmd_refuse(argv[2]);
    }
    if (cmd == "date") {
        if (argc != 3) m1_chat_fatal("date <YYYY-MM-DD>");
        return cmd_date(argv[2]);
    }
    std::fprintf(stderr, "unknown command %s\n", cmd.c_str());
    return 2;
}

#endif  // M1_CHAT_DRIVER
