// engine/m1_json.cpp — strict JSON reader + streaming escaper (see m1_json.h).

#include "m1_json.h"

#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace m1json {

// ---------------------------------------------------------------------------
// UTF-8 encode one code point (used only to materialise a \u escape)
// ---------------------------------------------------------------------------
static void append_utf8(std::string & o, unsigned cp) {
    if (cp < 0x80u) {
        o.push_back((char) cp);
    } else if (cp < 0x800u) {
        o.push_back((char) (0xC0u | (cp >> 6)));
        o.push_back((char) (0x80u | (cp & 0x3Fu)));
    } else if (cp < 0x10000u) {
        o.push_back((char) (0xE0u | (cp >> 12)));
        o.push_back((char) (0x80u | ((cp >> 6) & 0x3Fu)));
        o.push_back((char) (0x80u | (cp & 0x3Fu)));
    } else {
        o.push_back((char) (0xF0u | (cp >> 18)));
        o.push_back((char) (0x80u | ((cp >> 12) & 0x3Fu)));
        o.push_back((char) (0x80u | ((cp >> 6) & 0x3Fu)));
        o.push_back((char) (0x80u | (cp & 0x3Fu)));
    }
}

static int hexval(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

// ---------------------------------------------------------------------------
// the recursive-descent reader
// ---------------------------------------------------------------------------
namespace {

struct Parser {
    const char * b = nullptr;
    size_t       n = 0;
    size_t       i = 0;
    Error *      err = nullptr;

    bool fail(const char * what) {
        err->what   = what;
        err->offset = i;
        return false;
    }
    bool fail_at(const char * what, size_t at) {
        err->what   = what;
        err->offset = at;
        return false;
    }

    void ws() {
        while (i < n) {
            const char c = b[i];
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r') ++i;
            else break;
        }
    }

    bool parse_value(Value & v, int depth);
    bool parse_string(std::string & s);
    bool parse_number(Value & v);
    bool parse_literal(const char * lit, size_t len, Value & v, Type t, bool bv);

    // \uXXXX (the caller has consumed the backslash and the 'u')
    bool parse_unicode_escape(std::string & s);
};

bool Parser::parse_unicode_escape(std::string & s) {
    if (i + 4 > n) return fail("truncated \\u escape: need 4 hex digits");
    unsigned cp = 0;
    for (int k = 0; k < 4; ++k) {
        const int h = hexval(b[i + (size_t) k]);
        if (h < 0) return fail("invalid hex digit in \\u escape");
        cp = (cp << 4) | (unsigned) h;
    }
    i += 4;
    if (cp >= 0xD800u && cp <= 0xDBFFu) {  // high surrogate: a low one MUST follow
        if (i + 2 > n || b[i] != '\\' || b[i + 1] != 'u')
            return fail("lone high surrogate in \\u escape (no \\u low surrogate follows)");
        i += 2;
        if (i + 4 > n) return fail("truncated \\u escape after a high surrogate");
        unsigned lo = 0;
        for (int k = 0; k < 4; ++k) {
            const int h = hexval(b[i + (size_t) k]);
            if (h < 0) return fail("invalid hex digit in the low \\u escape");
            lo = (lo << 4) | (unsigned) h;
        }
        i += 4;
        if (lo < 0xDC00u || lo > 0xDFFFu) return fail("high surrogate not followed by a low surrogate");
        const unsigned full = 0x10000u + ((cp - 0xD800u) << 10) + (lo - 0xDC00u);
        append_utf8(s, full);
        return true;
    }
    if (cp >= 0xDC00u && cp <= 0xDFFFu) return fail("lone low surrogate in \\u escape");
    append_utf8(s, cp);
    return true;
}

bool Parser::parse_string(std::string & s) {
    // the caller has verified b[i] == '"'
    const size_t start = i;
    ++i;  // opening quote
    s.clear();
    for (;;) {
        if (i >= n) return fail_at("unterminated string", start);
        const unsigned char c = (unsigned char) b[i];
        if (c == '"') { ++i; return true; }
        // An escape is checked FIRST: '\\' is < 0x80, so a plain-ASCII branch above this one
        // would swallow every escape as a literal byte (that bug shipped once and U3-arm-45
        // exists to catch it).
        if (c == '\\') {
            ++i;
            if (i >= n) return fail_at("unterminated escape in string", start);
            const char e = b[i++];
            switch (e) {
                case '"':  s.push_back('"');  break;
                case '\\': s.push_back('\\'); break;
                case '/':  s.push_back('/');  break;
                case 'b':  s.push_back('\b'); break;
                case 'f':  s.push_back('\f'); break;
                case 'n':  s.push_back('\n'); break;
                case 'r':  s.push_back('\r'); break;
                case 't':  s.push_back('\t'); break;
                case 'u':  if (!parse_unicode_escape(s)) return false; break;
                default:   --i; return fail("invalid escape character in string");
            }
            continue;
        }
        if (c < 0x20u) return fail("unescaped control byte in string (must be \\u00XX escaped)");
        if (c < 0x80u) { s.push_back((char) c); ++i; continue; }
        // A JSON string must be valid UTF-8. Copying raw bytes without this check would let an
        // invalid sequence through the parser and out again in an error message, where it would
        // make the RESPONSE invalid JSON — so the rejection belongs here.
        size_t need = 0;
        if ((c & 0xE0u) == 0xC0u) need = 2;
        else if ((c & 0xF0u) == 0xE0u) need = 3;
        else if ((c & 0xF8u) == 0xF0u) need = 4;
        if (need == 0) return fail("invalid UTF-8 in string (stray continuation or bad lead byte)");
        if (i + need > n) return fail("truncated UTF-8 sequence in string");
        for (size_t k = 1; k < need; ++k) {
            if (((unsigned char) b[i + k] & 0xC0u) != 0x80u)
                return fail("invalid UTF-8 continuation byte in string");
        }
        s.append(b + i, need);
        i += need;
    }
}

bool Parser::parse_number(Value & v) {
    const size_t start = i;
    if (i < n && b[i] == '-') ++i;
    if (i >= n) return fail_at("truncated number", start);
    if (b[i] == '0') {
        ++i;
        if (i < n && b[i] >= '0' && b[i] <= '9') return fail("leading zero in number");
    } else if (b[i] >= '1' && b[i] <= '9') {
        while (i < n && b[i] >= '0' && b[i] <= '9') ++i;
    } else {
        return fail_at("expected a digit in number", start);
    }
    bool is_int = true;
    if (i < n && b[i] == '.') {
        is_int = false;
        ++i;
        if (i >= n || b[i] < '0' || b[i] > '9') return fail("expected a digit after '.'");
        while (i < n && b[i] >= '0' && b[i] <= '9') ++i;
    }
    if (i < n && (b[i] == 'e' || b[i] == 'E')) {
        is_int = false;
        ++i;
        if (i < n && (b[i] == '+' || b[i] == '-')) ++i;
        if (i >= n || b[i] < '0' || b[i] > '9') return fail("expected a digit in the exponent");
        while (i < n && b[i] >= '0' && b[i] <= '9') ++i;
    }
    // copy exactly the accepted span (strtod would accept a longer, non-JSON one)
    char tmp[64];
    const size_t len = i - start;
    if (len >= sizeof(tmp)) {
        // a legal JSON number this long is not one a request can usefully carry; still parse it
        std::string big(b + start, len);
        errno = 0;
        v.number = std::strtod(big.c_str(), nullptr);
        v.number_is_int = false;
        v.type = JT_NUMBER;
        return true;
    }
    std::memcpy(tmp, b + start, len);
    tmp[len] = '\0';
    v.type = JT_NUMBER;
    v.number = std::strtod(tmp, nullptr);
    v.number_is_int = false;
    if (is_int) {
        errno = 0;
        char * endp = nullptr;
        const long long iv = std::strtoll(tmp, &endp, 10);
        if (errno == 0 && endp && *endp == '\0') {
            v.integer = iv;
            v.number_is_int = true;
        }
    }
    if (!std::isfinite(v.number)) return fail_at("number is out of range", start);
    return true;
}

bool Parser::parse_literal(const char * lit, size_t len, Value & v, Type t, bool bv) {
    if (i + len > n || std::memcmp(b + i, lit, len) != 0) return fail("invalid literal");
    i += len;
    v.type = t;
    v.boolean = bv;
    return true;
}

bool Parser::parse_value(Value & v, int depth) {
    if (depth > kMaxDepth) return fail_at("maximum nesting depth (64) exceeded", i);
    ws();
    if (i >= n) return fail("unexpected end of input (expected a value)");
    const char c = b[i];
    switch (c) {
        case '{': {
            v.type = JT_OBJECT;
            ++i;
            ws();
            if (i < n && b[i] == '}') { ++i; return true; }
            for (;;) {
                ws();
                if (i >= n) return fail("unterminated object");
                if (b[i] != '"') return fail("expected a string key in object");
                std::string key;
                if (!parse_string(key)) return false;
                ws();
                if (i >= n || b[i] != ':') return fail("expected ':' after object key");
                ++i;
                Value child;
                if (!parse_value(child, depth + 1)) return false;
                v.obj.emplace_back(std::move(key), std::move(child));
                ws();
                if (i >= n) return fail("unterminated object");
                if (b[i] == ',') { ++i; continue; }
                if (b[i] == '}') { ++i; return true; }
                return fail("expected ',' or '}' in object");
            }
        }
        case '[': {
            v.type = JT_ARRAY;
            ++i;
            ws();
            if (i < n && b[i] == ']') { ++i; return true; }
            for (;;) {
                Value child;
                if (!parse_value(child, depth + 1)) return false;
                v.arr.push_back(std::move(child));
                ws();
                if (i >= n) return fail("unterminated array");
                if (b[i] == ',') { ++i; continue; }
                if (b[i] == ']') { ++i; return true; }
                return fail("expected ',' or ']' in array");
            }
        }
        case '"':
            v.type = JT_STRING;
            return parse_string(v.str);
        case 't': return parse_literal("true", 4, v, JT_BOOL, true);
        case 'f': return parse_literal("false", 5, v, JT_BOOL, false);
        case 'n': return parse_literal("null", 4, v, JT_NULL, false);
        default:
            if (c == '-' || (c >= '0' && c <= '9')) return parse_number(v);
            return fail("unexpected character (expected a value)");
    }
}

}  // namespace

const Value * Value::find(const char * key) const {
    if (type != JT_OBJECT) return nullptr;
    const Value * hit = nullptr;
    for (const auto & kv : obj) {
        if (kv.first == key) hit = &kv.second;
    }
    return hit;
}

const Value * Value::find(const std::string & key) const { return find(key.c_str()); }

bool parse(const char * buf, size_t len, Value & out, Error & err) {
    err.what   = nullptr;
    err.offset = 0;
    out = Value();
    Parser p;
    p.b   = buf;
    p.n   = len;
    p.i   = 0;
    p.err = &err;
    if (!p.parse_value(out, 0)) return false;
    p.ws();
    if (p.i != p.n) {
        // trailing bytes after the top-level value: a second JSON document, a stray byte, or
        // a body a client forgot to trim. Reject with the offset so the caller can be told.
        err.what   = "trailing bytes after the JSON value";
        err.offset = p.i;
        return false;
    }
    return true;
}

bool parse(const std::string & buf, Value & out, Error & err) {
    return parse(buf.data(), buf.size(), out, err);
}

// ---------------------------------------------------------------------------
// escaper
// ---------------------------------------------------------------------------
void append_string(std::string & out, const char * s, size_t n) {
    static const char * HEX = "0123456789abcdef";
    out.push_back('"');
    size_t run = 0;  // start of the current unescaped run
    for (size_t k = 0; k < n; ++k) {
        const unsigned char c = (unsigned char) s[k];
        const char * repl = nullptr;
        char buf[7];
        switch (c) {
            case '"':  repl = "\\\""; break;
            case '\\': repl = "\\\\"; break;
            case '\n': repl = "\\n";  break;
            case '\r': repl = "\\r";  break;
            case '\t': repl = "\\t";  break;
            case '\b': repl = "\\b";  break;
            case '\f': repl = "\\f";  break;
            default:
                if (c < 0x20u) {
                    buf[0] = '\\';
                    buf[1] = 'u';
                    buf[2] = '0';
                    buf[3] = '0';
                    buf[4] = HEX[(c >> 4) & 0xF];
                    buf[5] = HEX[c & 0xF];
                    buf[6] = '\0';
                    repl = buf;
                }
                break;
        }
        if (!repl) continue;
        if (k > run) out.append(s + run, k - run);
        out.append(repl);
        run = k + 1;
    }
    if (n > run) out.append(s + run, n - run);
    out.push_back('"');
}

void append_string(std::string & out, const std::string & s) { append_string(out, s.data(), s.size()); }

void append_int(std::string & out, long long v) {
    char buf[32];
    const int n = std::snprintf(buf, sizeof(buf), "%lld", v);
    if (n > 0) out.append(buf, (size_t) n);
}

void append_double(std::string & out, double v) {
    if (!std::isfinite(v)) { out.append("null"); return; }
    char buf[40];
    // 17 significant digits round-trips a double; strip a trailing '.0' so "2" stays "2"
    int n = std::snprintf(buf, sizeof(buf), "%.17g", v);
    if (n <= 0) { out.append("0"); return; }
    std::string s(buf, (size_t) n);
    if (s.find('.') == std::string::npos && s.find('e') == std::string::npos &&
        s.find("inf") == std::string::npos && s.find("nan") == std::string::npos) {
        s.append(".0");
    }
    out.append(s);
}

bool get_int(const Value & v, long long & out) {
    if (v.type != JT_NUMBER) return false;
    if (v.number_is_int) { out = v.integer; return true; }
    if (v.number != std::floor(v.number)) return false;
    if (v.number < -9.007199254740992e15 || v.number > 9.007199254740992e15) return false;
    out = (long long) v.number;
    return true;
}

bool get_number(const Value & v, double & out) {
    if (v.type != JT_NUMBER) return false;
    out = v.number;
    return true;
}

}  // namespace m1json
