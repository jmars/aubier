// engine/m1_json.h — strict JSON (RFC 8259) reader + a streaming string escaper.
//
// Server plan §F, unit U3. Hand-rolled, because MEASURED: no third-party JSON library is
// installed on this box and none may be added. This is the ONLY JSON code in the server: the
// request bodies are parsed here and every response/SSE chunk is built here.
//
// The reader is RECURSIVE DESCENT and STRICT. It rejects
//   * a leading '+', a leading zero, '.5', '5.', 'NaN', 'Infinity' — only the RFC number
//     grammar is accepted;
//   * unescaped control bytes (< 0x20) inside strings;
//   * a lone surrogate in a \u escape (a high surrogate must be followed by a low one);
//   * trailing bytes after the top-level value;
//   * nesting deeper than kMaxDepth (64) — the depth cap is what keeps a hostile body from
//     recursing the parser to a stack overflow;
// and every error carries the BYTE OFFSET into the buffer, because the prefix fuzz in
// scripts/engine_gate_http.sh has to be able to say WHERE a truncation was rejected.
//
// Unknown keys are NOT an error here (that is what lets ordinary OpenAI clients talk to us).
// Refusing an unknown/unsupported SAMPLING parameter BY NAME is the request validator's job
// (engine/m1_server.cpp), not the parser's.

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace m1json {

// Nesting cap shared by the parser and documented in the error text. 64 is far beyond any
// OpenAI request shape and far below any stack worry.
static const int kMaxDepth = 64;

enum Type : uint8_t {
    JT_NULL = 0,
    JT_BOOL,
    JT_NUMBER,
    JT_STRING,
    JT_ARRAY,
    JT_OBJECT,
};

struct Error {
    const char * what   = nullptr;  // nullptr => no error; otherwise a static string
    size_t       offset = 0;        // BYTE offset into the parsed buffer
    bool ok() const { return what == nullptr; }
};

struct Value {
    Type        type          = JT_NULL;
    bool        boolean       = false;
    double      number        = 0.0;
    bool        number_is_int = false;  // the literal had no '.'/'e' and fits in long long
    long long   integer       = 0;
    std::string str;                    // JT_STRING (UTF-8 bytes, escapes decoded)
    std::vector<Value> arr;
    std::vector<std::pair<std::string, Value>> obj;  // insertion order preserved

    bool is_null()   const { return type == JT_NULL; }
    bool is_bool()   const { return type == JT_BOOL; }
    bool is_number() const { return type == JT_NUMBER; }
    bool is_string() const { return type == JT_STRING; }
    bool is_array()  const { return type == JT_ARRAY; }
    bool is_object() const { return type == JT_OBJECT; }

    // A duplicate key resolves to its LAST occurrence (the JSON.parse convention).
    const Value * find(const char * key) const;
    const Value * find(const std::string & key) const;
};

// Parses exactly one value out of [buf, buf+len). On failure returns false and fills err.
bool parse(const char * buf, size_t len, Value & out, Error & err);
bool parse(const std::string & buf, Value & out, Error & err);

// ---------------------------------------------------------------------------
// Output side. append_string() is the streaming escaper: it appends the QUOTES and the escaped
// body in one pass, with no intermediate allocation, and it neither validates nor re-encodes
// UTF-8 (raw multi-byte bytes are already valid inside a JSON string). Control bytes are
// escaped, so the output is always a legal JSON string.
// ---------------------------------------------------------------------------
void append_string(std::string & out, const char * s, size_t n);
void append_string(std::string & out, const std::string & s);
void append_int(std::string & out, long long v);
void append_double(std::string & out, double v);  // non-finite => null (never emitted as 0)

// Strict accessors used by the request validator: they answer "is this value the type I need",
// never "coerce it anyway". get_int accepts only an integral number in long long range.
bool get_int(const Value & v, long long & out);
bool get_number(const Value & v, double & out);

}  // namespace m1json
