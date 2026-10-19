// engine/m1_chat.h — the Mistral Small 4 chat render, HARDCODED and HASH-PINNED (server U2).
//
// ONE model (models/Mistral-Small-4-119B-2603-UD-IQ2_XXS.gguf), ONE template. There is no
// Jinja engine here, no template generality and no "any model" surface: the GGUF's
// tokenizer.chat_template is pinned by sha256, and a boot check FATALs if the bytes in the
// file are not the bytes this renderer was written against. That is the whole safety model —
// a template change must be a code change, reviewed, not a silent behaviour change.
//
// WHAT THE PINNED TEMPLATE RENDERS (read out of the GGUF, template lines 144-274; MEASURED by
// running the real template through llama.cpp's own Jinja — tools/m1_chat_ref.cpp):
//
//   '<s>'
//   '[SYSTEM_PROMPT]' + (messages[0].content if messages[0].role=='system'
//                        else the default_system_message literal) + '[/SYSTEM_PROMPT]'
//   '[MODEL_SETTINGS]{"reasoning_effort": "none"}[/MODEL_SETTINGS]'      <- ALWAYS
//   per message, in order: user      -> '[INST]' + content + '[/INST]'
//                          assistant -> content + '</s>'
//                          tool      -> '[TOOL_RESULTS]' + content + '[/TOOL_RESULTS]'
//   and nothing after the last message: the template has NO add_generation_prompt branch
//   (grep 'add_generation_prompt' the template -> 0 hits), so no generation prompt is appended.
//
// TWO THINGS THE PLAN'S §C RENDER DOES NOT MENTION, both MEASURED from the template itself and
// confirmed against the oracle (reported to the orchestrator, not silently added):
//   * the [MODEL_SETTINGS] block above is emitted unconditionally by the template (line 189),
//     with reasoning_effort defaulting to 'none' — it is part of the real prompt;
//   * today_date is the ACTUAL date (set at template line 5, BEFORE the rollback ladder), while
//     yesterday_date is that date rolled back ONE DAY (ladder, lines 6-142). They are two
//     different dates; the plan's "BOTH of which are the same value" is wrong.
//
// BOS (the trap): the template ALREADY emits '<s>', and the tokenizer for this model ALSO
// prepends one (llama_vocab_get_add_bos() == true, pre type pixtral). The two policies:
//   M1_CHAT_BOS_TEMPLATE  (default): the '<s>' in the rendered text is THE bos and the tokenizer
//                         must be called with add_special == false -> exactly one id 1, and
//                         ids[0]==1 && ids[1]!=1 (assert this at integration).
//   M1_CHAT_BOS_TOKENIZER: add_special == true on the same text -> the DOUBLE-BOS prompt
//                         llama-server produces ([1,1,3,...]) — kept so a client that was
//                         benchmarked against llama-server can be reproduced exactly.
// MEASURED with builds/hip-fork/bin/llama-tokenize on '<s>[INST] Hi [/INST]': [1,1,3,24665,
// 1032,4] with BOS, [1,3,24665,1032,4] with --no-bos.

#pragma once

#include <cstddef>
#include <string>
#include <vector>

// A chat message exactly as the HTTP layer hands it over. roles: system (messages[0] only),
// user, assistant, tool — anything else is refused the way the template refuses it.
struct M1ChatMessage {
    std::string role;
    std::string content;
};

struct M1ChatDate {
    int year  = 1970;
    int month = 1;
    int day   = 1;
};

// MEASURED (U2, from the GGUF's raw tokenizer.chat_template KV bytes, by an independent python
// GGUF reader AND again through llama_model_chat_template() in tools/m1_chat_ref.cpp):
// 14121 bytes, sha256 below. A mismatch is a boot FATAL — see m1_chat_verify_template().
inline constexpr const char * M1_CHAT_TEMPLATE_SHA256 =
    "d579fedfe0a78b11376089da957bbd4299ffb3f95cf827ab00c9cdc77298f954";
inline constexpr std::size_t M1_CHAT_TEMPLATE_LEN = 14121;

// MEASURED: the rendered default_system_message length (256 + 844 + 1254 literal bytes from
// m1_chat_dsm.inc + two 10-byte dates). The boot check asserts it.
inline constexpr std::size_t M1_CHAT_DEFAULT_SYSTEM_MESSAGE_LEN = 2374;

// The [MODEL_SETTINGS] block is a literal of the template (line 189) with reasoning_effort
// fixed at 'none'. Hardcoded, like everything else here.
inline constexpr const char * M1_CHAT_MODEL_SETTINGS =
    "[MODEL_SETTINGS]{\"reasoning_effort\": \"none\"}[/MODEL_SETTINGS]";

enum M1ChatBos {
    M1_CHAT_BOS_TEMPLATE  = 0,  // add_special = false: the template's '<s>' is the one BOS
    M1_CHAT_BOS_TOKENIZER = 1,  // add_special = true : llama-server's double BOS
};

// The tokenizer's add_special flag for a policy (U5/the server passes this straight through).
inline bool m1_chat_tokenizer_add_special(M1ChatBos bos) {
    return bos == M1_CHAT_BOS_TOKENIZER;
}
const char * m1_chat_bos_name(M1ChatBos bos);

// How the caller wants the render built. Defaults are the product path.
struct M1ChatOpts {
    // nullptr  -> use the pinned default_system_message literal (the product path)
    // ""       -> omit the system block entirely (the template's `if default_system_message != ''`)
    // anything -> use it verbatim
    const char * default_system_message = nullptr;
    // "YYYY-MM-DD" overrides "now" for tests only (the boundary-day arms); nullptr -> wall clock.
    const char * now_ymd = nullptr;
};

// ---------------------------------------------------------------------------
// SHA-256 (real, ~60 lines — not a cheaper analogue under the name) + the pin check
// ---------------------------------------------------------------------------
std::string m1_chat_sha256_hex(const void * data, std::size_t n);

// Boot check #1: FATAL unless sha256(bytes) == M1_CHAT_TEMPLATE_SHA256 (and len is the pinned
// length). Called with the raw tokenizer.chat_template KV bytes.
void m1_chat_verify_template(const void * tmpl_bytes, std::size_t len);

// Boot check #2 (independent of the hash): FATAL unless the default_system_message pieces
// extracted from those same template bytes equal the compiled-in literal (m1_chat_dsm.inc),
// and the rendered length is M1_CHAT_DEFAULT_SYSTEM_MESSAGE_LEN.
void m1_chat_verify_default_system_message(const void * tmpl_bytes, std::size_t len);

// ---------------------------------------------------------------------------
// Dates — the template's own rollback ladder
// ---------------------------------------------------------------------------
M1ChatDate m1_chat_today();
std::string m1_chat_format_ymd(const M1ChatDate & d);
bool        m1_chat_parse_ymd(const char * ymd, M1ChatDate * out);

// The template's yesterday ladder, EXACTLY (lines 6-142), including where it gives up:
//   day != 01            -> day-1
//   day 01, month != 01  -> the previous month's last day as the template HARDCODES it
//                           (Feb -> 31 Jan; Mar -> 28 Feb, or 29 Feb for years 2024/2028/2032/1970)
//   day 01, month 01     -> 31 Dec of year-1, for years 2024..2032 (and 1970 -> 1970)
// Returns false where the template calls raise_exception() — i.e. 1 March of a year outside
// {2024, 2028, 2032, 1970} and 1 January of a year outside {1970, 2024..2032}. That is a real
// bug in the pinned template, reproduced deliberately: see m1_chat_prev_day()'s FATAL, and
// gate arm "refuse" in scripts/engine_gate_chat.sh, which proves the oracle refuses on the
// same days. NEVER widen this silently.
bool m1_chat_prev_day(M1ChatDate in, M1ChatDate * out);

// The default_system_message rendered for `now` (today_date = now, yesterday_date = prev_day(now)).
// FATALs (like the template's raise_exception) when prev_day() refuses.
std::string m1_chat_default_system_message(const M1ChatDate & now);

// CAN THIS DATE BE RENDERED AT ALL? The pre-check every caller of render_chat() on a
// CLIENT-REACHABLE path must run first (U2's wiring note 2): render_chat() is a FATAL where the
// pinned template's own ladder gives up — 1 March of a year outside {2024,2028,2032,1970} and
// 1 January outside {1970,2024..2032} — which the server has to answer as a 500, not as process
// death. Returns false with `why` describing the refusal (a server-side condition, not a client's).
bool m1_chat_can_render(const M1ChatOpts & opts, std::string * why);

// ---------------------------------------------------------------------------
// THE RENDER
// ---------------------------------------------------------------------------
// render_chat(messages, default_system_message) — the product entry point. Byte-for-byte what
// the pinned template renders (see the header comment). FATALs — refusing rather than
// rendering something the template would refuse — on:
//   * a role outside {system (first message only), user, assistant, tool};
//   * user/assistant turns that do not alternate after the optional system message
//     (template: 'conversation roles must alternate user and assistant roles ...');
//   * a "now" date the template's ladder cannot roll back (see m1_chat_prev_day).
// An assistant message with EMPTY content is rendered as just '</s>' — the raw template raises
// there, but llama.cpp patches exactly that condition out (common/chat.cpp: the
// "[TOOL_CALLS]"/"if (message['content'] is none or" fixup rewrites it to `{%- if false %}`),
// and the ORACLE is llama.cpp. MEASURED against it in gate arm "empty_assistant".
std::string render_chat(const std::vector<M1ChatMessage> & messages, const M1ChatOpts & opts = M1ChatOpts{});
