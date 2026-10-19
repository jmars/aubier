// tools/m1_chat_ref.cpp — THE ORACLE for U2: the Mistral Small 4 chat render, produced by
// llama.cpp's OWN Jinja runtime from the GGUF's OWN template bytes. TEST-ONLY.
//
// It loads VOCAB ONLY (llama_model_params.vocab_only = true, vendor/llama.cpp/include/llama.h:351),
// so it reads no weights, touches no GPU, takes no gpu.lock and costs ~0.5 s — it is runnable
// while a measurement lane holds the window, and on a loadavg-20 host.
//
// It exists so that scripts/engine_gate_chat.sh can BYTE-DIFF engine/m1_chat.cpp's render against
// an implementation we did not write. Everything here is deliberately independent of
// engine/m1_chat.*: this file does not include m1_chat.h and shares no argument parsing, no case
// parsing and no rendering code with the unit under test.
//
// TWO RENDERS PER CASE, because this model has TWO BOS policies (see engine/m1_chat.h):
//
//   <name>.ref-nobos.bin   model=nullptr + the template bytes as an OVERRIDE, so
//                          common_chat_templates_init leaves add_bos false: the rendered '<s>'
//                          SURVIVES. This is the string our renderer must equal byte for byte
//                          (M1_CHAT_BOS_TEMPLATE: the template's '<s>' is the one BOS).
//   <name>.ref.bin         model-driven init (add_bos = llama_vocab_get_add_bos() = TRUE for this
//                          vocab), which makes common_chat_templates_apply STRIP a leading BOS from
//                          the rendered text (vendor/llama.cpp/common/chat.cpp:931-933) and leave
//                          the tokenizer to add it back. This is what llama-server actually feeds
//                          the tokenizer, so the gate asserts ref.bin == our render MINUS '<s>'.
//
// Both come out of the same template bytes, so a difference between them is only the BOS.
//
// Usage:
//   m1_chat_ref source <model>                     print len + sha256 of llama_model_chat_template()
//   m1_chat_ref template <model> <out-file>        dump those raw bytes (the gate hashes them)
//   m1_chat_ref render <model> <cases> <outdir> [--now YYYY-MM-DD] [--refuse]
//   m1_chat_ref render ... --refuse                every case must RAISE (exit 0 only if all did)
//
// --now replaces the template's three strftime_now() calls with quoted constants for that date
// (the jinja runtime reads the clock itself: jinja/runtime.h:73 current_time = std::time(nullptr),
// and common_chat_templates_inputs::now does NOT reach it on this path — MEASURED). The ladder
// bytes are untouched, which the gate proves by render(--now today) == render(no --now).

#include "chat.h"
#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// the case file (the SAME grammar engine/m1_chat.cpp's driver reads, parsed independently here)
//   # comment
//   case <name>
//   <role> TAB <content>          escapes: backslash-n, backslash-t, backslash-s (space), double-backslash
//   (a blank line, a comment or the next "case " line ends the case)
// ---------------------------------------------------------------------------
struct RefCase {
    std::string                name;
    std::vector<common_chat_msg> msgs;
};

static std::string slurp(const char * path) {
    FILE * f = std::fopen(path, "rb");
    if (!f) { std::fprintf(stderr, "m1_chat_ref: cannot open %s\n", path); std::exit(70); }
    std::string s;
    char        buf[65536];
    size_t      n;
    while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) s.append(buf, n);
    std::fclose(f);
    return s;
}

static void spit(const std::string & path, const std::string & data) {
    FILE * f = std::fopen(path.c_str(), "wb");
    if (!f) { std::fprintf(stderr, "m1_chat_ref: cannot write %s\n", path.c_str()); std::exit(70); }
    if (!data.empty()) std::fwrite(data.data(), 1, data.size(), f);
    std::fclose(f);
}

static std::string unescape(const std::string & raw) {
    std::string o;
    for (size_t i = 0; i < raw.size(); i++) {
        if (raw[i] != '\\') { o += raw[i]; continue; }
        i++;
        if (i >= raw.size()) { std::fprintf(stderr, "m1_chat_ref: trailing backslash in a case\n"); std::exit(70); }
        switch (raw[i]) {
            case 'n': o += '\n'; break;
            case 't': o += '\t'; break;
            case 's': o += ' ';  break;
            case '\\': o += '\\'; break;
            default:
                std::fprintf(stderr, "m1_chat_ref: unsupported escape \\%c in a case\n", raw[i]);
                std::exit(70);
        }
    }
    return o;
}

static std::vector<RefCase> parse_cases(const char * path) {
    std::vector<std::string> lines;
    {
        const std::string t = slurp(path);
        size_t            i = 0;
        while (i < t.size()) {
            size_t e = t.find('\n', i);
            if (e == std::string::npos) e = t.size();
            std::string l = t.substr(i, e - i);
            i = e + 1;
            if (!l.empty() && l.back() == '\r') l.pop_back();
            lines.push_back(l);
        }
    }
    std::vector<RefCase> out;
    for (const std::string & l : lines) {
        if (l.empty() || l[0] == '#') continue;
        if (l.rfind("case ", 0) == 0) {
            out.push_back(RefCase{ l.substr(5), {} });
            if (out.back().name.empty()) { std::fprintf(stderr, "m1_chat_ref: unnamed case\n"); std::exit(70); }
            continue;
        }
        if (out.empty()) { std::fprintf(stderr, "m1_chat_ref: message before any 'case'\n"); std::exit(70); }
        const size_t tab = l.find('\t');
        if (tab == std::string::npos) { std::fprintf(stderr, "m1_chat_ref: no TAB in '%s'\n", l.c_str()); std::exit(70); }
        common_chat_msg m;
        m.role    = l.substr(0, tab);
        m.content = unescape(l.substr(tab + 1));
        out.back().msgs.push_back(m);
    }
    if (out.empty()) { std::fprintf(stderr, "m1_chat_ref: no cases in %s\n", path); std::exit(70); }
    return out;
}

// ---------------------------------------------------------------------------
// the date injection: strftime_now("%d"|"%m"|"%Y") -> a quoted constant. The ladder code itself
// is untouched, so the arms exercise the template's own branches.
// ---------------------------------------------------------------------------
static std::string inject_date(const std::string & tmpl, const char * ymd) {
    int y = 0, mo = 0, d = 0;
    if (std::sscanf(ymd, "%d-%d-%d", &y, &mo, &d) != 3 || y < 1970 || mo < 1 || mo > 12 || d < 1 || d > 31) {
        std::fprintf(stderr, "m1_chat_ref: bad --now '%s'\n", ymd);
        std::exit(70);
    }
    char py[16], pm[8], pd[8];
    std::snprintf(py, sizeof py, "'%04d'", y);
    std::snprintf(pm, sizeof pm, "'%02d'", mo);
    std::snprintf(pd, sizeof pd, "'%02d'", d);
    struct { const char * from; std::string to; } subs[] = {
        { "strftime_now(\"%Y\")", py },
        { "strftime_now(\"%m\")", pm },
        { "strftime_now(\"%d\")", pd },
    };
    std::string s = tmpl;
    for (auto & sub : subs) {
        const size_t at = s.find(sub.from);
        if (at == std::string::npos || s.find(sub.from, at + 1) != std::string::npos) {
            std::fprintf(stderr, "m1_chat_ref: '%s' does not appear exactly once in the template — "
                                 "the date injection cannot be faithful\n", sub.from);
            std::exit(70);
        }
        s = s.replace(at, std::strlen(sub.from), sub.to);
    }
    return s;
}

static common_chat_params apply_template(const llama_model * model, const char * override_src,
                                        const std::vector<common_chat_msg> & msgs, bool use_model) {
    common_chat_templates_ptr tmpls = use_model
        ? common_chat_templates_init(model, override_src ? override_src : "", "", "")
        : common_chat_templates_init(nullptr, override_src, "", "");
    common_chat_templates_inputs in;
    in.messages              = msgs;
    in.add_generation_prompt = false;   // this template has no generation-prompt branch; state it
    in.use_jinja             = true;
    return common_chat_templates_apply(tmpls.get(), in);
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        std::fprintf(stderr,
                     "usage: %s source <model>\n"
                     "       %s template <model> <out-file>\n"
                     "       %s render <model> <cases> <outdir> [--now YYYY-MM-DD] [--refuse]\n",
                     argv[0], argv[0], argv[0]);
        return 70;
    }
    const std::string cmd   = argv[1];
    const char *      model_path = argv[2];

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.vocab_only = true;                       // vocab only: no weights, no GPU (~0.5 s)
    llama_model * model = llama_model_load_from_file(model_path, mp);
    if (model == nullptr) { std::fprintf(stderr, "m1_chat_ref: cannot load vocab of %s\n", model_path); return 70; }

    const char * tmpl_src = llama_model_chat_template(model, nullptr);
    if (tmpl_src == nullptr) { std::fprintf(stderr, "m1_chat_ref: the GGUF has no tokenizer.chat_template\n"); return 70; }
    const std::string tmpl(tmpl_src);

    if (cmd == "source") {
        std::printf("ORACLE source len=%zu\n", tmpl.size());
        std::fwrite(tmpl.data(), 1, tmpl.size(), stdout);
        std::printf("\nORACLE source end\n");
        return 0;
    }
    if (cmd == "template") {
        if (argc < 4) { std::fprintf(stderr, "m1_chat_ref: template <model> <out-file>\n"); return 70; }
        spit(argv[3], tmpl);
        std::printf("ORACLE template %s len=%zu\n", argv[3], tmpl.size());
        return 0;
    }
    if (cmd != "render") { std::fprintf(stderr, "m1_chat_ref: unknown command %s\n", cmd.c_str()); return 70; }
    if (argc < 5) { std::fprintf(stderr, "m1_chat_ref: render <model> <cases> <outdir> [--now YMD]\n"); return 70; }

    const char * cases_path = argv[3];
    const std::string outdir = argv[4];
    const char * now_ymd = nullptr;
    bool         require_refusal = false;
    for (int i = 5; i < argc; i++) {
        const std::string a = argv[i];
        if (a == "--now" && i + 1 < argc) now_ymd = argv[++i];
        else if (a == "--refuse") require_refusal = true;
        else { std::fprintf(stderr, "m1_chat_ref: unknown arg %s\n", a.c_str()); return 70; }
    }
    const std::string src = now_ymd ? inject_date(tmpl, now_ymd) : tmpl;
    if (now_ymd) std::printf("ORACLE date injection now=%s (template bytes %zu -> %zu)\n", now_ymd, tmpl.size(), src.size());

    int bad = 0;
    for (const RefCase & c : parse_cases(cases_path)) {
        bool refused = false;
        std::string why;
        std::string nobos, modeldriven;
        std::string format_name;
        try {
            common_chat_params a = apply_template(model, src.c_str(), c.msgs, /*use_model=*/true);
            modeldriven   = a.prompt;
            format_name   = common_chat_format_name(a.format);
            common_chat_params b = apply_template(nullptr, src.c_str(), c.msgs, /*use_model=*/false);
            nobos         = b.prompt;
        } catch (const std::exception & e) {
            refused = true;
            why     = e.what();
        } catch (...) {
            refused = true;
            why     = "(non-std exception)";
        }
        if (refused) {
            std::printf("ORACLE %-18s REFUSED  %s\n", c.name.c_str(), why.c_str());
            if (!require_refusal) bad = 1;      // in render mode a refusal is a case that cannot be compared
            continue;
        }
        if (require_refusal) {
            std::printf("ORACLE %-18s RENDERED but a refusal was required (bad)\n", c.name.c_str());
            bad = 1;
            continue;
        }
        const std::string p1 = outdir + "/" + c.name + ".ref-nobos.bin";
        const std::string p2 = outdir + "/" + c.name + ".ref.bin";
        spit(p1, nobos);
        spit(p2, modeldriven);
        std::printf("ORACLE %-18s format=%-14s ref-nobos=%zu ref=%zu  head=%.24s\n", c.name.c_str(), format_name.c_str(),
                    nobos.size(), modeldriven.size(), nobos.c_str());
    }
    llama_model_free(model);
    llama_backend_free();
    return bad;
}
