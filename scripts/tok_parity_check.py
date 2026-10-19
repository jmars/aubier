#!/usr/bin/env python3
"""tok_parity_check.py — corpus builder + comparators for scripts/engine_gate_tok.sh (U1).

Three subcommands, each exit-code-decisive:

  build  <casedir> <root> <tok_test> <model>
        Write the parity corpus and its manifest.  The 1000 special-token texts are taken
        from an INDEPENDENT python GGUF reader here, NOT from tok_test, and then CROSS-CHECKED
        against `tok_test --special-list` (id, text): a disagreement is a FAIL (exit 1),
        because a corpus built from the implementation under test could not detect the
        implementation dropping specials.

  ids    <ref_ids> <our_ids>
        Compare two `[1, 234, ...]` arrays byte-for-byte after parsing; report the first
        differing index with both ids.  exit 0 / 1.

  pieces <ref_pieces> <our_pieces> <casefile> <idsfile>
        Compare per-token detokenized BYTES.  The reference prints `%6d -> '<piece>'` lines,
        which a piece containing a newline or a quote breaks as line-oriented text, so records
        are parsed SEQUENTIALLY against the known id list (idsfile, one id per line, our own
        order — already proven equal to the reference's by the ids arm), and the parse is then
        validated by requiring concat(pieces) == the case file's bytes.  A misparse therefore
        cannot pass silently: it shows up as a concat mismatch (exit 1) or as a piece mismatch
        (exit 1), never as a false PASS.

Exit codes: 0 pass, 1 fail (with a specific line on stderr), 2 insufficient/unusable input.
"""
import os
import struct
import sys


# ---------------------------------------------------------------------------
# GGUF — just enough to read the tokenizer's metadata block (no tensors)
# ---------------------------------------------------------------------------
_GGUF_SCALARS = {
    0: ("<B", 1),   # u8
    1: ("<b", 1),   # i8
    2: ("<H", 2),   # u16
    3: ("<h", 2),   # i16
    4: ("<I", 4),   # u32
    5: ("<i", 4),   # i32
    6: ("<f", 4),   # f32
    7: ("<?", 1),   # bool
    10: ("<Q", 8),  # u64
    11: ("<q", 8),  # i64
    12: ("<d", 8),  # f64
}
_GGUF_STRING = 8
_GGUF_ARRAY = 9


class Reader:
    def __init__(self, path):
        self.f = open(path, "rb")
        self.n = os.fstat(self.f.fileno()).st_size

    def read(self, k):
        b = self.f.read(k)
        if len(b) != k:
            raise SystemExit("tok_parity_check.py: GGUF truncated")
        return b

    def u32(self):
        return struct.unpack("<I", self.read(4))[0]

    def u64(self):
        return struct.unpack("<Q", self.read(8))[0]

    def string(self):
        n = self.u64()
        return self.read(n)

    def scalar(self, t):
        fmt, sz = _GGUF_SCALARS[t]
        return struct.unpack(fmt, self.read(sz))[0]


ATTR_UNKNOWN = 1 << 0
ATTR_UNUSED = 1 << 1
ATTR_NORMAL = 1 << 2
ATTR_CONTROL = 1 << 3
ATTR_USER_DEFINED = 1 << 4
ATTR_BYTE = 1 << 5
_TOKTYPE_TO_ATTR = {
    0: 0,  # UNDEFINED
    1: ATTR_NORMAL,
    2: ATTR_UNKNOWN,
    3: ATTR_CONTROL,
    4: ATTR_USER_DEFINED,
    5: ATTR_UNUSED,
    6: ATTR_BYTE,
}


def read_vocab(path):
    """Return dict(tokens=[bytes], attrs=[int], n_merges=int, kv={name: value})."""
    r = Reader(path)
    if r.read(4) != b"GGUF":
        raise SystemExit("tok_parity_check.py: not a GGUF file")
    version = r.u32()
    if version not in (2, 3):
        raise SystemExit("tok_parity_check.py: unexpected GGUF version %d" % version)
    n_tensors = r.u64()
    n_kv = r.u64()

    tokens = None
    token_types = None
    n_merges = None
    kv = {}
    want_scalars = {
        b"tokenizer.ggml.model", b"tokenizer.ggml.pre",
        b"tokenizer.ggml.bos_token_id", b"tokenizer.ggml.eos_token_id",
        b"tokenizer.ggml.add_bos_token",
    }
    want_arrays = {b"tokenizer.ggml.tokens", b"tokenizer.ggml.token_type",
                   b"tokenizer.ggml.merges"}

    for _ in range(n_kv):
        key = r.string()
        t = r.u32()
        if t == _GGUF_STRING:
            v = r.string()
            if key in want_scalars:
                kv[key] = v
        elif t == _GGUF_ARRAY:
            et = r.u32()
            n = r.u64()
            if key == b"tokenizer.ggml.tokens" or key == b"tokenizer.ggml.merges":
                if et != _GGUF_STRING:
                    raise SystemExit("tok_parity_check.py: %r is not a string array" % key)
                keep = key == b"tokenizer.ggml.tokens"
                vals = [] if keep else None
                for _i in range(n):
                    s = r.string()
                    if keep:
                        vals.append(s)
                if keep:
                    tokens = vals
                else:
                    n_merges = n
            elif key == b"tokenizer.ggml.token_type":
                if et != 5:  # int32
                    raise SystemExit("tok_parity_check.py: token_type is not int32[]")
                token_types = list(struct.unpack("<%di" % n, r.read(4 * n)))
            else:
                # skip: readable payload only for the scalar element types
                if et == _GGUF_STRING:
                    for _i in range(n):
                        r.read(r.u64())
                elif et in _GGUF_SCALARS:
                    r.read(_GGUF_SCALARS[et][1] * n)
                else:
                    raise SystemExit("tok_parity_check.py: cannot skip array type %d" % et)
        elif t in _GGUF_SCALARS:
            v = r.scalar(t)
            if key in want_scalars:
                kv[key] = v
        else:
            raise SystemExit("tok_parity_check.py: unknown GGUF kv type %d" % t)

    if tokens is None or token_types is None or n_merges is None:
        raise SystemExit("tok_parity_check.py: tokenizer metadata incomplete")
    attrs = [_TOKTYPE_TO_ATTR.get(t, 0) for t in token_types]
    return {"tokens": tokens, "attrs": attrs, "n_merges": n_merges,
            "n_tensors": n_tensors, "kv": kv}


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------
SPECIAL_ATTRS = ATTR_CONTROL | ATTR_USER_DEFINED | ATTR_UNKNOWN

# 3-byte codepoints (CJK/Hangul), 4-byte codepoints (emoji), combining accents
CJK3 = "你好世界日本語한국어中文測試"
EMOJI4 = "😀🎉🔥🌍🚀🧠"
COMBINING = "e\u0301a\u0300n\u0303o\u0308u\u0327"
WS_MIX = "a\tb\u00a0c\u2028d\u2003e\u3000f \u2009g"


def build(casedir, root, tok_test_bin, model):
    os.makedirs(casedir, exist_ok=True)
    vocab = read_vocab(model)

    # ---- independent control: our C++ reader must agree with this reader -------------------
    tokens = vocab["tokens"]
    attrs = vocab["attrs"]
    if len(tokens) != 131072:
        sys.stderr.write("tok_parity_check.py: FAIL python reader: %d tokens\n" % len(tokens))
        return 1
    for i, want in ((1, b"<s>"), (2, b"</s>"), (3, b"[INST]")):
        if tokens[i] != want:
            sys.stderr.write("tok_parity_check.py: FAIL python reader: tokens[%d]=%r != %r\n"
                             % (i, tokens[i], want))
            return 1
    special_ids = [i for i, a in enumerate(attrs) if a & SPECIAL_ATTRS]
    print("# python GGUF reader (independent): tokens=%d merges=%d control=%d specials=%d "
          "model=%r pre=%r"
          % (len(tokens), vocab["n_merges"],
             sum(1 for a in attrs if a & ATTR_CONTROL), len(special_ids),
             vocab["kv"].get(b"tokenizer.ggml.model", b"?").decode("utf-8", "replace"),
             vocab["kv"].get(b"tokenizer.ggml.pre", b"?").decode("utf-8", "replace")))
    if len(special_ids) != 1000:
        sys.stderr.write("tok_parity_check.py: FAIL python reader: %d specials, expected 1000\n"
                         % len(special_ids))
        return 1

    # ---- cross-check against tok_test --special-list (id, text, hex) ---------------------
    # Compare as SETS ordered by id: the invariant is that the same 1000 (id, text) PAIRS are
    # special, not that the two readers agree on the length-descending permutation (ours is
    # std::sort's, deliberately, and it is a property of our list, checked separately below).
    out = os.popen("%s --special-list 2>/dev/null" % tok_test_bin).read()
    got = []
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) == 2:
            got.append((int(parts[0]), bytes.fromhex(parts[1])))
    want = [(i, tokens[i]) for i in special_ids]
    if len(got) != len(want) or sorted(got) != want:
        sys.stderr.write("tok_parity_check.py: FAIL the C++ special-token cache does not match "
                         "the independent reader: %d vs %d entries"
                         % (len(got), len(want)))
        for a, b in zip(sorted(got), want):
            if a != b:
                sys.stderr.write("; first (id-sorted) difference: %r vs %r" % (a, b))
                break
        sys.stderr.write("\n")
        return 1
    # and OUR list must be sorted by DESCENDING text length (llama-vocab.cpp:3169-3173) — a
    # property a wrong comparator would break while leaving the set intact
    lens = [len(tokens[i]) for i, _t in got]
    for k in range(1, len(lens)):
        if lens[k] > lens[k - 1]:
            sys.stderr.write("tok_parity_check.py: FAIL the C++ special list is not sorted by "
                             "descending text length at position %d (%d > %d)\n"
                             % (k, lens[k], lens[k - 1]))
            return 1
    print("# special-token cache: %d entries, the C++ reader and the independent reader AGREE "
          "on the set, and the list is length-descending" % len(want))

    def write(name, data):
        if isinstance(data, str):
            data = data.encode("utf-8")
        with open(os.path.join(casedir, name + ".txt"), "wb") as f:
            f.write(data)

    cases = []   # (name, flags)   flags: "" or "sp0"

    known = [
        ("known-01-hello-world", "Hello world"),
        ("known-02-nihao", "你好"),
        ("known-03-digits-7", "1234567"),
        ("known-04-two-spaces", "  two  spaces  "),
        ("known-05-inst-token", "a [INST] b"),
        ("known-06-context", "Hello, world! Ça va?"),
        ("known-07-x-newline-y", "x\ny"),
        ("known-08-apostrophe", "don't"),
        ("known-09-special-alone", "[INST]"),
        ("known-10-bos-both", "<s>[INST] Hi [/INST]"),
    ]
    for name, text in known:
        write(name, text)
        cases.append((name, ""))

    write("ascii-printable", bytes(range(0x20, 0x7f)))
    cases.append(("ascii-printable", ""))

    write("utf8-cjk3", CJK3)
    write("utf8-emoji4", EMOJI4)
    write("utf8-combining", COMBINING)
    write("utf8-ws-mix", WS_MIX)
    write("utf8-mixed", "Preço: €1,50 — naïve café · 日本語 · 😀 · e\u0301\n")
    for n in ("utf8-cjk3", "utf8-emoji4", "utf8-combining", "utf8-ws-mix", "utf8-mixed"):
        cases.append((n, ""))

    write("spaces-40", " " * 40)
    write("digits-300", "1234567890" * 30)
    write("crlf", "a\r\nb\r\nc\r\n")
    write("lone-cr", "a\rb\rc")
    write("empty", "")
    for n in ("spaces-40", "digits-300", "crlf", "lone-cr", "empty"):
        cases.append((n, ""))

    # ---- the 1000 special texts, both arms, chunked so a failure names 50 candidates -----
    CHUNK = 50
    for c in range(0, len(want), CHUNK):
        chunk = want[c:c + CHUNK]
        blob = b"\n".join(t for _i, t in chunk)
        name = "special-%04d-%04d" % (c, min(c + CHUNK, len(want)) - 1)
        write(name, blob)
        cases.append((name, ""))        # parse_special = true  (the specials must become ids)
        cases.append((name, "sp0"))     # parse_special = false (the same bytes as literal text)
    # and every one of the 1000 in ONE file (interaction between adjacent specials)
    write("special-all", b"\n".join(t for _i, t in want))
    cases.append(("special-all", ""))
    cases.append(("special-all", "sp0"))

    # ---- real text: every scripts/*.sh, plus one ~100 KB slice of LEDGER.md ---------------
    scriptdir = os.path.join(root, "scripts")
    for fn in sorted(os.listdir(scriptdir)):
        if not fn.endswith(".sh"):
            continue
        with open(os.path.join(scriptdir, fn), "rb") as f:
            write("script-" + fn[:-3], f.read())
        cases.append(("script-" + fn[:-3], ""))

    # ---- the WHOLE vocab in one case: every NORMAL token text, one per line -----------------
    # 130072 texts / 1.39 MB: each is a pre-tokenized word the regex can produce, so this
    # exercises the BPE stage over effectively the entire vocab (MEASURED: the reference needs
    # ~1.1 s and emits 700k+ ids).  It is the largest case and the perf sanity arm.
    write("vocab-all-normal", b"\n".join(
        t for t, a in zip(tokens, attrs) if a & ATTR_NORMAL))
    cases.append(("vocab-all-normal", ""))

    ledger = os.path.join(root, "LEDGER.md")
    if os.path.isfile(ledger):
        with open(ledger, "rb") as f:
            blob = f.read(100000)
        # never cut a UTF-8 sequence in half: trim trailing continuation bytes
        cut = len(blob)
        while cut > 0 and (blob[cut - 1] & 0xC0) == 0x80:
            cut -= 1
        blob = blob[:cut]
        write("ledger-100k", blob)
        cases.append(("ledger-100k", ""))
        print("# corpus: LEDGER.md slice %d bytes (internal file, read at RUN TIME only)" % cut)
    else:
        print("# corpus: LEDGER.md ABSENT -> the 100 KB real-text case is SKIPPED (reported)")

    with open(os.path.join(casedir, "manifest.tsv"), "w") as f:
        for name, flags in cases:
            f.write("%s\t%s\n" % (name, flags))
    print("# corpus: %d cases in %s" % (len(cases), casedir))
    return 0


# ---------------------------------------------------------------------------
# ids
# ---------------------------------------------------------------------------
def parse_ids(text):
    """Parse '[...]' or a bare comma list into ints; None if it does not parse."""
    s = text.strip()
    if not s.startswith("[") or not s.endswith("]"):
        return None
    body = s[1:-1].strip()
    if not body:
        return []
    out = []
    for tok in body.replace(",", " ").split():
        try:
            out.append(int(tok))
        except ValueError:
            return None
    return out


def cmp_ids(ref_path, our_path):
    with open(ref_path, "r", errors="replace") as f:
        ref_txt = f.read()
    with open(our_path, "r", errors="replace") as f:
        our_txt = f.read()
    ref = parse_ids(ref_txt)
    our = parse_ids(our_txt)
    if ref is None:
        sys.stderr.write("  reference output is not an id array: %r\n" % ref_txt[:120])
        return 2
    if our is None:
        sys.stderr.write("  tok_test output is not an id array: %r\n" % our_txt[:120])
        return 2
    if len(ref) != len(our):
        k = 0
        while k < min(len(ref), len(our)) and ref[k] == our[k]:
            k += 1
        sys.stderr.write("  ID COUNT differs: reference %d, ours %d (first divergence at "
                         "[%d]: ref=%s ours=%s)\n"
                         % (len(ref), len(our), k,
                            ref[k] if k < len(ref) else "EOF",
                            our[k] if k < len(our) else "EOF"))
        return 1
    for k in range(len(ref)):
        if ref[k] != our[k]:
            lo = max(0, k - 3)
            sys.stderr.write("  ID MISMATCH at index %d: reference %d, ours %d\n"
                             "    ref  [%d:%d] = %s\n    ours [%d:%d] = %s\n"
                             % (k, ref[k], our[k], lo, k + 3, ref[lo:k + 3], lo, k + 3,
                                our[lo:k + 3]))
            return 1
    print("  %d ids identical" % len(ref))
    return 0


# ---------------------------------------------------------------------------
# pieces
# ---------------------------------------------------------------------------
def parse_ref_pieces(blob, ids):
    """Sequentially parse `%6d -> '<piece>'` records against the known id list."""
    pieces = []
    pos = 0
    n = len(ids)
    for k, tid in enumerate(ids):
        prefix = b"%6d -> '" % tid
        if not blob.startswith(prefix, pos):
            raise ValueError("record %d: expected %r at byte %d, found %r"
                             % (k, prefix, pos, blob[pos:pos + 40]))
        start = pos + len(prefix)
        nxt = (b"%6d -> '" % ids[k + 1]) if k + 1 < n else None
        j = start
        found = -1
        while True:
            j = blob.find(b"'", j)
            if j < 0:
                raise ValueError("record %d: no closing quote" % k)
            if blob.startswith(b"\n", j + 1):
                after = j + 2
                if nxt is None:
                    if after == len(blob):
                        found = j
                        break
                elif blob.startswith(nxt, after):
                    found = j
                    break
            j += 1
        pieces.append(blob[start:found])
        pos = found + 2
    if pos != len(blob):
        raise ValueError("%d trailing byte(s) after the last record" % (len(blob) - pos))
    return pieces


def cmp_pieces(ref_path, our_path, case, ids_path):
    with open(ref_path, "rb") as f:
        ref_blob = f.read()
    with open(our_path, "rb") as f:
        our_lines = f.read().splitlines()
    with open(ids_path, "r") as f:
        ids = [int(x) for x in f.read().split()]
    with open(case, "rb") as f:
        case_bytes = f.read()

    if not ids:
        sys.stderr.write("  no ids (empty case) — nothing to compare\n")
        return 0 if not our_lines and not ref_blob.strip() else 1
    try:
        ref = parse_ref_pieces(ref_blob, ids)
    except ValueError as e:
        sys.stderr.write("  REFERENCE PARSE FAILED (%s) — the harness, not the tokenizer; "
                         "the gate cannot decide from this arm\n" % e)
        return 2
    # the parse's own validity control: the pieces must reproduce the input bytes exactly
    joined = b"".join(ref)
    if joined != case_bytes:
        sys.stderr.write("  PARSE CONTROL FAILED: concat(reference pieces) is %d bytes, the case "
                         "is %d — a misparse cannot be allowed to pass\n"
                         % (len(joined), len(case_bytes)))
        return 2

    ours = []
    for line in our_lines:
        parts = line.split(b"\t")
        if len(parts) != 2:
            sys.stderr.write("  tok_test --pieces line malformed: %r\n" % line)
            return 2
        ours.append((int(parts[0]), bytes.fromhex(parts[1].decode())))

    if len(ours) != len(ids):
        sys.stderr.write("  PIECE COUNT differs: reference %d, ours %d\n"
                         % (len(ids), len(ours)))
        return 1
    for k, tid in enumerate(ids):
        if ours[k][0] != tid:
            sys.stderr.write("  ID ORDER differs at %d: reference %d, ours %d\n"
                             % (k, tid, ours[k][0]))
            return 1
        if ours[k][1] != ref[k]:
            sys.stderr.write("  PIECE MISMATCH at index %d (id %d): reference %r, ours %r\n"
                             % (k, tid, ref[k], ours[k][1]))
            return 1
    print("  %d pieces identical (%d bytes)" % (len(ids), len(joined)))
    return 0


def nobos_check(ref_path, nobos_path):
    """The CONTROL: --no-bos must be exactly "the default array minus its leading BOS"."""
    with open(ref_path, "r", errors="replace") as f:
        ref = parse_ids(f.read())
    with open(nobos_path, "r", errors="replace") as f:
        nb = parse_ids(f.read())
    if ref is None or nb is None:
        sys.stderr.write("  unparsable id array\n")
        return 2
    if not ref:
        sys.stderr.write("  the default array is empty\n")
        return 2
    expect = ref[1:] if ref[0] == 1 else ref
    if nb != expect:
        sys.stderr.write("  --no-bos gave %r, the default array minus its leading BOS is %r\n"
                         % (nb, expect))
        return 1
    print("  --no-bos == default minus the leading BOS (%d ids)" % len(nb))
    return 0


def nobos(ref_path, out_path):
    """The reference's own default id array minus its leading BOS: the anchor that the piece
    arm parses against (`--no-bos` means exactly "do not prepend BOS"; the gate re-derives that
    identity as a CONTROL on known-01 rather than assuming it)."""
    with open(ref_path, "r", errors="replace") as f:
        ids = parse_ids(f.read())
    if ids is None:
        sys.stderr.write("  reference output is not an id array\n")
        return 2
    if ids and ids[0] == 1:
        ids = ids[1:]
    with open(out_path, "w") as f:
        f.write("\n".join(str(i) for i in ids))
        if ids:
            f.write("\n")
    return 0


# ---------------------------------------------------------------------------
def main(argv):
    if len(argv) < 2:
        sys.stderr.write(__doc__)
        return 2
    cmd = argv[1]
    if cmd == "build":
        if len(argv) != 6:
            sys.stderr.write("build <casedir> <root> <tok_test> <model>\n")
            return 2
        return build(argv[2], argv[3], argv[4], argv[5])
    if cmd == "ids":
        if len(argv) != 4:
            sys.stderr.write("ids <ref> <ours>\n")
            return 2
        return cmp_ids(argv[2], argv[3])
    if cmd == "nobos":
        if len(argv) != 4:
            sys.stderr.write("nobos <ref_ids> <out_ids>\n")
            return 2
        return nobos(argv[2], argv[3])
    if cmd == "nobos-check":
        if len(argv) != 4:
            sys.stderr.write("nobos-check <ref_ids> <nobos_ids>\n")
            return 2
        return nobos_check(argv[2], argv[3])
    if cmd == "pieces":
        if len(argv) != 6:
            sys.stderr.write("pieces <ref> <ours> <casefile> <idsfile>\n")
            return 2
        return cmp_pieces(argv[2], argv[3], argv[4], argv[5])
    sys.stderr.write("unknown subcommand %r\n" % cmd)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
