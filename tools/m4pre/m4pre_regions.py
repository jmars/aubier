#!/usr/bin/env python3
"""m4pre_regions.py — emit the exact byte regions of ONE decode token's CPU-side
(miss) expert volume, as a region file for tools/m4pre/m4pre_contention.cpp.

At the shipping pin (`-ncmoe 12 -ot ffn_down_exps=CPU --no-op-offload`) the CPU
computes, per token:
    12 layers x 4 experts of blk.N.ffn_gate_up_exps.weight
    36 layers x 4 experts of blk.N.ffn_down_exps.weight
(the 12 gate_up layers are the -ncmoe 12 tail, blk.24..35).

GGUF stores the expert axis as the OUTERMOST dimension, so the first 4 experts of
a tensor are one contiguous run at the tensor's start. That is why a "first 4
experts" read is a real per-expert-block read and not a synthetic stride.

Sizes are computed from the GGUF tensor dims + the ggml block table, and are
CROSS-CHECKED against `llama-gguf r` output when a cached listing is available
(logs/m4pre/gguf_r.txt); a mismatch is fatal, never silently reconciled.

Usage: m4pre_regions.py MODEL.gguf [--listing logs/m4pre/gguf_r.txt] [--out FILE]
Prints the region file on stdout (or writes --out) and a summary on stderr.
"""
import os
import struct
import sys

# (blck_size, type_size) per ggml_type, from vendor/llama.cpp/ggml/src/ggml.c
# type_traits[]. Only the types this model actually uses are validated against
# the cached llama-gguf listing below.
TYPE_TABLE = {
    0:  (1, 4, "F32"),
    1:  (1, 2, "F16"),
    2:  (32, 18, "Q4_0"),
    3:  (32, 20, "Q4_1"),
    6:  (32, 22, "Q5_0"),
    7:  (32, 24, "Q5_1"),
    8:  (32, 34, "Q8_0"),
    10: (256, 84, "Q2_K"),
    11: (256, 110, "Q3_K"),
    12: (256, 144, "Q4_K"),
    13: (256, 176, "Q5_K"),
    14: (256, 210, "Q6_K"),
    16: (256, 66, "IQ2_XXS"),
    17: (256, 74, "IQ2_XS"),
    18: (256, 98, "IQ3_XXS"),
    19: (256, 50, "IQ1_S"),
    20: (32, 18, "IQ4_NL"),
    21: (256, 110, "IQ3_S"),
    22: (256, 82, "IQ2_S"),
    23: (256, 136, "IQ4_XS"),
    24: (1, 1, "I8"),
    25: (1, 2, "I16"),
    26: (1, 4, "I32"),
    27: (1, 8, "I64"),
    28: (1, 8, "F64"),
    29: (256, 56, "IQ1_M"),
    30: (1, 2, "BF16"),
    34: (256, 54, "TQ1_0"),
    35: (256, 66, "TQ2_0"),
    39: (32, 17, "MXFP4"),
}

GU_TENSOR = "ffn_gate_up_exps.weight"
DN_TENSOR = "ffn_down_exps.weight"


class RD:
    def __init__(self, f):
        self.f = f

    def raw(self, n):
        b = self.f.read(n)
        if len(b) != n:
            raise EOFError("short read")
        return b

    def u32(self):
        return struct.unpack("<I", self.raw(4))[0]

    def u64(self):
        return struct.unpack("<Q", self.raw(8))[0]

    def i32(self):
        return struct.unpack("<i", self.raw(4))[0]

    def i64(self):
        return struct.unpack("<q", self.raw(8))[0]

    def f32(self):
        return struct.unpack("<f", self.raw(4))[0]

    def f64(self):
        return struct.unpack("<d", self.raw(8))[0]

    def string(self):
        return self.raw(self.u64()).decode("utf-8", "replace")

    def value(self, t):
        if t == 0:
            return struct.unpack("<B", self.raw(1))[0]
        if t == 1:
            return struct.unpack("<b", self.raw(1))[0]
        if t == 2:
            return struct.unpack("<H", self.raw(2))[0]
        if t == 3:
            return struct.unpack("<h", self.raw(2))[0]
        if t == 4:
            return self.u32()
        if t == 5:
            return self.i32()
        if t == 6:
            return self.f32()
        if t == 7:
            return struct.unpack("<?", self.raw(1))[0]
        if t == 8:
            return self.string()
        if t == 9:
            et = self.u32()
            return [self.value(et) for _ in range(self.u64())]
        if t == 10:
            return self.u64()
        if t == 11:
            return self.i64()
        if t == 12:
            return self.f64()
        raise ValueError("gguf kv type %d" % t)


def parse_gguf(path):
    with open(path, "rb") as f:
        r = RD(f)
        if r.raw(4) != b"GGUF":
            raise ValueError("not a GGUF file")
        ver = r.u32()
        n_tensors = r.u64()
        n_kv = r.u64()
        kv = {}
        for _ in range(n_kv):
            k = r.string()
            kv[k] = r.value(r.u32())
        infos = []
        for _ in range(n_tensors):
            name = r.string()
            nd = r.u32()
            dims = [r.u64() for _ in range(nd)]
            ttype = r.u32()
            off = r.u64()
            infos.append((name, dims, ttype, off))
        data_start = (f.tell() + kv.get("general.alignment", 32) - 1) // kv.get("general.alignment", 32) * kv.get("general.alignment", 32)
    return kv, infos, data_start, ver


def tensor_bytes(dims, ttype):
    blck, tsize, _ = TYPE_TABLE[ttype]
    row_elems = dims[0]
    if row_elems % blck:
        raise ValueError("row %d not a multiple of block %d" % (row_elems, blck))
    row_bytes = row_elems // blck * tsize
    nrows = 1
    for d in dims[1:]:
        nrows *= d
    return row_bytes * nrows, row_bytes


def load_listing(path):
    """name -> size, from a cached `llama-gguf r` dump (authoritative cross-check)."""
    if not path or not os.path.exists(path):
        return {}
    sizes = {}
    for line in open(path):
        if "tensor[" not in line or "name = " not in line:
            continue
        try:
            name = line.split("name = ")[1].split(",")[0]
            size = int(line.split("size = ")[1].split(",")[0])
        except (IndexError, ValueError):
            continue
        sizes.setdefault(name, size)
    return sizes


def main():
    argv = sys.argv[1:]
    if not argv:
        print(__doc__, file=sys.stderr)
        return 12
    model = argv[0]
    listing = None
    out = None
    i = 1
    while i < len(argv):
        if argv[i] == "--listing":
            listing = argv[i + 1]
            i += 2
        elif argv[i] == "--out":
            out = argv[i + 1]
            i += 2
        else:
            print("unknown arg %s" % argv[i], file=sys.stderr)
            return 12
    if listing is None:
        listing = os.path.join(os.path.dirname(os.path.abspath(model)), "..",
                               "logs", "m4pre", "gguf_r.txt")

    kv, infos, data_start, ver = parse_gguf(model)
    ref = load_listing(listing)
    n_layer = int(kv["mistral4.block_count"])
    n_expert = int(kv["mistral4.expert_count"])
    print("# gguf v%d data_start=%d n_layer=%d n_expert=%d arch=%s"
          % (ver, data_start, n_layer, n_expert, kv.get("general.architecture")),
          file=sys.stderr)
    print("# cross-check listing: %s (%d tensors)%s"
          % (listing, len(ref), "" if ref else " -- MISSING, sizes unverified"),
          file=sys.stderr)

    gu, dn = {}, {}
    for name, dims, ttype, off in infos:
        if name.endswith(GU_TENSOR):
            gu[int(name.split(".")[1])] = (dims, ttype, off)
        elif name.endswith(DN_TENSOR):
            dn[int(name.split(".")[1])] = (dims, ttype, off)
    if len(gu) != n_layer or len(dn) != n_layer:
        print("FATAL: expected %d of each expert tensor, found gu=%d dn=%d"
              % (n_layer, len(gu), len(dn)), file=sys.stderr)
        return 2

    # the -ncmoe 12 tail: the last 12 layers keep their MoE on the CPU
    gu_layers = sorted(gu)[-12:]
    # -ot ffn_down_exps=CPU moves every down tensor to the CPU
    dn_layers = sorted(dn)

    regions = []
    total = 0
    for layer in gu_layers:
        dims, ttype, off = gu[layer]
        size, _ = tensor_bytes(dims, ttype)
        nbytes = size // n_expert
        name = "blk.%d.%s" % (layer, GU_TENSOR)
        if ref and ref.get(name) != size:
            print("FATAL: %s size %d computed vs %s in listing"
                  % (name, size, ref.get(name)), file=sys.stderr)
            return 2
        for e in range(4):
            regions.append((data_start + off + e * nbytes, nbytes))
        total += 4 * nbytes
    for layer in dn_layers:
        dims, ttype, off = dn[layer]
        size, _ = tensor_bytes(dims, ttype)
        nbytes = size // n_expert
        name = "blk.%d.%s" % (layer, DN_TENSOR)
        if ref and ref.get(name) != size:
            print("FATAL: %s size %d computed vs %s in listing"
                  % (name, size, ref.get(name)), file=sys.stderr)
            return 2
        for e in range(4):
            regions.append((data_start + off + e * nbytes, nbytes))
        total += 4 * nbytes

    print("# %d gate_up layers x 4 experts + %d down layers x 4 experts = %d regions"
          % (len(gu_layers), len(dn_layers), len(regions)), file=sys.stderr)
    print("# total = %d bytes = %.2f MiB" % (total, total / 1048576.0), file=sys.stderr)
    print("# (the brief's ~500 MiB uses 8.3 MiB per down layer; 4 down experts are "
          "2.5625 MiB each = 10.25 MiB, so the real volume is higher)", file=sys.stderr)

    body = "".join("%d %d\n" % (o, l) for o, l in regions)
    if out:
        with open(out, "w") as f:
            f.write("# m4pre regions: one token's CPU-side expert miss volume\n")
            f.write(body)
    else:
        sys.stdout.write(body)
    return 0


if __name__ == "__main__":
    sys.exit(main())
