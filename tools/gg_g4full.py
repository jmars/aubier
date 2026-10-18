#!/usr/bin/env python3
# tools/gg_g4full.py — build the FULL G=4 fixture (engine g1-load unit, Tier 0.2e).
#
# Stream-copies a GGUF, splitting every flat expert tensor into G members of 128/G experts
# each and adding the taillis.expert_groups key (taillis/layout.md §2).  G=4 is the unit's
# fixture: a real partition (G>1) that still groups experts (unlike g128's per-expert split).
#
# The data section is copied BYTE-IDENTICAL: each member's byte range is a slice of the
# original flat tensor's range (expert-major layout, so expert e's rows sit at
# toff + e*expert_stride), and the G member ranges TILE the original range exactly.
# Only the info section (1 entry -> G entries per expert tensor) and the one new KV change.
# This is the same trick gg_onegroup_full.py uses: real model, real bytes, real loader,
# real loop — only the grouping metadata differs.
#
# Usage: gg_g4full.py <src.gguf> <dst.gguf> [G]      (G default 4; must divide 128)
# The destination is a NEW file (never under models/).  ~35 GB of sequential I/O.
#
# Hand-rolled GGUF walking (v2/v3), no gguf-py dependency.  Tensor SIZES are COMPUTED from
# (ne, type) via ggml's block table and ASSERTED against the file layout (toff + size <=
# next toff, gap < alignment) — a wrong table entry trips immediately.

import struct
import sys

EXPERT_KEY = "taillis.expert_groups"
GROUP_SUFFIXES = ("ffn_gate_up_exps.weight", "ffn_down_exps.weight")

SCALAR_SIZE = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 8: None, 9: None, 10: 8, 11: 8, 12: 8}

# ggml block table (ggml-common.h, QK_K=256; type ids from the fork's ggml.h enum —
# IQ2_XXS=16 .. IQ4_XS=23, NOT the pre-renumber enum).  Only types a model file may carry
# in an EXPERT tensor + the common ones; an unknown type aborts (the table is load-bearing
# for the member split).
TYPE_BLOCKS = {
    0: (1, 4),       # F32
    1: (1, 2),       # F16
    2: (32, 18),     # Q4_0
    3: (32, 20),     # Q4_1
    6: (32, 22),     # Q5_0
    7: (32, 24),     # Q5_1
    8: (32, 34),     # Q8_0
    9: (32, 36),     # Q8_1
    10: (256, 84),   # Q2_K
    11: (256, 110),  # Q3_K
    12: (256, 144),  # Q4_K
    13: (256, 176),  # Q5_K
    14: (256, 210),  # Q6_K
    15: (256, 292),  # Q8_K
    16: (256, 66),   # IQ2_XXS  (2.0625 bpw)
    17: (256, 74),   # IQ2_XS   (2.3125 bpw)
    18: (256, 98),   # IQ3_XXS  (3.0625 bpw)
    19: (256, 50),   # IQ1_S    (1.5625 bpw)
    20: (32, 20),    # IQ4_NL
    21: (256, 110),  # IQ3_S    (3.4375 bpw)
    22: (256, 82),   # IQ2_S    (2.5625 bpw)
    23: (256, 134),  # IQ4_XS   (4.25 bpw)
    30: (1, 2),      # BF16
}
TYPE_NAMES = {0: "f32", 1: "f16", 2: "q4_0", 3: "q4_1", 6: "q5_0", 7: "q5_1", 8: "q8_0",
              9: "q8_1", 10: "q2_K", 11: "q3_K", 12: "q4_K", 13: "q5_K", 14: "q6_K",
              15: "q8_K", 16: "iq2_xxs", 17: "iq2_xs", 18: "iq3_xxs", 19: "iq1_s",
              20: "iq4_nl", 21: "iq3_s", 22: "iq2_s", 23: "iq4_xs", 30: "bf16"}

def read_str(f):
    (n,) = struct.unpack("<Q", f.read(8))
    return f.read(n).decode("utf-8")

def skip_value(f, t):
    if t == 8:
        read_str(f)
    elif t == 9:
        (et,) = struct.unpack("<I", f.read(4))
        (n,) = struct.unpack("<Q", f.read(8))
        for _ in range(n):
            skip_value(f, et)
    else:
        f.read(SCALAR_SIZE[t])

def value_bytes_for_str_array(members):
    out = struct.pack("<I", 8) + struct.pack("<Q", len(members))
    for m in members:
        b = m.encode("utf-8")
        out += struct.pack("<Q", len(b)) + b
    return out

def kv_bytes(key, members):
    kb = key.encode("utf-8")
    return struct.pack("<Q", len(kb)) + kb + struct.pack("<I", 9) + value_bytes_for_str_array(members)

def row_size(ttype, ne0):
    blck, tsize = TYPE_BLOCKS[ttype]
    assert ne0 % blck == 0, f"ne0 {ne0} not a multiple of block size {blck} (type {TYPE_NAMES.get(ttype, ttype)})"
    return ne0 // blck * tsize

def main(src, dst, G=4):
    with open(src, "rb") as f:
        magic = f.read(4)
        assert magic == b"GGUF", f"not a GGUF file: {src}"
        (version,) = struct.unpack("<I", f.read(4))
        assert version in (2, 3), f"unsupported GGUF version {version}"
        if version == 2:
            n_tensors, n_kv = struct.unpack("<II", f.read(8))
        else:
            n_tensors, n_kv = struct.unpack("<QQ", f.read(16))

        kv_start = f.tell()
        alignment = 32
        for _ in range(n_kv):
            kname = read_str(f)
            (ktype,) = struct.unpack("<I", f.read(4))
            if kname == "general.alignment" and ktype == 4:
                alignment = struct.unpack("<I", f.read(4))[0]
            else:
                skip_value(f, ktype)
        kv_len = f.tell() - kv_start

        infos_start = f.tell()
        tensors = []            # (name, n_dims, ne, ttype, toff)
        for _ in range(n_tensors):
            name = read_str(f)
            (n_dims,) = struct.unpack("<I", f.read(4))
            ne = struct.unpack("<" + "Q" * n_dims, f.read(8 * n_dims))
            (ttype,) = struct.unpack("<I", f.read(4))
            (toff,) = struct.unpack("<Q", f.read(8))
            tensors.append((name, n_dims, ne, ttype, toff))
        infos_len = f.tell() - infos_start

        # per-tensor sizes from the block table, validated against the file layout
        by_off = sorted(tensors, key=lambda t: t[4])
        import os
        file_size = os.path.getsize(src)
        sizes = {}
        for i, (name, n_dims, ne, ttype, toff) in enumerate(by_off):
            assert ttype in TYPE_BLOCKS, f"{name}: type {ttype} not in the tool's block table"
            row = row_size(ttype, ne[0])
            size = row
            for d in range(1, n_dims):
                size *= ne[d]
            sizes[name] = size
            end = toff + size
            nxt = by_off[i + 1][4] if i + 1 < len(by_off) else None
            if nxt is not None:
                assert end <= nxt, f"{name}: computed size overruns the next tensor ({toff}+{size} > {nxt})"
                assert nxt - end < alignment, f"{name}: gap {nxt - end} >= alignment {alignment} — block table wrong?"
            else:
                assert end <= file_size, f"{name}: last tensor overruns the file"

        expert_names = []       # the key's members, in ARRAY ORDER (the reader's expert order)
        out_infos = []          # (name, n_dims, ne, ttype, toff) after the split
        n_split = 0
        for (name, n_dims, ne, ttype, toff) in tensors:
            if name.endswith(GROUP_SUFFIXES):
                assert ne[2] == 128, f"{name}: ne[2]={ne[2]}, expected 128 experts"
                assert 128 % G == 0, f"G={G} does not divide 128"
                per = ne[2] // G
                row = row_size(ttype, ne[0])
                stride = row * ne[1]                      # bytes per expert
                assert stride * per % alignment == 0, \
                    f"{name}: member span {stride * per} not {alignment}-aligned"
                base = name[: -len(".weight")]
                for k in range(G):
                    member = f"{base}.g{k}.weight"
                    mne = (ne[0], ne[1], per)
                    mtoff = toff + k * stride * per
                    out_infos.append((member, 3, mne, ttype, mtoff))
                    expert_names.append(f"{base}.g{k}")
                n_split += 1
            else:
                out_infos.append((name, n_dims, ne, ttype, toff))

        assert n_split == 72, f"expected the 72 model expert tensors, saw {n_split}"

    new_kv = kv_bytes(EXPERT_KEY, expert_names)
    # info entries recomputed exactly (G-1 extra per split tensor, names grew by ".g<k>")
    def info_entry_bytes(name, n_dims):
        return 8 + len(name.encode()) + 4 + 8 * n_dims + 4 + 8
    new_infos_len = sum(info_entry_bytes(nm, nd) for nm, nd, *_ in out_infos)
    new_data_start = (kv_start + kv_len + len(new_kv) + new_infos_len + alignment - 1) // alignment * alignment
    data_start = (infos_start + infos_len + alignment - 1) // alignment * alignment

    with open(src, "rb") as f, open(dst, "wb") as g:
        g.write(b"GGUF" + struct.pack("<I", version)
                + struct.pack("<QQ", len(out_infos), n_kv + 1))
        f.seek(kv_start)
        remaining = kv_len
        while remaining:
            chunk = f.read(min(1 << 24, remaining))
            g.write(chunk)
            remaining -= len(chunk)
        g.write(new_kv)
        for (name, n_dims, ne, ttype, toff) in out_infos:
            nb = name.encode()
            g.write(struct.pack("<Q", len(nb)) + nb)
            g.write(struct.pack("<I", n_dims) + struct.pack("<" + "Q" * n_dims, *ne))
            g.write(struct.pack("<I", ttype))
            g.write(struct.pack("<Q", toff))   # offsets stay RELATIVE to the data section
        pos = g.tell()
        g.write(b"\0" * (new_data_start - pos))
        f.seek(data_start)
        while True:
            chunk = f.read(1 << 24)
            if not chunk:
                break
            g.write(chunk)
    print(f"wrote {dst}: {n_split} expert tensors split into G={G} members "
          f"({len(expert_names)} key entries, {len(out_infos)} tensor infos, "
          f"data shifted by {new_data_start - data_start} bytes, alignment {alignment}, "
          f"data section byte-identical)")

if __name__ == "__main__":
    if len(sys.argv) not in (3, 4):
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) == 4 else 4)
