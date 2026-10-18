#!/usr/bin/env python3
# tools/gg_onegroup_full.py — build the FULL one-group fixture (loader-groups unit, SEAM 3).
#
# Stream-copies a GGUF, renaming every expert tensor to its ".g0" member name and adding
# the taillis.expert_groups key (taillis/layout.md §2, the trivial partition: ONE group per
# logical expert tensor).  Only names + the one new KV change; the tensor DATA section is
# copied byte-identical (shifted as a whole by the header delta).  This is what makes the
# fallback (b) proof strong: the real model, real bytes, real loader, real loop — only the
# grouping metadata differs.
#
# Usage: gg_onegroup_full.py <src.gguf> <dst.gguf>
# The destination is a NEW file (never under models/).  ~35 GB of sequential I/O for the
# real model; run it when the host window allows.
#
# Hand-rolled GGUF walking (v2/v3): the only KV values interpreted are "general.alignment"
# (u32, optional) — everything else is copied raw.  No gguf-py dependency.

import struct
import sys

KEY = b"GGUF"  # magic check only; the real key lives in EXPERT_KEY below

EXPERT_KEY = "taillis.expert_groups"
GROUP_SUFFIXES = ("ffn_gate_up_exps.weight", "ffn_down_exps.weight")

SCALAR_SIZE = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 8: None, 9: None, 10: 8, 11: 8, 12: 8}
# 0..7 scalars, 8 string, 9 array, 10 u64, 11 i64, 12 f64 (gguf.h enum order)

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
    # the VALUE type (9 = ARRAY) is written by kv_bytes; here: elem type (8 = STRING), count, strings
    out = struct.pack("<I", 8) + struct.pack("<Q", len(members))
    for m in members:
        b = m.encode("utf-8")
        out += struct.pack("<Q", len(b)) + b
    return out

def kv_bytes(key, members):
    kb = key.encode("utf-8")
    return struct.pack("<Q", len(kb)) + kb + struct.pack("<I", 9) + value_bytes_for_str_array(members)

def main(src, dst):
    with open(src, "rb") as f:
        magic = f.read(4)
        assert magic == b"GGUF", f"not a GGUF file: {src}"
        (version,) = struct.unpack("<I", f.read(4))
        assert version in (2, 3), f"unsupported GGUF version {version}"
        header_fmt = "<QI" if version == 2 else "<QQ"
        # v2 had u32 counts; v3 has u64.  The repo's toolchain writes v3; support both.
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
        expert_names = []
        renames = 0
        for _ in range(n_tensors):
            name = read_str(f)
            (n_dims,) = struct.unpack("<I", f.read(4))
            f.read(8 * n_dims)          # ne[]
            f.read(4)                   # type
            f.read(8)                   # offset
            if name.endswith(GROUP_SUFFIXES):
                expert_names.append(name[: -len(".weight")] + ".g0")   # the member name, not the logical
                renames += 1
        infos_len = f.tell() - infos_start
        data_start = (f.tell() + alignment - 1) // alignment * alignment
        assert renames in (2, 72), f"expected the 2 fixture or 72 model expert tensors, saw {renames}"

    new_kv = kv_bytes(EXPERT_KEY, expert_names)
    new_data_start = (kv_start + kv_len + len(new_kv) + infos_len + 3 * renames + alignment - 1) // alignment * alignment
    shift = new_data_start - data_start

    with open(src, "rb") as f, open(dst, "wb") as g:
        g.write(b"GGUF" + struct.pack("<I", version) + struct.pack("<QQ", n_tensors, n_kv + 1))
        f.seek(kv_start)
        remaining = kv_len
        while remaining:
            chunk = f.read(min(1 << 24, remaining))
            g.write(chunk)
            remaining -= len(chunk)
        g.write(new_kv)
        f.seek(infos_start)
        for _ in range(n_tensors):
            name = read_str(f)
            (n_dims,) = struct.unpack("<I", f.read(4))
            ne = struct.unpack("<" + "Q" * n_dims, f.read(8 * n_dims))
            (ttype,) = struct.unpack("<I", f.read(4))
            (toff,) = struct.unpack("<Q", f.read(8))
            if name.endswith(GROUP_SUFFIXES):
                name = name[: -len(".weight")] + ".g0.weight"
            nb = name.encode()
            g.write(struct.pack("<Q", len(nb)) + nb)
            g.write(struct.pack("<I", n_dims) + struct.pack("<" + "Q" * n_dims, *ne))
            g.write(struct.pack("<I", ttype))
            g.write(struct.pack("<Q", toff))   # offsets are RELATIVE to the data section: unchanged
        pos = g.tell()
        g.write(b"\0" * (new_data_start - pos))
        f.seek(data_start)
        while True:
            chunk = f.read(1 << 24)
            if not chunk:
                break
            g.write(chunk)
    print(f"wrote {dst}: {renames} expert tensors renamed to .g0, key '{EXPERT_KEY}' "
          f"with {len(expert_names)} members, data shifted by {shift} bytes, alignment {alignment}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
