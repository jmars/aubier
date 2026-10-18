#!/usr/bin/env python3
# tools/m1_permute_fixture.py — build the U6 TREATMENT fixture: a byte-copy of the model GGUF
# in which, PER LAYER, the expert axis of ffn_gate_up_exps / ffn_down_exps is PERMUTED so that
# the layer's hot experts occupy fixture ids [0, S).  NOTHING else changes: same header, same
# KVs, same tensor infos (names, shapes, types, offsets) — the only bytes that move are the
# expert slabs inside the 72 expert tensors.  The engine then keeps its fixed [0,n_res) store
# cut (same VRAM, same store count, same graph structure) while the COMPOSITION of the
# resident set becomes the per-layer hot set.  The id permutation table P (the same file the
# engine consumes via --route-perm) maps original expert id e -> fixture id P[e]; fixture
# slab P[e] must carry original expert e's bytes.
#
# Usage:
#   m1_permute_fixture.py build <src.gguf> <dst.gguf> <perm.txt>   (perm.txt: 36 lines "<l> <128 ints>")
#   m1_permute_fixture.py verify <src.gguf> <dst.gguf> <perm.txt>
#
# The GGUF walk is hand-rolled like tools/gg_g4full.py (no gguf-py); tensor SIZES are computed
# from the block table and ASSERTED against the file layout.  A wrong table trips immediately.
#
# ~35 GB copy + ~30 GB of slab rewrites.  The destination is a NEW file (never under models/).

import os
import struct
import sys

SCALAR_SIZE = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 8: None, 9: None, 10: 8, 11: 8, 12: 8}
# ggml block table (ggml-common.h, QK_K=256; fork's type ids — same table as tools/gg_g4full.py)
TYPE_BLOCKS = {
    0: (1, 4), 1: (1, 2), 2: (32, 18), 3: (32, 20), 6: (32, 22), 7: (32, 24), 8: (32, 34),
    9: (32, 36), 10: (256, 84), 11: (256, 110), 12: (256, 144), 13: (256, 176), 14: (256, 210),
    15: (256, 292), 16: (256, 66), 17: (256, 74), 18: (256, 98), 19: (256, 50), 20: (32, 20),
    21: (256, 110), 22: (256, 82), 23: (256, 134), 30: (1, 2),
}
EXPERT_SUFFIXES = ("ffn_gate_up_exps.weight", "ffn_down_exps.weight")
N_EXPERT = 128
N_LAYERS = 36


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


def row_size(ttype, ne0):
    blck, tsize = TYPE_BLOCKS[ttype]
    assert ne0 % blck == 0, f"ne0 {ne0} not a multiple of block size {blck}"
    return ne0 // blck * tsize


def parse_header(path):
    with open(path, "rb") as f:
        magic = f.read(4)
        assert magic == b"GGUF", f"not a GGUF file: {path}"
        (version,) = struct.unpack("<I", f.read(4))
        assert version in (2, 3), f"unsupported GGUF version {version}"
        if version == 2:
            n_tensors, n_kv = struct.unpack("<II", f.read(8))
        else:
            n_tensors, n_kv = struct.unpack("<QQ", f.read(16))
        alignment = 32
        kv_start = f.tell()
        for _ in range(n_kv):
            kname = read_str(f)
            (ktype,) = struct.unpack("<I", f.read(4))
            if kname == "general.alignment" and ktype == 4:
                alignment = struct.unpack("<I", f.read(4))[0]
            else:
                skip_value(f, ktype)
        kv_len = f.tell() - kv_start
        infos_start = f.tell()
        tensors = []
        for _ in range(n_tensors):
            name = read_str(f)
            (n_dims,) = struct.unpack("<I", f.read(4))
            ne = struct.unpack("<" + "Q" * n_dims, f.read(8 * n_dims))
            (ttype,) = struct.unpack("<I", f.read(4))
            (toff,) = struct.unpack("<Q", f.read(8))
            tensors.append((name, n_dims, ne, ttype, toff))
        infos_len = f.tell() - infos_start
        data_start = (infos_start + infos_len + alignment - 1) // alignment * alignment

        by_off = sorted(tensors, key=lambda t: t[4])
        file_size = os.path.getsize(path)
        sizes = {}
        for i, (name, n_dims, ne, ttype, toff) in enumerate(by_off):
            assert ttype in TYPE_BLOCKS, f"{name}: type {ttype} not in block table"
            size = row_size(ttype, ne[0])
            for d in range(1, n_dims):
                size *= ne[d]
            sizes[name] = size
            end = toff + size
            nxt = by_off[i + 1][4] if i + 1 < len(by_off) else None
            if nxt is not None:
                assert end <= nxt, f"{name}: size overruns next tensor"
                assert nxt - end < alignment, f"{name}: gap {nxt-end} >= alignment"
            else:
                assert data_start + end <= file_size, f"{name}: last tensor overruns the file"
    return {"version": version, "n_tensors": n_tensors, "n_kv": n_kv,
            "alignment": alignment, "data_start": data_start, "tensors": tensors,
            "sizes": sizes, "file_size": file_size}


def load_perm(path):
    perm = []            # perm[l][e] = fixture id of original expert e
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            l = int(parts[0])
            row = [int(x) for x in parts[1:]]
            assert l == len(perm), f"perm row {l} out of order"
            assert len(row) == N_EXPERT, f"perm row {l}: {len(row)} entries"
            assert sorted(row) == list(range(N_EXPERT)), f"perm row {l} not a permutation"
            perm.append(row)
    assert len(perm) == N_LAYERS, f"perm has {len(perm)} rows"
    return perm


def expert_tensors(hdr):
    """the 72 (layer, suffix) expert tensors -> [(name, toff, slab_bytes)]"""
    out = []
    for (name, n_dims, ne, ttype, toff) in hdr["tensors"]:
        for l in range(N_LAYERS):
            for suf in EXPERT_SUFFIXES:
                if name == f"blk.{l}.{suf}":
                    assert n_dims == 3 and ne[2] == N_EXPERT, f"{name}: {ne} unexpected"
                    slab = hdr["sizes"][name] // N_EXPERT
                    out.append((name, toff, slab))
    assert len(out) == 2 * N_LAYERS, f"expected {2*N_LAYERS} expert tensors, saw {len(out)}"
    return out


def build(src, dst, perm_path):
    hdr = parse_header(src)
    perm = load_perm(perm_path)
    experts = expert_tensors(hdr)
    # covered byte ranges (absolute) of the expert tensors — everything else must be verbatim
    covered = sorted((hdr["data_start"] + toff, hdr["data_start"] + toff + hdr["sizes"][name], name)
                     for (name, toff, _slab) in experts)
    for i in range(1, len(covered)):
        assert covered[i][0] >= covered[i - 1][1], "expert ranges overlap"
    print(f"# {src} -> {dst}: {len(experts)} expert tensors, "
          f"{sum(hi - lo for lo, hi, _ in covered) / 2**30:.2f} GiB permuted of "
          f"{hdr['file_size'] / 2**30:.2f} GiB", flush=True)

    with open(src, "rb") as f:
        if os.path.exists(dst):
            os.remove(dst)
        with open(dst, "wb") as g:
            remaining = hdr["file_size"]
            f.seek(0)
            while remaining:
                chunk = f.read(min(1 << 24, remaining))
                g.write(chunk)
                remaining -= len(chunk)

    src_fd = os.open(src, os.O_RDONLY)
    dst_fd = os.open(dst, os.O_RDWR)
    total = 0
    for (name, toff, slab) in experts:
        l = int(name.split(".")[1])
        P = perm[l]                     # P[e] = fixture id of original expert e
        inv = [0] * N_EXPERT
        for e in range(N_EXPERT):
            inv[P[e]] = e               # inv[j] = the original expert that fixture id j carries
        base = hdr["data_start"] + toff
        buf = os.pread(src_fd, slab * N_EXPERT, base)
        assert len(buf) == slab * N_EXPERT
        for j in range(N_EXPERT):
            e = inv[j]
            os.pwrite(dst_fd, buf[e * slab:(e + 1) * slab], base + j * slab)
        total += slab * N_EXPERT
        print(f"  {name}: permuted ({slab * N_EXPERT / 2**20:.1f} MiB)", flush=True)
    os.close(src_fd)
    os.close(dst_fd)
    print(f"# wrote {total / 2**30:.2f} GiB of permuted slabs; now verifying", flush=True)
    verify(src, dst, perm_path)


def verify(src, dst, perm_path):
    hdr_s = parse_header(src)
    hdr_d = parse_header(dst)
    # (1) the header must be byte-identical (same names/shapes/types/offsets)
    assert hdr_s == hdr_d, "dst header differs from src"
    # (2) every non-expert byte region must be verbatim
    experts = expert_tensors(hdr_s)
    covered = sorted((hdr_s["data_start"] + toff, hdr_s["data_start"] + toff + hdr_s["sizes"][name])
                     for (name, toff, _s) in experts)
    fs = open(src, "rb")
    fd = open(dst, "rb")
    pos = 0
    checked_verbatim = 0
    for (lo, hi) in covered:
        while pos < lo:
            n = min(1 << 24, lo - pos)
            fs.seek(pos); a = fs.read(n)
            fd.seek(pos); b = fd.read(n)
            assert a == b, f"non-expert region at {pos} differs"
            checked_verbatim += n
            pos += n
        pos = hi
    while pos < hdr_s["file_size"]:
        n = min(1 << 24, hdr_s["file_size"] - pos)
        fs.seek(pos); a = fs.read(n)
        fd.seek(pos); b = fd.read(n)
        assert a == b, f"non-expert region at {pos} differs"
        checked_verbatim += n
        pos += n
    # (3) every expert slab: fixture slab P[e] == source slab e, byte for byte
    perm = load_perm(perm_path)
    src_fd = os.open(src, os.O_RDONLY)
    dst_fd = os.open(dst, os.O_RDONLY)
    n_checked = 0
    for (name, toff, slab) in experts:
        l = int(name.split(".")[1])
        P = perm[l]
        base = hdr_s["data_start"] + toff
        sbuf = os.pread(src_fd, slab * N_EXPERT, base)
        dbuf = os.pread(dst_fd, slab * N_EXPERT, base)
        for e in range(N_EXPERT):
            j = P[e]
            assert dbuf[j * slab:(j + 1) * slab] == sbuf[e * slab:(e + 1) * slab], \
                f"{name}: fixture slab {j} != source expert {e}"
            n_checked += 1
    os.close(src_fd)
    os.close(dst_fd)
    print(f"VERIFIED: header identical; {checked_verbatim / 2**30:.2f} GiB non-expert bytes "
          f"verbatim; {n_checked} expert slabs match the permutation.", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 5 or sys.argv[1] not in ("build", "verify"):
        print(__doc__)
        sys.exit(2)
    if sys.argv[1] == "build":
        build(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        verify(sys.argv[2], sys.argv[3], sys.argv[4])
