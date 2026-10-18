#!/usr/bin/env python3
# scripts/analyse_route_u6.py — U6: derive the per-layer oracle top-S hot set from the engine's
# own routing traces (--route-dump lines "R <layer> <pos> <sel0..3> <w0..w3>"), and report
# COVERAGE (the fraction of routed expert slots served by a resident set) both FITTED (the
# derivation traces) and HELD OUT (a different trace set).  This is p-pe's methodology rule:
# the allocation is derived from a real routing trace and evaluated against traces it never saw.
#
#   analyse_route_u6.py hotset  --traces 'trace-A*.txt' --n-res 48 --out-perm perm-48.txt \
#                                            --out-hotset hotset-A48.json
#       per layer: count selections of each expert over the whole trace set; hot set = the
#       top-48 ids by count (ties broken by lower id — stated, deterministic); writes the perm
#       table (entry e = the fixture id carrying expert e: hot experts get 0..47 in count-desc
#       order, the rest keep their original relative order) + a JSON hotset record.
#
#   analyse_route_u6.py coverage --traces 'trace-B*.txt' --hotset hotset-A48.json --n-res 48
#       coverage of the resident set on these traces: control = the fixed range [0,n_res),
#       treatment = the hotset; per-set mean + per-layer min/mean/max, misses per token.
#
# GGUF header parsing (for exact per-expert bytes -> the store drive's GiB/s) reuses the
# block-table walk of tools/m1_permute_fixture.py via --gguf, optional:
#   analyse_route_u6.py bytes --gguf <path> --trace <one trace> --n-res 48 [--hotset js]
#       prints per-trace-set miss GiB and, given --drive-s (from the engine's decomposition
#       print), the achieved GiB/s.  --gguf: the CONTROL model for control traces, the
#       FIXTURE for treatment traces (slab sizes are identical — same shapes and types —
#       so either header gives the same per-expert bytes; asserted).

import argparse
import glob
import json
import struct
import sys

N_LAYERS = 36
N_EXPERT = 128
N_USED = 4


def parse_traces(paths):
    """-> rows[l] = list of (file_idx, [ids]) one per (trace file, position);
    n_rows = total rows.  Positions from DIFFERENT files never collide (the file index
    is part of the identity), so a set of traces contributes every routed token."""
    rows = [[] for _ in range(N_LAYERS)]
    for fi, path in enumerate(paths):
        with open(path) as f:
            for line in f:
                if not line.startswith("R "):
                    continue
                p = line.split()
                l = int(p[1])
                ids = [int(x) for x in p[3:3 + N_USED]]
                for e in ids:
                    assert 0 <= e < N_EXPERT, f"{path}: id {e} out of range"
                rows[l].append((fi, ids))
    n_rows = len(rows[0])
    for l in range(1, N_LAYERS):
        assert len(rows[l]) == n_rows, f"layer {l} has {len(rows[l])} rows, layer 0 has {n_rows}"
    return rows, n_rows, n_rows


def hotset_from(rows, n_res):
    hot = []
    for l in range(N_LAYERS):
        counts = [0] * N_EXPERT
        for _fi, ids in rows[l]:
            for e in ids:
                counts[e] += 1
        order = sorted(range(N_EXPERT), key=lambda e: (-counts[e], e))
        hot.append((order[:n_res], counts))
    return hot


def cmd_hotset(args):
    paths = sorted(sum([glob.glob(p) for p in args.traces], []))
    paths = [p for p in paths if ".stdout" not in p]   # the rows live in trace-<tag>.txt only
    assert paths, "no trace files matched"
    print(f"# hotset from {len(paths)} traces: {', '.join(p.split('/')[-1] for p in paths)}")
    rows, n_rows, _ = parse_traces(paths)
    hot = hotset_from(rows, args.n_res)
    total = n_rows * N_LAYERS * N_USED
    cov = sum(sum(1 for _fi, ids in rows[l] for e in ids if e in set(hot[l][0]))
              for l in range(N_LAYERS)) / total
    print(f"# rows={n_rows} slots={total}  FITTED coverage at S={args.n_res}: {cov:.4f}")
    with open(args.out_perm, "w") as f:
        f.write("# U6 perm table: line l entry e = the PERMUTED FIXTURE id carrying original\n")
        f.write("# expert e.  Hot experts (top-%d by selection count in the derivation traces)\n" % args.n_res)
        f.write("# take ids 0..%d in count-desc order (ties: lower id); the rest keep their\n" % (args.n_res - 1))
        f.write("# original relative order.  Derived by scripts/analyse_route_u6.py from:\n")
        for p in paths:
            f.write(f"#   {p}\n")
        for l in range(N_LAYERS):
            P = [0] * N_EXPERT
            for j, e in enumerate(hot[l][0]):
                P[e] = j
            nxt = args.n_res
            for e in range(N_EXPERT):
                if e not in hot[l][0]:
                    P[e] = nxt
                    nxt += 1
            f.write(str(l) + " " + " ".join(str(x) for x in P) + "\n")
    with open(args.out_hotset, "w") as f:
        json.dump({"n_res": args.n_res,
                   "traces": paths,
                   "fitted_coverage": cov,
                   "hot": [hot[l][0] for l in range(N_LAYERS)],
                   "counts": [hot[l][1] for l in range(N_LAYERS)]}, f)
    print(f"# wrote {args.out_perm} and {args.out_hotset}")
    print("# per-layer fitted coverage (min/mean/max):")
    per = []
    for l in range(N_LAYERS):
        tot_l = len(rows[l]) * N_USED
        c_l = sum(1 for _fi, ids in rows[l] for e in ids if e in set(hot[l][0])) / tot_l
        per.append(c_l)
    print(f"#   {min(per):.4f} / {sum(per)/len(per):.4f} / {max(per):.4f}")


def cmd_coverage(args):
    paths = sorted(sum([glob.glob(p) for p in args.traces], []))
    paths = [p for p in paths if ".stdout" not in p]
    assert paths, "no trace files matched"
    hs = json.load(open(args.hotset))
    assert hs["n_res"] == args.n_res, "hotset n_res mismatch"
    hot = [set(h) for h in hs["hot"]]
    rows, n_rows, _ = parse_traces(paths)
    tot = n_rows * N_LAYERS * N_USED
    # control: the fixed range [0, n_res) on the ORIGINAL ids (what today's engine serves)
    cov_ctrl = sum(sum(1 for _fi, ids in rows[l] for e in ids if e < args.n_res)
                   for l in range(N_LAYERS)) / tot
    # treatment: the hot set
    cov_hot = sum(sum(1 for _fi, ids in rows[l] for e in ids if e in hot[l])
                  for l in range(N_LAYERS)) / tot
    print(f"# coverage on {len(paths)} traces ({n_rows} rows, {tot} slots)")
    print(f"#   CONTROL  fixed [0,{args.n_res}) : {cov_ctrl:.4f}")
    print(f"#   TREATMENT hotset (S={args.n_res}) : {cov_hot:.4f}")
    print(f"#   misses/token: control {N_LAYERS*N_USED*(1-cov_ctrl)/N_LAYERS:.3f}/layer "
          f"({N_LAYERS*N_USED*(1-cov_ctrl):.1f}/token), treatment "
          f"{N_LAYERS*N_USED*(1-cov_hot)/N_LAYERS:.3f}/layer ({N_LAYERS*N_USED*(1-cov_hot):.1f}/token)")
    per_c, per_h = [], []
    for l in range(N_LAYERS):
        t_l = len(rows[l]) * N_USED
        per_c.append(sum(1 for _fi, ids in rows[l] for e in ids if e < args.n_res) / t_l)
        per_h.append(sum(1 for _fi, ids in rows[l] for e in ids if e in hot[l]) / t_l)
    print("#   per-layer control  min/mean/max: %.4f / %.4f / %.4f" %
          (min(per_c), sum(per_c) / N_LAYERS, max(per_c)))
    print("#   per-layer hotset  min/mean/max: %.4f / %.4f / %.4f" %
          (min(per_h), sum(per_h) / N_LAYERS, max(per_h)))
    # per-trace breakdown
    for path in paths:
        r1, n1, _ = parse_traces([path])
        t1 = n1 * N_LAYERS * N_USED
        c1 = sum(sum(1 for _fi, ids in r1[l] for e in ids if e < args.n_res) for l in range(N_LAYERS)) / t1
        h1 = sum(sum(1 for _fi, ids in r1[l] for e in ids if e in hot[l]) for l in range(N_LAYERS)) / t1
        print(f"#     {path.split('/')[-1]}: rows={n1} control={c1:.4f} hotset={h1:.4f}")


TYPE_BLOCKS = {
    0: (1, 4), 1: (1, 2), 2: (32, 18), 3: (32, 20), 6: (32, 22), 7: (32, 24), 8: (32, 34),
    9: (32, 36), 10: (256, 84), 11: (256, 110), 12: (256, 144), 13: (256, 176), 14: (256, 210),
    15: (256, 292), 16: (256, 66), 17: (256, 74), 18: (256, 98), 19: (256, 50), 20: (32, 20),
    21: (256, 110), 22: (256, 82), 23: (256, 134), 30: (1, 2),
}


def expert_slab_bytes(gguf_path):
    with open(gguf_path, "rb") as f:
        assert f.read(4) == b"GGUF"
        (version,) = struct.unpack("<I", f.read(4))
        if version == 2:
            n_tensors, n_kv = struct.unpack("<II", f.read(8))
        else:
            n_tensors, n_kv = struct.unpack("<QQ", f.read(16))
        alignment = 32

        def rstr():
            (n,) = struct.unpack("<Q", f.read(8))
            return f.read(n).decode()

        def skip(t):
            if t == 8:
                rstr()
            elif t == 9:
                (et,) = struct.unpack("<I", f.read(4))
                (n,) = struct.unpack("<Q", f.read(8))
                for _ in range(n):
                    skip(et)
            else:
                f.read({0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}[t])

        for _ in range(n_kv):
            kname = rstr()
            (ktype,) = struct.unpack("<I", f.read(4))
            if kname == "general.alignment" and ktype == 4:
                alignment = struct.unpack("<I", f.read(4))[0]
            else:
                skip(ktype)
        slabs = {}
        for _ in range(n_tensors):
            name = rstr()
            (n_dims,) = struct.unpack("<I", f.read(4))
            ne = struct.unpack("<" + "Q" * n_dims, f.read(8 * n_dims))
            (ttype,) = struct.unpack("<I", f.read(4))
            f.read(8)
            for suf in ("ffn_gate_up_exps.weight", "ffn_down_exps.weight"):
                if name.endswith(suf):
                    blck, tsize = TYPE_BLOCKS[ttype]
                    assert ne[0] % blck == 0
                    row = ne[0] // blck * tsize
                    slabs[name] = row * ne[1] * ne[2] // N_EXPERT
        return slabs


def cmd_bytes(args):
    slabs = expert_slab_bytes(args.gguf)
    assert len(slabs) == 2 * N_LAYERS, f"{len(slabs)} expert tensors"
    rows, n_rows, _ = parse_traces([args.trace])
    hot = None
    if args.hotset:
        hs = json.load(open(args.hotset))
        hot = [set(h) for h in hs["hot"]]
    total_gib = 0.0
    miss_gib = 0.0
    for l in range(N_LAYERS):
        gu = slabs[f"blk.{l}.ffn_gate_up_exps.weight"]
        dn = slabs[f"blk.{l}.ffn_down_exps.weight"]
        for _fi, ids in rows[l]:
            for e in ids:
                total_gib += gu + dn
                resident = (e < args.n_res) if hot is None else (e in hot[l])
                if not resident:
                    miss_gib += gu + dn
    print(f"# trace {args.trace.split('/')[-1]} ({n_rows} positions): routed "
          f"{total_gib / 2**30:.2f} GiB of expert weights; miss (non-resident) "
          f"{miss_gib / 2**30:.3f} GiB")
    if args.drive_s:
        print(f"#   at drive {args.drive_s} s -> achieved {miss_gib / 2**30 / args.drive_s:.2f} GiB/s "
              f"(CPU store's mul_mat_id reads = miss bytes)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("hotset")
    h.add_argument("--traces", nargs="+", required=True)
    h.add_argument("--n-res", type=int, default=48)
    h.add_argument("--out-perm", required=True)
    h.add_argument("--out-hotset", required=True)
    c = sub.add_parser("coverage")
    c.add_argument("--traces", nargs="+", required=True)
    c.add_argument("--hotset", required=True)
    c.add_argument("--n-res", type=int, default=48)
    b = sub.add_parser("bytes")
    b.add_argument("--gguf", required=True)
    b.add_argument("--trace", required=True)
    b.add_argument("--n-res", type=int, default=48)
    b.add_argument("--hotset", default=None)
    b.add_argument("--drive-s", type=float, default=None)
    args = ap.parse_args()
    {"hotset": cmd_hotset, "coverage": cmd_coverage, "bytes": cmd_bytes}[args.cmd](args)
