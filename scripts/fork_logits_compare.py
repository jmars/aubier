#!/usr/bin/env python3
# fork_logits_compare.py — compare two moe-logits-dump outputs (ref vs cand).
# Metrics: top-token agreement %, mean KL(cand||ref) in nats, cross-perplexity
# of cand against ref, max |logit| delta, first divergence position, and the
# per-position divergences (near-tie vs substantive, by logit gap).
#
# Usage: fork_logits_compare.py ref.logits cand.logits [ref.tokens cand.tokens]
import sys, struct, math

def read_logits(path):
    with open(path, 'rb') as f:
        n_decode = struct.unpack('<Q', f.read(8))[0]
        n_vocab = struct.unpack('<I', f.read(4))[0]
        rows = []
        for _ in range(n_decode):
            rows.append(struct.unpack('<%df' % n_vocab, f.read(4*n_vocab)))
    return n_vocab, rows

def read_tokens(path):
    with open(path, 'rb') as f:
        n = struct.unpack('<I', f.read(4))[0]
        return struct.unpack('<%di' % n, f.read(4*n))

def logsumexp(v):
    m = max(v)
    return m + math.log(sum(math.exp(x - m) for x in v))

def main():
    ref_path, cand_path = sys.argv[1], sys.argv[2]
    ref_tok_path = sys.argv[3] if len(sys.argv) > 3 else ref_path + '.tokens'
    cand_tok_path = sys.argv[4] if len(sys.argv) > 4 else cand_path + '.tokens'

    nv_r, ref = read_logits(ref_path)
    nv_c, cand = read_logits(cand_path)
    assert nv_r == nv_c, "n_vocab mismatch"
    nv = nv_r
    n = min(len(ref), len(cand))
    assert n > 0, "empty logits"

    ref_toks = read_tokens(ref_tok_path)
    cand_toks = read_tokens(cand_tok_path)

    agrees = 0
    total_kl = 0.0
    total_ce = 0.0   # cross-entropy of cand under ref, in nats
    max_abs_delta = 0.0
    first_div = None
    divergences = []

    for i in range(n):
        r, c = ref[i], cand[i]
        rt, ct = ref_toks[i] if i < len(ref_toks) else None, cand_toks[i] if i < len(cand_toks) else None
        r_top = max(range(nv), key=lambda j: r[j])
        c_top = max(range(nv), key=lambda j: c[j])
        if r_top == c_top:
            agrees += 1
        else:
            if first_div is None:
                first_div = i
            # near-tie if the ref's top-2 logit gap is tiny; substantive otherwise
            r_sorted = sorted(r, reverse=True)
            gap = r_sorted[0] - r_sorted[1]
            c_top_gap = c[c_top] - r[c_top]
            divergences.append((i, rt, ct, gap, c_top_gap))

        # KL(cand || ref) = sum_j cand_softmax_j * (log cand_softmax_j - log ref_softmax_j)
        lr = logsumexp(r); lc = logsumexp(c)
        kl = 0.0; ce = 0.0
        for j in range(nv):
            pr = math.exp(r[j] - lr)
            pc = math.exp(c[j] - lc)
            ce += pr * (c[j] - lc)   # cross-entropy: ref distribution, cand log-probs
            if pc > 0:
                kl += pc * ((c[j] - lc) - (r[j] - lr))
        total_kl += kl
        total_ce += -ce
        d = max(abs(c[j] - r[j]) for j in range(nv))
        max_abs_delta = max(max_abs_delta, d)

    print(f"positions            : {n}")
    print(f"top-token agreement  : {agrees}/{n} = {100.0*agrees/n:.4f}%")
    print(f"mean KL(cand||ref)   : {total_kl/n:.6f} nats")
    print(f"cross-entropy (ref)  : {total_ce/n:.6f} nats  -> cand perp vs ref = {math.exp(total_ce/n):.6f}")
    print(f"max |logit| delta    : {max_abs_delta:.6f}")
    print(f"first divergence     : {first_div if first_div is not None else 'none'}")
    if divergences:
        print(f"divergent positions  : {len(divergences)}")
        near = sum(1 for d in divergences if d[3] < 0.02)
        print(f"  near-tie (ref gap<0.02): {near}, substantive: {len(divergences)-near}")
        for d in divergences[:10]:
            print(f"    pos {d[0]}: ref_tok={d[1]} cand_tok={d[2]} ref_gap={d[3]:.5f} cand_ref_delta={d[4]:.5f}")

if __name__ == '__main__':
    main()
