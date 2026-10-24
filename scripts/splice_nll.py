#!/usr/bin/env python3
"""splice_nll.py — concatenate per-chunk .nll files into ONE side .nll.

The instrument's `compare-nll` takes two compact files; a chunked corpus is concatenated in
CHUNK ORDER with positions renumbered globally (the instrument ignores the pos column for the
statistic; the targets must line up, which chunk order preserves).  Same splice arithmetic as
run-chunked-ext.sh (the L58 pattern).

    splice.py <out.nll> <in.nll> [<in.nll> ...]
"""
import sys

out_path = sys.argv[1]
base = 0
with open(out_path, 'w') as out:
    out.write('# pos target nll  (spliced from %d chunk .nll files in order)\n' % (len(sys.argv) - 2))
    for p in sys.argv[2:]:
        n = 0
        with open(p) as f:
            for ln in f:
                if ln.startswith('#') or not ln.strip():
                    continue
                w = ln.split()
                out.write('%d %s %s\n' % (base + n, w[1], w[2]))
                n += 1
        base += n
print('spliced %d positions -> %s' % (base, out_path))
