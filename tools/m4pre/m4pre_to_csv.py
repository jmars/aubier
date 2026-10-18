#!/usr/bin/env python3
"""m4pre_to_csv.py RAW CSV — flatten m4pre_contention.sh's raw row log into CSV."""
import re
import sys

ARM = re.compile(r'^(\S+)\s+bytes=(\d+) workers=(\d+)\s+GiBps_completion=([\d.]+) \(sec=([\d.]+)\)'
                 r'\s+GiBps_med=([\d.]+) \(sec=([\d.]+)\)\s+GiBps_bestthread=([\d.]+)'
                 r'(?:\s+GiBps_worst=([\d.]+))?\s+sec_min=([\d.]+) sec_max=([\d.]+)')
SUM = re.compile(r'^combined_concurrent\s+GiBps_sum=([\d.]+)\s+\(cpu_completion ([\d.]+) \+ copy ([\d.]+)\)'
                 r'\s+sec_cpu=([\d.]+) sec_copy=([\d.]+) ratio_copy_over_cpu=([\d.]+)')

HDR = ["label", "mode", "source", "copy_source", "threads", "chunk_kib", "arm", "bytes", "workers",
       "GiBps_completion", "GiBps_med", "GiBps_bestthread", "GiBps_worst", "sec_min", "sec_max",
       "load1_before", "load1_after", "row_rc"]


def main():
    src, dst = sys.argv[1], sys.argv[2]
    out = [",".join(HDR)]
    cur = {}

    def flush():
        if not cur:
            return
        for arm, v in cur["arms"].items():
            out.append(",".join(str(x) for x in
                                [cur["label"], cur["mode"], cur["source"], cur["copy_source"],
                                 cur["threads"], cur["chunk_kib"], arm] + v +
                                [cur["load_before"], cur["load_after"], cur["rc"]]))
        for k, v in cur["extra"].items():
            out.append(",".join(str(x) for x in
                                [cur["label"], cur["mode"], cur["source"], cur["copy_source"],
                                 cur["threads"], cur["chunk_kib"], k, "", "", v] + [""] * 5 +
                                [cur["load_before"], cur["load_after"], cur["rc"]]))
        cur.clear()

    for line in open(src):
        if line.startswith("# m4pre "):
            flush()
            m = dict(re.findall(r'(\w+)=(\S+)', line))
            cur.update({"label": m.get("label", ""), "mode": m.get("mode", ""),
                        "source": m.get("source", ""), "copy_source": m.get("copy_source", ""),
                        "threads": m.get("threads", ""), "chunk_kib": m.get("chunk_kib", ""),
                        "arms": {}, "extra": {}, "load_before": "", "load_after": "", "rc": ""})
        elif not cur:
            continue
        elif line.startswith("# device:"):
            m = re.search(r'load1_before=(\S+)', line)
            if m:
                cur["load_before"] = m.group(1)
        elif line.startswith("# load1_after="):
            cur["load_after"] = line.split("=")[1].strip()
        elif line.startswith("# row_rc="):
            cur["rc"] = line.split("=")[1].strip()
            flush()
        else:
            m = ARM.match(line)
            if m:
                cur["arms"][m.group(1)] = [m.group(2), m.group(3), m.group(4), m.group(6),
                                           m.group(8), m.group(9) or "", m.group(10), m.group(11)]
                continue
            m = SUM.match(line)
            if m:
                cur["extra"]["combined_GiBps_sum"] = m.group(1)
    flush()
    with open(dst, "w") as f:
        f.write("\n".join(out) + "\n")
    print("wrote %s (%d lines)" % (dst, len(out)))


if __name__ == "__main__":
    main()
