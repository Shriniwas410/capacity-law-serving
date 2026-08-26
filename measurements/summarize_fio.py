#!/usr/bin/env python3
"""Summarize week1 fio JSON, skipping empty/corrupt files."""
import json, glob, os, sys
out = sys.argv[1]
rows = []
for p in sorted(glob.glob(os.path.join(out, "*.json"))):
    if os.path.getsize(p) < 100:
        continue
    try:
        with open(p) as f:
            j = json.load(f)
    except json.JSONDecodeError:
        continue
    job = j["jobs"][0]
    bw = job["read"]["bw_bytes"] / 1e9
    iops = job["read"]["iops"]
    lat = job["read"]["lat_ns"]["mean"] / 1e6
    rows.append((os.path.basename(p)[:-5], bw, iops, lat))
print(f"{'job':<22} {'GB/s':>8} {'iops':>10} {'lat_ms':>8}")
for n, b, i, l in rows:
    print(f"{n:<22} {b:8.3f} {i:10.1f} {l:8.2f}")
tsv = os.path.join(out, "summary.tsv")
with open(tsv, "w") as f:
    f.write("job\tGBps\tiops\tlat_ms\n")
    for n, b, i, l in rows:
        f.write(f"{n}\t{b:.4f}\t{i:.1f}\t{l:.2f}\n")
print("wrote", tsv)
