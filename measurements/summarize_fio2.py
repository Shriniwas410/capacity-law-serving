import json, glob, os, re
d = "/home/shri/week1_fio"
print(f"{'job':<22} {'GB/s':>8}")
rows = []
for p in sorted(glob.glob(d + "/*.json")):
    n = os.path.basename(p)[:-5]
    raw = open(p, encoding="utf-8", errors="replace").read()
    i = raw.find("{")
    if i < 0:
        print(f"{n:<22} NOJSON")
        continue
    j = json.loads(raw[i:])
    bw = j["jobs"][0]["read"]["bw_bytes"] / 1e9
    err = j["jobs"][0].get("error", 0)
    rows.append((n, bw, err))
    print(f"{n:<22} {bw:8.3f}  err={err}")
print()
for prefix in ("mmap", "pread", "odirect"):
    xs = [b for n, b, e in rows if n.startswith(prefix)]
    if xs:
        print(f"{prefix:10} min {min(xs):.3f}  max {max(xs):.3f} GB/s")
od8 = [b for n, b, e in rows if n.startswith("odirect") and ("10m" in n or "16m" in n)]
print(f"G1 O_DIRECT >=8MB max {max(od8):.3f}  need >=2.0  -> {'PASS' if max(od8)>=2.0 else 'FAIL'}")
