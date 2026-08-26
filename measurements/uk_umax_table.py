#!/usr/bin/env python3
"""Print a compact (k, U_max) table from uk_umax_235b.json."""
import json
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "measurements/uk_umax_235b.json"
r = json.load(open(path, encoding="utf-8"))
print(f"{'domain':10s} {'k2@8':>6} {'k2@12':>6} {'k2@16':>6} {'k4@8':>6} {'k4@12':>6} {'k4@16':>6} {'k4@24':>6}  n2fit8  n2fit12 n4fit12 n4fit16  u/tok2 u/tok4")
rows = []
for d, rec in r["domains"].items():
    h, n = rec["horizon_prefix"], rec["naive_full_k"]

    def t(k, u):
        return h[str(k)][str(u)]["mean_admitted_T"]

    row = (
        t(2, 8), t(2, 12), t(2, 16),
        t(4, 8), t(4, 12), t(4, 16), t(4, 24),
        n["2"]["fit"]["8"], n["2"]["fit"]["12"],
        n["4"]["fit"]["12"], n["4"]["fit"]["16"],
        n["2"]["unique_per_emitted"], n["4"]["unique_per_emitted"],
        h["2"]["12"]["demand_vs_greedy"],
        h["4"]["12"]["demand_vs_greedy"],
        h["4"]["16"]["demand_vs_greedy"],
    )
    rows.append(row)
    print(
        f"{d:10s} {row[0]:6.3f} {row[1]:6.3f} {row[2]:6.3f} "
        f"{row[3]:6.3f} {row[4]:6.3f} {row[5]:6.3f} {row[6]:6.3f}  "
        f"{row[7]:6.4f} {row[8]:7.3f} {row[9]:7.3f} {row[10]:7.3f}  "
        f"{row[11]:6.2f} {row[12]:6.2f}"
    )

def mean(i):
    return sum(x[i] for x in rows) / len(rows)

print("--- pooled mean_T / naive fit / unique-per-emitted (greedy=8) ---")
print(
    f"HSpec mean_T  k=2 Umax 8/12/16 = {mean(0):.3f} / {mean(1):.3f} / {mean(2):.3f}"
)
print(
    f"HSpec mean_T  k=4 Umax 8/12/16/24 = {mean(3):.3f} / {mean(4):.3f} / {mean(5):.3f} / {mean(6):.3f}"
)
print(f"naive k=2 fit U<=8 / U<=12 / = {mean(7):.4f} / {mean(8):.3f}")
print(f"naive k=4 fit U<=12 / U<=16 = {mean(9):.3f} / {mean(10):.3f}")
print(f"naive unique/tok k=2 / k=4 = {mean(11):.2f} / {mean(12):.2f}")
print(
    f"HSpec demand_vs_greedy k2@12={mean(13):.3f} k4@12={mean(14):.3f} k4@16={mean(15):.3f}"
)
print(json.dumps(r["pooled"], indent=2))
