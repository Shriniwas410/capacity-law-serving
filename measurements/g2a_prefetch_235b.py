#!/usr/bin/env python3
"""G2a on Qwen3-235B traces: LRU vs Belady vs causal prefetch.

Prefetch loads CHARGE a slot and evict LRU (Demand-MIN / ISCA'18). Objective
is DEMAND misses, not prefetch traffic. Oracle next-token prefetch is labeled
non-causal.

Gate G2a: some CAUSAL arm recovers >=10% of the LRU -> bounded-OPT (W=8) gap
at cap 12.5% (16 experts). Else W=0.

Usage:
  python measurements/g2a_prefetch_235b.py --traces measurements/traces --out measurements/g2a_235b.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, "/mnt/c/Users/shrin/Desktop/AI/sticky-moe")
from analyze import miss_per_tok

CAPS = (0.0625, 12 / 128, 0.125, 0.25)  # 8, 12, 16, 32 experts
LOADS_PER_TOK = None  # set from shape


def load_trace(path):
    z = np.load(path, allow_pickle=True)
    meta = json.loads(str(z["meta"]))
    return z["experts"], meta


def lru_with_prefetch(seq: np.ndarray, cap: int, kind: str):
    """seq (T, k). kind in {none, lag1, lag2, oracle_next}.
    Returns demand_misses_per_token, prefetch_loads_per_token."""
    T, k = seq.shape
    resident = []  # LRU: MRU at end
    demand = 0
    pf_loads = 0

    def insert(e: int):
        nonlocal resident
        if e in resident:
            resident.remove(e)
            resident.append(e)
            return False  # hit
        miss = True
        if len(resident) >= cap:
            resident.pop(0)
        resident.append(e)
        return miss

    for t in range(T):
        needed = [int(e) for e in seq[t]]
        for e in needed:
            if insert(e):
                demand += 1
        extra = []
        if kind == "none":
            extra = []
        elif kind == "lag1":
            extra = needed  # copy of current set; no-op if already resident
        elif kind == "lag2":
            extra = needed
            if t > 0:
                extra = extra + [int(e) for e in seq[t - 1]]
        elif kind == "oracle_next":
            if t + 1 < T:
                extra = [int(e) for e in seq[t + 1]]
        seen = set()
        for e in extra:
            if e in seen:
                continue
            seen.add(e)
            if e not in resident:
                pf_loads += 1
                insert(e)
    return demand / T, pf_loads / T


def layer_mean_prefetch(experts, cap, kind):
    L = experts.shape[0]
    d, p = [], []
    for l in range(L):
        a, b = lru_with_prefetch(experts[l], cap, kind)
        d.append(a)
        p.append(b)
    return float(np.mean(d)), float(np.mean(p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.traces, "*.npz")))
    if not paths:
        sys.exit(f"no npz in {args.traces}")

    report = {
        "traces_dir": os.path.abspath(args.traces),
        "n_experts": 128,
        "prefetch_charges_slots": True,
        "objective": "demand_misses",
        "oracle_next_noncausal": True,
        "domains": {},
        "started": time.time(),
    }
    t0 = time.time()
    recoveries_125 = []

    for path in paths:
        experts, meta = load_trace(path)
        E = int(meta.get("num_experts") or 128)
        L, T, k = experts.shape
        loads = L * k
        domain = os.path.splitext(os.path.basename(path))[0]
        rec = {"path": path, "meta": meta, "shape": [L, T, k], "loads_per_tok": loads, "by_cap": {}}
        print(f"[{domain}] L={L} T={T} k={k} loads/tok={loads}")
        for capf in CAPS:
            cap = max(1, int(round(capf * E)))
            lru = miss_per_tok(experts, capf, "lru", n_experts=E)
            static = miss_per_tok(experts, capf, "static", n_experts=E)
            opt = miss_per_tok(experts, capf, "belady", n_experts=E)
            opt8 = miss_per_tok(experts, capf, "belady", n_experts=E, lookahead=8)
            pf = {}
            for kind in ("none", "lag1", "lag2", "oracle_next"):
                dmean, pmean = layer_mean_prefetch(experts, cap, kind)
                # layer_mean is misses per layer; miss_per_tok sums layers
                # convert: lru_with_prefetch returns per-layer demand/T, mean over L
                # miss_per_tok returns sum_layers demand/T = L * per-layer
                pf[kind] = {
                    "demand_miss_per_tok_sum_layers": dmean * L,
                    "prefetch_loads_per_tok_mean_layer": pmean,
                }
            gap_opt8 = lru - opt8
            lag2_demand = pf["lag2"]["demand_miss_per_tok_sum_layers"]
            recov = None
            if gap_opt8 > 1e-9:
                recov = (lru - lag2_demand) / gap_opt8
            cell = {
                "cap_experts": cap,
                "cap_frac": capf,
                "miss": {
                    "lru": lru,
                    "static": static,
                    "belady": opt,
                    "belady_W8": opt8,
                    "lru_plus_lag1_prefetch": pf["lag1"]["demand_miss_per_tok_sum_layers"],
                    "lru_plus_lag2_prefetch": lag2_demand,
                    "lru_plus_oracle_next_prefetch": pf["oracle_next"]["demand_miss_per_tok_sum_layers"],
                },
                "hit_lru": 1.0 - lru / loads,
                "hit_opt": 1.0 - opt / loads,
                "hit_opt_W8": 1.0 - opt8 / loads,
                "prefetch": pf,
                "lag2_frac_of_LRU_W8_gap": recov,
            }
            rec["by_cap"][str(capf)] = cell
            print(
                f"  cap={cap:2d}  LRU={lru:.2f}  OPT={opt:.2f}  W8={opt8:.2f}  "
                f"lag2={lag2_demand:.2f}  recov={recov if recov is not None else float('nan'):.3f}"
            )
            if abs(capf - 0.125) < 1e-9:
                recoveries_125.append(recov if recov is not None else 0.0)
        report["domains"][domain] = rec

    mean_recov = float(np.mean(recoveries_125)) if recoveries_125 else None
    report["pooled"] = {
        "G2a_cap_12.5pct_lag2_frac_of_LRU_W8_gap": mean_recov,
        "G2a_causal_ge_10pct": bool(mean_recov is not None and mean_recov >= 0.10),
        "elapsed_s": round(time.time() - t0, 1),
        "note": (
            "lag1 persist-prefetch of the current set is nearly a no-op (already resident). "
            "lag2 union is the causal arm. oracle_next is NON-causal. "
            "Do not transplant 30B miss fractions. Not a tok/s claim."
        ),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    print(json.dumps(report["pooled"], indent=2))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
