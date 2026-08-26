#!/usr/bin/env python3
"""U(k) expert-union curves from routing traces.

U(k) = mean |union of top-k expert sets over k consecutive tokens|,
averaged over layers and time. G2b gate: U(4) <= 16 on the *model the
number will be cited for* — 30B numbers are not 235B numbers.

Usage:
  python measurements/uk_union.py --traces /path/to/npz_dir --out measurements/uk_30b.json
"""
from __future__ import annotations

import argparse
import json
import glob
import os
import sys

import numpy as np


KS = (1, 2, 3, 4, 6, 8)


def load_trace(path):
    z = np.load(path, allow_pickle=True)
    meta = json.loads(str(z["meta"]))
    return z["experts"], meta  # (L, T, k)


def union_curve(experts: np.ndarray, ks=KS):
    L, T, k = experts.shape
    # per-token frozensets; int16 ids
    sets = [[frozenset(int(e) for e in experts[l, t]) for t in range(T)] for l in range(L)]
    out = {}
    for kk in ks:
        if kk > T:
            continue
        layer_means = []
        for l in range(L):
            sizes = []
            for t in range(T - kk + 1):
                u = set()
                for j in range(kk):
                    u |= sets[l][t + j]
                sizes.append(len(u))
            layer_means.append(float(np.mean(sizes)))
        arr = np.asarray(layer_means)
        out[str(kk)] = {
            "mean": float(arr.mean()),
            "std_over_layers": float(arr.std()),
            "min_layer": float(arr.min()),
            "max_layer": float(arr.max()),
            "demand_vs_naive": float(arr.mean() / (k * kk)),  # 1.0 = no sharing
        }
    return out


def lag1(experts: np.ndarray, n_experts: int):
    L, T, k = experts.shape
    reuse = []
    chance = []
    jaccard = []
    for l in range(L):
        a = [set(int(e) for e in experts[l, t]) for t in range(T)]
        inter = [len(a[t] & a[t + 1]) for t in range(T - 1)]
        uni = [len(a[t] | a[t + 1]) for t in range(T - 1)]
        reuse.append(float(np.mean(inter) / k))
        jaccard.append(float(np.mean([i / u if u else 0.0 for i, u in zip(inter, uni)])))
        # frequency chance: sum p_e^2 / (k/E * E) = sum p_e^2 / mean_active
        # token incidence p_e = fraction of tokens where e is active
        inc = np.zeros(n_experts, dtype=np.float64)
        for t in range(T):
            for e in a[t]:
                if 0 <= e < n_experts:
                    inc[e] += 1
        inc /= T
        # E[|A_t ∩ A_{t+1}|]/k = (sum_e inc_e^2) / k
        chance.append(float((inc ** 2).sum() / k))
    return {
        "reuse": float(np.mean(reuse)),
        "chance": float(np.mean(chance)),
        "ratio": float(np.mean(reuse) / max(np.mean(chance), 1e-12)),
        "jaccard": float(np.mean(jaccard)),
    }


def sanity(experts: np.ndarray, n_experts: int):
    """Catch the argsort-view bug: every id appearing exactly k times, or ids out of range."""
    flat = experts.reshape(-1)
    bad = int(((flat < 0) | (flat >= n_experts)).sum())
    uniq, counts = np.unique(flat, return_counts=True)
    # tell-tale of the view bug: every expert appears exactly k * T * something uniform
    return {
        "shape": list(experts.shape),
        "id_min": int(flat.min()),
        "id_max": int(flat.max()),
        "n_out_of_range": bad,
        "n_unique_ids": int(len(uniq)),
        "count_min": int(counts.min()),
        "count_max": int(counts.max()),
        "count_std": float(counts.std()),
        "uniform_8x_bug": bool(len(uniq) == n_experts and counts.min() == counts.max()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-experts", type=int, default=0)
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.traces, "*.npz")))
    if not paths:
        sys.exit(f"no npz in {args.traces}")

    report = {"traces_dir": os.path.abspath(args.traces), "domains": {}}
    u4_means = []
    for path in paths:
        experts, meta = load_trace(path)
        n_experts = args.num_experts or int(meta.get("num_experts") or 0)
        if not n_experts:
            n_experts = int(experts.max()) + 1
        domain = os.path.splitext(os.path.basename(path))[0]
        san = sanity(experts, n_experts)
        u = union_curve(experts)
        lag = lag1(experts, n_experts)
        rec = {
            "path": path,
            "meta": meta,
            "sanity": san,
            "U(k)": u,
            "lag1": lag,
            "G2b_U4_le_16": bool(u.get("4", {}).get("mean", 999) <= 16),
        }
        report["domains"][domain] = rec
        if "4" in u:
            u4_means.append(u["4"]["mean"])
        print(
            f"{domain:12s}  shape={tuple(experts.shape)}  "
            f"U(4)={u.get('4', {}).get('mean', float('nan')):.2f}  "
            f"U(4)/32={u.get('4', {}).get('demand_vs_naive', float('nan')):.3f}  "
            f"lag1_reuse={lag['reuse']:.3f} vs chance {lag['chance']:.3f} ({lag['ratio']:.2f}x)  "
            f"unique_ids={san['n_unique_ids']}  uniform_bug={san['uniform_8x_bug']}"
        )

    report["pooled"] = {
        "U4_mean_over_domains": float(np.mean(u4_means)) if u4_means else None,
        "G2b_U4_le_16": bool(u4_means and float(np.mean(u4_means)) <= 16),
        "note": "30B U(k) is NOT a 235B result. Do not transplant.",
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    print(f"wrote {args.out}")
    print(json.dumps(report["pooled"], indent=2))


if __name__ == "__main__":
    main()
