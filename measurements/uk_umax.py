#!/usr/bin/env python3
"""HorizonSpec (k, U_max) tables on routing traces.

Oracle = the *target* model's recorded top-8 sets. That is an upper bound on
union sharing if a drafter routed identically; a real 0.6B draft will match
less. Admission is longest prefix (draft-simple), not a tree.

Engine constraint on this box (PR#25294 stream cache `8s`):
  n_slots = 8. A verify batch whose unique-expert count exceeds n_slots aborts
  unless waves are enabled, and waves currently require 24 slots/layer (~24 GB
  on 235B). So U_max must be <= 8 for stock k>1 verify to even *run* here.

U_max=8 with top-8 routing admits k>1 only when consecutive tokens use the
*same* 8 experts (identical sets). That rate is the number that matters.

Demand accounting (α=1 oracle): unique experts per emitted token = U / T.
Greedy is 8. Ratio < 1 is an IO-byte win *if* the batch fits in cache.
Does not include LRU hits, GEMM, or draft cost. Not a tok/s claim.

Usage:
  python measurements/uk_umax.py --traces measurements/traces --out measurements/uk_umax_235b.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from uk_union import load_trace, sanity

KS = (1, 2, 3, 4, 6, 8)
UMAXS = (8, 12, 16, 24)
N_SLOTS_BOX = 8
# Qwen3-235B-A22B Q4_K_M: ~127.5 GB expert table / (94 layers * 128 experts)
EXPERT_MB = 127.5 * 1024.0 / (94.0 * 128.0)  # ~10.61 MiB


def token_sets(experts_l: np.ndarray):
    T = experts_l.shape[0]
    return [frozenset(int(e) for e in experts_l[t]) for t in range(T)]


def union_size(sets, t0, kk):
    u = set()
    for j in range(kk):
        u |= sets[t0 + j]
    return len(u)


def naive_fit(sets, kk, caps):
    T = len(sets)
    if kk > T:
        return None
    n = T - kk + 1
    sizes = np.empty(n, dtype=np.float64)
    ident_lag1 = 0
    ident_denom = 0
    for t in range(n):
        sizes[t] = union_size(sets, t, kk)
        if kk >= 2:
            ident_denom += 1
            if sets[t] == sets[t + 1]:
                ident_lag1 += 1
    out = {
        "mean_U": float(sizes.mean()),
        "p50_U": float(np.median(sizes)),
        "p90_U": float(np.percentile(sizes, 90)),
        "max_U": float(sizes.max()),
        "demand_vs_greedy": float(sizes.mean() / (8.0 * kk)),  # unique/tok vs 8
        "unique_per_emitted": float(sizes.mean() / kk),
        "fit": {},
    }
    for cap in caps:
        out["fit"][str(cap)] = float((sizes <= cap).mean())
    if kk >= 2:
        out["p_identical_first_two"] = float(ident_lag1 / max(ident_denom, 1))
    return out


def admit_prefix(sets, t0, k_max, u_max):
    """Longest prefix of k_max tokens whose union <= u_max. Always >= 1 if |s0|<=u_max."""
    u = set(sets[t0])
    if len(u) > u_max:
        return 0, len(u)
    admitted = 1
    for j in range(1, k_max):
        nxt = u | sets[t0 + j]
        if len(nxt) <= u_max:
            u = nxt
            admitted = j + 1
        else:
            break
    return admitted, len(u)


def horizon_table(sets, k_max, u_max):
    T = len(sets)
    if k_max > T:
        return None
    n = T - k_max + 1
    admitted = np.empty(n, dtype=np.int16)
    unions = np.empty(n, dtype=np.float64)
    full_k = 0
    for t in range(n):
        a, u = admit_prefix(sets, t, k_max, u_max)
        admitted[t] = a
        unions[t] = u
        if a == k_max:
            full_k += 1
    mean_t = float(admitted.mean())
    mean_u = float(unions.mean())
    # skip windows that could not even admit token 0 (should not happen at U_max>=8)
    ok = admitted > 0
    unique_per = float(unions[ok].mean() / admitted[ok].mean()) if ok.any() else float("nan")
    return {
        "mean_admitted_T": mean_t,
        "mean_U_admitted": mean_u,
        "p_full_k": float(full_k / n),
        "unique_per_emitted": unique_per,
        "demand_vs_greedy": unique_per / 8.0,
        "p_T_eq_1": float((admitted == 1).mean()),
        "p_T_ge_2": float((admitted >= 2).mean()),
    }


def layer_reduce(experts, fn):
    """fn(sets) -> dict of floats/dicts; mean over layers for scalars."""
    L = experts.shape[0]
    recs = []
    for l in range(L):
        recs.append(fn(token_sets(experts[l])))
    recs = [r for r in recs if r is not None]
    if not recs:
        return None
    keys = recs[0].keys()
    out = {}
    for k in keys:
        vals = [r[k] for r in recs]
        if isinstance(vals[0], dict):
            subkeys = vals[0].keys()
            out[k] = {
                sk: float(np.mean([v[sk] for v in vals]))
                for sk in subkeys
            }
        else:
            arr = np.asarray(vals, dtype=np.float64)
            out[k] = float(arr.mean())
    return out


def domain_report(experts, meta):
    L, T, k = experts.shape
    n_layers = L
    greedy_gb = n_layers * 8 * EXPERT_MB / 1024.0  # GB/tok unique-expert demand
    naive = {}
    for kk in KS:
        naive[str(kk)] = layer_reduce(
            experts, lambda s, kk=kk: naive_fit(s, kk, UMAXS)
        )
    horiz = {}
    for kk in KS:
        horiz[str(kk)] = {}
        for um in UMAXS:
            horiz[str(kk)][str(um)] = layer_reduce(
                experts, lambda s, kk=kk, um=um: horizon_table(s, kk, um)
            )
    # identical-set stickiness (lag-1), pooled over layers
    ident = []
    for l in range(L):
        s = token_sets(experts[l])
        ident.append(float(np.mean([s[t] == s[t + 1] for t in range(T - 1)])))
    p_ident = float(np.mean(ident))

    # engine GO/NO-GO at this box's 8 slots: naive k=2/4 must fit
    n2 = naive["2"]["fit"]["8"] if naive.get("2") else 0.0
    n4 = naive["4"]["fit"]["8"] if naive.get("4") else 0.0
    h2 = horiz["2"]["8"] if horiz.get("2") else {}
    return {
        "meta": meta,
        "n_layers": L,
        "n_tokens": T,
        "top_k": k,
        "expert_mb": EXPERT_MB,
        "greedy_unique_GB_per_tok": greedy_gb,
        "p_identical_lag1_sets": p_ident,
        "naive_full_k": naive,
        "horizon_prefix": horiz,
        "box_8slot": {
            "n_slots": N_SLOTS_BOX,
            "waves_require_slots": 24,
            "naive_k2_fit_frac": n2,
            "naive_k4_fit_frac": n4,
            "horizon_k2_Umax8_mean_T": h2.get("mean_admitted_T"),
            "horizon_k2_Umax8_p_T_ge_2": h2.get("p_T_ge_2"),
            "note": (
                "naive k>1 verify fits 8-slot cache only when U(k)<=8, i.e. all "
                "k tokens share an identical 8-expert set. HorizonSpec U_max=8 "
                "truncates to T=1 otherwise."
            ),
        },
    }


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
        "n_slots_this_box": N_SLOTS_BOX,
        "U_max_grid": list(UMAXS),
        "k_grid": list(KS),
        "oracle": "target-model recorded routing (upper bound on union sharing)",
        "domains": {},
    }
    p_ident = []
    k2_fit8 = []
    k2_t = []
    k4_fit8 = []
    k4_t_u12 = []
    for path in paths:
        experts, meta = load_trace(path)
        n_experts = int(meta.get("num_experts") or 128)
        domain = os.path.splitext(os.path.basename(path))[0]
        san = sanity(experts, n_experts)
        rec = domain_report(experts, meta)
        rec["path"] = path
        rec["sanity"] = san
        report["domains"][domain] = rec
        p_ident.append(rec["p_identical_lag1_sets"])
        k2_fit8.append(rec["box_8slot"]["naive_k2_fit_frac"])
        k2_t.append(rec["box_8slot"]["horizon_k2_Umax8_mean_T"])
        k4_fit8.append(rec["box_8slot"]["naive_k4_fit_frac"])
        k4_t_u12.append(rec["horizon_prefix"]["4"]["12"]["mean_admitted_T"])
        print(
            f"{domain:12s}  ident_lag1={rec['p_identical_lag1_sets']:.4f}  "
            f"naive k=2 fit@8={rec['box_8slot']['naive_k2_fit_frac']:.4f}  "
            f"HSpec k=2 Umax=8 mean_T={rec['box_8slot']['horizon_k2_Umax8_mean_T']:.3f}  "
            f"naive k=4 fit@8={rec['box_8slot']['naive_k4_fit_frac']:.4f}  "
            f"HSpec k=4 Umax=12 mean_T={rec['horizon_prefix']['4']['12']['mean_admitted_T']:.3f}  "
            f"U(4)/4 unique/tok={rec['naive_full_k']['4']['unique_per_emitted']:.2f}"
        )

    report["pooled"] = {
        "p_identical_lag1_sets": float(np.mean(p_ident)),
        "naive_k2_fit_8slot": float(np.mean(k2_fit8)),
        "horizon_k2_Umax8_mean_T": float(np.mean(k2_t)),
        "naive_k4_fit_8slot": float(np.mean(k4_fit8)),
        "horizon_k4_Umax12_mean_T": float(np.mean(k4_t_u12)),
        "G2b_engine_8slot": (
            "NO-GO: naive k>=2 does not fit 8-slot stream cache "
            "(needs identical expert sets or 24-slot waves)."
        ),
        "note": (
            "Do not cite as tok/s. Unique-expert demand only, target-oracle routing. "
            "G2b U(4)<=16 is a different gate (mean union, not 8-slot fit)."
        ),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    print(f"wrote {args.out}")
    print(json.dumps(report["pooled"], indent=2))


if __name__ == "__main__":
    main()
