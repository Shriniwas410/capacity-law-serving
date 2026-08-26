#!/usr/bin/env python3
"""Engine-semantic HorizonSpec T: min over 94 layers vs F4 mean-over-layers."""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np


def token_sets(experts_l):
    return [frozenset(int(e) for e in experts_l[t]) for t in range(experts_l.shape[0])]


def admit_prefix(sets, t0, k_max, u_max):
    u = set(sets[t0])
    if len(u) > u_max:
        return 0
    admitted = 1
    for j in range(1, k_max):
        nxt = u | sets[t0 + j]
        if len(nxt) <= u_max:
            u = nxt
            admitted = j + 1
        else:
            break
    return admitted


def main():
    traces = sys.argv[1] if len(sys.argv) > 1 else "measurements/traces"
    k_max = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    u_maxs = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else [12]
    out = {"k_max": k_max, "by_umax": {}}
    for u_max in u_maxs:
        doms = {}
        print("==== U_max", u_max, "k", k_max)
        for path in sorted(glob.glob(os.path.join(traces, "*.npz"))):
            experts = np.load(path, allow_pickle=True)["experts"]
            L, T, _ = experts.shape
            n = T - k_max + 1
            layer_sets = [token_sets(experts[l]) for l in range(L)]
            min_T = np.empty(n, dtype=np.int16)
            mean_T = np.empty(n, dtype=np.float64)
            n_overflow = np.empty(n, dtype=np.int16)
            for t in range(n):
                ts = np.array(
                    [admit_prefix(layer_sets[l], t, k_max, u_max) for l in range(L)],
                    dtype=np.int16,
                )
                min_T[t] = int(ts.min())
                mean_T[t] = float(ts.mean())
                n_overflow[t] = int((ts < 2).sum())
            rec = {
                "n_windows": int(n),
                "n_layers": int(L),
                "F4_style_mean_T": float(mean_T.mean()),
                "engine_min_layer_mean_T": float(min_T.mean()),
                "p_engine_T_ge_2": float((min_T >= 2).mean()),
                "mean_overflow_layers": float(n_overflow.mean()),
                "p50_overflow_layers": float(np.median(n_overflow)),
            }
            name = os.path.splitext(os.path.basename(path))[0]
            doms[name] = rec
            print(
                name,
                "F4_mean_T={:.3f}".format(rec["F4_style_mean_T"]),
                "engine_min_T={:.3f}".format(rec["engine_min_layer_mean_T"]),
                "p_minT_ge2={:.4f}".format(rec["p_engine_T_ge_2"]),
                "mean_overflow_L={:.1f}".format(rec["mean_overflow_layers"]),
            )
        keys = list(doms.values())
        pooled = {
            k: float(np.mean([r[k] for r in keys]))
            for k in (
                "F4_style_mean_T",
                "engine_min_layer_mean_T",
                "p_engine_T_ge_2",
                "mean_overflow_layers",
            )
        }
        print("POOLED", json.dumps(pooled))
        out["by_umax"][str(u_max)] = {"domains": doms, "pooled": pooled}
    dest = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_min_layer_T.json")
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
        f.write("\n")
    print("wrote", dest)


if __name__ == "__main__":
    main()
