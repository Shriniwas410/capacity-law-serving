#!/usr/bin/env python3
"""Emit docs/data.js from measurements/*.json so the Pages explainer stays sourced."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
M = ROOT / "measurements"


def load(name: str):
    return json.loads((M / name).read_text(encoding="utf-8"))


def mean(xs):
    return sum(xs) / len(xs)


def main() -> None:
    uk = load("uk_235b.json")
    um = load("uk_umax_235b.json")
    mn = load("_min_layer_T.json")
    spec_ub1 = load("spec235b_verdict.json")
    spec_ub8 = load("spec235b_ub8/verdict.json")
    spec_umax = load("spec235b_umax/verdict.json")
    greedy8 = load("ab235b_verdict.json")
    greedy12 = load("ab235b_12s_verdict.json")

    domains = ["code", "general", "math", "medical"]
    ks = ["1", "2", "3", "4", "6", "8"]
    slots = ["8", "12", "16", "24"]

    uk_series = {}
    lag1 = {}
    for d in domains:
        uk_series[d] = {k: uk["domains"][d]["U(k)"][k]["mean"] for k in ks}
        lag1[d] = uk["domains"][d]["lag1"]

    naive_fit = {s: {"k2": [], "k4": []} for s in slots}
    hspec = {s: {"k2": [], "k4": []} for s in slots}
    for d in domains:
        rec = um["domains"][d]
        for s in slots:
            naive_fit[s]["k2"].append(rec["naive_full_k"]["2"]["fit"][s])
            naive_fit[s]["k4"].append(rec["naive_full_k"]["4"]["fit"][s])
            hspec[s]["k2"].append(rec["horizon_prefix"]["2"][s]["mean_admitted_T"])
            hspec[s]["k4"].append(rec["horizon_prefix"]["4"][s]["mean_admitted_T"])

    naive_out = {
        s: {"k2": mean(naive_fit[s]["k2"]), "k4": mean(naive_fit[s]["k4"])} for s in slots
    }
    hspec_out = {
        s: {"k2": mean(hspec[s]["k2"]), "k4": mean(hspec[s]["k4"])} for s in slots
    }

    engine = {}
    for s, rec in mn["by_umax"].items():
        p = rec["pooled"]
        engine[s] = {
            "mean_T": p["engine_min_layer_mean_T"],
            "p_T_ge_2": p["p_engine_T_ge_2"],
            "overflow_layers": p["mean_overflow_layers"],
            "F4_mean_T": p["F4_style_mean_T"],
        }

    expert_mb = um["domains"]["code"]["expert_mb"]
    out = {
        "title": "Capacity-law serving of Qwen3-235B-A22B",
        "repo": "https://github.com/Shriniwas410/capacity-law-serving",
        "paper": "paper/paper.pdf",
        "geometry": {
            "model": "Qwen3-235B-A22B-Q4_K_M",
            "n_moe_layers": 94,
            "n_experts": 128,
            "top_k": 8,
            "expert_mb": expert_mb,
            "gguf_gb": 133,
            "mmap_repack_mb": 105495.75,
            "trace_shape": [94, 2048, 8],
        },
        "uk": uk_series,
        "uk_pooled_U4": uk["pooled"]["U4_mean_over_domains"],
        "lag1": lag1,
        "p_identical_lag1_sets": um["pooled"]["p_identical_lag1_sets"],
        "naive_fit": naive_out,
        "hspec_mean_T": hspec_out,
        "engine_min": engine,
        "greedy": {
            "s8_tok_s": greedy8["arms"]["stream_direct"]["eval_tok_s"],
            "s12_tok_s": greedy12["cache_12s"]["eval_tok_s"],
            "s12_rss_gib": greedy12["cache_12s"]["max_rss_gb"],
            "s8_rss_gib": greedy12["cache_8s"]["max_rss_gb"],
        },
        "spec": {
            "ub1": {
                "baseline": spec_ub1["arms"]["baseline"]["eval_tok_s"],
                "k2_tok_s": spec_ub1["arms"]["spec_k2"]["eval_tok_s"],
                "k2_alpha": spec_ub1["arms"]["spec_k2"]["alpha"],
                "k2_x": spec_ub1["arms"]["spec_k2"]["speedup_vs_baseline"],
                "k4_tok_s": spec_ub1["arms"]["spec_k4"]["eval_tok_s"],
                "k4_alpha": spec_ub1["arms"]["spec_k4"]["alpha"],
                "k4_x": spec_ub1["arms"]["spec_k4"]["speedup_vs_baseline"],
            },
            "ub8": {
                "baseline": spec_ub8["arms"]["baseline"]["eval_tok_s"],
                "k2_tok_s": spec_ub8["arms"]["spec_k2"]["eval_tok_s"],
                "k2_alpha": spec_ub8["spec_k2_alpha"],
                "k2_x": spec_ub8["spec_k2_speedup"],
                "k4_tok_s": spec_ub8["arms"]["spec_k4"]["eval_tok_s"],
                "k4_alpha": spec_ub8["spec_k4_alpha"],
                "k4_x": spec_ub8["spec_k4_speedup"],
            },
            "umax": {
                "k2_tok_s": spec_umax["arms"]["spec_k2"]["eval_tok_s"],
                "k2_alpha": spec_umax["spec_k2_alpha"],
                "k2_x": spec_umax["spec_k2_speedup"],
                "k4_tok_s": spec_umax["arms"]["spec_k4"]["eval_tok_s"],
                "k4_alpha": spec_umax["spec_k4_alpha"],
                "k4_x": spec_umax["spec_k4_speedup"],
            },
        },
        "io_odirect_qd8_gbs": [2.368, 2.339, 2.354],
        "files": {
            "uk": "measurements/uk_235b.json",
            "umax": "measurements/uk_umax_235b.json",
            "engine_min": "measurements/_min_layer_T.json",
            "ub1": "measurements/spec235b_verdict.json",
            "ub8": "measurements/spec235b_ub8/verdict.json",
            "umax_run": "measurements/spec235b_umax/verdict.json",
        },
    }

    js = "window.CAPACITY = " + json.dumps(out, indent=2) + ";\n"
    dest = Path(__file__).resolve().parent / "data.js"
    dest.write_text(js, encoding="utf-8")
    print("wrote", dest, "bytes", dest.stat().st_size)


if __name__ == "__main__":
    main()
