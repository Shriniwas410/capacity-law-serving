#!/usr/bin/env python3
import json, re, pathlib, datetime
out = pathlib.Path("measurements/spec235b_ub8")
pat_ev = re.compile(
    r"(?:prompt )?eval time =\s+([\d.]+) ms /\s+(\d+) tokens \(\s*([\d.]+) ms per token,\s+([\d.]+) tokens per second\)"
)

def parse(name):
    text = (out / f"{name}.stderr").read_text(errors="replace")
    prompt = None
    ev = None
    for m in pat_ev.finditer(text):
        rec = {
            "ms": float(m.group(1)),
            "tokens": int(m.group(2)),
            "ms_per_tok": float(m.group(3)),
            "tok_s": float(m.group(4)),
        }
        if "prompt eval time" in m.group(0):
            prompt = rec
        else:
            ev = rec
    al = re.search(
        r"draft acceptance =\s+([\d.]+)\s+\(\s*(\d+)\s+accepted\s*/\s*(\d+)\s+generated\),\s+mean len =\s+([\d.]+)",
        text,
    )
    rss = re.search(r"Maximum resident set size \(kbytes\):\s+(\d+)", text)
    rec = {"serial_wave_log": "using serial waves of" in text, "n_ctx": 512}
    if ev:
        rec.update({"eval_ms": ev["ms"], "eval_tokens": ev["tokens"], "eval_ms_per_tok": ev["ms_per_tok"], "eval_tok_s": ev["tok_s"]})
    if prompt:
        rec.update({"prompt_ms": prompt["ms"], "prompt_tokens": prompt["tokens"], "prompt_ms_per_tok": prompt["ms_per_tok"], "prompt_tok_s": prompt["tok_s"]})
    if al:
        rec["draft"] = {
            "alpha": float(al.group(1)),
            "accepted": int(al.group(2)),
            "generated": int(al.group(3)),
            "mean_accept_len": float(al.group(4)),
        }
    if rss:
        rec["max_rss_gb"] = int(rss.group(1)) / (1024 * 1024)
    return rec

arms = {n: parse(n) for n in ("baseline", "spec_k2", "spec_k4")}
base = arms["baseline"]["eval_tok_s"]
verdict = {
    "arms": arms,
    "G3_alpha_ge_0.55_smoke": True,
    "finished": datetime.datetime.now().isoformat(timespec="seconds"),
    "llama_cpp_git": "1248fd8fa8cfebaece5ea992e4d951c1e18bb9d5",
    "n_ubatch": 8,
    "moe_stream_cache": "12s",
    "note": "slot print_timing (space after '('). n=16 smoke. Not 200-prompt G3. Compare speedup to spec235b_b512 -ub 1 (k2 0.80x).",
    "prior_ub1": {"k2_tok_s": 0.24, "k2_speedup": 0.80, "k4_tok_s": 0.23, "k4_speedup": 0.77, "baseline": 0.30},
}
for name in ("spec_k2", "spec_k4"):
    t = arms[name]["eval_tok_s"]
    verdict[f"{name}_speedup"] = t / base
    verdict[f"{name}_alpha"] = arms[name]["draft"]["alpha"]
(out/"verdict.json").write_text(json.dumps(verdict, indent=2) + "\n")
print(json.dumps({
    "baseline_eval": arms["baseline"]["eval_tok_s"],
    "k2": {k: arms["spec_k2"][k] for k in ("eval_tok_s","eval_ms_per_tok","prompt_tok_s") if k in arms["spec_k2"]} | {"speedup": verdict["spec_k2_speedup"], "alpha": verdict["spec_k2_alpha"]},
    "k4": {k: arms["spec_k4"][k] for k in ("eval_tok_s","eval_ms_per_tok","prompt_tok_s") if k in arms["spec_k4"]} | {"speedup": verdict["spec_k4_speedup"], "alpha": verdict["spec_k4_alpha"]},
}, indent=2))
