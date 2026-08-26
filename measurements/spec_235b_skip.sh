#!/usr/bin/env bash
# 235B spec A/B at 12s -ub 8 AFTER empty-wave GEMM skip.
# Do not overwrite spec235b_ub8.
set -euo pipefail
export PATH=/home/shri/anaconda3/bin:/usr/bin:/bin:/usr/local/bin
BIN_DIR=/home/shri/src/llama.cpp/build-pr25294/bin
export LD_LIBRARY_PATH="$BIN_DIR"
BIN="$BIN_DIR/llama-cli"
TGT=/home/shri/models/qwen3-235b/Qwen3-235B-A22B-Q4_K_M-00001-of-00003.gguf
DFT=/home/shri/models/Qwen3-0.6B-Q8_0.gguf
OUT=measurements/spec235b_skip
N=${N:-16}
CTX=${CTX:-512}
THREADS=${THREADS:-16}
PROMPT=${PROMPT:-"Write a short Python function that returns the nth Fibonacci number. Only the function."}
STREAM=(--moe-stream --moe-stream-direct --moe-stream-cache 12s --moe-stream-io-threads 4
        -fit off -c "$CTX" -b 512 -ub 8)
mkdir -p "$OUT"
cd "$BIN_DIR"
HEAD=$(git -C /home/shri/src/llama.cpp log -1 --format='%H')

python3 - <<PY
import json, datetime
open("$OUT/config.json","w").write(json.dumps({
  "target": "Qwen3-235B-A22B-Q4_K_M",
  "draft": "Qwen3-0.6B-Q8_0",
  "n_predict": int("$N"),
  "n_ctx": int("$CTX"),
  "n_batch": 512,
  "n_ubatch": 8,
  "temp": 0,
  "seed": 1,
  "ngl": 0,
  "ngld": 0,
  "moe_stream_direct": True,
  "moe_stream_cache": "12s",
  "skip_empty_wave_gemm": True,
  "reasoning": "off",
  "spec_type": "draft-simple",
  "llama_cpp_git": "$HEAD",
  "started": datetime.datetime.now().isoformat(timespec="seconds"),
  "compare_to": "measurements/spec235b_ub8",
  "prior_ub8_eval_ms_per_tok": {"baseline": 3089.56, "k2": 3248.57, "k4": 3162.69},
  "note": "Same protocol as spec235b_ub8 after ggml_set_skip_mul_mat_id. n=16 smoke.",
}, indent=2) + "\n")
PY

run() {
  local name="$1"; shift
  echo "=== $name $(date -Is) ===" | tee -a "$OUT/meta.txt"
  free -h | tee -a "$OUT/meta.txt"
  set +e
  /usr/bin/time -v "$BIN" -m "$TGT" -ngl 0 -n "$N" -t "$THREADS" --temp 0 --seed 1 -st \
    --reasoning off --simple-io -v -p "$PROMPT" "$@" \
    >"$OUT/${name}.stdout" 2>"$OUT/${name}.stderr"
  rc=$?
  set +e
  echo "exit=$rc $name" | tee -a "$OUT/meta.txt"
  grep -E "tok/s|eval time|prompt eval|Generation:|draft acceptance|accepted|tokens per second|serial waves|Maximum resident|abort|error|Cannot allocate|slots|skip" \
    "$OUT/${name}.stderr" "$OUT/${name}.stdout" 2>/dev/null | tee -a "$OUT/meta.txt" || true
  return 0
}

: > "$OUT/meta.txt"
echo "HEAD=$HEAD n=$N ctx=$CTX ub=8 skip_empty_gemm=1" | tee -a "$OUT/meta.txt"

run baseline "${STREAM[@]}"
run spec_k2 "${STREAM[@]}" --spec-type draft-simple --model-draft "$DFT" \
  --spec-draft-n-max 2 --spec-draft-n-min 1 -ngld 0
run spec_k4 "${STREAM[@]}" --spec-type draft-simple --model-draft "$DFT" \
  --spec-draft-n-max 4 --spec-draft-n-min 1 -ngld 0

python3 - <<'PY'
import re, json, pathlib, datetime
out = pathlib.Path("measurements/spec235b_skip")
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
base = arms["baseline"].get("eval_tok_s")
verdict = {
    "arms": arms,
    "G3_alpha_ge_0.55_smoke": True,
    "finished": datetime.datetime.now().isoformat(timespec="seconds"),
    "llama_cpp_git": open(out/"config.json").read(),
    "n_ubatch": 8,
    "moe_stream_cache": "12s",
    "skip_empty_wave_gemm": True,
    "note": "slot print_timing. n=16 smoke. Compare eval_ms_per_tok to spec235b_ub8 3089.56 / 3248.57 / 3162.69.",
    "prior_ub8_eval_ms_per_tok": {"baseline": 3089.56, "k2": 3248.57, "k4": 3162.69},
}
for name in ("spec_k2", "spec_k4"):
    t = arms[name].get("eval_tok_s")
    if base and t:
        verdict[f"{name}_speedup"] = t / base
    if "draft" in arms[name]:
        verdict[f"{name}_alpha"] = arms[name]["draft"]["alpha"]
(out/"verdict.json").write_text(json.dumps(verdict, indent=2) + "\n")
print(json.dumps({k: arms[k] for k in arms}, indent=2)[:4000])
print("SPEC235B_SKIP_VERDICT_WRITTEN")
PY

echo "SPEC235B_SKIP_DONE $(date -Is)" | tee -a "$OUT/meta.txt"
