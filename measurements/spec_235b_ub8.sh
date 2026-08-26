#!/usr/bin/env bash
# 235B stock spec at 12s WITH batched verify (-ub 8) after serial-wave patch.
# Prior smoke used -ub 1 and lost wall-clock (0.80x) despite alpha ~0.75.
set -euo pipefail
export PATH=/home/shri/anaconda3/bin:/usr/bin:/bin:/usr/local/bin
BIN_DIR=/home/shri/src/llama.cpp/build-pr25294/bin
export LD_LIBRARY_PATH="$BIN_DIR"
BIN="$BIN_DIR/llama-cli"
TGT=/home/shri/models/qwen3-235b/Qwen3-235B-A22B-Q4_K_M-00001-of-00003.gguf
DFT=/home/shri/models/Qwen3-0.6B-Q8_0.gguf
OUT=measurements/spec235b_ub8
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
  "reasoning": "off",
  "spec_type": "draft-simple",
  "llama_cpp_git": "$HEAD",
  "started": datetime.datetime.now().isoformat(timespec="seconds"),
  "prior_spec_ub1_k2_tok_s": 0.24,
  "prior_spec_ub1_k2_speedup": 0.80,
  "prior_greedy_12s_eval_tok_s": 0.38,
  "note": "Batched verify via serial waves. Not 200-prompt G3. Compare to spec235b_b512 (-ub 1).",
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
  set -e
  echo "exit=$rc $name" | tee -a "$OUT/meta.txt"
  grep -E "tok/s|eval time|prompt eval|Generation:|draft acceptance|accepted|tokens per second|serial waves|Maximum resident|abort|error|Cannot allocate|slots" \
    "$OUT/${name}.stderr" "$OUT/${name}.stdout" 2>/dev/null | tee -a "$OUT/meta.txt" || true
  return 0
}

: > "$OUT/meta.txt"
echo "HEAD=$HEAD n=$N ctx=$CTX ub=8" | tee -a "$OUT/meta.txt"

run baseline "${STREAM[@]}"
run spec_k2 "${STREAM[@]}" --spec-type draft-simple --model-draft "$DFT" \
  --spec-draft-n-max 2 --spec-draft-n-min 1 -ngld 0
run spec_k4 "${STREAM[@]}" --spec-type draft-simple --model-draft "$DFT" \
  --spec-draft-n-max 4 --spec-draft-n-min 1 -ngld 0

python3 - <<'PY'
import re, json, pathlib, datetime
out = pathlib.Path("measurements/spec235b_ub8")

def read(name):
    return (out/f"{name}.stderr").read_text(errors="replace") + "\n" + (out/f"{name}.stdout").read_text(errors="replace")

def tps(text):
    m = re.findall(r"Generation:\s*([\d.]+)\s*t/s", text)
    if m:
        return float(m[-1])
    gen = re.findall(r"eval time\s*=.*?([\d.]+)\s*tokens per second", text, re.S)
    if gen:
        return float(gen[-1])
    return None

def alpha(text):
    m = re.findall(r"draft acceptance\s*=\s*([\d.]+)\s*\(\s*(\d+)\s*accepted\s*/\s*(\d+)\s*generated\)", text)
    if not m:
        return None
    a, acc, tot = m[-1]
    return {"alpha": float(a), "accepted": int(acc), "generated": int(tot)}

def rss_gb(text):
    m = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", text)
    return int(m.group(1)) / (1024 * 1024) if m else None

def aborted(text):
    return bool(re.search(r"GGML_ABORT|GGML_ASSERT|Cannot allocate|insufficient memory|unique experts", text))

arms = {}
for name in ("baseline", "spec_k2", "spec_k4"):
    text = read(name)
    arms[name] = {
        "tok_s": tps(text),
        "draft": alpha(text),
        "max_rss_gb": rss_gb(text),
        "looks_aborted": aborted(text),
        "serial_wave_log": bool(re.search(r"using serial waves of", text)),
        "exit_guess": 0 if "SPEC235B" not in text else None,
        "n_ctx": (lambda m: int(m[-1]) if m else None)(re.findall(r"n_ctx\s*=\s*(\d+)", text)),
    }
    print(name, arms[name])

base = arms["baseline"]["tok_s"]
verdict = {
    "arms": arms,
    "G3_alpha_ge_0.55_smoke": None,
    "finished": datetime.datetime.now().isoformat(timespec="seconds"),
    "note": "n=16 smoke at 12s with -ub 8 serial waves. Not 200-prompt G3. Compare to spec235b_b512 -ub 1.",
    "prior_ub1": {"k2_tok_s": 0.24, "k2_speedup": 0.80, "k4_tok_s": 0.23, "k4_speedup": 0.77, "baseline": 0.30},
}
for name in ("spec_k2", "spec_k4"):
    d = arms[name]["draft"]
    t = arms[name]["tok_s"]
    if base and t:
        verdict[f"{name}_speedup"] = t / base
    if d:
        verdict[f"{name}_alpha"] = d["alpha"]
        ok = d["alpha"] >= 0.55
        if verdict["G3_alpha_ge_0.55_smoke"] is None:
            verdict["G3_alpha_ge_0.55_smoke"] = ok
        else:
            verdict["G3_alpha_ge_0.55_smoke"] = verdict["G3_alpha_ge_0.55_smoke"] or ok
(out/"verdict.json").write_text(json.dumps(verdict, indent=2) + "\n")
print(json.dumps(verdict, indent=2))
PY

echo "SPEC235B_UB8_DONE $(date -Is)" | tee -a "$OUT/meta.txt"
