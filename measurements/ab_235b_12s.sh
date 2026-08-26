#!/usr/bin/env bash
# 235B greedy smoke: does --moe-stream-cache 12s fit in 21 Gi WSL?
# CPU-only, same flags as ab235b stream-direct except slot count.
# 12s first (OOM gate). If it lives, run 8s after for a same-day tok/s pair.
# 8s second is warmer — do not treat the ratio as a cold A/B.
set -euo pipefail
export PATH=/home/shri/anaconda3/bin:/usr/bin:/bin:/usr/local/bin
BIN_DIR=/home/shri/src/llama.cpp/build-pr25294/bin
export LD_LIBRARY_PATH="$BIN_DIR"
BIN="$BIN_DIR/llama-completion"
MODEL=/home/shri/models/qwen3-235b/Qwen3-235B-A22B-Q4_K_M-00001-of-00003.gguf
OUT=measurements/ab235b_12s
N=${N:-16}
CTX=${CTX:-512}
THREADS=${THREADS:-16}
PROMPT=${PROMPT:-"The capital of France is"}
mkdir -p "$OUT"
cd "$BIN_DIR"
HEAD=$(git -C /home/shri/src/llama.cpp log -1 --format='%H')

python3 - <<PY
import json, datetime
open("$OUT/config.json","w").write(json.dumps({
  "model": "Qwen3-235B-A22B-Q4_K_M",
  "model_path": "$MODEL",
  "n_predict": int("$N"),
  "n_ctx": int("$CTX"),
  "n_batch": 1,
  "n_ubatch": 1,
  "temp": 0,
  "seed": 1,
  "ngl": 0,
  "threads": int("$THREADS"),
  "prompt": "$PROMPT",
  "reasoning": "off",
  "fit": "off",
  "moe_stream": True,
  "moe_stream_direct": True,
  "arms": ["cache_12s", "cache_8s"],
  "order": "12s first (OOM gate), 8s only if 12s exit=0",
  "llama_cpp_git": "$HEAD",
  "started": datetime.datetime.now().isoformat(timespec="seconds"),
  "prior_8s_eval_tok_s": 0.29,
  "prior_8s_note": "measurements/ab235b_verdict.json stream_direct, 2026-08-25T00:13:33",
}, indent=2) + "\n")
PY

run() {
  local name="$1"; shift
  echo "=== $name $(date -Is) ===" | tee -a "$OUT/meta.txt"
  free -h | tee -a "$OUT/meta.txt"
  set +e
  /usr/bin/time -v "$BIN" -m "$MODEL" -ngl 0 -n "$N" -t "$THREADS" --temp 0 --seed 1 \
    -no-cnv -st --reasoning off -p "$PROMPT" \
    --moe-stream --moe-stream-direct --moe-stream-io-threads 4 \
    -fit off -c "$CTX" -b 1 -ub 1 "$@" \
    >"$OUT/${name}.stdout" 2>"$OUT/${name}.stderr"
  rc=$?
  set -e
  echo "exit=$rc $name" | tee -a "$OUT/meta.txt"
  echo "--- timings/rss ---" | tee -a "$OUT/meta.txt"
  grep -E "tok/s|eval time|prompt eval|tokens per second|load time|n_ctx|Maximum resident|Cannot allocate|OOM|CPU_REPACK|abort|error" \
    "$OUT/${name}.stderr" 2>/dev/null | tee -a "$OUT/meta.txt" || true
  return "$rc"
}

: > "$OUT/meta.txt"
echo "bin=$BIN HEAD=$HEAD n=$N ctx=$CTX" | tee -a "$OUT/meta.txt"

set +e
run cache_12s --moe-stream-cache 12s
rc12=$?
set -e
echo "cache_12s_rc=$rc12" | tee -a "$OUT/meta.txt"

if [[ "$rc12" -eq 0 ]]; then
  set +e
  run cache_8s --moe-stream-cache 8s
  rc8=$?
  set -e
  echo "cache_8s_rc=$rc8" | tee -a "$OUT/meta.txt"
else
  echo "skip cache_8s (12s failed)" | tee -a "$OUT/meta.txt"
  rc8=""
fi

python3 - <<PY
import re, json, pathlib, datetime
out = pathlib.Path("measurements/ab235b_12s")

def text(name):
    p = out / f"{name}.stderr"
    if not p.exists():
        return ""
    return p.read_text(errors="replace")

def tps(blob, kind="eval"):
    if kind == "eval":
        gen = re.findall(r"eval time\s*=.*?([\d.]+)\s*tokens per second", blob, re.S)
        return float(gen[-1]) if gen else None
    m = re.findall(r"prompt eval time\s*=.*?([\d.]+)\s*tokens per second", blob, re.S)
    return float(m[-1]) if m else None

def rss_kb(blob):
    m = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", blob)
    return int(m.group(1)) if m else None

def n_ctx(blob):
    m = re.findall(r"n_ctx\s*=\s*(\d+)", blob)
    return int(m[-1]) if m else None

blob12 = text("cache_12s")
rec12 = {
    "exit": int("$rc12"),
    "eval_tok_s": tps(blob12, "eval"),
    "prompt_tok_s": tps(blob12, "prompt"),
    "max_rss_gb": (rss_kb(blob12) / (1024*1024)) if rss_kb(blob12) else None,
    "n_ctx": n_ctx(blob12),
    "oom": bool(re.search(r"Cannot allocate|insufficient memory|CPU_REPACK|std::bad_alloc", blob12, re.I)),
}
verdict = {
    "cache_12s": rec12,
    "cache_8s": None,
    "G_12s_fits": rec12["exit"] == 0 and not rec12["oom"],
    "finished": datetime.datetime.now().isoformat(timespec="seconds"),
    "note": "12s first. 8s second is warmer. Compare 12s eval tok/s to prior 0.29 (ab235b stream_direct).",
}
if "$rc8" != "":
    blob8 = text("cache_8s")
    verdict["cache_8s"] = {
        "exit": int("$rc8" or -1),
        "eval_tok_s": tps(blob8, "eval"),
        "prompt_tok_s": tps(blob8, "prompt"),
        "max_rss_gb": (rss_kb(blob8) / (1024*1024)) if rss_kb(blob8) else None,
        "n_ctx": n_ctx(blob8),
    }
    a, b = rec12["eval_tok_s"], verdict["cache_8s"]["eval_tok_s"]
    if a and b and b > 0:
        verdict["ratio_12s_over_8s"] = a / b
(out/"verdict.json").write_text(json.dumps(verdict, indent=2) + "\n")
print(json.dumps(verdict, indent=2))
PY

echo "AB235B12S_DONE $(date -Is)" | tee -a "$OUT/meta.txt"
