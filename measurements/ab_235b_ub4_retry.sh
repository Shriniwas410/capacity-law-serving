#!/usr/bin/env bash
# 235B 12s, -ub 4 only (ub1 already 0.16 prompt tok/s). timeout detects hang.
set -euo pipefail
export PATH=/home/shri/anaconda3/bin:/usr/bin:/bin
BIN_DIR=/home/shri/src/llama.cpp/build-pr25294/bin
export LD_LIBRARY_PATH="$BIN_DIR"
BIN="$BIN_DIR/llama-completion"
MODEL=/home/shri/models/qwen3-235b/Qwen3-235B-A22B-Q4_K_M-00001-of-00003.gguf
OUT=measurements/ab235b_ubatch
PROMPT=${PROMPT:-"Write a short Python function that returns the nth Fibonacci number. Only the function."}
mkdir -p "$OUT"
cd "$BIN_DIR"
HEAD=$(git -C /home/shri/src/llama.cpp log -1 --format='%H')
echo "=== ub4_retry $(date -Is) HEAD=$HEAD ===" | tee -a "$OUT/meta.txt"
set +e
/usr/bin/timeout --signal=KILL 180 /usr/bin/time -v "$BIN" -m "$MODEL" -ngl 0 -n 1 -t 16 --temp 0 --seed 1 \
  -no-cnv -st --reasoning off -p "$PROMPT" \
  --moe-stream --moe-stream-direct --moe-stream-cache 12s --moe-stream-io-threads 4 \
  -fit off -c 512 -b 8 -ub 4 \
  >"$OUT/ub4.stdout" 2>"$OUT/ub4.stderr"
rc=$?
set -e
echo "exit=$rc ub4_retry" | tee -a "$OUT/meta.txt"
grep -E "tok/s|eval time|prompt eval|serial waves|Maximum resident|abort|error|waves =" \
  "$OUT/ub4.stderr" | tee -a "$OUT/meta.txt" || true
python3 - <<PY
import re, json, pathlib, datetime
out = pathlib.Path("measurements/ab235b_ubatch")
b = (out/"ub4.stderr").read_text(errors="replace")
def tps(kind):
    pat = r"eval time\s*=.*?([\d.]+)\s*tokens per second" if kind=="eval" else r"prompt eval time\s*=.*?([\d.]+)\s*tokens per second"
    m = re.findall(pat, b, re.S)
    return float(m[-1]) if m else None
rss = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", b)
v = {
  "ub1_prior_prompt_tok_s": 0.16,
  "ub4": {
    "exit": int("$rc"),
    "timed_out": int("$rc")==137 or int("$rc")==124,
    "prompt_tok_s": tps("prompt"),
    "eval_tok_s": tps("eval"),
    "max_rss_gb": (int(rss.group(1))/(1024*1024)) if rss else None,
    "serial_wave_log": bool(re.search(r"using serial waves of", b)),
    "wave_stats": bool(re.search(r"moe stream: waves", b)),
  },
  "finished": datetime.datetime.now().isoformat(timespec="seconds"),
}
a, bps = 0.16, v["ub4"]["prompt_tok_s"]
if bps and a:
    v["ratio_ub4_over_ub1_prompt"] = bps / a
(out/"verdict.json").write_text(json.dumps(v, indent=2)+"\n")
print(json.dumps(v, indent=2))
PY
echo "UB4_RETRY_DONE $(date -Is)" | tee -a "$OUT/meta.txt"
exit "$rc"
