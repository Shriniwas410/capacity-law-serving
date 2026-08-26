#!/usr/bin/env bash
# 235B prefill A/B at 12s: -ub 1 vs -ub 4 after serial-wave patch.
# n=1 so we do not pay 16 decode tokens. Prompt is the spec Fibonacci text.
set -euo pipefail
export PATH=/home/shri/anaconda3/bin:/usr/bin:/bin:/usr/local/bin
BIN_DIR=/home/shri/src/llama.cpp/build-pr25294/bin
export LD_LIBRARY_PATH="$BIN_DIR"
BIN="$BIN_DIR/llama-completion"
MODEL=/home/shri/models/qwen3-235b/Qwen3-235B-A22B-Q4_K_M-00001-of-00003.gguf
OUT=measurements/ab235b_ubatch
N=${N:-1}
CTX=${CTX:-512}
THREADS=${THREADS:-16}
PROMPT=${PROMPT:-"Write a short Python function that returns the nth Fibonacci number. Only the function."}
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
  "temp": 0,
  "seed": 1,
  "ngl": 0,
  "threads": int("$THREADS"),
  "prompt": "$PROMPT",
  "reasoning": "off",
  "fit": "off",
  "moe_stream": True,
  "moe_stream_direct": True,
  "moe_stream_cache": "12s",
  "arms": ["ub1", "ub4"],
  "order": "ub1 first (serial 1-token graphs), ub4 second (serial waves, warmer)",
  "llama_cpp_git": "$HEAD",
  "started": datetime.datetime.now().isoformat(timespec="seconds"),
  "note": "Measures prompt tok/s. ub4 second is warmer. Do not treat ratio as a cold A/B.",
}, indent=2) + "\n")
PY

run() {
  local name="$1"; shift
  echo "=== $name $(date -Is) ===" | tee -a "$OUT/meta.txt"
  free -h | tee -a "$OUT/meta.txt"
  set +e
  /usr/bin/time -v "$BIN" -m "$MODEL" -ngl 0 -n "$N" -t "$THREADS" --temp 0 --seed 1 \
    -no-cnv -st --reasoning off -p "$PROMPT" \
    --moe-stream --moe-stream-direct --moe-stream-cache 12s --moe-stream-io-threads 4 \
    -fit off -c "$CTX" "$@" \
    >"$OUT/${name}.stdout" 2>"$OUT/${name}.stderr"
  rc=$?
  set -e
  echo "exit=$rc $name" | tee -a "$OUT/meta.txt"
  grep -E "tok/s|eval time|prompt eval|tokens per second|serial waves|Maximum resident|abort|error|Cannot allocate" \
    "$OUT/${name}.stderr" 2>/dev/null | tee -a "$OUT/meta.txt" || true
  return "$rc"
}

: > "$OUT/meta.txt"
echo "HEAD=$HEAD n=$N ctx=$CTX" | tee -a "$OUT/meta.txt"

set +e
run ub1 -b 1 -ub 1
rc1=$?
run ub4 -b 8 -ub 4
rc4=$?
set -e
echo "ub1_rc=$rc1 ub4_rc=$rc4" | tee -a "$OUT/meta.txt"

python3 - <<PY
import re, json, pathlib, datetime
out = pathlib.Path("measurements/ab235b_ubatch")

def blob(name):
    return (out/f"{name}.stderr").read_text(errors="replace")

def rec(name, rc):
    b = blob(name)
    def tps(kind):
        pat = r"eval time\s*=.*?([\d.]+)\s*tokens per second" if kind=="eval" else r"prompt eval time\s*=.*?([\d.]+)\s*tokens per second"
        m = re.findall(pat, b, re.S)
        return float(m[-1]) if m else None
    rss = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", b)
    return {
        "exit": int(rc),
        "eval_tok_s": tps("eval"),
        "prompt_tok_s": tps("prompt"),
        "max_rss_gb": (int(rss.group(1))/(1024*1024)) if rss else None,
        "serial_wave_log": bool(re.search(r"using serial waves of", b)),
        "aborted": bool(re.search(r"GGML_ABORT|GGML_ASSERT|Aborted", b)),
    }

v = {
    "ub1": rec("ub1", "$rc1"),
    "ub4": rec("ub4", "$rc4"),
    "finished": datetime.datetime.now().isoformat(timespec="seconds"),
    "note": "ub4 second is warmer. Headline is prompt_tok_s and that ub4 did not abort.",
}
a, b = v["ub1"]["prompt_tok_s"], v["ub4"]["prompt_tok_s"]
if a and b and a > 0:
    v["ratio_ub4_over_ub1_prompt"] = b / a
(out/"verdict.json").write_text(json.dumps(v, indent=2)+"\n")
print(json.dumps(v, indent=2))
PY
echo "AB235B_UBATCH_DONE $(date -Is)" | tee -a "$OUT/meta.txt"
