#!/usr/bin/env bash
# G1 30B A/B. One-shot (-no-cnv). CPU-only. Visible logs.
set -euo pipefail
BIN_DIR=/home/shri/src/llama.cpp/build-pr25294/bin
export LD_LIBRARY_PATH="$BIN_DIR"
BIN="$BIN_DIR/llama-completion"
MODEL=/home/shri/models/Qwen3-30B-A3B-Q4_K_M.gguf
OUT=measurements/ab30b
N=16
THREADS=16
PROMPT="The capital of France is"
mkdir -p "$OUT"
cd "$BIN_DIR"

run() {
  local name="$1"; shift
  echo "=== $name $(date -Is) ===" | tee -a "$OUT/meta.txt"
  local start end
  start=$(date +%s.%N)
  "$BIN" -m "$MODEL" -ngl 0 -n "$N" -t "$THREADS" --temp 0 --seed 1 \
    -no-cnv -st -p "$PROMPT" "$@" \
    >"$OUT/${name}.stdout" 2>"$OUT/${name}.stderr" || {
      echo "RUN_FAILED $name exit=$?" | tee -a "$OUT/meta.txt"
    }
  end=$(date +%s.%N)
  python3 -c "print('wall_sec=' + str(round(float('$end')-float('$start'), 3)))" | tee "$OUT/${name}.time" | tee -a "$OUT/meta.txt"
  echo "--- stdout ---" | tee -a "$OUT/meta.txt"
  cat "$OUT/${name}.stdout" | tee -a "$OUT/meta.txt"
  echo "--- timings ---" | tee -a "$OUT/meta.txt"
  grep -E "tok/s|eval time|prompt eval|llama_perf|tokens per second|total time|load time" "$OUT/${name}.stderr" | tee -a "$OUT/meta.txt" || true
}

: > "$OUT/meta.txt"
echo "bin=$BIN HEAD=$(cd /home/shri/src/llama.cpp && git log -1 --oneline) n=$N" | tee -a "$OUT/meta.txt"

run mmap
run stream_direct --moe-stream --moe-stream-direct --moe-stream-cache 4 --moe-stream-io-threads 4

python3 - <<'PY'
import re, pathlib
out = pathlib.Path("measurements/ab30b")
def tps(name):
    text = (out/f"{name}.stderr").read_text(errors="replace")
    # prefer eval (generation) tok/s over prompt eval
    gen = re.findall(r"eval time\s*=.*?([\d.]+)\s*tokens per second", text, re.S)
    if gen:
        return float(gen[-1])
    all_ = re.findall(r"([\d.]+)\s*tokens per second", text)
    return float(all_[-1]) if all_ else None
a, b = tps("mmap"), tps("stream_direct")
print(f"mmap_toks={a}  stream_direct_toks={b}")
open(out/"verdict.txt","w").write(f"mmap={a}\nstream={b}\n")
if a and b and a > 0:
    r = b/a
    v = "PASS" if r >= 1.30 else "FAIL"
    extra = f"ratio={r:.3f}  G1 need >=1.30  -> {v}\n"
    print(extra)
    open(out/"verdict.txt","a").write(extra)
PY
