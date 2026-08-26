#!/usr/bin/env bash
# 235B greedy A/B on THIS box: page-cache stream vs O_DIRECT stream.
#
# Full mmap is impossible here (CPU_REPACK tried to allocate ~105 GB and died).
# The 0.44 tok/s figure is from 2026-08-05 Windows llama-server -ngl 99 -ncmoe 94,
# not CPU-only llama-completion — do not treat it as this A/B's mmap arm.
#
# Both arms: CPU, -fit off -c 512 -b 1 -ub 1 --moe-stream-cache 8s.
# n=16 like 30B G1. Do not mix with G0's ~0.24 tok/s sequential 512-token prefill.
set -euo pipefail
export PATH=/home/shri/anaconda3/bin:/usr/bin:/bin:/usr/local/bin
BIN_DIR=/home/shri/src/llama.cpp/build-pr25294/bin
export LD_LIBRARY_PATH="$BIN_DIR"
BIN="$BIN_DIR/llama-completion"
MODEL=/home/shri/models/qwen3-235b/Qwen3-235B-A22B-Q4_K_M-00001-of-00003.gguf
OUT=measurements/ab235b
N=${N:-16}
CTX=${CTX:-512}
THREADS=${THREADS:-16}
PROMPT=${PROMPT:-"The capital of France is"}
COMMON=(--moe-stream --moe-stream-cache 8s --moe-stream-io-threads 4 -fit off -c "$CTX" -b 1 -ub 1)
mkdir -p "$OUT"
# keep the failed full-mmap log if present
if [[ -f "$OUT/mmap.stderr" ]] && grep -q 'CPU_REPACK' "$OUT/mmap.stderr" 2>/dev/null; then
  mv -f "$OUT/mmap.stderr" "$OUT/mmap_repack_oom.stderr"
  mv -f "$OUT/mmap.stdout" "$OUT/mmap_repack_oom.stdout" 2>/dev/null || true
fi
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
  "moe_stream_cache": "8s",
  "pagecache": {"moe_stream": True, "moe_stream_direct": False},
  "stream_direct": {"moe_stream": True, "moe_stream_direct": True},
  "not_compared": {
    "full_mmap": "CPU_REPACK ~105GB OOM — see mmap_repack_oom.stderr",
    "legacy_0.44_tok_s": "2026-08-05 Windows llama-server -ngl 99 -ncmoe 94, not this binary",
  },
  "llama_cpp_git": "$HEAD",
  "started": datetime.datetime.now().isoformat(timespec="seconds"),
}, indent=2) + "\n")
PY

run() {
  local name="$1"; shift
  echo "=== $name $(date -Is) ===" | tee -a "$OUT/meta.txt"
  set +e
  "$BIN" -m "$MODEL" -ngl 0 -n "$N" -t "$THREADS" --temp 0 --seed 1 \
    -no-cnv -st --reasoning off -p "$PROMPT" "$@" \
    >"$OUT/${name}.stdout" 2>"$OUT/${name}.stderr"
  rc=$?
  set -e
  echo "exit=$rc $name" | tee -a "$OUT/meta.txt"
  echo "--- timings ---" | tee -a "$OUT/meta.txt"
  grep -E "tok/s|eval time|prompt eval|llama_perf|tokens per second|total time|load time|Generation:|n_ctx" \
    "$OUT/${name}.stderr" "$OUT/${name}.stdout" 2>/dev/null | tee -a "$OUT/meta.txt" || true
}

: > "$OUT/meta.txt"
echo "bin=$BIN HEAD=$HEAD n=$N ctx=$CTX" | tee -a "$OUT/meta.txt"
free -h | tee -a "$OUT/meta.txt"

run pagecache "${COMMON[@]}"
run stream_direct "${COMMON[@]}" --moe-stream-direct

python3 - <<'PY'
import re, json, pathlib, datetime
out = pathlib.Path("measurements/ab235b")

def text(name):
    return (out/f"{name}.stderr").read_text(errors="replace") + "\n" + (out/f"{name}.stdout").read_text(errors="replace")

def tps(blob):
    m = re.findall(r"Generation:\s*([\d.]+)\s*t/s", blob)
    if m:
        return float(m[-1])
    gen = re.findall(r"eval time\s*=.*?([\d.]+)\s*tokens per second", blob, re.S)
    if gen:
        return float(gen[-1])
    all_ = re.findall(r"([\d.]+)\s*tokens per second", blob)
    return float(all_[-1]) if all_ else None

def prompt_tps(blob):
    m = re.findall(r"prompt eval time\s*=.*?([\d.]+)\s*tokens per second", blob, re.S)
    return float(m[-1]) if m else None

def n_ctx(blob):
    m = re.findall(r"n_ctx\s*=\s*(\d+)", blob)
    return int(m[-1]) if m else None

arms = {}
for name in ("pagecache", "stream_direct"):
    blob = text(name)
    arms[name] = {
        "eval_tok_s": tps(blob),
        "prompt_tok_s": prompt_tps(blob),
        "n_ctx": n_ctx(blob),
    }
    print(name, arms[name])

a, b = arms["pagecache"]["eval_tok_s"], arms["stream_direct"]["eval_tok_s"]
verdict = {
    "arms": arms,
    "ratio_direct_over_pagecache": (b / a) if a and b and a > 0 else None,
    "finished": datetime.datetime.now().isoformat(timespec="seconds"),
    "note": (
        "CPU-only stream A/B, n=16, ctx=512, cache=8s. "
        "Not comparable to 0.44 GPU llama-server -ncmoe, nor to G0 2048-token prefill."
    ),
}
(out/"verdict.json").write_text(json.dumps(verdict, indent=2) + "\n")
print(json.dumps(verdict, indent=2))
PY

echo "AB235B_DONE $(date -Is)" | tee -a "$OUT/meta.txt"
