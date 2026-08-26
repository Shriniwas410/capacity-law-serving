#!/usr/bin/env bash
# Fast tracer sanity on 30B (64 tokens) before the 235B hours-long run.
set -euo pipefail
export PATH=/home/shri/anaconda3/bin:/usr/bin:/bin:/usr/local/bin
BIN_DIR=/home/shri/src/llama.cpp/build-pr25294/bin
export LD_LIBRARY_PATH="$BIN_DIR"
BIN="$BIN_DIR/llama-moe-trace"
MODEL=/home/shri/models/Qwen3-30B-A3B-Q4_K_M.gguf
CORPUS=/mnt/c/Users/shrin/Desktop/AI/moe-routing-lab/data/code/corpus.txt
OUT=measurements/g0_smoke30
CONVERT=/mnt/c/Users/shrin/Desktop/AI/moe-routing-lab/06_convert_gguf_trace.py
VALIDATE=measurements/validate_trace.py
mkdir -p "$OUT"
cd "$BIN_DIR"
export MOE_TRACE_OUT="$OUT/trace_code.bin"
export MOE_TRACE_MAX_TOKENS=64
export MOE_TRACE_WINDOW=64
"$BIN" -m "$MODEL" -f "$CORPUS" -ngl 0 -t 16 -c 64 -b 1 -ub 1 -fit off --temp 0 --seed 1 \
  --moe-stream --moe-stream-direct --moe-stream-cache 8s \
  >"$OUT/stdout" 2>"$OUT/stderr"
python3 "$CONVERT" --bin "$OUT/trace_code.bin" --out "$OUT/code.npz" \
  --model-id Qwen3-30B-A3B-Q4_K_M --num-experts 128
python3 "$VALIDATE" --npz "$OUT/code.npz" --num-experts 128
python3 measurements/uk_union.py --traces "$OUT" --out "$OUT/uk.json"
echo SMOKE_OK
