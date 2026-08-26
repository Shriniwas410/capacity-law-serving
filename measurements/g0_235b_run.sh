#!/usr/bin/env bash
# 32-token 235B smoke, then 2048 x 4 domains. CPU-only, setsid-safe.
set -euo pipefail
export PATH=/home/shri/anaconda3/bin:/usr/bin:/bin
BIN_DIR=/home/shri/src/llama.cpp/build-pr25294/bin
export LD_LIBRARY_PATH="$BIN_DIR"
BIN="$BIN_DIR/llama-moe-trace"
MODEL=/home/shri/models/qwen3-235b/Qwen3-235B-A22B-Q4_K_M-00001-of-00003.gguf
CORPUS=/mnt/c/Users/shrin/Desktop/AI/moe-routing-lab/data/code/corpus.txt
SMOKE=measurements/traces_smoke
CONVERT=/mnt/c/Users/shrin/Desktop/AI/moe-routing-lab/06_convert_gguf_trace.py
VALIDATE=measurements/validate_trace.py
mkdir -p "$SMOKE" measurements/traces

echo "=== 235B 32-token smoke $(date -Is) ==="
export MOE_TRACE_OUT="$SMOKE/trace_code.bin"
export MOE_TRACE_MAX_TOKENS=32
export MOE_TRACE_WINDOW=32
cd "$BIN_DIR"
"$BIN" -m "$MODEL" -f "$CORPUS" -ngl 0 -t 16 -c 32 -b 1 -ub 1 -fit off \
  --temp 0 --seed 1 --moe-stream --moe-stream-direct --moe-stream-cache 8s \
  --moe-stream-io-threads 4 \
  >"$SMOKE/stdout" 2>"$SMOKE/stderr"
python3 "$CONVERT" --bin "$SMOKE/trace_code.bin" --out "$SMOKE/code.npz" \
  --model-id Qwen3-235B-A22B-Q4_K_M --num-experts 128
python3 "$VALIDATE" --npz "$SMOKE/code.npz" --num-experts 128
python3 measurements/uk_union.py --traces "$SMOKE" --out "$SMOKE/uk.json"
echo "SMOKE_235B_OK $(date -Is)"

echo "=== 235B 2048 x 4 domains $(date -Is) ==="
unset MOE_TRACE_OUT MOE_TRACE_MAX_TOKENS MOE_TRACE_WINDOW
bash measurements/trace_235b.sh
echo "G0_235B_DONE $(date -Is)"
