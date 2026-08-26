#!/usr/bin/env bash
set -euo pipefail
export PATH=/home/shri/anaconda3/bin:/usr/bin:/bin
BIN_DIR=/home/shri/src/llama.cpp/build-pr25294/bin
export LD_LIBRARY_PATH="$BIN_DIR"
OUT=measurements/smoke_umax_30b
mkdir -p "$OUT"
cd "$BIN_DIR"
set +e
timeout 120 /usr/bin/time -v ./llama-cli \
  -m /home/shri/models/Qwen3-30B-A3B-Q4_K_M.gguf \
  --model-draft /home/shri/models/Qwen3-0.6B-Q8_0.gguf \
  --spec-type draft-simple --spec-draft-n-max 2 --spec-draft-n-min 1 \
  -ngl 0 -ngld 0 -n 8 -t 16 --temp 0 --seed 1 -st --reasoning off --simple-io \
  -c 512 -b 512 -ub 8 \
  --moe-stream --moe-stream-direct --moe-stream-cache 8s --moe-stream-umax \
  --moe-stream-io-threads 4 -fit off \
  -p "The capital of France is" \
  >"$OUT/stdout.txt" 2>"$OUT/stderr.txt"
echo EXIT=$? | tee "$OUT/meta.txt"
grep -E "HorizonSpec|serial waves|draft acceptance|eval time|GGML_|umax trunc|Generation:" \
  "$OUT/stderr.txt" "$OUT/stdout.txt" 2>/dev/null | tee -a "$OUT/meta.txt" || true
