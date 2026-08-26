#!/usr/bin/env bash
# G0: log Qwen3-235B-A22B routing. CPU-only, stream-direct, sequential decode.
# Do not transplant 30B miss rates. Writes config.json next to each .bin.
set -euo pipefail
export PATH=/home/shri/anaconda3/bin:/usr/bin:/bin:/usr/local/bin
BIN_DIR=/home/shri/src/llama.cpp/build-pr25294/bin
export LD_LIBRARY_PATH="$BIN_DIR"
BIN="$BIN_DIR/llama-moe-trace"
MODEL=/home/shri/models/qwen3-235b/Qwen3-235B-A22B-Q4_K_M-00001-of-00003.gguf
DATA=/mnt/c/Users/shrin/Desktop/AI/moe-routing-lab/data
OUT=measurements/traces
CONVERT=/mnt/c/Users/shrin/Desktop/AI/moe-routing-lab/06_convert_gguf_trace.py
VALIDATE=measurements/validate_trace.py
MAX_TOKENS=${MAX_TOKENS:-2048}
WINDOW=${WINDOW:-512}
THREADS=${THREADS:-16}
DOMAINS=${DOMAINS:-code math medical general}

mkdir -p "$OUT"
cd "$BIN_DIR"
HEAD=$(git -C /home/shri/src/llama.cpp log -1 --format='%H')
echo "HEAD=$HEAD max_tokens=$MAX_TOKENS window=$WINDOW domains=$DOMAINS" | tee -a "$OUT/meta.txt"

for domain in $DOMAINS; do
  corpus="$DATA/$domain/corpus.txt"
  bin="$OUT/trace_${domain}.bin"
  npz="$OUT/${domain}.npz"
  cfg="$OUT/trace_${domain}.config.json"
  if [[ -s "$npz" ]]; then
    echo "skip $domain (npz exists)"
    continue
  fi
  python3 - <<PY
import json, os, datetime
open("$cfg","w").write(json.dumps({
  "model": "Qwen3-235B-A22B-Q4_K_M",
  "model_path": "$MODEL",
  "domain": "$domain",
  "tokens": int("$MAX_TOKENS"),
  "window": int("$WINDOW"),
  "seq": int("$WINDOW"),
  "n_batch": 1,
  "n_ubatch": 1,
  "ngl": 0,
  "moe_stream": True,
  "moe_stream_direct": True,
  "moe_stream_cache": "8s",
  "llama_cpp_git": "$HEAD",
  "started": datetime.datetime.now().isoformat(timespec="seconds"),
}, indent=2) + "\n")
PY
  echo "=== $domain $(date -Is) ===" | tee -a "$OUT/meta.txt"
  export MOE_TRACE_OUT="$bin"
  export MOE_TRACE_MAX_TOKENS="$MAX_TOKENS"
  export MOE_TRACE_WINDOW="$WINDOW"
  set +e
  "$BIN" -m "$MODEL" -f "$corpus" -ngl 0 -t "$THREADS" -c "$WINDOW" -b 1 -ub 1 -fit off \
    --temp 0 --seed 1 --moe-stream --moe-stream-direct --moe-stream-cache 8s \
    --moe-stream-io-threads 4 \
    >"$OUT/${domain}.stdout" 2>"$OUT/${domain}.stderr"
  rc=$?
  set -e
  echo "exit=$rc $domain" | tee -a "$OUT/meta.txt"
  if [[ $rc -ne 0 ]]; then
    echo "FAILED $domain" | tee -a "$OUT/meta.txt"
    tail -n 40 "$OUT/${domain}.stderr" | tee -a "$OUT/meta.txt"
    exit $rc
  fi
  python3 "$CONVERT" --bin "$bin" --out "$npz" \
    --model-id Qwen3-235B-A22B-Q4_K_M --num-experts 128
  python3 "$VALIDATE" --npz "$npz" --num-experts 128
done

echo "G0 done $(date -Is)" | tee -a "$OUT/meta.txt"
ls -lh "$OUT"
