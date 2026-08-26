#!/usr/bin/env bash
set -euo pipefail
export PATH=/home/shri/anaconda3/bin:/usr/bin:/bin
sed 's/\r$//' measurements/llama-moe-trace/moe-trace.cpp \
  > /home/shri/src/llama.cpp/tools/moe-trace/moe-trace.cpp
cd /home/shri/src/llama.cpp
cmake --build build-pr25294 -j"$(nproc)" --target llama-moe-trace
# re-smoke
sed -i 's/\r$//' measurements/*.sh
rm -rf measurements/g0_smoke30
bash measurements/smoke_trace_30b.sh
python3 measurements/dump_trace_head.py measurements/g0_smoke30/trace_code.bin
