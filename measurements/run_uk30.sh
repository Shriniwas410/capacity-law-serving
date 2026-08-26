#!/usr/bin/env bash
set -euo pipefail
export PATH=/home/shri/anaconda3/bin:/usr/bin:/bin
python3 --version
python3 measurements/uk_union.py \
  --traces /mnt/c/Users/shrin/Desktop/AI/moe-routing-lab/traces \
  --out measurements/uk_30b.json \
  --num-experts 128
