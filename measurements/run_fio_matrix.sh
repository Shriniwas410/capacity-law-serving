#!/usr/bin/env bash
# G1 fio matrix on the real 30B GGUF (already on ext4).
# {mmap, pread, O_DIRECT} x {QD1, QD8} x {3.5MB, 10MB, 16MB}
set -euo pipefail
FILE="${1:-/home/shri/models/Qwen3-30B-A3B-Q4_K_M.gguf}"
OUT="${2:-$HOME/week1_fio}"
FIO="${FIO:-$HOME/.local/bin/fio}"
mkdir -p "$OUT"
SIZE=$(stat -c %s "$FILE")
echo "file=$FILE bytes=$SIZE fio=$FIO"
echo "start=$(date -Is)" | tee "$OUT/meta.txt"

run() {
  local name="$1"; shift
  echo "=== $name ===" | tee -a "$OUT/meta.txt"
  "$FIO" --name="$name" --filename="$FILE" --output="$OUT/${name}.json" --output-format=json+ \
    --time_based --runtime=20s --ramp_time=3s --group_reporting --norandommap \
    --thread --ioengine="$1" --direct="$2" --iodepth="$3" --rw=randread \
    --bs="$4" --size="$SIZE" --offset=0 --numjobs=1
}

# mmap cannot use O_DIRECT
run mmap_qd1_3m5   mmap  0 1 3584k
run mmap_qd8_3m5   mmap  0 8 3584k
run mmap_qd1_10m   mmap  0 1 10m
run mmap_qd8_10m   mmap  0 8 10m
run mmap_qd1_16m   mmap  0 1 16m
run mmap_qd8_16m   mmap  0 8 16m

run pread_qd1_3m5  psync 0 1 3584k
run pread_qd8_3m5  pvsync 0 8 3584k
run pread_qd1_10m  psync 0 1 10m
run pread_qd8_10m  pvsync 0 8 10m
run pread_qd1_16m  psync 0 1 16m
run pread_qd8_16m  pvsync 0 8 16m

# libaio-dev is not installed (no sudo). io_uring is compiled into our user-local fio.
run odirect_qd1_3m5  psync    1 1 3584k
run odirect_qd8_3m5  io_uring 1 8 3584k
run odirect_qd1_10m  psync    1 1 10m
run odirect_qd8_10m  io_uring 1 8 10m
run odirect_qd1_16m  psync    1 1 16m
run odirect_qd8_16m  io_uring 1 8 16m

python3 "$HOME/measurements/summarize_fio.py" "$OUT"
echo "done=$(date -Is)" | tee -a "$OUT/meta.txt"
