#!/usr/bin/env python3
"""Print first records of a moe-trace bin to catch duplicate per-token callbacks."""
import struct
import sys
from collections import defaultdict

path = sys.argv[1]
recs = []
with open(path, "rb") as f:
    while True:
        head = f.read(12)
        if len(head) < 12:
            break
        layer, n_tokens, k = struct.unpack("<iii", head)
        data = list(struct.unpack("<" + "i" * (n_tokens * k), f.read(4 * n_tokens * k)))
        recs.append((layer, n_tokens, k, data))

print(f"n_records={len(recs)}")
layers = sorted({r[0] for r in recs})
print(f"layers={layers[:8]}... n_layers={len(layers)}")
per = defaultdict(int)
for layer, n_tokens, k, data in recs:
    per[layer] += n_tokens
print("tokens_per_layer sample", {l: per[l] for l in layers[:5]})
# first 6 records
for r in recs[:12]:
    print(f" layer={r[0]} n={r[1]} k={r[2]} ids={r[3]}")
# consecutive identical?
same = 0
for i in range(1, min(200, len(recs))):
    if recs[i][0] == recs[i-1][0] and recs[i][3] == recs[i-1][3]:
        same += 1
print(f"identical consecutive same-layer in first 200: {same}")
