#!/usr/bin/env python3
"""Fail a llama-moe-trace .bin / .npz if the argsort-view bug is back."""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys
from collections import defaultdict

import numpy as np


def read_records(path):
    with open(path, "rb") as f:
        while True:
            head = f.read(12)
            if len(head) < 12:
                break
            layer, n_tokens, k = struct.unpack("<iii", head)
            raw = f.read(4 * n_tokens * k)
            if len(raw) < 4 * n_tokens * k:
                raise SystemExit(f"truncated record layer={layer}")
            data = np.frombuffer(raw, dtype="<i4")
            yield layer, data.reshape(n_tokens, k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default=None)
    ap.add_argument("--npz", default=None)
    ap.add_argument("--num-experts", type=int, required=True)
    args = ap.parse_args()
    if args.npz:
        z = np.load(args.npz, allow_pickle=True)
        experts = z["experts"]
        meta = json.loads(str(z["meta"]))
        print("npz", experts.shape, meta)
        flat = experts.reshape(-1)
    elif args.bin:
        per = defaultdict(list)
        for layer, ids in read_records(args.bin):
            per[layer].append(ids)
        stacked = [np.concatenate(per[l], axis=0) for l in sorted(per)]
        experts = np.stack(stacked)
        print("bin", experts.shape, "layers", sorted(per))
        flat = experts.reshape(-1)
    else:
        sys.exit("need --bin or --npz")

    n = args.num_experts
    bad = int(((flat < 0) | (flat >= n)).sum())
    uniq, counts = np.unique(flat, return_counts=True)
    uniform = bool(len(uniq) == n and counts.min() == counts.max())
    print(
        f"id_range=[{int(flat.min())},{int(flat.max())}]  out_of_range={bad}  "
        f"n_unique={len(uniq)}  count_min={int(counts.min())} count_max={int(counts.max())}  "
        f"uniform_bug={uniform}"
    )
    if bad or uniform or int(flat.min()) < 0 or int(flat.max()) >= n:
        sys.exit("TRACE_INVALID")
    print("TRACE_OK")


if __name__ == "__main__":
    main()
