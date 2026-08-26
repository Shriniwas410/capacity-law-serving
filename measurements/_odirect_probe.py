#!/usr/bin/env python3
import os, sys, errno
paths = sys.argv[1:] or [
    "/home/shri/models/Qwen3-30B-A3B-Q4_K_M.gguf",
    "/home/shri/models/qwen3-235b/Qwen3-235B-A22B-Q4_K_M-00001-of-00003.gguf",
]
for p in paths:
    try:
        fd = os.open(p, os.O_RDONLY | os.O_DIRECT)
        os.close(fd)
        print(f"O_DIRECT OK  {p}")
    except OSError as e:
        print(f"O_DIRECT FAIL {p}: {e}")
        sys.exit(1)
