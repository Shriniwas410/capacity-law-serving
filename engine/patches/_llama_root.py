"""Resolve the llama.cpp checkout the patch scripts edit."""
from __future__ import annotations

import os
from pathlib import Path


def llama_root() -> Path:
    return Path(os.environ.get("LLAMA_CPP", "/home/shri/src/llama.cpp"))
