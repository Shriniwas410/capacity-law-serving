#!/usr/bin/env bash
set -euo pipefail
export PATH=/home/shri/anaconda3/bin:/usr/bin:/bin:/usr/local/bin
cd /home/shri/src/llama.cpp
echo "HEAD=$(git log -1 --oneline)"
# llama-cli lives under tools/cli, which is gated on LLAMA_BUILD_SERVER.
# llama-completion uses common_params_parse (has --moe-stream-direct) and is
# already in this tree. Reconfigure to add server/cli; keep existing object files.
cmake -S . -B build-pr25294 \
  -DGGML_CUDA=OFF -DGGML_VULKAN=OFF -DGGML_BLAS=OFF -DGGML_CCACHE=OFF \
  -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_EXAMPLES=OFF \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_MAKE_PROGRAM=/usr/bin/gmake \
  -G "Unix Makefiles"
# completion is enough for G1; cli is the long-term binary
cmake --build build-pr25294 -j"$(nproc)" --target llama-completion llama-cli
ls -lh build-pr25294/bin/llama-completion build-pr25294/bin/llama-cli
./build-pr25294/bin/llama-completion --help 2>/dev/null | grep -i moe | head -40 || true
