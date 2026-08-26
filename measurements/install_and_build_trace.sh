#!/usr/bin/env bash
# Install moe-trace into llama.cpp tools, reconfigure, build.
set -euo pipefail
export PATH=/home/shri/anaconda3/bin:/usr/bin:/bin:/usr/local/bin
SRC_CPP=measurements/llama-moe-trace/moe-trace.cpp
SRC_CMA=measurements/llama-moe-trace/CMakeLists.txt
DST=/home/shri/src/llama.cpp/tools/moe-trace
mkdir -p "$DST"
# strip CR so bash/cpp are unix
sed 's/\r$//' "$SRC_CPP" > "$DST/moe-trace.cpp"
sed 's/\r$//' "$SRC_CMA" > "$DST/CMakeLists.txt"
# strip CR from week1 shell scripts
for f in measurements/*.sh; do
  sed -i 's/\r$//' "$f" || true
done
# ensure tools/CMakeLists has moe-trace
if ! grep -q 'add_subdirectory(moe-trace)' /home/shri/src/llama.cpp/tools/CMakeLists.txt; then
  echo "ERROR: tools/CMakeLists.txt missing moe-trace subdirectory"
  grep -n completion /home/shri/src/llama.cpp/tools/CMakeLists.txt
  exit 1
fi
cd /home/shri/src/llama.cpp
echo "HEAD=$(git log -1 --oneline)"
cmake -S . -B build-pr25294 \
  -DGGML_CUDA=OFF -DGGML_VULKAN=OFF -DGGML_BLAS=OFF -DGGML_CCACHE=OFF \
  -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_EXAMPLES=OFF \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_MAKE_PROGRAM=/usr/bin/gmake \
  -G "Unix Makefiles"
cmake --build build-pr25294 -j"$(nproc)" --target llama-moe-trace
ls -lh build-pr25294/bin/llama-moe-trace
