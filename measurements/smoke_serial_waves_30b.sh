#!/usr/bin/env bash
# 30B smoke: serial waves on an 8-slot cache (n_slots < 3*n_expert_used=24).
# If the graph patch works, -ub 4 must not GGML_ABORT.
set -euo pipefail
export PATH=/home/shri/anaconda3/bin:/usr/bin:/bin:/usr/local/bin
BIN_DIR=/home/shri/src/llama.cpp/build-pr25294/bin
export LD_LIBRARY_PATH="$BIN_DIR"
BIN="$BIN_DIR/llama-completion"
MODEL=/home/shri/models/Qwen3-30B-A3B-Q4_K_M.gguf
OUT=measurements/smoke_serial_waves_30b
N=${N:-8}
CTX=${CTX:-512}
THREADS=${THREADS:-16}
PROMPT=${PROMPT:-"The capital of France is"}
mkdir -p "$OUT"
cd "$BIN_DIR"
HEAD=$(git -C /home/shri/src/llama.cpp log -1 --format='%H')
DIFF=$(git -C /home/shri/src/llama.cpp diff --stat -- src/llama-graph.cpp src/llama-context.cpp | tr '\n' ' ')

python3 - <<PY
import json, datetime
open("$OUT/config.json","w").write(json.dumps({
  "model": "Qwen3-30B-A3B-Q4_K_M",
  "model_path": "$MODEL",
  "n_predict": int("$N"),
  "n_ctx": int("$CTX"),
  "n_batch": 8,
  "n_ubatch": 4,
  "moe_stream_cache": "8s",
  "temp": 0,
  "seed": 1,
  "ngl": 0,
  "threads": int("$THREADS"),
  "prompt": "$PROMPT",
  "fit": "off",
  "moe_stream": True,
  "moe_stream_direct": True,
  "llama_cpp_git": "$HEAD",
  "llama_cpp_diffstat": "$DIFF",
  "purpose": "serial-wave smoke: 8 slots + ub=4 must exit 0 and log serial waves",
  "started": datetime.datetime.now().isoformat(timespec="seconds"),
}, indent=2) + "\n")
PY

: > "$OUT/meta.txt"
echo "HEAD=$HEAD n=$N" | tee -a "$OUT/meta.txt"
set +e
/usr/bin/time -v "$BIN" -m "$MODEL" -ngl 0 -n "$N" -t "$THREADS" --temp 0 --seed 1 \
  -no-cnv -st --reasoning off -p "$PROMPT" \
  --moe-stream --moe-stream-direct --moe-stream-cache 8s --moe-stream-io-threads 4 \
  -fit off -c "$CTX" -b 8 -ub 4 \
  >"$OUT/run.stdout" 2>"$OUT/run.stderr"
rc=$?
set -e
echo "exit=$rc" | tee -a "$OUT/meta.txt"

python3 - <<PY
import re, json, pathlib, datetime
out = pathlib.Path("measurements/smoke_serial_waves_30b")
err = (out/"run.stderr").read_text(errors="replace")
out_s = (out/"run.stdout").read_text(errors="replace")
blob = err + "\n" + out_s
def tps(kind):
    pat = r"eval time\s*=.*?([\d.]+)\s*tokens per second" if kind=="eval" else r"prompt eval time\s*=.*?([\d.]+)\s*tokens per second"
    m = re.findall(pat, blob, re.S)
    return float(m[-1]) if m else None
verdict = {
  "exit": int("$rc"),
  "serial_wave_log": bool(re.search(r"using serial waves of", err)),
  "aborted": bool(re.search(r"GGML_ABORT|GGML_ASSERT|Aborted", blob)),
  "eval_tok_s": tps("eval"),
  "prompt_tok_s": tps("prompt"),
  "n_ctx": (lambda m: int(m[-1]) if m else None)(re.findall(r"n_ctx\s*=\s*(\d+)", blob)),
  "PASS": int("$rc")==0 and bool(re.search(r"using serial waves of", err)),
  "finished": datetime.datetime.now().isoformat(timespec="seconds"),
}
(out/"verdict.json").write_text(json.dumps(verdict, indent=2)+"\n")
print(json.dumps(verdict, indent=2))
PY
echo "SMOKE30B_SERIAL_WAVES_DONE $(date -Is)" | tee -a "$OUT/meta.txt"
exit "$rc"
