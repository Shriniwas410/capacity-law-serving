#!/usr/bin/env bash
# Stock llama.cpp speculative decoding: 0.6B drafts, 30B verifies.
# CPU-only. llama-cli needs -st (rejects -no-cnv).
# Cache 4 GiB (not 8s): verify batches are k+1 tokens and need >=24 slots/layer.
# --reasoning off so Qwen3 thinking does not eat the token budget.
set -euo pipefail
export PATH=/home/shri/anaconda3/bin:/usr/bin:/bin:/usr/local/bin
BIN_DIR=/home/shri/src/llama.cpp/build-pr25294/bin
export LD_LIBRARY_PATH="$BIN_DIR"
BIN="$BIN_DIR/llama-cli"
TGT=/home/shri/models/Qwen3-30B-A3B-Q4_K_M.gguf
DFT=/home/shri/models/Qwen3-0.6B-Q8_0.gguf
OUT=measurements/spec30b
N=${N:-128}
THREADS=${THREADS:-16}
PROMPT=${PROMPT:-"Write a short Python function that returns the nth Fibonacci number. Only the function."}
STREAM=(--moe-stream --moe-stream-direct --moe-stream-cache 4 --moe-stream-io-threads 4 -fit off)
mkdir -p "$OUT"
cd "$BIN_DIR"
HEAD=$(git -C /home/shri/src/llama.cpp log -1 --format='%H')

python3 - <<PY
import json, datetime
open("$OUT/config.json","w").write(json.dumps({
  "target": "Qwen3-30B-A3B-Q4_K_M",
  "draft": "Qwen3-0.6B-Q8_0",
  "n_predict": int("$N"),
  "temp": 0,
  "seed": 1,
  "ngl": 0,
  "moe_stream_direct": True,
  "moe_stream_cache": "4GiB",
  "reasoning": "off",
  "llama_cpp_git": "$HEAD",
  "started": datetime.datetime.now().isoformat(timespec="seconds"),
}, indent=2) + "\n")
PY

run() {
  local name="$1"; shift
  echo "=== $name $(date -Is) ===" | tee -a "$OUT/meta.txt"
  set +e
  "$BIN" -m "$TGT" -ngl 0 -n "$N" -t "$THREADS" --temp 0 --seed 1 -st \
    --reasoning off --simple-io -v -p "$PROMPT" "$@" \
    >"$OUT/${name}.stdout" 2>"$OUT/${name}.stderr"
  rc=$?
  set -e
  echo "exit=$rc $name" | tee -a "$OUT/meta.txt"
  echo "--- timings/spec ---" | tee -a "$OUT/meta.txt"
  grep -E "tok/s|eval time|prompt eval|Generation:|draft acceptance|accepted|tokens per second|load time" \
    "$OUT/${name}.stderr" "$OUT/${name}.stdout" 2>/dev/null | tee -a "$OUT/meta.txt" || true
}

: > "$OUT/meta.txt"
echo "HEAD=$HEAD n=$N bin=$BIN" | tee -a "$OUT/meta.txt"

run baseline "${STREAM[@]}"
# --model-draft loads the file; --spec-type draft-simple actually turns spec on
# (default types = NONE — first run loaded the 0.6B and never drafted).
run spec_k2 "${STREAM[@]}" --spec-type draft-simple --model-draft "$DFT" \
  --spec-draft-n-max 2 --spec-draft-n-min 1 -ngld 0
run spec_k4 "${STREAM[@]}" --spec-type draft-simple --model-draft "$DFT" \
  --spec-draft-n-max 4 --spec-draft-n-min 1 -ngld 0

python3 - <<'PY'
import re, json, pathlib
out = pathlib.Path("measurements/spec30b")

def read(name):
    return (out/f"{name}.stderr").read_text(errors="replace") + "\n" + (out/f"{name}.stdout").read_text(errors="replace")

def tps(text):
    m = re.findall(r"Generation:\s*([\d.]+)\s*t/s", text)
    if m:
        return float(m[-1])
    gen = re.findall(r"eval time\s*=.*?([\d.]+)\s*tokens per second", text, re.S)
    if gen:
        return float(gen[-1])
    all_ = re.findall(r"([\d.]+)\s*tokens per second", text)
    return float(all_[-1]) if all_ else None

def alpha(text):
    m = re.findall(r"draft acceptance\s*=\s*([\d.]+)\s*\(\s*(\d+)\s*accepted\s*/\s*(\d+)\s*generated\)", text)
    if not m:
        return None
    a, acc, tot = m[-1]
    return {"alpha": float(a), "accepted": int(acc), "generated": int(tot)}

arms = {}
for name in ("baseline", "spec_k2", "spec_k4"):
    text = read(name)
    rec = {"tok_s": tps(text), "draft": alpha(text)}
    arms[name] = rec
    print(name, rec)

base = arms["baseline"]["tok_s"]
verdict = {"arms": arms, "G3_alpha_ge_0.55": None}
for name in ("spec_k2", "spec_k4"):
    d = arms[name]["draft"]
    t = arms[name]["tok_s"]
    if base and t:
        verdict[f"{name}_speedup"] = t / base
    if d:
        verdict[f"{name}_alpha"] = d["alpha"]
        if verdict["G3_alpha_ge_0.55"] is None:
            verdict["G3_alpha_ge_0.55"] = d["alpha"] >= 0.55
        else:
            verdict["G3_alpha_ge_0.55"] = verdict["G3_alpha_ge_0.55"] or d["alpha"] >= 0.55
(out/"verdict.json").write_text(json.dumps(verdict, indent=2) + "\n")
print(json.dumps(verdict, indent=2))
PY

echo "SPEC30B_DONE $(date -Is)" | tee -a "$OUT/meta.txt"
