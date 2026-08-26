from pathlib import Path
p = Path("/home/shri/src/llama.cpp/build-pr25294/CMakeFiles/CMakeConfigureLog.yaml")
t = p.read_text(errors="replace")
print("len", len(t))
lines = t.splitlines()
for i, l in enumerate(lines):
    low = l.lower()
    if "error" in low or "failed" in low or "gmake" in low or "No such" in l:
        print(f"{i}: {l[:240]}")
print("---TAIL---")
print("\n".join(lines[-80:]))
