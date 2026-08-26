# Measured findings — Qwen3-235B-A22B on one consumer box

**Date:** 2026-08-23 – 2026-08-25  
**Machine:** Windows 11 + WSL2 Ubuntu, i9-12900 (16C/24T), RTX 3070 8 GB, ~32 GB host RAM, `.wslconfig` 22 GB (WSL `free` ≈ 21 Gi), WD NVMe.  
**Rule:** every number below traces to a `config.json` / verdict file you can open. Do not overlay series that differ in `tokens`, `params`, `seq`, device split, or binary.

This is a **measurement report** for a systems paper, not a trained-model result and not a TMLR resubmission of Paper 1. DirectRing / PR#25294 is **substrate**, not a novelty claim.

Paper 1 (*Cacheable by Design?*, arXiv 2608.18261) is a separate artifact. Do not transplant its 30B / 137M cache numbers onto 235B.

---

## What you can claim (measured)

### F1. O_DIRECT on this VHDX is ~2.35 GB/s (G1a PASS)

Random reads of the 30B GGUF on ext4, fio 3.39, 20 s, `measurements/fio_summary.tsv`, `/home/shri/week1_fio/`.

| job | GB/s |
|---|---:|
| mmap QD1 10 MB | 0.064 |
| pread QD1 10 MB | 1.377 |
| O_DIRECT QD1 10 MB | 1.856 |
| **O_DIRECT QD8 3.5 / 10 / 16 MB** | **2.368 / 2.339 / 2.354** |

G1a bar was ≥ 2.0 GB/s on ≥ 8 MB reads. **PASS.** mmap-fault in this matrix is ~0.06 GB/s (page-fault path), not the 30B *decode* mmap number below.

### F2. 30B decode: stream-direct 1.99× vs mmap (G1b PASS)

`measurements/ab30b/verdict.txt`. `llama-completion`, n=16, temp 0, CPU, HEAD `1248fd8fa`.

| arm | eval tok/s |
|---|---:|
| mmap | 1.71 |
| `--moe-stream-direct` cache 4 GiB | **3.41** |

Ratio **1.994**. G1b bar ≥ 1.30. **PASS.**

### F3. 235B routing traces exist (G0 PASS) — 94 MoE layers, not 48

Provenance: `/home/shri/measurements/g0_235b/` + `measurements/uk_235b.json`.  
`llama-moe-trace`, CPU, `--moe-stream-direct --moe-stream-cache 8s -fit off -b 1 -ub 1`, 2048 tokens × 4 domains, window 512, TRACE_OK, git `1248fd8fa`.

Shape **(94, 2048, 8)** top-8 of 128. Do not use 30B `moe-routing-lab/traces/` as 235B.

| domain | U(4) | U(4)/32 | lag-1 reuse / chance |
|---|---:|---:|---|
| code | 18.35 | 0.573 | 0.479 / 0.256 (**1.87×**) |
| general | 18.87 | 0.590 | 0.463 / 0.203 (**2.27×**) |
| math | 20.00 | 0.625 | 0.406 / 0.186 (**2.19×**) |
| medical | 18.71 | 0.585 | 0.459 / 0.236 (**1.95×**) |
| **mean** | **18.98** | ~0.59 | ~2.1× |

Roadmap G2b `U(4)≤16`: **FAIL** (mean 18.98). Union sharing is real (`demand_vs_naive` ~0.59) but the union is larger than 16.

### F4. HorizonSpec `U_max` is a slot count, not a slogan

`measurements/uk_umax_235b.json` — target-oracle routing, longest-prefix admission. Not tok/s.

Consecutive tokens share an **identical** 8-expert set **0.26%** of the time. With top-8 routing, `U_max=8` admits k>1 only then. Mean admitted T at 8 slots = **1.003** (greedy).

| cap | naive k=2 fit | HSpec k=2 mean T | naive k=4 fit | HSpec k=4 mean T |
|---|---:|---:|---:|---:|
| 8 slots | 0.26% | 1.003 | ~0% | 1.003 |
| 12 slots | 54% | 1.54 | 1.2% | 1.64 |
| 16 slots | 100% | 2.00 | 23% | 2.83 |
| 24 slots | 100% | 2.00 | 94.6% | 3.95 |

Naive k=4 unique experts/token = **4.75** vs greedy 8 (union win) **if** the batch can load ~19 experts. Stock overlap waves want **24** slots/layer; F13 serial waves already run at 12.

**Engine vs F4 aggregation.** F4 `mean_admitted_T` averages over 94 layers (`uk_umax.py` `layer_reduce`). The umax patch takes `min(admitted_n)` across layers. On `g0_235b`, 12-slot k=2 engine T = **1.001**, p(T≥2) = **0.12%**, mean overflow layers **43** (`measurements/_min_layer_T.json`). Do not treat F4 T=1.54 as what `--moe-stream-umax` will admit.

### F5. Full CPU mmap of 235B is impossible here

`measurements/ab235b/mmap_repack_oom.stderr`: `ggml_aligned_malloc` **105 495.75 MB** `CPU_REPACK` buffer (cited as 105 496 MB). WSL has 21 Gi.

### F6. The 0.44 tok/s figure is a different stack

2026-08-05 Windows `llama-server -ngl 99 -ncmoe 94`: warm decode **0.441 tok/s**, cold **0.128 tok/s** (`SESSION-LOG-2026-08.md`). GPU non-MoE + CPU MoE. **Not comparable** to CPU-only `llama-completion` / `llama-cli` in this week.

### F7. CPU 235B greedy decode (the number to cite for this binary)

**A.** `llama-completion`, n=16, ctx=512, `-fit off -b 1 -ub 1`, temp 0, prompt “The capital of France is” (5 prompt / 15 eval tokens). `measurements/ab235b_verdict.json`, `measurements/ab235b_12s_verdict.json`.

| date | cache | eval tok/s | prompt tok/s | peak RSS |
|---|---|---:|---:|---:|
| 2026-08-25 00:13 | 8s page-cache stream | 0.26 | 0.12 (cold) | — |
| 2026-08-25 00:13 | 8s O_DIRECT | 0.29 | 0.24 (warm, 2nd) | — |
| 2026-08-25 11:51 | **12s O_DIRECT** (1st, cold) | **0.38** | 0.11 | **16.70 GiB** |
| 2026-08-25 11:51 | 8s O_DIRECT (2nd) | 0.30 | 0.27 | 12.70 GiB |

12s **fits**. Same-day 12s/8s eval **1.27×** (12s was colder). GDN 16-token probe is skipped at 8s and 12s (`slots < 24`).

**B.** `llama-cli` same 12s, Fibonacci chat prompt (28 prompt / 16 eval tokens), `-b 512 -ub 1`. `measurements/spec235b_verdict.json`. Baseline eval **0.30 tok/s**, RSS 16.70 GiB. Different prompt/binary — do not average with F7-A.

### F8. `llama-cli -b 1` cannot even greedy-decode 235B stream

`/home/shri/measurements/spec235b/`, exit 134.  
`llama-context.cpp:1781 GGML_ASSERT(n_tokens_all <= cparams.n_batch)` after skipping the GDN probe. Chat template submits >1 token while `-b 1`. **Fix:** `-b 512 -ub 1` (physical ubatch stays 1). `llama-completion -b 1` already splits and works (F7-A).

### F9. 30B speculation: α PASS, speedup null

`measurements/spec30b_verdict.json`. 0.6B draft, 30B target, `--spec-type draft-simple`, cache 4 GiB, 37 gen tokens, 1 prompt.

| arm | tok/s | α |
|---|---:|---|
| baseline | 1.96 | — |
| k=2 | 1.99 | **0.857** (24/28) |
| k=4 | 2.01 | **0.833** (30/36) |

G3 α≥0.55 **smoke PASS**. Wall-clock **~1.0×**: 30B at 4 GiB expert cache is not SSD-bound. **Not** the 200-prompt G3 gate.

### F10. 235B speculation at 12s: α PASS; `-ub 1` loses; serial-wave `-ub 8` recovers to ~1.0×

**F10a — stock verify (`-ub 1`), 2026-08-25 18:51 ET.** `measurements/spec235b_verdict.json`. 0.6B draft, 12s, `-b 512 -ub 1`, n=16, Fibonacci prompt. All exit 0. RSS ~17.6 GiB.

| arm | eval tok/s | α | mean accept len | vs baseline |
|---|---:|---:|---:|---:|
| baseline | **0.30** | — | — | 1.00× |
| spec k=2 | 0.24 | **0.75** (9/12) | 2.50 | **0.80×** |
| spec k=4 | 0.23 | **0.79** (11/14) | 3.75 | **0.77×** |

Same-family 0.6B is a usable drafter (α≥0.55 smoke). Stock spec **loses** because `-ub 1` serializes verify: no union amortization, plus draft cost.

**F10b — batched verify (serial waves, `-ub 8`), 2026-08-25 20:22–20:29 ET.** `measurements/spec235b_ub8/verdict.json`. Same prompt, n=16, 12s, `-b 512 -ub 8`, llama.cpp `1248fd8fa` + local serial-wave patches. Slot `print_timing` (not stdout `0.3 t/s` rounding).

| arm | eval tok/s | eval ms/tok | prompt tok/s (28 tok) | α | vs this baseline |
|---|---:|---:|---:|---:|---:|
| baseline | **0.32** | 3089.56 | 0.44 | — | 1.00× |
| spec k=2 | 0.31 | 3248.57 | 0.42 | **0.75** (9/12) | **0.97×** |
| spec k=4 | 0.32 | 3162.69 | 0.42 | **0.79** (11/14) | **1.00×** |

Two-decimal tok/s give k=4 **1.00×**; eval ms/tok is **0.951× / 0.977×**. Prompt **0.44** is llama-cli prefill (28 tok), not F6 GPU decode. Batched verify **stops the 0.80× tax**. It does **not** deliver the F4 T≈1.54 projection: this run still GEMMs masked pairs, has no `U_max` (F15 is a later lose), n=16 is a smoke. Do not cite 0.97× as HorizonSpec speedup.

### F12. Causal prefetch on 235B traces is a NO-GO (G2a FAIL)

`measurements/g2a_235b.json`. Per-layer cache, prefetch **charges a slot**, objective = **demand** misses (not prefetch traffic). `miss_per_tok` with `n_experts=128` from meta.

At **cap 12.5% (16 experts)**:

| domain | LRU miss/tok | OPT | OPT W=8 | lag-2 prefetch | oracle-next (non-causal) | LRU hit |
|---|---:|---:|---:|---:|---:|---:|
| code | 279.8 | 175.5 | 179.4 | 279.8 | 22.7 | 0.628 |
| general | 304.2 | 200.7 | 205.4 | 304.2 | 22.7 | 0.596 |
| math | 348.4 | 228.4 | 232.5 | 348.4 | 27.8 | 0.537 |
| medical | 293.6 | 188.6 | 193.2 | 293.6 | 25.1 | 0.610 |

Lag-2 recovers **0%** of the LRU→W8 gap. G2a bar (≥10%) **FAIL**. Prefetch W=0.

At cap **8 and 12**, lag-2 **raises** demand misses (recovery −0.5 to −1.0): prefetches evict what the next token needs. Frequency-pin `static` is worse than LRU at cap=16 (code 307 vs 280) — same direction as Paper 1 Result 1b, now on **235B traces**.

**oracle-next** (load the actual next token’s 8 experts) cuts demand misses ~280 → ~23. That is unreachable as causal prefetch and is exactly the lookahead a draft model can supply if verify can hold the union. It is the case for HorizonSpec, not for lag-1/lag-2 prefetch.

### F13. 12-slot serial waves make multi-token ubatch legal; 235B prefill 0.16 → 0.28 tok/s (warm)

llama.cpp stock aborts a multi-token graph when `n_slots < 3*n_expert_used` (24). Local patch: serial waves with `cap = n_slots`, wait if a later wave custom-op is entered first (OpenMP assert otherwise hangs), and **do not pin the next wave’s resident experts when `cap == n_slots`** (that deadlock showed up on the second 4-token ubatch, layer 1: `/tmp/moe_wave.log` `victim-wait`).

30B smoke: 8s + `-ub 4`, exit 0, `measurements/smoke_serial_waves_30b/verdict.json`.

235B `llama-completion` 12s, Fibonacci prompt **16 prompt tokens**, n=1, `measurements/ab235b_ubatch/` (`verdict.json` tok/s; `ub1.stderr` / `ub4.stderr` ms/tok and RSS):

| arm | prompt tok/s | ms/tok | peak RSS | note |
|---|---:|---:|---:|---|
| `-b 1 -ub 1` | **0.16** | 6400.29 | 16.69 GiB | first (colder) |
| `-b 8 -ub 4` | **0.28** | 3523.06 | 16.70 GiB | later; **1.75× vs that ub1**, warmer |

Do not treat 1.75× as a cold A/B. Direction is real: ub4 **exits 0** (ub4 without the overlap-guard hung). llama-cli baseline prompt 0.44 tok/s at `-ub 8` (28 tokens, F10b) is a different binary/token count — do not overlay on the 16-token llama-completion pair.

### F14. Empty-wave GEMM skip does not move 235B spec (2026-08-25 21:04–21:12 ET)

`measurements/spec235b_skip/verdict.json`. Same protocol as F10b after `ggml_set_skip_mul_mat_id`. Slot `print_timing`.

| arm | F10b eval ms/tok | skip eval ms/tok | skip eval tok/s | α |
|---|---:|---:|---:|---:|
| baseline | 3089.56 | 3074.59 | 0.33 | — |
| spec k=2 | 3248.57 | 3251.26 | 0.31 | 0.75 |
| spec k=4 | 3162.69 | 3038.80 | 0.33 | 0.79 |

k=2 is unchanged. k=4 is 4% faster on a later warmer arm; **do not claim a skip speedup**. The ~1.0× F10b result is not an empty-wave GEMM artifact.

### F15. HorizonSpec `--moe-stream-umax` v1 on 235B: α=0, **0.79× / 0.73×** (FAIL as speedup)

`measurements/spec235b_umax/verdict.json`. Log: `U_max=12: admitting 1/3 tokens` (k=2) and `1/5` (k=4), uniq=8. That is T=1 of [sampled, drafts] — **expected** layout, not a row-index bug. Eval 0.33 → 0.26 / 0.24 tok/s. Draft 0/27 and 0/50 accepted. T=1 blocks F4 T≈1.54 / later drafts; it does **not** skip the first draft compare. α=0 means row-0 logits ≠ F10b. Do not combine F14 empty-wave skip with umax (`plan_n_waves=1` vs compiled 2-wave graph). The flag is in the engine; it is **not** F4. Do not put these tok/s in Paper 2 as a win.

### F11. Expert-table bytes ÷ RAM slab is the feasibility variable

235B Q4_K_M expert table ~127 GB. 12 slots × 94 layers × 10.85 MB = **11.96 GB** expert cache; measured peak **16.70 GiB**. 8 slots peak **12.70 GiB**. 16s would add ~4 GiB on top of 16.7 — too tight for 21 Gi without a dedicated OOM budget. DeepSeek-V3 is 671B total / 37B activated; a 4-bit table for that geometry is larger than this 133 GB checkpoint. That is slab arithmetic, not a timing result. Do not cite a 355–600 GB figure as if it came from the DeepSeek PDF.

---

## What you must not claim

- **Do not** cite 0.44 tok/s as this week’s CPU baseline (F6 vs F7).
- **Do not** cite 30B U(4) or 30B miss fractions as 235B (G0).
- **Do not** cite 30B spec 1.0× as “speculation does not work”; it does not move wall-clock when the cache already holds the working set (F9). On 235B, α is high and `-ub 1` wall-clock **drops** (F10a); serial-wave `-ub 8` returns to ~1.0× (F10b), not a 1.5× win.
- **Do not** headline DirectRing / O_DIRECT; it is taken (PR#25294 + mbolt). Publish it as the measured BW matrix (F1–F2).
- **Do not** claim HorizonSpec engine speedup. F4 (`T≈1.54`) is trace-oracle simulation. Engine `--moe-stream-umax` v1 **did run** and **lost** (F15: α=0, 0.79×/0.73×). Serial waves are a **batched-verify substrate**, not HorizonSpec. Do not write “U_max is unimplemented.”
- **Do not** claim G2b PASS: `U(4)≤16` failed.
- **Do not** claim causal prefetch helps at 12.5% cap (F12).
- n=16 / 1-prompt spec numbers are **smokes**, not the 200-prompt G3 gate.
- **Do not** claim empty-wave GEMM skip sped up 235B spec (F14). k=2 unchanged; k=4 later/warmer.

---

## Paper-shaped contribution (honest)

**Title angle:** *Capacity-law serving of a 235B-class MoE on 32 GB host (GPU idle): routing traces, union law, and why stock speculation loses.*

1. **Qwen3-235B-A22B top-8 routing traces** (94×2048×8 × 4 domains) with provenance (F3). Do not claim “first public” without a literature search that survived PDF ingest.
2. **Measured U(k) / identical-set rate** showing 8-slot speculation is a no-op. 12-slot k=2 **per-layer-mean** fit is ~54% / T=1.54 (F4). **Engine min-over-94-layers** at the same cap is T=1.001, p(T≥2)=0.12% (`measurements/_min_layer_T.json`).
3. **CPU tok/s ladder** on one box: 8s 0.29 → 12s 0.38 greedy; mmap impossible (F5, F7).
4. **Paired speculation result:** α high on 30B and 235B; wall-clock null on 30B (cache-resident); **negative** on 235B with `-ub 1`; **parity** with serial-wave `-ub 8` (F9–F10). That motivates admission-time `U_max` on top of batched verify, not a bigger draft k.
5. **Negatives:** G2b `U(4)≤16` fail; G2a causal prefetch **NO-GO** (F12); llama-cli `-b 1` assert (F8).

Venue: measurement / systems (MLSys-shaped only after batched-verify + `U_max` actually moves 235B tok/s). Not a 2026 TMLR resubmit of Paper 1.

---

## Remaining work (not done; required for an end-to-end speedup claim)

1. **Done (substrate):** serial waves at 12 slots so `-ub > 1` does not abort/hang (F13). Empty-wave GEMM skip **measured** (F14): no k=2 move.
2. **HorizonSpec v1 ran and lost (F15):** T=1 (1/3, 1/5) plus α=0 from row-0 logit mismatch. A `U(2)≤12` gate will not admit T=2 (engine p(T≥2)=0.12%). v2: pin row 0; always FFN rows 0–1 (serial waves if U(2)>12); mask suffix by token row `t≥T`, not `expert_wave=0xff`; **do not** `ggml_set_skip_mul_mat_id` when umax truncated; keep `n_loop=min(k, admitted_n)` (not T−1); 30B n=16 `-v` smoke first. Until α recovers, Paper 2 spec table stays F10b.
3. G3: ~200 prompts for α (current n=16 / 37 tokens are smokes).
4. Optional: 16s cache only with more WSL RAM; GateTail/ColdTier only after IO share is profiled.

Paper 2 draft: `paper/paper2.tex` / `paper/paper2.pdf`. Cites: `python paper/verify_refs.py` → 0 HARD. Not a TMLR resubmit of Paper 1.

---

## Artifact index

| File | What |
|---|---|
| `measurements/fio_summary.tsv` | F1 |
| `measurements/ab30b/verdict.txt` | F2 |
| `measurements/uk_235b.json` + `/home/shri/measurements/g0_235b/*.npz` | F3 |
| `measurements/uk_umax_235b.json` | F4 |
| `measurements/_min_layer_T.json` | engine min-over-layers T vs F4 mean |
| `measurements/ab235b/mmap_repack_oom.stderr` | F5 |
| `measurements/ab235b_verdict.json` | F7 8s page vs direct |
| `measurements/ab235b_12s_verdict.json` | F7 12s vs 8s |
| `measurements/spec30b_verdict.json` | F9 |
| `/home/shri/measurements/spec235b/` | F8 abort |
| `measurements/spec235b_verdict.json` | F10a `-ub 1` spec |
| `measurements/spec235b_ub8/verdict.json` | F10b `-ub 8` serial-wave spec |
| `measurements/spec235b_skip/verdict.json` | F14 empty-wave GEMM skip |
| `measurements/spec235b_umax/verdict.json` | F15 HorizonSpec umax v1 (α=0 lose) |
| `paper/paper2.tex` / `paper/paper2.pdf` | Paper 2 draft |
| `measurements/ab235b_ubatch/{verdict.json,ub1.stderr,ub4.stderr}` | F13 ub1 vs ub4 prefill |
| `measurements/smoke_serial_waves_30b/verdict.json` | F13 30B wave smoke |
| `measurements/g2a_235b.json` | G2a prefetch sim |
| `measurements/STATUS.md` | running log |
| llama.cpp `1248fd8fa` `pr-25294` + GDN-skip + serial-wave patches | engine |

GPU was idle (~1 GB) for all of the above. Eval/decode was **CPU-only** (`-ngl 0`, `GGML_CUDA=OFF`).
