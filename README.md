# capacity-law-serving

**Can stock speculative decoding speed up a 235B MoE on a 32 GB PC?**
A measurement study: routing traces, the expert-union law, and why `-ub 1` loses
then serial-wave `-ub 8` only returns to parity. All decode numbers are CPU-only
llama.cpp on one Windows/WSL2 box (RTX 3070 left idle).

This is **Paper 2**, not [Cacheable by Design?](https://github.com/Shriniwas410/cacheable-by-design)
(arXiv:2608.18261, 137M router training). Do not mix those tok/s tables.

📄 Paper: [`paper/paper.pdf`](paper/paper.pdf) · Results: [`RESULTS.md`](RESULTS.md)

## TL;DR

Qwen3-235B-A22B Q4_K_M is 133 GB. Full CPU mmap dies at a 105 GB `CPU_REPACK`
buffer. Stream-direct decode is **0.29 tok/s** (8-slot) and **0.38 tok/s**
(12-slot, 16.70 GiB RSS). Top-8 traces have shape `(94, 2048, 8)`; mean
`U(4)=18.98`. A 0.6B draft has α **0.75–0.79** on a 16-token smoke, but
`-ub 1` verify is **0.80×**; serial waves at `-ub 8` are ~parity, not 1.5×.
`--moe-stream-umax` v1 **lost** (α=0). Per-layer-mean HorizonSpec T=1.54 is
**not** engine min T=1.001.

## Repository layout

| Path | What |
|---|---|
| `RESULTS.md` | Measured findings F1–F15 with provenance |
| `measurements/` | Verdicts, configs, fio TSV, union JSON, prefetch sim |
| `measurements/traces/` | Four domain `.npz` traces (94×2048×8) |
| `engine/patches/` | llama.cpp patches on `1248fd8fa` (serial waves, skip-empty GEMM, umax v1) |
| `llama-moe-trace/` | Router-telemetry add-on (weights untouched) |
| `paper/` | LaTeX, bibliography, `verify_refs.py` |

Paper appendix tables point at files under `measurements/` (this used to be a local `week1/` dump).

## Reproduce analysis (no 235B weights required)

```bash
python measurements/uk_umax.py          # needs traces in measurements/traces/
python measurements/_min_layer_T.py measurements/traces 2 8,12,16
python measurements/g2a_prefetch_235b.py
python paper/verify_refs.py             # 0 HARD vs local PDF corpus (PDFs not in git)
```

Decode timings need Qwen3-235B-A22B Q4_K_M GGUF, Qwen3-0.6B-Q8_0, and llama.cpp
`1248fd8fa` (PR#25294 stream-direct) plus `engine/patches/`. CPU only (`-ngl 0`).
Never load 235B on an 8 GB GPU while other training holds VRAM.

## What is NOT in the repo (and why)

GGUF weights, llama.cpp binaries, and copyrighted reference PDFs
(`paper/refpdfs/*.pdf`) are excluded — same policy as Cacheable by Design.
Verdicts, configs, traces, and the I/O TSV **are** included so every headline
number has a file.

## Paper numbers → files

| Claim | File |
|---|---|
| O_DIRECT ~2.35 GB/s | `measurements/fio_summary.tsv` |
| 8s / 12s greedy 0.29 / 0.38 tok/s | `measurements/ab235b_verdict.json`, `ab235b_12s_verdict.json` |
| U(4)=18.98 | `measurements/uk_235b.json` |
| HSpec per-layer-mean T=1.54 | `measurements/uk_umax_235b.json` |
| Engine min T=1.001 | `measurements/_min_layer_T.json` |
| spec `-ub 1` 0.80× | `measurements/spec235b_verdict.json` |
| spec `-ub 8` ~parity | `measurements/spec235b_ub8/verdict.json` |
| umax v1 lose α=0 | `measurements/spec235b_umax/verdict.json` |
| lag-2 prefetch 0% | `measurements/g2a_235b.json` |

## Citation

Companion training paper: [arXiv:2608.18261](https://arxiv.org/abs/2608.18261).
This measurement preprint has no arXiv id yet.

Author: Shriniwas Ramesh Suram
([ORCID 0009-0009-0452-9407](https://orcid.org/0009-0009-0452-9407)).

## License

MIT — see [`LICENSE`](LICENSE).
