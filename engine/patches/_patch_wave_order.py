#!/usr/bin/env python3
from pathlib import Path

p = Path("/home/shri/src/llama.cpp/src/llama-moe-stream.cpp")
t = p.read_text()
old = """    mgr->stats.n_wave_calls++;

    if (w == 0) {
        mgr->plan_waves_locked(*sl, ids, n);
    }
    GGML_ASSERT(sl->plan_next_wave == w); // waves must run in order (enforced by the graph ordering token)

    const uint32_t n_ids = (uint32_t) a->ne[0]; // experts per token (n_expert_used)

    mgr->stage_wave_locked(lk, *sl, w, n_ids); // make this wave resident, preload the next, build the pool
    sl->plan_next_wave = w + 1;

    mgr->emit_wave_slots(*sl, ids, out, w, n_ids, a->ne[1]);
"""
new = """    mgr->stats.n_wave_calls++;

    // Wave custom-ops are supposed to run in order via a 1-element view of the previous
    // wave's GEMM output. On the CPU backend a later wave's ids op can still be entered
    // first on a fat graph (94 layers x several waves). GGML_ASSERT in an OpenMP region
    // then hangs the process instead of aborting. Wait (mtx released) for the predecessor.
    if (w == 0) {
        mgr->plan_waves_locked(*sl, ids, n);
    } else {
        while (sl->plan_next_wave != w) {
            if (mgr->load_failed) {
                GGML_ABORT("MoE expert streaming: expert load failed (I/O error)");
            }
            mgr->cv_done.wait(lk);
        }
    }

    const uint32_t n_ids = (uint32_t) a->ne[0]; // experts per token (n_expert_used)

    mgr->stage_wave_locked(lk, *sl, w, n_ids); // make this wave resident, preload the next, build the pool
    sl->plan_next_wave = w + 1;
    mgr->cv_done.notify_all();

    mgr->emit_wave_slots(*sl, ids, out, w, n_ids, a->ne[1]);
"""
old_mask = """    std::lock_guard<std::mutex> lock(mgr->mtx);

    GGML_ASSERT(sl->plan_next_wave > w); // this wave's ids op has already run
"""
new_mask = """    std::unique_lock<std::mutex> lock(mgr->mtx);

    while (sl->plan_next_wave <= w) {
        if (mgr->load_failed) {
            GGML_ABORT("MoE expert streaming: expert load failed (I/O error)");
        }
        mgr->cv_done.wait(lock);
    }
"""
if old not in t:
    raise SystemExit("wave_ids block not found")
if old_mask not in t:
    raise SystemExit("wave_mask block not found")
p.write_text(t.replace(old, new, 1).replace(old_mask, new_mask, 1))
print("patched wave_ids and wave_mask")
