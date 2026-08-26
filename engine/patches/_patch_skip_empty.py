#!/usr/bin/env python3
"""Empty-wave GEMM skip: ggml flag + wave_ids/wave_mask + mul_mat_id early-out."""
from pathlib import Path

from _llama_root import llama_root

ROOT = llama_root()
GGML_H = ROOT / "ggml/include/ggml.h"
GGML_C = ROOT / "ggml/src/ggml.c"
CPU_C = ROOT / "ggml/src/ggml-cpu/ggml-cpu.c"
MOE = ROOT / "src/llama-moe-stream.cpp"

h = GGML_H.read_text()
old = """    GGML_API void ggml_set_input(struct ggml_tensor * tensor);
    GGML_API void ggml_set_output(struct ggml_tensor * tensor);
"""
new = """    GGML_API void ggml_set_input(struct ggml_tensor * tensor);
    GGML_API void ggml_set_output(struct ggml_tensor * tensor);

    // MoE serial waves: skip expert mul_mat_id when this wave's expert slice is empty.
    GGML_API void ggml_set_skip_mul_mat_id(bool skip);
    GGML_API bool ggml_get_skip_mul_mat_id(void);
"""
if old not in h:
    raise SystemExit("ggml.h marker not found")
GGML_H.write_text(h.replace(old, new, 1))

c = GGML_C.read_text()
old = """uint64_t ggml_graph_next_uid(void) {
"""
new = """static int g_skip_mul_mat_id = 0;

void ggml_set_skip_mul_mat_id(bool skip) {
    __atomic_store_n(&g_skip_mul_mat_id, skip ? 1 : 0, __ATOMIC_RELEASE);
}

bool ggml_get_skip_mul_mat_id(void) {
    return __atomic_load_n(&g_skip_mul_mat_id, __ATOMIC_ACQUIRE) != 0;
}

uint64_t ggml_graph_next_uid(void) {
"""
if old not in c:
    raise SystemExit("ggml.c marker not found")
GGML_C.write_text(c.replace(old, new, 1))

cpu = CPU_C.read_text()
old = """static void ggml_compute_forward_mul_mat_id(
        const struct ggml_compute_params * params,
              struct ggml_tensor * dst) {

    const struct ggml_tensor * src0 = dst->src[0];
"""
new = """static void ggml_compute_forward_mul_mat_id(
        const struct ggml_compute_params * params,
              struct ggml_tensor * dst) {

    if (ggml_get_skip_mul_mat_id()) {
        if (params->ith == 0 && dst->data != NULL) {
            memset(dst->data, 0, ggml_nbytes(dst));
        }
        return;
    }

    const struct ggml_tensor * src0 = dst->src[0];
"""
if old not in cpu:
    raise SystemExit("ggml-cpu.c marker not found")
CPU_C.write_text(cpu.replace(old, new, 1))

moe = MOE.read_text()
old = """    mgr->stage_wave_locked(lk, *sl, w, n_ids); // make this wave resident, preload the next, build the pool
    moe_wlog("ids stage-end il=%d w=%d pool=%zu demand=%zu", sl->il, w, sl->plan_pool.size(), sl->demand_slots.size());
    sl->plan_next_wave = w + 1;
    mgr->cv_done.notify_all();

    mgr->emit_wave_slots(*sl, ids, out, w, n_ids, a->ne[1]);
    moe_wlog("ids emit-end il=%d w=%d", sl->il, w);
"""
new = """    mgr->stage_wave_locked(lk, *sl, w, n_ids); // make this wave resident, preload the next, build the pool
    moe_wlog("ids stage-end il=%d w=%d pool=%zu demand=%zu", sl->il, w, sl->plan_pool.size(), sl->demand_slots.size());
    sl->plan_next_wave = w + 1;
    mgr->cv_done.notify_all();

    mgr->emit_wave_slots(*sl, ids, out, w, n_ids, a->ne[1]);
    moe_wlog("ids emit-end il=%d w=%d", sl->il, w);
    // Empty waves still have a compiled GEMM (graph sized for worst-case n_touch_max).
    // Zero-fill via ggml_mul_mat_id skip instead of Q4 dots on parked slots.
    ggml_set_skip_mul_mat_id(w >= (int32_t) sl->plan_n_waves);
"""
if old not in moe:
    raise SystemExit("wave_ids marker not found")
moe = moe.replace(old, new, 1)

old = """    while (sl->plan_next_wave <= w) {
        if (mgr->load_failed) {
            GGML_ABORT("MoE expert streaming: expert load failed (I/O error)");
        }
        mgr->cv_done.wait(lock);
    }
"""
new = """    while (sl->plan_next_wave <= w) {
        if (mgr->load_failed) {
            GGML_ABORT("MoE expert streaming: expert load failed (I/O error)");
        }
        mgr->cv_done.wait(lock);
    }
    ggml_set_skip_mul_mat_id(false);
"""
if old not in moe:
    raise SystemExit("wave_mask marker not found")
moe = moe.replace(old, new, 1)
MOE.write_text(moe)
print("skip-empty-wave patches OK")
