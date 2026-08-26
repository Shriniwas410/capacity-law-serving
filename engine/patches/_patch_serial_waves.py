#!/usr/bin/env python3
"""Apply serial-wave fallback when n_slots < 3*n_expert_used."""
from pathlib import Path

from _llama_root import llama_root

ROOT = llama_root()
GRAPH = ROOT / "src/llama-graph.cpp"
CTX = ROOT / "src/llama-context.cpp"

OLD_GRAPH = """        if (n_touch_max > msl->n_slots) {
            // the cache must hold three sets at once: this wave's experts, the next wave's preloaded
            //   experts (so its loads overlap this wave's compute), and n_expert_used parking slots
            //   the masked-out pairs GEMM against (Metal needs a slot at most once per token row).
            //   so cap + cap + n_expert_used = n_slots -> cap = (n_slots - n_expert_used)/2
            stream_wave_cap = msl->n_slots > (uint32_t) n_expert_used ? (msl->n_slots - (uint32_t) n_expert_used)/2 : 0;
            if (stream_wave_cap < (uint32_t) n_expert_used) {
                // a wave must fit at least n_expert_used experts, i.e. n_slots >= 3*n_expert_used
                GGML_ABORT("MoE expert streaming: multi-pass expert GEMMs need an expert cache of at least "
                           "3*n_expert_used slots (have %u, need %u); increase --moe-stream-cache or reduce -ub",
                        msl->n_slots, 3*(uint32_t) n_expert_used);
            }
            n_stream_waves = (uint32_t) ((n_touch_max + stream_wave_cap - 1)/stream_wave_cap); // ceil(n_touch_max/cap)
        }"""

NEW_GRAPH = """        if (n_touch_max > msl->n_slots) {
            // Overlap preload + Metal parking wants three sets in cache at once:
            // this wave, the next wave's preloaded experts, and n_expert_used parking
            // slots (a slot at most once per token row). That is
            // cap + cap + n_expert_used = n_slots -> cap = (n_slots - n_expert_used)/2
            // and therefore n_slots >= 3*n_expert_used.
            stream_wave_cap = msl->n_slots > (uint32_t) n_expert_used ? (msl->n_slots - (uint32_t) n_expert_used)/2 : 0;
            if (stream_wave_cap < (uint32_t) n_expert_used) {
                // Small CPU cache (this box: 8-12 slots, Qwen3 top-8 needs 24 for overlap):
                // serial waves that fill every slot. Best-effort overlap preload finds no
                // victims and becomes the next wave's demand load. Last-wave parking still
                // borrows from plan_pool when the slice is < n_expert_used.
                static bool warned_serial_waves = false;
                if (!warned_serial_waves) {
                    warned_serial_waves = true;
                    LLAMA_LOG_WARN("%s: MoE stream cache has %u slots (< 3*n_expert_used=%u); "
                                   "using serial waves of %u (no overlap preload)\\n",
                            __func__, msl->n_slots, 3*(uint32_t) n_expert_used, msl->n_slots);
                }
                stream_wave_cap = msl->n_slots;
            }
            if (stream_wave_cap < 1) {
                GGML_ABORT("MoE expert streaming: need at least 1 cache slot");
            }
            n_stream_waves = (uint32_t) ((n_touch_max + stream_wave_cap - 1)/stream_wave_cap); // ceil(n_touch_max/cap)
        }"""

OLD_GDN = """            // Qwen3 MoE streaming with a small expert cache (n_slots < 3*n_expert_used)
            // cannot build a 16-token graph: llama-graph.cpp aborts because multi-pass
            // waves need 24 slots/layer (~24 GB on Qwen3-235B). The chunked GDN probe
            // is for Qwen3.5; skip it so 8-slot decode-time tracing can start."""

NEW_GDN = """            // Qwen3 MoE streaming with a small expert cache (n_slots < 3*n_expert_used)
            // can now build multi-token graphs via serial waves (cap = n_slots), but a
            // 16-token GDN probe is still a fat graph (n_touch_max = 128, many waves x
            // layers) and is only for Qwen3.5. Skip it so 8/12-slot decode can start."""

OLD_NODES = """        uint32_t cap = mstream->n_slots > n_eu ? (mstream->n_slots - n_eu)/2 : 0;
        cap = std::max<uint32_t>(cap, 1);"""

NEW_NODES = """        uint32_t cap = mstream->n_slots > n_eu ? (mstream->n_slots - n_eu)/2 : 0;
        if (cap < n_eu) {
            cap = mstream->n_slots; // serial waves; see llama-graph.cpp
        }
        cap = std::max<uint32_t>(cap, 1);"""


def must_replace(path: Path, old: str, new: str, label: str) -> None:
    t = path.read_text()
    n = t.count(old)
    if n != 1:
        raise SystemExit(f"{label}: expected 1 match in {path}, found {n}")
    path.write_text(t.replace(old, new, 1))
    print(f"patched {label}: {path}")


def main() -> None:
    must_replace(GRAPH, OLD_GRAPH, NEW_GRAPH, "graph waves")
    must_replace(CTX, OLD_GDN, NEW_GDN, "gdn comment")
    must_replace(CTX, OLD_NODES, NEW_NODES, "graph_max_nodes")
    print("OK")


if __name__ == "__main__":
    main()
