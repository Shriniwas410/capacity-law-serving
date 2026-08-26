#!/usr/bin/env python3
from pathlib import Path
p = Path("/home/shri/src/llama.cpp/src/llama-moe-stream.cpp")
t = p.read_text()

old = """    // protect the next wave's already-resident experts so this wave's victims do not evict them
    const size_t nfirst = first + sl.plan_capacity;
    const size_t ncount = nfirst < sl.uniq.size() ? std::min<size_t>(sl.plan_capacity, sl.uniq.size() - nfirst) : 0;
    for (size_t i = nfirst; i < nfirst + ncount; i++) {
        const auto it = sl.expert_slot.find(sl.uniq[i]);
        if (it != sl.expert_slot.end()) {
            sl.keep[it->second] = 1;
        }
    }
"""
new = """    // Overlap mode (n_slots > cap) keeps the next wave's already-resident experts so this
    // wave does not evict them, and preloads the rest. Serial waves use cap == n_slots:
    // the next wave can already occupy every slot (second ubatch, warm cache). Pinning
    // them makes pick_victim return -1 with an empty I/O queue -> OpenMP hang.
    const bool overlap = sl.n_slots > sl.plan_capacity;
    const size_t nfirst = first + sl.plan_capacity;
    const size_t ncount = nfirst < sl.uniq.size() ? std::min<size_t>(sl.plan_capacity, sl.uniq.size() - nfirst) : 0;
    if (overlap) {
        for (size_t i = nfirst; i < nfirst + ncount; i++) {
            const auto it = sl.expert_slot.find(sl.uniq[i]);
            if (it != sl.expert_slot.end()) {
                sl.keep[it->second] = 1;
            }
        }
    }
"""
if old not in t:
    raise SystemExit("keep-next block not found")
t = t.replace(old, new, 1)

old = """    if (std::getenv("LLAMA_MOE_STREAM_NO_PRELOAD") == nullptr) {
        for (size_t i = nfirst; i < nfirst + ncount; i++) {
"""
new = """    if (overlap && std::getenv("LLAMA_MOE_STREAM_NO_PRELOAD") == nullptr) {
        for (size_t i = nfirst; i < nfirst + ncount; i++) {
"""
if old not in t:
    raise SystemExit("preload block not found")
t = t.replace(old, new, 1)

old = """static void moe_wlog(const char * fmt, ...) {
    FILE * f = fopen("/tmp/moe_wave.log", "a");
    if (!f) {
        return;
    }
"""
new = """static void moe_wlog(const char * fmt, ...) {
    static int on = -1;
    if (on < 0) {
        on = std::getenv("LLAMA_MOE_STREAM_WAVE_LOG") != nullptr;
    }
    if (!on) {
        return;
    }
    FILE * f = fopen("/tmp/moe_wave.log", "a");
    if (!f) {
        return;
    }
"""
if old not in t:
    raise SystemExit("moe_wlog not found")
t = t.replace(old, new, 1)

p.write_text(t)
print("overlap guard + gated log OK")
