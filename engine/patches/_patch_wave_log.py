#!/usr/bin/env python3
from pathlib import Path
p = Path("/home/shri/src/llama.cpp/src/llama-moe-stream.cpp")
t = p.read_text()
inc = """#include <algorithm>
#include <cinttypes>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
"""
inc2 = """#include <algorithm>
#include <cstdarg>
#include <cinttypes>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

static void moe_wlog(const char * fmt, ...) {
    FILE * f = fopen("/tmp/moe_wave.log", "a");
    if (!f) {
        return;
    }
    va_list ap;
    va_start(ap, fmt);
    fprintf(f, "%lld ", (long long) ggml_time_us());
    vfprintf(f, fmt, ap);
    fputc('\\n', f);
    va_end(ap);
    fclose(f);
}
"""
if inc not in t:
    raise SystemExit("includes not found")
t = t.replace(inc, inc2, 1)

old = """    mgr->stats.n_wave_calls++;

    // Wave custom-ops are supposed to run in order via a 1-element view of the previous
"""
new = """    mgr->stats.n_wave_calls++;
    moe_wlog("ids enter il=%d w=%d next=%d n=%ld", sl->il, w, sl->plan_next_wave, (long) n);

    // Wave custom-ops are supposed to run in order via a 1-element view of the previous
"""
if old not in t:
    raise SystemExit("enter log site not found")
t = t.replace(old, new, 1)

old = """    if (w == 0) {
        mgr->plan_waves_locked(*sl, ids, n);
    } else {
        while (sl->plan_next_wave != w) {
"""
new = """    if (w == 0) {
        mgr->plan_waves_locked(*sl, ids, n);
        moe_wlog("ids planned il=%d uniq=%zu nwaves=%u cap=%u", sl->il, sl->uniq.size(), sl->plan_n_waves, sl->plan_capacity);
    } else {
        moe_wlog("ids wait il=%d w=%d next=%d", sl->il, w, sl->plan_next_wave);
        while (sl->plan_next_wave != w) {
"""
if old not in t:
    raise SystemExit("plan log site not found")
t = t.replace(old, new, 1)

old = """    mgr->stage_wave_locked(lk, *sl, w, n_ids); // make this wave resident, preload the next, build the pool
    sl->plan_next_wave = w + 1;
    mgr->cv_done.notify_all();

    mgr->emit_wave_slots(*sl, ids, out, w, n_ids, a->ne[1]);
"""
new = """    moe_wlog("ids stage-begin il=%d w=%d n_ids=%u", sl->il, w, n_ids);
    mgr->stage_wave_locked(lk, *sl, w, n_ids); // make this wave resident, preload the next, build the pool
    moe_wlog("ids stage-end il=%d w=%d pool=%zu demand=%zu", sl->il, w, sl->plan_pool.size(), sl->demand_slots.size());
    sl->plan_next_wave = w + 1;
    mgr->cv_done.notify_all();

    mgr->emit_wave_slots(*sl, ids, out, w, n_ids, a->ne[1]);
    moe_wlog("ids emit-end il=%d w=%d", sl->il, w);
"""
if old not in t:
    raise SystemExit("stage log site not found")
t = t.replace(old, new, 1)

old = """            while ((v = pick_victim_locked(sl, sl.keep.data())) < 0) {
                    cv_done.wait(lk);
"""
# two occurrences (remap and stage). replace all with logged version
old_v = """while ((v = pick_victim_locked(sl, sl.keep.data())) < 0) {
                    cv_done.wait(lk);"""
new_v = """while ((v = pick_victim_locked(sl, sl.keep.data())) < 0) {
                    moe_wlog("victim-wait il=%d keep_or_loading", sl.il);
                    cv_done.wait(lk);"""
c = t.count(old_v)
if c < 1:
    raise SystemExit(f"victim-wait site count {c}")
t = t.replace(old_v, new_v)

p.write_text(t)
print("debug logs inserted, victim sites", c)
