#!/usr/bin/env python3
"""HorizonSpec U_max: longest causal prefix whose expert union fits in n_slots.

Prefix FFN stays bit-exact (causal attn). Suffix expert GEMMs are skipped so we
do not pay extra serial waves. Sampler must not accept beyond admitted rows.
Opt-in: --moe-stream-umax, enabled only on spec-verify batches (not prefill).
"""
from pathlib import Path

MOE_H = Path("/home/shri/src/llama.cpp/src/llama-moe-stream.h")
MOE_C = Path("/home/shri/src/llama.cpp/src/llama-moe-stream.cpp")
LLAMA_H = Path("/home/shri/src/llama.cpp/include/llama.h")
MODEL_C = Path("/home/shri/src/llama.cpp/src/llama-model.cpp")
CTX_C = Path("/home/shri/src/llama.cpp/src/llama-context.cpp")
ARG = Path("/home/shri/src/llama.cpp/common/arg.cpp")
COMMON_H = Path("/home/shri/src/llama.cpp/common/common.h")
SAMP_H = Path("/home/shri/src/llama.cpp/common/sampling.h")
SAMP_C = Path("/home/shri/src/llama.cpp/common/sampling.cpp")
SRV = Path("/home/shri/src/llama.cpp/tools/server/server-context.cpp")


def must_replace(path: Path, old: str, new: str, label: str) -> None:
    t = path.read_text()
    n = t.count(old)
    if n != 1:
        raise SystemExit(f"{label}: expected 1 match in {path}, found {n}")
    path.write_text(t.replace(old, new, 1))
    print(f"patched {label}")


# --- header: umax state on the manager ---
must_replace(
    MOE_H,
    """    bool debug = false;

    struct {""",
    """    bool debug = false;

    // HorizonSpec: longest prefix whose unique experts fit in n_slots.
    // Opt-in; only safe on speculative-verify ubatches (causal suffix).
    bool    umax_enabled = false;
    bool    umax_seen    = false; // first MoE layer of this ubatch has planned
    int32_t admitted_n   = 0;     // min prefix length over MoE layers
    int64_t n_umax_trunc = 0;     // tokens dropped from the union

    void begin_ubatch() {
        umax_seen  = false;
        admitted_n = 0;
    }

    struct {""",
    "moe-stream.h umax fields",
)

must_replace(
    MOE_H,
    """    void print_stats() const;""",
    """    void print_stats() const;

    void plan_waves_locked(llama_moe_stream_layer & sl, const int32_t * ids, int64_t n, uint32_t n_ids);""",
    "moe-stream.h plan_waves proto",
)

# If plan_waves is only defined in cpp as a method, check - it was
# void llama_moe_stream::plan_waves_locked without a header decl perhaps.
# Grep showed definition in cpp. Adding to header is fine if it wasn't there.
# If it was already in header I'd double. The first grep of .h didn't show plan_waves
# as a public method - it was only in cpp. Good.

# --- plan_waves body ---
OLD_PLAN = """void llama_moe_stream::plan_waves_locked(llama_moe_stream_layer & sl, const int32_t * ids, int64_t n) {
    stats.n_calls++;
    start_workers_locked();

    sl.touched.assign(sl.n_expert, 0);
    sl.uniq.clear();
    for (int64_t i = 0; i < n; i++) {
        const int32_t e = ids[i];
        GGML_ASSERT(e >= 0 && (uint32_t) e < sl.n_expert);
        if (!sl.touched[e]) {
            sl.touched[e] = 1;
            sl.uniq.push_back(e);
        }
    }

    GGML_ASSERT(sl.plan_capacity > 0);
    sl.expert_wave.assign(sl.n_expert, 0xff);
    for (size_t i = 0; i < sl.uniq.size(); i++) {
        GGML_ASSERT(i/sl.plan_capacity < 0xff);
        sl.expert_wave[sl.uniq[i]] = (uint8_t) (i/sl.plan_capacity);
    }
    sl.plan_n_waves   = (uint32_t) ((sl.uniq.size() + sl.plan_capacity - 1)/sl.plan_capacity);
    sl.plan_next_wave = 0;
}"""

NEW_PLAN = """void llama_moe_stream::plan_waves_locked(llama_moe_stream_layer & sl, const int32_t * ids, int64_t n, uint32_t n_ids) {
    stats.n_calls++;
    start_workers_locked();

    GGML_ASSERT(n_ids > 0 && n % n_ids == 0);
    const int64_t n_tok = n / (int64_t) n_ids;

    sl.touched.assign(sl.n_expert, 0);
    sl.uniq.clear();

    int64_t T = n_tok;
    const bool umax = umax_enabled && n_tok > 1 && n_ids <= sl.n_slots;
    if (umax) {
        T = 0;
        for (int64_t t = 0; t < n_tok; t++) {
            int32_t added[64];
            uint32_t n_added = 0;
            GGML_ASSERT(n_ids <= 64);
            for (uint32_t kk = 0; kk < n_ids; kk++) {
                const int32_t e = ids[t*(int64_t) n_ids + kk];
                GGML_ASSERT(e >= 0 && (uint32_t) e < sl.n_expert);
                if (sl.touched[e]) {
                    continue;
                }
                bool dup = false;
                for (uint32_t a = 0; a < n_added; a++) {
                    if (added[a] == e) { dup = true; break; }
                }
                if (!dup) {
                    added[n_added++] = e;
                }
            }
            if (sl.uniq.size() + n_added > sl.n_slots) {
                break;
            }
            for (uint32_t a = 0; a < n_added; a++) {
                sl.touched[added[a]] = 1;
                sl.uniq.push_back(added[a]);
            }
            T++;
        }
        if (T == 0) {
            // token 0 itself does not fit (n_expert_used > n_slots); fall back
            T = n_tok;
            sl.touched.assign(sl.n_expert, 0);
            sl.uniq.clear();
        }
    }
    if (sl.uniq.empty()) {
        for (int64_t i = 0; i < n; i++) {
            const int32_t e = ids[i];
            GGML_ASSERT(e >= 0 && (uint32_t) e < sl.n_expert);
            if (!sl.touched[e]) {
                sl.touched[e] = 1;
                sl.uniq.push_back(e);
            }
        }
    }

    if (umax && T < n_tok) {
        n_umax_trunc += (n_tok - T);
        static bool warned = false;
        if (!warned) {
            warned = true;
            LLAMA_LOG_WARN("%s: HorizonSpec U_max=%u: admitting %ld/%ld tokens (uniq=%zu); suffix FFN skipped\\n",
                    __func__, sl.n_slots, (long) T, (long) n_tok, sl.uniq.size());
        }
    }

    if (!umax_seen) {
        admitted_n = (int32_t) T;
        umax_seen  = true;
    } else {
        admitted_n = std::min(admitted_n, (int32_t) T);
    }

    GGML_ASSERT(sl.plan_capacity > 0);
    sl.expert_wave.assign(sl.n_expert, 0xff);
    for (size_t i = 0; i < sl.uniq.size(); i++) {
        GGML_ASSERT(i/sl.plan_capacity < 0xff);
        sl.expert_wave[sl.uniq[i]] = (uint8_t) (i/sl.plan_capacity);
    }
    sl.plan_n_waves   = (uint32_t) ((sl.uniq.size() + sl.plan_capacity - 1)/sl.plan_capacity);
    sl.plan_next_wave = 0;
}"""

must_replace(MOE_C, OLD_PLAN, NEW_PLAN, "plan_waves umax prefix")

must_replace(
    MOE_C,
    """            GGML_ASSERT(sl.expert_wave[e] != 0xff);
            if (sl.expert_wave[e] == (uint8_t) w) {""",
    """            if (sl.expert_wave[e] == 0xff) {
                continue; // HorizonSpec suffix-only expert: park in pass 2
            }
            if (sl.expert_wave[e] == (uint8_t) w) {""",
    "emit skip 0xff",
)

must_replace(
    MOE_C,
    """    if (w == 0) {
        mgr->plan_waves_locked(*sl, ids, n);
        moe_wlog("ids planned il=%d uniq=%zu nwaves=%u cap=%u", sl->il, sl->uniq.size(), sl->plan_n_waves, sl->plan_capacity);""",
    """    const uint32_t n_ids = (uint32_t) a->ne[0]; // experts per token (n_expert_used)

    if (w == 0) {
        mgr->plan_waves_locked(*sl, ids, n, n_ids);
        moe_wlog("ids planned il=%d uniq=%zu nwaves=%u cap=%u admitted=%d", sl->il, sl->uniq.size(), sl->plan_n_waves, sl->plan_capacity, mgr->admitted_n);""",
    "wave_ids call plan with n_ids",
)

# n_ids was declared later - remove duplicate
must_replace(
    MOE_C,
    """    const uint32_t n_ids = (uint32_t) a->ne[0]; // experts per token (n_expert_used)

    moe_wlog("ids stage-begin il=%d w=%d n_ids=%u", sl->il, w, n_ids);""",
    """    moe_wlog("ids stage-begin il=%d w=%d n_ids=%u", sl->il, w, n_ids);""",
    "drop duplicate n_ids",
)

# print_stats line
must_replace(
    MOE_C,
    """        LLAMA_LOG_INFO("%s: moe stream: waves = %" PRId64 " (%" PRId64 " non-empty), preloads issued = %" PRId64 " (ready on arrival = %" PRId64 "), wave stall = %.2f ms\\n",
                __func__, stats.n_wave_calls, stats.n_waves_run, stats.n_preload_issued, stats.n_preload_ready, stats.t_stall_wave_us/1000.0);""",
    """        LLAMA_LOG_INFO("%s: moe stream: waves = %" PRId64 " (%" PRId64 " non-empty), preloads issued = %" PRId64 " (ready on arrival = %" PRId64 "), wave stall = %.2f ms, umax trunc tokens = %" PRId64 " (admitted_n=%d)\\n",
                __func__, stats.n_wave_calls, stats.n_waves_run, stats.n_preload_issued, stats.n_preload_ready, stats.t_stall_wave_us/1000.0, n_umax_trunc, admitted_n);""",
    "print_stats umax",
)

# llama.h API
must_replace(
    LLAMA_H,
    """    LLAMA_API void llama_moe_stream_print_stats(const struct llama_model * model);""",
    """    LLAMA_API void llama_moe_stream_print_stats(const struct llama_model * model);
    // HorizonSpec: when enabled, MoE streaming admits the longest prefix whose
    // unique experts fit in the slot cache. Prefix outputs stay bit-exact under
    // causal attention; suffix expert GEMMs are skipped. Enable only for
    // speculative-verify ubatches, never prompt prefill.
    LLAMA_API void llama_moe_stream_set_umax(const struct llama_model * model, bool enable);
    LLAMA_API int32_t llama_moe_stream_admitted_n(const struct llama_model * model);""",
    "llama.h umax API",
)

must_replace(
    MODEL_C,
    """void llama_moe_stream_print_stats(const llama_model * model) {
    if (model && model->moe_stream()) {
        model->moe_stream()->print_stats();
    }
}""",
    """void llama_moe_stream_print_stats(const llama_model * model) {
    if (model && model->moe_stream()) {
        model->moe_stream()->print_stats();
    }
}

void llama_moe_stream_set_umax(const llama_model * model, bool enable) {
    if (model && model->moe_stream()) {
        model->moe_stream()->umax_enabled = enable;
    }
}

int32_t llama_moe_stream_admitted_n(const llama_model * model) {
    if (model && model->moe_stream()) {
        return model->moe_stream()->admitted_n;
    }
    return 0;
}""",
    "model.cpp umax API",
)

must_replace(
    CTX_C,
    """int llama_context::decode(const llama_batch & batch_inp) {
    // MTP hook batches carry both token (next-token id) and embd (h_nextn row),
    // so accept either present rather than requiring exactly one.
    GGML_ASSERT(batch_inp.token || batch_inp.embd);""",
    """int llama_context::decode(const llama_batch & batch_inp) {
    if (llama_moe_stream * ms = model.moe_stream()) {
        ms->begin_ubatch();
    }

    // MTP hook batches carry both token (next-token id) and embd (h_nextn row),
    // so accept either present rather than requiring exactly one.
    GGML_ASSERT(batch_inp.token || batch_inp.embd);""",
    "decode begin_ubatch",
)

# common.h flag
must_replace(
    COMMON_H,
    """    bool     moe_stream_direct     = false; // use O_DIRECT for expert reads (bypass page cache)""",
    """    bool     moe_stream_direct     = false; // use O_DIRECT for expert reads (bypass page cache)
    bool     moe_stream_umax       = false; // HorizonSpec: admit spec-verify prefix to slot cap""",
    "common.h moe_stream_umax",
)

# arg.cpp: insert after moe-stream-direct block.
must_replace(
    ARG,
    """    add_opt(common_arg(
        {"--moe-stream-direct"},
        "use O_DIRECT for --moe-stream expert reads (bypass the page cache); implies --moe-stream. "
        "falls back to buffered reads if O_DIRECT is unsupported by the OS or filesystem",
        [](common_params & params) {
            params.moe_stream = true;
            params.moe_stream_direct = true;
        }
    ).set_env("LLAMA_ARG_MOE_STREAM_DIRECT"));""",
    """    add_opt(common_arg(
        {"--moe-stream-direct"},
        "use O_DIRECT for --moe-stream expert reads (bypass the page cache); implies --moe-stream. "
        "falls back to buffered reads if O_DIRECT is unsupported by the OS or filesystem",
        [](common_params & params) {
            params.moe_stream = true;
            params.moe_stream_direct = true;
        }
    ).set_env("LLAMA_ARG_MOE_STREAM_DIRECT"));
    add_opt(common_arg(
        {"--moe-stream-umax"},
        "HorizonSpec: during speculative verify, load only the longest token prefix whose unique experts "
        "fit in the stream cache (U_max = n_slots). Prefix is bit-exact under causal attention. "
        "Do not use for prompt prefill. Implies --moe-stream.",
        [](common_params & params) {
            params.moe_stream = true;
            params.moe_stream_umax = true;
        }
    ).set_env("LLAMA_ARG_MOE_STREAM_UMAX"));""",
    "arg.cpp --moe-stream-umax",
)

must_replace(
    SAMP_H,
    """std::vector<llama_token> common_sampler_sample_and_accept_n(struct common_sampler * gsmpl, struct llama_context * ctx, const std::vector<int> & idxs, const llama_tokens & draft, bool grammar_first = false);""",
    """std::vector<llama_token> common_sampler_sample_and_accept_n(struct common_sampler * gsmpl, struct llama_context * ctx, const std::vector<int> & idxs, const llama_tokens & draft, bool grammar_first = false, int32_t n_ubatch_admitted = -1);""",
    "sampling.h admitted arg",
)

must_replace(
    SAMP_C,
    """std::vector<llama_token> common_sampler_sample_and_accept_n(struct common_sampler * gsmpl, struct llama_context * ctx, const std::vector<int> & idxs, const llama_tokens & draft, bool grammar_first) {
    GGML_ASSERT(idxs.size() == draft.size() + 1 && "idxs.size() must be draft.size() + 1");

    std::vector<llama_token> result;
    result.reserve(idxs.size());

    size_t i = 0;
    for (; i < draft.size(); i++) {
        const llama_token id = common_sampler_sample(gsmpl, ctx, idxs[i], grammar_first);

        common_sampler_accept(gsmpl, id, true);

        result.push_back(id);

        if (draft[i] != id) {
            break;
        }
    }

    if (i == draft.size()) {
        const llama_token id = common_sampler_sample(gsmpl, ctx, idxs[i], grammar_first);

        common_sampler_accept(gsmpl, id, true);

        result.push_back(id);
    }

    return result;
}""",
    """std::vector<llama_token> common_sampler_sample_and_accept_n(struct common_sampler * gsmpl, struct llama_context * ctx, const std::vector<int> & idxs, const llama_tokens & draft, bool grammar_first, int32_t n_ubatch_admitted) {
    GGML_ASSERT(idxs.size() == draft.size() + 1 && "idxs.size() must be draft.size() + 1");

    std::vector<llama_token> result;
    result.reserve(idxs.size());

    size_t n_loop = draft.size();
    if (n_ubatch_admitted > 0) {
        n_loop = std::min(n_loop, (size_t) n_ubatch_admitted);
    }

    size_t i = 0;
    for (; i < n_loop; i++) {
        const llama_token id = common_sampler_sample(gsmpl, ctx, idxs[i], grammar_first);

        common_sampler_accept(gsmpl, id, true);

        result.push_back(id);

        if (draft[i] != id) {
            break;
        }
    }

    if (i == draft.size() && (n_ubatch_admitted <= 0 || (size_t) n_ubatch_admitted > draft.size())) {
        const llama_token id = common_sampler_sample(gsmpl, ctx, idxs[i], grammar_first);

        common_sampler_accept(gsmpl, id, true);

        result.push_back(id);
    }

    return result;
}""",
    "sampling.cpp umax cap",
)

must_replace(
    SRV,
    """        const int ret = llama_decode(ctx_tgt, batch_view);""",
    """        if (params_base.moe_stream_umax) {
            bool spec_only = true;
            iterate(slots, [&](server_slot & s) {
                if (s.is_processing() && s.spec_draft.empty()) {
                    spec_only = false;
                }
            });
            llama_moe_stream_set_umax(llama_get_model(ctx_tgt), spec_only);
        }

        const int ret = llama_decode(ctx_tgt, batch_view);

        llama_moe_stream_set_umax(llama_get_model(ctx_tgt), false);""",
    "server decode umax gate",
)

must_replace(
    SRV,
    """                auto accepted = common_sampler_sample_and_accept_n(slot.smpl.get(), slot.ctx_tgt, slot.spec_i_batch, slot.spec_draft);""",
    """                const int32_t admitted = llama_moe_stream_admitted_n(llama_get_model(slot.ctx_tgt));
                auto accepted = common_sampler_sample_and_accept_n(slot.smpl.get(), slot.ctx_tgt, slot.spec_i_batch, slot.spec_draft, false, admitted);""",
    "server accept admitted cap",
)

print("UMAX_PATCH_OK")
