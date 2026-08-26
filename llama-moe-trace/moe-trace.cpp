// moe-trace: capture MoE router expert selections ("ffn_moe_topk-<layer>")
// for every token via the backend-scheduler eval callback, with the model
// weights untouched. Output is a raw binary stream of records:
//     int32 layer, int32 n_tokens, int32 k, then n_tokens*k int32 expert ids
// (one record per MoE layer per decoded chunk), parsed downstream into the
// moe-routing-lab trace format.
//
// Decode is always one token at a time (required for --moe-stream: a 512-token
// prefill can touch all 128 experts/layer and abort unless the cache has
// >= 24 slots/layer, which is ~24 GB on Qwen3-235B). Windows of W tokens
// are independent prefills (memory cleared between them) so the trace stays
// comparable to the original 30B 512-chunk captures.
//
// Usage:
//   MOE_TRACE_OUT=trace.bin llama-moe-trace -m model.gguf -f corpus.txt \
//       -c 512 -b 1 -ub 1 -ngl 0 --moe-stream --moe-stream-direct --moe-stream-cache 8s

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

struct moe_trace_state {
    FILE * out = nullptr;
    size_t records = 0;
    bool have_prev = false;
    int32_t prev_layer = -1;
    std::vector<int32_t> prev_ids;
};

static bool moe_trace_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    // exact prefix "ffn_moe_topk-" — must NOT match remapped "ffn_moe_topk_stream-"
    const bool match = strncmp(t->name, "ffn_moe_topk-", 13) == 0;
    if (ask) {
        return match;
    }
    if (!match) {
        return true;
    }
    auto * st = (moe_trace_state *) user_data;

    const int32_t layer    = atoi(t->name + 13);
    const int32_t k        = (int32_t) t->ne[0];
    const int32_t n_tokens = (int32_t) t->ne[1];

    // ffn_moe_topk is a non-contiguous view (top-k slice of the wide
    // argsort result): a flat copy would read whole argsort rows instead of
    // the selected experts, so copy row by row honoring the stride
    std::vector<int32_t> ids((size_t) k * n_tokens);
    for (int32_t i = 0; i < n_tokens; i++) {
        ggml_backend_tensor_get(t, ids.data() + (size_t) i * k,
                                (size_t) i * t->nb[1], k * sizeof(int32_t));
    }

    // --moe-stream evaluates ffn_moe_topk twice per token (identical ids).
    // Drop the immediate consecutive duplicate; do not key on ids alone or
    // genuine lag-1 full-set reuse would collapse two tokens into one.
    if (st->have_prev && st->prev_layer == layer && st->prev_ids == ids) {
        return true;
    }
    st->have_prev = true;
    st->prev_layer = layer;
    st->prev_ids = ids;

    fwrite(&layer,    sizeof(int32_t), 1, st->out);
    fwrite(&n_tokens, sizeof(int32_t), 1, st->out);
    fwrite(&k,        sizeof(int32_t), 1, st->out);
    fwrite(ids.data(), sizeof(int32_t), ids.size(), st->out);
    st->records++;
    return true;
}

int main(int argc, char ** argv) {
    moe_trace_state st;

    common_params params;
    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    const char * out_path = getenv("MOE_TRACE_OUT");
    if (!out_path) {
        out_path = "moe_trace.bin";
    }
    st.out = fopen(out_path, "wb");
    if (!st.out) {
        LOG_ERR("cannot open %s for writing\n", out_path);
        return 1;
    }

    long max_trace_tokens = 0;   // 0 = whole corpus
    if (const char * s = getenv("MOE_TRACE_MAX_TOKENS")) {
        max_trace_tokens = atol(s);
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    params.cb_eval           = moe_trace_cb;
    params.cb_eval_user_data = &st;
    params.warmup            = false;

    auto llama_init = common_init_from_params(params);
    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();
    if (!model || !ctx) {
        LOG_ERR("failed to init model/context\n");
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::vector<llama_token> tokens =
        common_tokenize(ctx, params.prompt, llama_vocab_get_add_bos(vocab), true);
    if (max_trace_tokens > 0 && (long) tokens.size() > max_trace_tokens) {
        tokens.resize(max_trace_tokens);
    }

    const int n_ctx = llama_n_ctx(ctx);
    int window = n_ctx;
    if (const char * s = getenv("MOE_TRACE_WINDOW")) {
        window = atoi(s);
    }
    if (window < 8) {
        window = 8;
    }
    if (window > n_ctx) {
        window = n_ctx;
    }

    LOG_INF("tracing %zu tokens, window=%d, n_ctx=%d -> %s\n",
            tokens.size(), window, n_ctx, out_path);

    // one-token batches: required for --moe-stream with a small expert cache
    llama_batch batch = llama_batch_init(1, 0, 1);
    const std::vector<llama_seq_id> seq_ids = {0};

    for (size_t start = 0; start < tokens.size(); start += (size_t) window) {
        const int n = (int) std::min((size_t) window, tokens.size() - start);
        if (n < 8) {
            break;
        }
        llama_memory_clear(llama_get_memory(ctx), true);
        for (int i = 0; i < n; i++) {
            common_batch_clear(batch);
            common_batch_add(batch, tokens[start + i], i, seq_ids, true);
            if (llama_decode(ctx, batch)) {
                LOG_ERR("decode failed at offset %zu (window start %zu pos %d)\n",
                        start + (size_t) i, start, i);
                return 1;
            }
        }
        LOG_INF("  %zu / %zu tokens  records=%zu\n",
                std::min(start + (size_t) n, tokens.size()), tokens.size(), st.records);
        fflush(st.out);
    }

    llama_batch_free(batch);
    fclose(st.out);
    LOG_INF("wrote %zu records to %s\n", st.records, out_path);

    llama_backend_free();
    return 0;
}
