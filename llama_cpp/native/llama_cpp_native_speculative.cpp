#include "llama_cpp_native_speculative.h"

#include "common.h"
#include "llama.h"
#include "log.h"
#include "speculative.h"

#include "../../vendor/llama.cpp/src/llama-ext.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

enum class llama_cpp_native_speculative_provider_kind {
    external_draft,
    external_mtp,
    internal_mtp,
};

struct llama_cpp_native_speculative {
    common_params params;
    common_speculative_init_result_ptr draft_init;
    common_speculative * speculative = nullptr;
    common_speculative_type type = COMMON_SPECULATIVE_TYPE_NONE;
    llama_cpp_native_speculative_provider_kind provider =
            llama_cpp_native_speculative_provider_kind::external_draft;

    llama_context * target_context = nullptr;
    llama_context * draft_context  = nullptr;

    llama_tokens prompt;
    llama_tokens result;
    std::string last_error;
    bool has_last_draft = false;
    bool last_draft_verified = false;
    bool shares_target_memory = false;

    ~llama_cpp_native_speculative() {
        if (speculative != nullptr) {
            common_speculative_free(speculative);
            speculative = nullptr;
        }
        // draft_init is destroyed after the speculative implementation, so its
        // model/context remain alive for the entire speculative lifetime.
        draft_context = nullptr;
        draft_init.reset();
    }
};

namespace {

void copy_error(char * dst, size_t capacity, const std::string & message) {
    if (dst == nullptr || capacity == 0) {
        return;
    }

    const size_t n = std::min(capacity - 1, message.size());
    std::memcpy(dst, message.data(), n);
    dst[n] = '\0';
}

bool set_error(llama_cpp_native_speculative * speculative, const std::string & message) {
    if (speculative != nullptr) {
        speculative->last_error = message;
    }
    return false;
}

bool is_supported_type(common_speculative_type type) {
    switch (type) {
        case COMMON_SPECULATIVE_TYPE_DRAFT_MTP:
        case COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH:
        case COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK:
            return true;
        default:
            return false;
    }
}

const char * provider_kind_name(llama_cpp_native_speculative_provider_kind provider) {
    switch (provider) {
        case llama_cpp_native_speculative_provider_kind::external_draft:
            return "external draft";
        case llama_cpp_native_speculative_provider_kind::external_mtp:
            return "external MTP";
        case llama_cpp_native_speculative_provider_kind::internal_mtp:
            return "internal MTP";
    }

    return "unknown";
}

bool is_internal_mtp(const llama_cpp_native_speculative * speculative) {
    return speculative->provider == llama_cpp_native_speculative_provider_kind::internal_mtp;
}

void assign_tokens(llama_tokens & dst, const int32_t * tokens, size_t count) {
    if (count == 0) {
        dst.clear();
        return;
    }
    if (tokens == nullptr) {
        throw std::invalid_argument("token pointer is null while token count is non-zero");
    }
    dst.assign(tokens, tokens + count);
}

bool validate_batch(llama_cpp_native_speculative * speculative, const llama_batch & batch) {
    if (batch.n_tokens < 0) {
        return set_error(speculative, "target batch has a negative token count");
    }
    if (batch.n_tokens == 0) {
        return true;
    }
    if (batch.pos == nullptr || batch.n_seq_id == nullptr || batch.seq_id == nullptr) {
        return set_error(speculative, "target batch is missing position or sequence metadata");
    }
    if ((batch.token == nullptr) == (batch.embd == nullptr)) {
        return set_error(speculative, "target batch must contain exactly one of tokens or embeddings");
    }
    if (speculative->type == COMMON_SPECULATIVE_TYPE_DRAFT_MTP && batch.embd != nullptr) {
        return set_error(
                speculative,
                "draft-mtp currently supports text token batches only; multimodal embeddings are unsupported");
    }
    for (int32_t i = 0; i < batch.n_tokens; ++i) {
        if (batch.n_seq_id[i] != 1 || batch.seq_id[i] == nullptr || batch.seq_id[i][0] != 0) {
            return set_error(speculative, "experimental native speculative decoding only supports sequence 0");
        }
    }
    return true;
}

std::string model_architecture(const llama_model * model) {
    char value[64] = {};
    if (llama_model_meta_val_str(model, "general.architecture", value, sizeof(value)) < 0) {
        return {};
    }
    return value;
}

bool mtp_vocabs_are_compatible(
        const llama_model * target_model,
        const llama_model * draft_model,
        std::string & reason) {
    const llama_vocab * target_vocab = llama_model_get_vocab(target_model);
    const llama_vocab * draft_vocab  = llama_model_get_vocab(draft_model);

    if (target_vocab == nullptr || draft_vocab == nullptr) {
        reason = "target or assistant vocabulary is unavailable";
        return false;
    }
    if (llama_vocab_type(target_vocab) != llama_vocab_type(draft_vocab)) {
        reason = "target and assistant vocabulary types differ";
        return false;
    }
    if (llama_vocab_get_add_bos(target_vocab) != llama_vocab_get_add_bos(draft_vocab) ||
        (llama_vocab_get_add_bos(target_vocab) &&
         llama_vocab_bos(target_vocab) != llama_vocab_bos(draft_vocab))) {
        reason = "target and assistant BOS configuration differs";
        return false;
    }
    if (llama_vocab_get_add_eos(target_vocab) != llama_vocab_get_add_eos(draft_vocab) ||
        (llama_vocab_get_add_eos(target_vocab) &&
         llama_vocab_eos(target_vocab) != llama_vocab_eos(draft_vocab))) {
        reason = "target and assistant EOS configuration differs";
        return false;
    }

    const int32_t n_target = llama_vocab_n_tokens(target_vocab);
    const int32_t n_draft  = llama_vocab_n_tokens(draft_vocab);
    if (std::abs(n_target - n_draft) > 128) {
        reason = "target and assistant vocabulary sizes differ by more than 128 tokens";
        return false;
    }

    for (int32_t token = 5; token < std::min(n_target, n_draft); ++token) {
        const char * target_text = llama_vocab_get_text(target_vocab, token);
        const char * draft_text  = llama_vocab_get_text(draft_vocab, token);
        if (target_text == nullptr || draft_text == nullptr || std::strcmp(target_text, draft_text) != 0) {
            reason = "target and assistant token text differs at token " + std::to_string(token);
            return false;
        }
    }

    return true;
}

} // namespace

extern "C" {

uint32_t llama_cpp_native_speculative_abi_version(void) {
    return LLAMA_CPP_NATIVE_SPECULATIVE_ABI_VERSION;
}

llama_cpp_native_speculative * llama_cpp_native_speculative_init(
        llama_model * target_model,
        llama_context * target_context,
        const llama_cpp_native_speculative_params * params,
        char * error,
        size_t error_capacity) {
    try {
        if (target_model == nullptr || target_context == nullptr) {
            throw std::invalid_argument("target model and context are required");
        }
        if (params == nullptr) {
            throw std::invalid_argument("native speculative parameters are required");
        }
        if (params->struct_size != sizeof(llama_cpp_native_speculative_params)) {
            throw std::invalid_argument("native speculative parameter ABI mismatch");
        }
        if (params->spec_type == nullptr || params->spec_type[0] == '\0') {
            throw std::invalid_argument("speculative type is required");
        }
        if (params->n_max <= 0) {
            throw std::invalid_argument("n_max must be greater than zero");
        }
        if (params->n_min < 0 || params->n_min > params->n_max) {
            throw std::invalid_argument("n_min must be between zero and n_max");
        }
        if (params->p_min < 0.0f || params->p_min > 1.0f) {
            throw std::invalid_argument("p_min must be between zero and one");
        }

        const common_speculative_type type = common_speculative_type_from_name(params->spec_type);
        if (!is_supported_type(type)) {
            throw std::invalid_argument(
                    std::string("unsupported experimental native speculative type: ") + params->spec_type);
        }

        const bool has_model_path = params->model_path != nullptr && params->model_path[0] != '\0';
        llama_cpp_native_speculative_provider_kind provider;
        if (type == COMMON_SPECULATIVE_TYPE_DRAFT_MTP) {
            provider = has_model_path
                    ? llama_cpp_native_speculative_provider_kind::external_mtp
                    : llama_cpp_native_speculative_provider_kind::internal_mtp;
        } else {
            if (!has_model_path) {
                throw std::invalid_argument(
                        std::string("a draft model path is required for ") + params->spec_type);
            }
            provider = llama_cpp_native_speculative_provider_kind::external_draft;
        }

        if (provider == llama_cpp_native_speculative_provider_kind::internal_mtp &&
            llama_model_n_layer_nextn(target_model) <= 0) {
            throw std::runtime_error(
                    "embedded native MTP requires a target GGUF containing usable NextN/MTP layers");
        }

        auto result = std::make_unique<llama_cpp_native_speculative>();
        result->target_context = target_context;
        result->type           = type;
        result->provider       = provider;

        common_params & base = result->params;
        base.n_ctx      = params->n_ctx > 0 ? params->n_ctx : (int32_t) llama_n_ctx(target_context);
        base.n_batch    = params->n_batch > 0 ? params->n_batch : (int32_t) llama_n_batch(target_context);
        base.n_ubatch   = params->n_ubatch > 0 ? params->n_ubatch : (int32_t) llama_n_ubatch(target_context);
        base.n_parallel = 1;

        base.n_outputs_max         = 1;
        base.n_outputs_max_per_seq = 1;
        base.flash_attn_type       = (llama_flash_attn_type) params->flash_attn_type;
        base.no_kv_offload         = !params->offload_kqv;
        base.no_op_offload         = !params->op_offload;
        base.kv_unified            = params->kv_unified;
        base.no_perf               = params->no_perf;

        auto & draft = base.speculative.draft;
        base.speculative.types = { type };
        if (has_model_path) {
            draft.mparams.path = params->model_path;
        } else {
            // An empty draft model descriptor is how llama.cpp/common selects
            // an embedded target-model MTP context instead of loading a second
            // GGUF. In particular, never assign a null C string to std::string.
            draft.mparams.path.clear();
        }
        draft.n_gpu_layers      = params->n_gpu_layers;
        draft.n_max             = params->n_max;
        draft.n_min             = std::max(0, params->n_min);
        draft.p_min             = params->p_min;
        draft.cache_type_k      = (ggml_type) params->cache_type_k;
        draft.cache_type_v      = (ggml_type) params->cache_type_v;

        if (params->n_threads > 0) {
            draft.cpuparams.n_threads = params->n_threads;
        }
        if (params->n_threads_batch > 0) {
            draft.cpuparams_batch.n_threads = params->n_threads_batch;
        }

        common_params draft_params = common_base_params_to_speculative(base);
        result->draft_init = common_speculative_init_from_params(draft_params, target_model, target_context);
        if (!result->draft_init) {
            throw std::runtime_error("failed to allocate draft model/context owner");
        }

        llama_model * draft_model = result->draft_init->model();
        result->draft_context     = result->draft_init->context();
        if (result->draft_context == nullptr) {
            if (provider == llama_cpp_native_speculative_provider_kind::internal_mtp) {
                throw std::runtime_error(
                        "failed to create an embedded MTP context from the target model; "
                        "ensure its NextN/MTP tensors were loaded and this architecture supports an MTP graph");
            }
            throw std::runtime_error(std::string("failed to load draft model: ") + params->model_path);
        }
        if (provider != llama_cpp_native_speculative_provider_kind::internal_mtp &&
            draft_model == nullptr) {
            throw std::runtime_error(std::string("failed to load draft model: ") + params->model_path);
        }

        if (provider == llama_cpp_native_speculative_provider_kind::external_mtp) {
            const std::string target_arch = model_architecture(target_model);
            const std::string draft_arch  = model_architecture(draft_model);
            if (target_arch != "gemma4") {
                throw std::runtime_error(
                        "external draft-mtp currently requires a Gemma 4 target "
                        "(general.architecture=gemma4)");
            }
            if (draft_arch != "gemma4-assistant") {
                throw std::runtime_error(
                        "external draft-mtp currently requires a Gemma 4 assistant GGUF "
                        "(general.architecture=gemma4-assistant)");
            }

            result->shares_target_memory =
                    llama_get_ctx_other(result->draft_context) == target_context;
            if (!result->shares_target_memory) {
                throw std::runtime_error(
                        "Gemma 4 MTP assistant context did not attach to the target context");
            }

            const int32_t n_mtp_layers = llama_model_n_layer_nextn(draft_model);
            if (n_mtp_layers <= 0) {
                throw std::runtime_error(
                        "Gemma 4 MTP assistant has no next-token prediction layers");
            }

            const int32_t target_width = llama_model_n_embd_out(target_model);
            const int32_t draft_width  = llama_model_n_embd_out(draft_model);
            if (target_width <= 0 || target_width != draft_width) {
                throw std::runtime_error(
                        "Gemma 4 target and MTP assistant hidden widths do not match (target=" +
                        std::to_string(target_width) + ", assistant=" +
                        std::to_string(draft_width) + ")");
            }

            std::string vocab_error;
            if (!mtp_vocabs_are_compatible(target_model, draft_model, vocab_error)) {
                throw std::runtime_error(
                        "Gemma 4 target and MTP assistant are incompatible: " + vocab_error);
            }

            COM_INF(
                    "native speculative bridge: Gemma 4 MTP assistant recognized "
                    "(%d heads, shared target KV, n_max=%d)\n",
                    n_mtp_layers,
                    params->n_max);
        } else if (provider == llama_cpp_native_speculative_provider_kind::internal_mtp) {
            // llama.cpp/common intentionally owns only the embedded MTP context
            // in this mode; its init-result model pointer is null because the
            // target model remains owned by Python's Llama instance.
            const llama_model * context_model = llama_get_model(result->draft_context);
            if (context_model != target_model) {
                throw std::runtime_error(
                        "embedded MTP context was not created against the target model");
            }
            if (llama_get_memory(target_context) == nullptr) {
                throw std::runtime_error(
                        "embedded native MTP requires a target context with usable memory");
            }
            if (llama_get_memory(result->draft_context) == nullptr) {
                throw std::runtime_error(
                        "embedded native MTP context was created without usable memory");
            }

            result->shares_target_memory =
                    llama_get_ctx_other(result->draft_context) == target_context;

            const int32_t n_mtp_layers = llama_model_n_layer_nextn(context_model);
            if (n_mtp_layers <= 0) {
                throw std::runtime_error(
                        "embedded native MTP requires a target GGUF containing usable NextN/MTP layers");
            }

            const int32_t target_width = llama_model_n_embd_out(target_model);
            const int32_t draft_width  = llama_model_n_embd_out(context_model);
            if (target_width <= 0 || target_width != draft_width) {
                throw std::runtime_error(
                        "target and embedded MTP context hidden widths do not match (target=" +
                        std::to_string(target_width) + ", MTP=" +
                        std::to_string(draft_width) + ")");
            }

            COM_INF(
                    "native speculative bridge: embedded MTP context recognized "
                    "(%d heads, %s memory, n_max=%d)\n",
                    n_mtp_layers,
                    result->shares_target_memory ? "shared target" : "independent",
                    params->n_max);
        }

        draft.ctx_tgt = target_context;
        draft.ctx_dft = result->draft_context;
        result->speculative = common_speculative_init(base.speculative, 1);
        if (result->speculative == nullptr) {
            throw std::runtime_error("failed to initialize llama.cpp common speculative decoder");
        }

        COM_INF(
                "native speculative bridge: initialized %s provider for %s\n",
                provider_kind_name(provider),
                params->spec_type);

        copy_error(error, error_capacity, "");
        return result.release();
    } catch (const std::exception & exc) {
        copy_error(error, error_capacity, exc.what());
        return nullptr;
    } catch (...) {
        copy_error(error, error_capacity, "unknown native speculative initialization error");
        return nullptr;
    }
}

void llama_cpp_native_speculative_free(llama_cpp_native_speculative * speculative) {
    delete speculative;
}

const char * llama_cpp_native_speculative_last_error(
        const llama_cpp_native_speculative * speculative) {
    return speculative == nullptr ? "native speculative handle is null" : speculative->last_error.c_str();
}

bool llama_cpp_native_speculative_begin(
        llama_cpp_native_speculative * speculative,
        const int32_t * prompt_tokens,
        size_t prompt_token_count) {
    if (speculative == nullptr) {
        return false;
    }

    try {
        assign_tokens(speculative->prompt, prompt_tokens, prompt_token_count);
        common_speculative_begin(speculative->speculative, 0, speculative->prompt);
        speculative->last_error.clear();
        return true;
    } catch (const std::exception & exc) {
        return set_error(speculative, exc.what());
    } catch (...) {
        return set_error(speculative, "unknown native speculative begin error");
    }
}

bool llama_cpp_native_speculative_process(
        llama_cpp_native_speculative * speculative,
        const llama_batch * batch) {
    if (speculative == nullptr || batch == nullptr) {
        return false;
    }

    try {
        if (!validate_batch(speculative, *batch)) {
            return false;
        }
        if (!common_speculative_process(speculative->speculative, *batch)) {
            return set_error(speculative, "llama.cpp common speculative batch processing failed");
        }
        if (speculative->has_last_draft) {
            speculative->last_draft_verified = true;
        }
        speculative->last_error.clear();
        return true;
    } catch (const std::exception & exc) {
        return set_error(speculative, exc.what());
    } catch (...) {
        return set_error(speculative, "unknown native speculative process error");
    }
}

int32_t llama_cpp_native_speculative_draft(
        llama_cpp_native_speculative * speculative,
        int32_t n_past,
        int32_t id_last,
        const int32_t * prompt_tokens,
        size_t prompt_token_count,
        int32_t * output_tokens,
        size_t output_capacity) {
    if (speculative == nullptr || output_tokens == nullptr) {
        return -1;
    }

    try {
        if (output_capacity == 0) {
            speculative->result.clear();
            speculative->has_last_draft = false;
            speculative->last_draft_verified = false;
            speculative->last_error.clear();
            return 0;
        }
        if (speculative->has_last_draft) {
            throw std::runtime_error(
                    "cannot create a new native draft while the previous draft is outstanding");
        }
        assign_tokens(speculative->prompt, prompt_tokens, prompt_token_count);
        speculative->result.clear();
        speculative->has_last_draft = false;
        speculative->last_draft_verified = false;

        auto & draft_params = common_speculative_get_draft_params(speculative->speculative, 0);
        draft_params = {
            /* .drafting = */ true,
            /* .n_max    = */ (int32_t) output_capacity,
            /* .n_past   = */ n_past,
            /* .id_last  = */ id_last,
            /* .prompt   = */ &speculative->prompt,
            /* .result   = */ &speculative->result,
        };

        auto rollback_temporary_block = [&]() -> bool {
            auto * memory = llama_get_memory(speculative->draft_context);
            return memory != nullptr && llama_memory_seq_rm(memory, 0, n_past, -1);
        };

        try {
            common_speculative_draft(speculative->speculative);
        } catch (...) {
            rollback_temporary_block();
            throw;
        }

        // Match llama.cpp server cleanup at checkpoint.pos_max + 1. Gemma 4's
        // shared-cache seq_rm is intentionally a no-op, while independent draft
        // contexts discard their temporary proposal block here. The target
        // verification batch then repopulates the accepted path through process().
        if (!rollback_temporary_block()) {
            set_error(speculative, "failed to roll back temporary draft-context block");
            return -1;
        }

        const size_t count = std::min(output_capacity, speculative->result.size());
        std::copy_n(speculative->result.data(), count, output_tokens);
        speculative->has_last_draft = count > 0;
        speculative->last_draft_verified = false;
        speculative->last_error.clear();
        return (int32_t) count;
    } catch (const std::exception & exc) {
        set_error(speculative, exc.what());
        return -1;
    } catch (...) {
        set_error(speculative, "unknown native speculative draft error");
        return -1;
    }
}

bool llama_cpp_native_speculative_accept(
        llama_cpp_native_speculative * speculative,
        uint16_t n_accepted) {
    if (speculative == nullptr) {
        return false;
    }

    try {
        if (speculative->has_last_draft) {
            if (n_accepted > speculative->result.size()) {
                return set_error(speculative, "accepted draft count exceeds the last draft size");
            }
            if (speculative->type == COMMON_SPECULATIVE_TYPE_DRAFT_MTP &&
                !speculative->last_draft_verified) {
                if (n_accepted != 0) {
                    return set_error(
                            speculative,
                            "cannot accept MTP tokens before target verification succeeds");
                }
                // Cancellation before verification must preserve the previous
                // pending hidden row. MTP accept(0) would otherwise copy stale
                // verification output into the next draft state.
            } else {
                common_speculative_accept(speculative->speculative, 0, n_accepted);
            }
            speculative->has_last_draft = false;
            speculative->last_draft_verified = false;
        }
        speculative->last_error.clear();
        return true;
    } catch (const std::exception & exc) {
        return set_error(speculative, exc.what());
    } catch (...) {
        return set_error(speculative, "unknown native speculative accept error");
    }
}

bool llama_cpp_native_speculative_memory_seq_rm(
        llama_cpp_native_speculative * speculative,
        int32_t p0,
        int32_t p1) {
    if (speculative == nullptr || speculative->draft_context == nullptr) {
        return false;
    }

    auto * memory = llama_get_memory(speculative->draft_context);
    if (memory == nullptr || !llama_memory_seq_rm(memory, 0, p0, p1)) {
        return set_error(speculative, "draft context does not support the requested sequence rollback");
    }
    speculative->last_error.clear();
    return true;
}

bool llama_cpp_native_speculative_memory_seq_add(
        llama_cpp_native_speculative * speculative,
        int32_t p0,
        int32_t p1,
        int32_t delta) {
    if (speculative == nullptr || speculative->draft_context == nullptr) {
        return false;
    }

    auto * memory = llama_get_memory(speculative->draft_context);
    if (memory == nullptr) {
        return set_error(speculative, "draft context has no memory object");
    }
    llama_memory_seq_add(memory, 0, p0, p1, delta);
    speculative->last_error.clear();
    return true;
}

bool llama_cpp_native_speculative_memory_clear(
        llama_cpp_native_speculative * speculative,
        bool clear_data) {
    if (speculative == nullptr || speculative->draft_context == nullptr) {
        return false;
    }

    try {
        auto * memory = llama_get_memory(speculative->draft_context);
        if (memory == nullptr) {
            return set_error(speculative, "draft context has no memory object");
        }
        // Gemma 4 assistant contexts share the target cache's cell ledger.
        // Python clears the target first; clearing this shared view again would
        // also erase live target state if callers invoked draft.clear()
        // independently. Embedded MTP contexts with independent memory are
        // cleared here exactly once.
        if (!speculative->shares_target_memory) {
            llama_memory_clear(memory, clear_data);
        }
        llama_synchronize(speculative->draft_context);

        speculative->prompt.clear();
        speculative->result.clear();
        speculative->has_last_draft = false;
        speculative->last_draft_verified = false;

        if (is_internal_mtp(speculative)) {
            // common_speculative_impl_draft_mtp keeps pending_h/verify_h outside
            // the context memory. Its begin() hook deliberately does not reset
            // them, so retaining the implementation after a full clear would
            // feed the previous request's final hidden row into position zero of
            // the next request. Recreate only the implementation; draft_init
            // continues to own the already-cleared MTP context.
            if (speculative->speculative != nullptr) {
                common_speculative_free(speculative->speculative);
                speculative->speculative = nullptr;
            }

            auto & draft = speculative->params.speculative.draft;
            draft.ctx_tgt = speculative->target_context;
            draft.ctx_dft = speculative->draft_context;
            speculative->speculative = common_speculative_init(
                    speculative->params.speculative, 1);
            if (speculative->speculative == nullptr) {
                return set_error(
                        speculative,
                        "failed to reset embedded MTP state after clearing its context");
            }
        }

        speculative->last_error.clear();
        return true;
    } catch (const std::exception & exc) {
        return set_error(speculative, exc.what());
    } catch (...) {
        return set_error(speculative, "unknown native speculative clear error");
    }
}

void llama_cpp_native_speculative_print_stats(
        const llama_cpp_native_speculative * speculative) {
    if (speculative != nullptr && speculative->speculative != nullptr) {
        common_speculative_print_stats(speculative->speculative);
    }
}

} // extern "C"
