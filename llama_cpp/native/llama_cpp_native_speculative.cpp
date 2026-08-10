#include "llama_cpp_native_speculative.h"

#include "common.h"
#include "llama.h"
#include "speculative.h"

#include <algorithm>
#include <cstring>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

struct llama_cpp_native_speculative {
    common_params params;
    common_speculative_init_result_ptr draft_init;
    common_speculative * speculative = nullptr;

    llama_context * target_context = nullptr;
    llama_context * draft_context  = nullptr;

    llama_tokens prompt;
    llama_tokens result;
    std::string last_error;
    bool has_last_draft = false;

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
        case COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH:
        case COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK:
            return true;
        default:
            return false;
    }
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
    for (int32_t i = 0; i < batch.n_tokens; ++i) {
        if (batch.n_seq_id[i] != 1 || batch.seq_id[i] == nullptr || batch.seq_id[i][0] != 0) {
            return set_error(speculative, "experimental native speculative decoding only supports sequence 0");
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
        if (params->model_path == nullptr || params->model_path[0] == '\0') {
            throw std::invalid_argument("draft model path is required");
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

        auto result = std::make_unique<llama_cpp_native_speculative>();
        result->target_context = target_context;

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
        draft.mparams.path      = params->model_path;
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
        if (draft_model == nullptr || result->draft_context == nullptr) {
            throw std::runtime_error(std::string("failed to load draft model: ") + params->model_path);
        }

        draft.ctx_tgt = target_context;
        draft.ctx_dft = result->draft_context;
        result->speculative = common_speculative_init(base.speculative, 1);
        if (result->speculative == nullptr) {
            throw std::runtime_error("failed to initialize llama.cpp common speculative decoder");
        }

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
            speculative->last_error.clear();
            return 0;
        }
        assign_tokens(speculative->prompt, prompt_tokens, prompt_token_count);
        speculative->result.clear();
        speculative->has_last_draft = false;

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

        // common_speculative_draft evaluates a temporary noise/proposal block
        // in the draft context. The target verification batch will populate the
        // accepted path again through process(), so discard the temporary block.
        if (!rollback_temporary_block()) {
            set_error(speculative, "failed to roll back temporary draft-context block");
            return -1;
        }

        const size_t count = std::min(output_capacity, speculative->result.size());
        std::copy_n(speculative->result.data(), count, output_tokens);
        speculative->has_last_draft = count > 0;
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
            common_speculative_accept(speculative->speculative, 0, n_accepted);
            speculative->has_last_draft = false;
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

    auto * memory = llama_get_memory(speculative->draft_context);
    if (memory == nullptr) {
        return set_error(speculative, "draft context has no memory object");
    }
    llama_memory_clear(memory, clear_data);
    llama_synchronize(speculative->draft_context);
    speculative->prompt.clear();
    speculative->result.clear();
    speculative->has_last_draft = false;
    speculative->last_error.clear();
    return true;
}

void llama_cpp_native_speculative_print_stats(
        const llama_cpp_native_speculative * speculative) {
    if (speculative != nullptr && speculative->speculative != nullptr) {
        common_speculative_print_stats(speculative->speculative);
    }
}

} // extern "C"
