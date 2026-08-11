#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#    define LLAMA_CPP_NATIVE_SPEC_API __declspec(dllexport)
#else
#    define LLAMA_CPP_NATIVE_SPEC_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

struct llama_context;
struct llama_model;
struct llama_batch;

// Experimental, llama-cpp-python-specific C ABI around llama.cpp/common's
// speculative decoding implementation. This is deliberately small and only
// targets a single text sequence for now.
typedef struct llama_cpp_native_speculative llama_cpp_native_speculative;

#define LLAMA_CPP_NATIVE_SPECULATIVE_ABI_VERSION 2u

typedef struct llama_cpp_native_speculative_params {
    uint32_t struct_size;

    // Null or empty selects embedded target-model MTP for draft-mtp only.
    // All external draft providers require a non-empty model path.
    const char * model_path;
    const char * spec_type;

    int32_t n_gpu_layers;
    int32_t n_ctx;
    int32_t n_batch;
    int32_t n_ubatch;
    int32_t n_threads;
    int32_t n_threads_batch;

    int32_t n_max;
    int32_t n_min;
    float   p_min;

    int32_t cache_type_k;
    int32_t cache_type_v;
    int32_t flash_attn_type;

    bool offload_kqv;
    bool op_offload;
    bool kv_unified;
    bool no_perf;
} llama_cpp_native_speculative_params;

LLAMA_CPP_NATIVE_SPEC_API uint32_t llama_cpp_native_speculative_abi_version(void);

LLAMA_CPP_NATIVE_SPEC_API llama_cpp_native_speculative * llama_cpp_native_speculative_init(
        struct llama_model * target_model,
        struct llama_context * target_context,
        const llama_cpp_native_speculative_params * params,
        char * error,
        size_t error_capacity);

LLAMA_CPP_NATIVE_SPEC_API void llama_cpp_native_speculative_free(
        llama_cpp_native_speculative * speculative);

LLAMA_CPP_NATIVE_SPEC_API const char * llama_cpp_native_speculative_last_error(
        const llama_cpp_native_speculative * speculative);

LLAMA_CPP_NATIVE_SPEC_API bool llama_cpp_native_speculative_begin(
        llama_cpp_native_speculative * speculative,
        const int32_t * prompt_tokens,
        size_t prompt_token_count);

// Must be called after every successful target llama_decode() batch.
LLAMA_CPP_NATIVE_SPEC_API bool llama_cpp_native_speculative_process(
        llama_cpp_native_speculative * speculative,
        const struct llama_batch * batch);

// Returns the number of tokens written, or -1 on error.
LLAMA_CPP_NATIVE_SPEC_API int32_t llama_cpp_native_speculative_draft(
        llama_cpp_native_speculative * speculative,
        int32_t n_past,
        int32_t id_last,
        const int32_t * prompt_tokens,
        size_t prompt_token_count,
        int32_t * output_tokens,
        size_t output_capacity);

LLAMA_CPP_NATIVE_SPEC_API bool llama_cpp_native_speculative_accept(
        llama_cpp_native_speculative * speculative,
        uint16_t n_accepted);

// Mirror target-context cache operations into the owned draft context.
LLAMA_CPP_NATIVE_SPEC_API bool llama_cpp_native_speculative_memory_seq_rm(
        llama_cpp_native_speculative * speculative,
        int32_t p0,
        int32_t p1);

LLAMA_CPP_NATIVE_SPEC_API bool llama_cpp_native_speculative_memory_seq_add(
        llama_cpp_native_speculative * speculative,
        int32_t p0,
        int32_t p1,
        int32_t delta);

LLAMA_CPP_NATIVE_SPEC_API bool llama_cpp_native_speculative_memory_clear(
        llama_cpp_native_speculative * speculative,
        bool clear_data);

LLAMA_CPP_NATIVE_SPEC_API void llama_cpp_native_speculative_print_stats(
        const llama_cpp_native_speculative * speculative);

#ifdef __cplusplus
}
#endif
