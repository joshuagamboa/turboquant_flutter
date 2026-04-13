#pragma once
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct tq_engine tq_engine_t;

typedef struct {
    const char* model_path;
    int32_t n_ctx;
    int32_t n_threads;
    const char* cache_type_k;
    const char* cache_type_v;
    int32_t n_gpu_layers;
    bool offload_kv;
} tq_config_t;

typedef struct {
    bool gpu_available;
    bool metal_available;
    bool vulkan_available;
    bool is_simulator;
    bool turbo3_supported;
    bool turbo4_supported;
    bool simdgroup_reduction_available;
    bool tensor_api_available;
    int32_t apple_gpu_family;
    int64_t system_ram_mb;
    int32_t recommended_n_ctx;
} tq_probe_result_t;

typedef struct {
    bool success;
    bool gpu_layers_enabled;
    bool offload_kv;
    bool cpu_kv_fallback;
    bool flash_attention_auto;
    bool flash_attention_required;
    int32_t n_gpu_layers;
} tq_validation_result_t;

typedef void (*tq_token_cb)(const char* token, bool is_end, void* user_data);

bool tq_probe(tq_probe_result_t* out_probe, char* err, int32_t err_cap);
bool tq_validate_config(
    tq_config_t config,
    tq_validation_result_t* out_result,
    char* err,
    int32_t err_cap
);

tq_engine_t* tq_init(tq_config_t config, char* err, int32_t err_cap);
void tq_free(tq_engine_t* engine);

bool tq_generate(
    tq_engine_t* engine,
    const char* prompt,
    tq_token_cb callback,
    void* user_data,
    char* err,
    int32_t err_cap
);

void tq_stop_generation(tq_engine_t* engine);

#ifdef __cplusplus
}
#endif
