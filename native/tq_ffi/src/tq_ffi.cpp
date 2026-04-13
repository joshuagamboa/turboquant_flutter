#include "tq_ffi.h"

#include "common.h"
#include "ggml-backend.h"
#include "llama.h"

#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#include "ggml/src/ggml-metal/ggml-metal-device.h"
#endif

#ifdef __APPLE__
#include <TargetConditionals.h>
#endif

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>
#include <unistd.h>

struct tq_engine {
    llama_model * model = nullptr;
    llama_context * ctx = nullptr;
    std::atomic<bool> stop_generation{false};
    std::mutex mutex;

    ~tq_engine() {
        if (ctx) {
            llama_free(ctx);
        }
        if (model) {
            llama_model_free(model);
        }
    }
};

struct tq_runtime_config {
    ggml_type type_k = GGML_TYPE_Q8_0;
    ggml_type type_v = GGML_TYPE_Q8_0;
    int32_t n_gpu_layers = 0;
    bool offload_kv = false;
    bool gpu_layers_enabled = false;
    llama_flash_attn_type flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;
    bool flash_attention_auto = true;
    bool flash_attention_required = false;
};

static std::once_flag backend_init_flag;
static thread_local std::string g_log_capture;

#ifdef ANDROID
#include <android/log.h>
#define LOG_TAG "TurboQuantNative"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)

static void llama_log_callback(ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    (void) user_data;
    if (text) {
        g_log_capture.append(text);
        if (g_log_capture.size() > 16384) {
            g_log_capture.erase(0, g_log_capture.size() - 16384);
        }
        __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "%s", text);
    }
}
#else
#define LOGD(...) printf(__VA_ARGS__)

static void llama_log_callback(ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    (void) user_data;
    if (text) {
        g_log_capture.append(text);
        if (g_log_capture.size() > 16384) {
            g_log_capture.erase(0, g_log_capture.size() - 16384);
        }
        printf("%s", text);
    }
}
#endif

static ggml_type parse_ggml_type(const char * name) {
    if (name == nullptr) return GGML_TYPE_F16;
    if (strcmp(name, "f32") == 0) return GGML_TYPE_F32;
    if (strcmp(name, "f16") == 0) return GGML_TYPE_F16;
    if (strcmp(name, "q4_0") == 0) return GGML_TYPE_Q4_0;
    if (strcmp(name, "q4_1") == 0) return GGML_TYPE_Q4_1;
    if (strcmp(name, "q5_0") == 0) return GGML_TYPE_Q5_0;
    if (strcmp(name, "q5_1") == 0) return GGML_TYPE_Q5_1;
    if (strcmp(name, "q8_0") == 0) return GGML_TYPE_Q8_0;
    if (strcmp(name, "turbo3") == 0) return GGML_TYPE_TURBO3_0;
    if (strcmp(name, "turbo4") == 0) return GGML_TYPE_TURBO4_0;
    return GGML_TYPE_F16;
}

static void clear_log_capture() {
    g_log_capture.clear();
}

static std::string log_capture_tail() {
    constexpr size_t kMaxTailBytes = 4096;
    if (g_log_capture.size() <= kMaxTailBytes) {
        return g_log_capture;
    }
    return g_log_capture.substr(g_log_capture.size() - kMaxTailBytes);
}

static void set_error(char * err, int32_t err_cap, const std::string & message) {
    if (err && err_cap > 0) {
        snprintf(err, err_cap, "%s", message.c_str());
    }
}

static std::string build_error_message(const std::string & message) {
    const std::string logs = log_capture_tail();
    if (logs.empty()) {
        return message;
    }

    return message + "\nNative log tail:\n" + logs;
}

static tq_runtime_config resolve_runtime_config(const tq_config_t & config) {
    tq_runtime_config resolved;
    resolved.type_k = parse_ggml_type(config.cache_type_k);
    resolved.type_v = parse_ggml_type(config.cache_type_v);
    resolved.n_gpu_layers = std::max(0, config.n_gpu_layers);
    resolved.offload_kv = config.offload_kv;
    resolved.gpu_layers_enabled = resolved.n_gpu_layers > 0;
    resolved.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;
    resolved.flash_attention_auto = true;
    resolved.flash_attention_required =
        ggml_is_quantized(resolved.type_k) || ggml_is_quantized(resolved.type_v);
    return resolved;
}

static const char * runtime_path_name(const tq_runtime_config & config) {
    if (config.gpu_layers_enabled && config.offload_kv) {
        return "GPU-KV";
    }
    if (config.gpu_layers_enabled && !config.offload_kv) {
        return "CPU-KV fallback";
    }
    return "CPU-only";
}

static void fill_validation_result(
    tq_validation_result_t * out_result,
    const tq_runtime_config & runtime_config,
    bool success
) {
    if (!out_result) {
        return;
    }

    out_result->success = success;
    out_result->gpu_layers_enabled = runtime_config.gpu_layers_enabled;
    out_result->offload_kv = runtime_config.offload_kv;
    out_result->cpu_kv_fallback = runtime_config.gpu_layers_enabled && !runtime_config.offload_kv;
    out_result->flash_attention_auto = runtime_config.flash_attention_auto;
    out_result->flash_attention_required = runtime_config.flash_attention_required;
    out_result->n_gpu_layers = runtime_config.n_gpu_layers;
}

static void log_effective_config(
    const char * stage,
    const tq_config_t & raw_config,
    const tq_runtime_config & runtime_config
) {
    LOGD(
        "[%s] model=%s n_ctx=%d n_threads=%d n_gpu_layers=%d offload_kv=%s path=%s "
        "type_k=%s type_v=%s flash_attn=%s%s\n",
        stage,
        raw_config.model_path ? raw_config.model_path : "(null)",
        raw_config.n_ctx,
        raw_config.n_threads,
        runtime_config.n_gpu_layers,
        runtime_config.offload_kv ? "true" : "false",
        runtime_path_name(runtime_config),
        ggml_type_name(runtime_config.type_k),
        ggml_type_name(runtime_config.type_v),
        llama_flash_attn_type_name(runtime_config.flash_attn_type),
        runtime_config.flash_attention_required ? " (required by quantized KV)" : ""
    );
}

static void ensure_backend_ready() {
    std::call_once(backend_init_flag, []() {
        llama_backend_init();
        llama_log_set(llama_log_callback, nullptr);
    });
}

static bool validate_model_file(const char * model_path, char * err, int32_t err_cap) {
    FILE * file = fopen(model_path, "rb");
    if (!file) {
        set_error(err, err_cap, "File not found or not readable: " + std::string(model_path));
        return false;
    }

    char magic[4];
    if (fread(magic, 1, sizeof(magic), file) != sizeof(magic) || memcmp(magic, "GGUF", sizeof(magic)) != 0) {
        fclose(file);
        set_error(err, err_cap, "Invalid GGUF file: " + std::string(model_path));
        return false;
    }

    fseek(file, 0, SEEK_END);
    const long size = ftell(file);
    fclose(file);
    LOGD("[tq_init] validated GGUF file, size=%ld bytes\n", size);
    return true;
}

static tq_engine * tq_init_internal(
    tq_config_t config,
    tq_validation_result_t * validation_result,
    char * err,
    int32_t err_cap
) {
    if (config.model_path == nullptr || strlen(config.model_path) == 0) {
        set_error(err, err_cap, "Model path is empty");
        return nullptr;
    }

    ensure_backend_ready();
    clear_log_capture();

    const tq_runtime_config runtime_config = resolve_runtime_config(config);
    fill_validation_result(validation_result, runtime_config, false);
    log_effective_config("tq_init", config, runtime_config);

    if (!validate_model_file(config.model_path, err, err_cap)) {
        return nullptr;
    }

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = runtime_config.n_gpu_layers;

    llama_model * model = llama_model_load_from_file(config.model_path, mparams);
    if (!model) {
        set_error(
            err,
            err_cap,
            build_error_message(
                "Failed to load model for " + std::string(runtime_path_name(runtime_config)) +
                " path: " + std::string(config.model_path)
            )
        );
        return nullptr;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = config.n_ctx;
    cparams.n_threads = config.n_threads;
    cparams.n_threads_batch = config.n_threads;
    cparams.type_k = runtime_config.type_k;
    cparams.type_v = runtime_config.type_v;
    cparams.flash_attn_type = runtime_config.flash_attn_type;
    cparams.offload_kqv = runtime_config.offload_kv;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        llama_model_free(model);
        set_error(
            err,
            err_cap,
            build_error_message(
                "Failed to create context for " + std::string(runtime_path_name(runtime_config)) +
                " path (type_k=" + std::string(ggml_type_name(runtime_config.type_k)) +
                ", type_v=" + std::string(ggml_type_name(runtime_config.type_v)) + ")"
            )
        );
        return nullptr;
    }

    fill_validation_result(validation_result, runtime_config, true);

    tq_engine * engine = new tq_engine();
    engine->model = model;
    engine->ctx = ctx;
    return engine;
}

bool tq_probe(tq_probe_result_t * out_probe, char * err, int32_t err_cap) {
    (void) err;
    (void) err_cap;

    if (!out_probe) {
        return false;
    }

    ensure_backend_ready();
    memset(out_probe, 0, sizeof(*out_probe));

    out_probe->gpu_available = llama_supports_gpu_offload();
    out_probe->turbo3_supported = parse_ggml_type("turbo3") == GGML_TYPE_TURBO3_0;
    out_probe->turbo4_supported = parse_ggml_type("turbo4") == GGML_TYPE_TURBO4_0;

#ifdef GGML_USE_METAL
    out_probe->metal_available = out_probe->gpu_available;
#endif

#ifdef GGML_USE_VULKAN
    out_probe->vulkan_available = out_probe->gpu_available;
#endif

#if defined(__APPLE__) && TARGET_OS_SIMULATOR
    out_probe->is_simulator = true;
#endif

#ifdef GGML_USE_METAL
    if (out_probe->metal_available) {
        ggml_backend_t metal_backend = ggml_backend_metal_init();
        if (metal_backend != nullptr) {
            for (int family = 20; family >= 1; --family) {
                if (ggml_backend_metal_supports_family(metal_backend, family)) {
                    out_probe->apple_gpu_family = family;
                    break;
                }
            }

            ggml_metal_device_t dev = ggml_metal_device_get(0);
            if (dev != nullptr) {
                const struct ggml_metal_device_props * props = ggml_metal_device_get_props(dev);
                if (props != nullptr) {
                    out_probe->simdgroup_reduction_available = props->has_simdgroup_reduction;
                    out_probe->tensor_api_available = props->has_tensor;
                }
            }

            ggml_backend_free(metal_backend);
        }
    }
#endif

    const long pages = sysconf(_SC_PHYS_PAGES);
    const long page_size = sysconf(_SC_PAGE_SIZE);
    const int64_t total_memory = (int64_t) pages * page_size;
    const int64_t total_mb = total_memory / (1024 * 1024);

    out_probe->system_ram_mb = total_mb;

    if (out_probe->is_simulator) {
        out_probe->recommended_n_ctx = 128;
    } else if (total_mb <= 4096) {
        out_probe->recommended_n_ctx = 2048;
    } else if (total_mb <= 8192) {
        out_probe->recommended_n_ctx = 4096;
    } else {
        out_probe->recommended_n_ctx = 8192;
    }

    LOGD(
        "[tq_probe] gpu=%s metal=%s vulkan=%s simulator=%s apple_gpu_family=%d "
        "simdgroup_reduction=%s tensor_api=%s recommended_n_ctx=%d\n",
        out_probe->gpu_available ? "true" : "false",
        out_probe->metal_available ? "true" : "false",
        out_probe->vulkan_available ? "true" : "false",
        out_probe->is_simulator ? "true" : "false",
        out_probe->apple_gpu_family,
        out_probe->simdgroup_reduction_available ? "true" : "false",
        out_probe->tensor_api_available ? "true" : "false",
        out_probe->recommended_n_ctx
    );

    return true;
}

bool tq_validate_config(
    tq_config_t config,
    tq_validation_result_t * out_result,
    char * err,
    int32_t err_cap
) {
    if (out_result) {
        memset(out_result, 0, sizeof(*out_result));
    }

    tq_engine * engine = tq_init_internal(config, out_result, err, err_cap);
    if (!engine) {
        return false;
    }

    delete engine;
    return true;
}

tq_engine_t * tq_init(tq_config_t config, char * err, int32_t err_cap) {
    return reinterpret_cast<tq_engine_t *>(tq_init_internal(config, nullptr, err, err_cap));
}

void tq_free(tq_engine_t * engine) {
    if (engine) {
        delete reinterpret_cast<tq_engine *>(engine);
    }
}

bool tq_generate(
    tq_engine_t * engine_ptr,
    const char * prompt,
    tq_token_cb callback,
    void * user_data,
    char * err,
    int32_t err_cap
) {
    tq_engine * engine = reinterpret_cast<tq_engine *>(engine_ptr);
    if (!engine) {
        set_error(err, err_cap, "Engine is null");
        return false;
    }

    std::lock_guard<std::mutex> lock(engine->mutex);
    engine->stop_generation = false;
    clear_log_capture();

    const llama_vocab * vocab = llama_model_get_vocab(engine->model);

    std::vector<llama_token> tokens_list;
    tokens_list.resize(strlen(prompt) + 1);
    int32_t n_tokens = llama_tokenize(vocab, prompt, strlen(prompt), tokens_list.data(), tokens_list.size(), true, true);
    if (n_tokens < 0) {
        tokens_list.resize(-n_tokens);
        n_tokens = llama_tokenize(vocab, prompt, strlen(prompt), tokens_list.data(), tokens_list.size(), true, true);
    }
    tokens_list.resize(n_tokens);

    llama_batch batch = llama_batch_init(llama_n_ctx(engine->ctx), 0, 1);
    for (size_t i = 0; i < tokens_list.size(); ++i) {
        common_batch_add(batch, tokens_list[i], i, {0}, i == tokens_list.size() - 1);
    }

    if (llama_decode(engine->ctx, batch) != 0) {
        llama_batch_free(batch);
        set_error(err, err_cap, build_error_message("Initial decode failed"));
        callback("", true, user_data);
        return false;
    }

    int32_t n_cur = tokens_list.size();

    // Sampling order matters: filter first, then select.
    auto sparams = llama_sampler_chain_default_params();
    llama_sampler * smpl = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(0.7f));
    llama_sampler_chain_add(smpl, llama_sampler_init_top_k(40));
    llama_sampler_chain_add(smpl, llama_sampler_init_top_p(0.95f, 1));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(42));

    bool sent_end = false;

    while (n_cur < llama_n_ctx(engine->ctx) && !engine->stop_generation) {
        const llama_token id = llama_sampler_sample(smpl, engine->ctx, -1);

        if (llama_vocab_is_eog(vocab, id)) {
            callback("", true, user_data);
            sent_end = true;
            break;
        }

        char buf[128];
        const int n = llama_token_to_piece(vocab, id, buf, sizeof(buf), 0, true);
        if (n > 0) {
            std::string token_str(buf, n);
            callback(token_str.c_str(), false, user_data);
        }

        common_batch_clear(batch);
        common_batch_add(batch, id, n_cur, {0}, true);
        ++n_cur;

        if (llama_decode(engine->ctx, batch) != 0) {
            llama_batch_free(batch);
            llama_sampler_free(smpl);
            set_error(
                err,
                err_cap,
                build_error_message("Decode failed at step " + std::to_string(n_cur))
            );
            callback("", true, user_data);
            return false;
        }
    }

    if (!sent_end) {
        callback("", true, user_data);
    }

    llama_batch_free(batch);
    llama_sampler_free(smpl);
    return true;
}

void tq_stop_generation(tq_engine_t * engine_ptr) {
    tq_engine * engine = reinterpret_cast<tq_engine *>(engine_ptr);
    if (engine) {
        engine->stop_generation = true;
    }
}
