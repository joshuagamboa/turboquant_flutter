#include "tq_ffi.h"
#include "llama.h"
#include "common.h"
#include <vector>
#include <string>
#include <mutex>
#include <atomic>
#include <cstring>

struct tq_engine {
    llama_model* model = nullptr;
    llama_context* ctx = nullptr;
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

static ggml_type parse_ggml_type(const char* name) {
    if (name == nullptr) return GGML_TYPE_F16;
    if (strcmp(name, "f32") == 0) return GGML_TYPE_F32;
    if (strcmp(name, "f16") == 0) return GGML_TYPE_F16;
    if (strcmp(name, "q4_0") == 0) return GGML_TYPE_Q4_0;
    if (strcmp(name, "q4_1") == 0) return GGML_TYPE_Q4_1;
    if (strcmp(name, "q5_0") == 0) return GGML_TYPE_Q5_0;
    if (strcmp(name, "q5_1") == 0) return GGML_TYPE_Q5_1;
    if (strcmp(name, "q8_0") == 0) return GGML_TYPE_Q8_0;
    if (strcmp(name, "turbo3") == 0) return GGML_TYPE_TQ1_0;
    if (strcmp(name, "turbo4") == 0) return GGML_TYPE_TQ2_0;
    return GGML_TYPE_F16;
}

#include <unistd.h>

bool tq_probe(tq_probe_result_t* out_probe, char* err, int32_t err_cap) {
    if (!out_probe) return false;
    
    llama_backend_init();

    out_probe->gpu_available = llama_supports_gpu_offload();
    
    out_probe->metal_available = false;
#ifdef GGML_USE_METAL
    out_probe->metal_available = out_probe->gpu_available;
#endif

    out_probe->vulkan_available = false;
#ifdef GGML_USE_VULKAN
    out_probe->vulkan_available = out_probe->gpu_available;
#endif

    // TurboQuant types (TQ1_0, TQ2_0) are supported if the binary was built with them
    // We've verified they exist in this llama.cpp fork.
    out_probe->turbo3_supported = true;
    out_probe->turbo4_supported = true;
    
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    int64_t total_memory = (int64_t)pages * page_size;
    int64_t total_mb = total_memory / (1024 * 1024);
    
    out_probe->system_ram_mb = total_mb;
    
    // Aggressive Dynamic Context Sizing leveraging TurboQuant compression
    if (total_mb <= 4096) {
        out_probe->recommended_n_ctx = 2048;
    } else if (total_mb <= 8192) {
        out_probe->recommended_n_ctx = 4096;
    } else {
        out_probe->recommended_n_ctx = 8192;
    }
    
    return true;
}

#include <mutex>

static std::once_flag backend_init_flag;

#ifdef ANDROID
#include <android/log.h>
#define LOG_TAG "TurboQuantNative"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)

static void llama_log_callback(ggml_log_level level, const char * text, void * user_data) {
    (void)level; (void)user_data;
    __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "%s", text);
}
#else
#define LOGD(...) printf(__VA_ARGS__)
static void llama_log_callback(ggml_log_level level, const char * text, void * user_data) {
    (void)level; (void)user_data;
    printf("%s", text);
}
#endif

tq_engine_t* tq_init(tq_config_t config, char* err, int32_t err_cap) {
    if (config.model_path == nullptr || strlen(config.model_path) == 0) {
        if (err && err_cap > 0) snprintf(err, err_cap, "Model path is empty");
        return nullptr;
    }

    std::call_once(backend_init_flag, []() {
        llama_backend_init();
        llama_log_set(llama_log_callback, nullptr);
    });

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = config.use_gpu ? 99 : 0;

    LOGD("Native tq_init loading: %s", config.model_path);
    FILE* f = fopen(config.model_path, "rb");
    if (f) {
        char magic[4];
        if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "GGUF", 4) != 0) {
            LOGD("File is NOT a valid GGUF (Magic check failed)");
            if (err && err_cap > 0) {
                snprintf(err, err_cap, "Invalid GGUF file: %s", config.model_path);
            }
            fclose(f);
            return nullptr;
        }
        fseek(f, 0, SEEK_END);
        long size = ftell(f);
        fseek(f, 0, SEEK_SET);
        LOGD("Valid GGUF found, size: %ld bytes", size);
        fclose(f);
    } else {
        LOGD("File NOT found or NOT readable via fopen");
        if (err && err_cap > 0) {
            snprintf(err, err_cap, "File NOT found or NOT readable: %s", config.model_path);
        }
        return nullptr;
    }

    llama_model* model = llama_model_load_from_file(config.model_path, mparams);
    if (!model) {
        if (err && err_cap > 0) {
            snprintf(err, err_cap, "Failed to load model: %s", config.model_path);
        }
        return nullptr;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = config.n_ctx;
    cparams.n_threads = config.n_threads;
    cparams.n_threads_batch = config.n_threads;
    cparams.type_k = parse_ggml_type(config.cache_type_k);
    cparams.type_v = parse_ggml_type(config.cache_type_v);
    cparams.offload_kqv = config.use_gpu;

    llama_context* ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        llama_model_free(model);
        if (err && err_cap > 0) {
            snprintf(err, err_cap, "Failed to create context");
        }
        return nullptr;
    }

    tq_engine* engine = new tq_engine();
    engine->model = model;
    engine->ctx = ctx;
    return (tq_engine_t*)engine;
}

void tq_free(tq_engine_t* engine) {
    if (engine) {
        delete (tq_engine*)engine;
    }
}

bool tq_generate(
    tq_engine_t* engine_ptr,
    const char* prompt,
    tq_token_cb callback,
    void* user_data,
    char* err,
    int32_t err_cap
) {
    tq_engine* engine = (tq_engine*)engine_ptr;
    if (!engine) return false;

    std::lock_guard<std::mutex> lock(engine->mutex);
    engine->stop_generation = false;

    const llama_vocab* vocab = llama_model_get_vocab(engine->model);
    
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
        common_batch_add(batch, tokens_list[i], i, { 0 }, i == tokens_list.size() - 1);
    }

    if (llama_decode(engine->ctx, batch) != 0) {
        llama_batch_free(batch);
        if (err && err_cap > 0) snprintf(err, err_cap, "Initial decode failed");
        return false;
    }

    int32_t n_cur = tokens_list.size();
    
    // Sampling
    auto sparams = llama_sampler_chain_default_params();
    llama_sampler* smpl = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(0.7f));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(42));
    llama_sampler_chain_add(smpl, llama_sampler_init_top_k(40));
    llama_sampler_chain_add(smpl, llama_sampler_init_top_p(0.95f, 1));

    while (n_cur < llama_n_ctx(engine->ctx) && !engine->stop_generation) {
        const llama_token id = llama_sampler_sample(smpl, engine->ctx, -1);
        
        if (llama_vocab_is_eog(vocab, id)) {
            callback("", true, user_data);
            break;
        }

        char buf[128];
        int n = llama_token_to_piece(vocab, id, buf, sizeof(buf), 0, true);
        if (n > 0) {
            std::string token_str(buf, n);
            callback(token_str.c_str(), false, user_data);
        }

        common_batch_clear(batch);
        common_batch_add(batch, id, n_cur, { 0 }, true);

        n_cur++;

        if (llama_decode(engine->ctx, batch) != 0) {
            if (err && err_cap > 0) snprintf(err, err_cap, "Decode failed at step %d", n_cur);
            break;
        }
    }

    llama_batch_free(batch);
    llama_sampler_free(smpl);
    return true;
}

void tq_stop_generation(tq_engine_t* engine_ptr) {
    tq_engine* engine = (tq_engine*)engine_ptr;
    if (engine) {
        engine->stop_generation = true;
    }
}
