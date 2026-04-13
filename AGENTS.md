# TurboQuant Flutter — Agent Instructions & Testing Guide

This file tracks the current state of the project, architecture decisions, and platform-specific testing procedures for future agents.

## 🏗 Project State & Architecture

*   **Engine Fork**: llama.cpp with TurboQuant patches located in `third_party/llama_cpp_turboquant`.
*   **Native Wrapper**: `native/tq_ffi` provides a stable C ABI (`tq_ffi.h`) for Flutter.
*   **Plugin Architecture**: 
    *   Uses **FFI** for high-performance bridge.
    *   Offloads inference to **Background Worker Isolates** to keep the UI thread fluid.
    *   Supports dynamic KV-Cache quantization selection (`turbo4`, `turbo3`, `q8_0`, `f16`).
    *   Implements **Dynamic Context Sizing** based on OS-level memory probing.

## 🧪 Testing How-Tos

### Standard Test Model
*   **Model**: `Gemma 4 E2B`
*   **Recommended Quant**: `Q4_K_M`
*   **Hugging Face Source**: `unsloth/gemma-4-2b-it-GGUF`

### Provisioning Models to Devices
Automated in-app downloading has been removed. Models must be provisioned manually:

#### **iOS (Simulator & Physical)**
1.  **Finder Method**: Connect device/start simulator. In Finder, select the device > Files tab. Drag the `.gguf` file into the `Turboquant Flutter` folder.
2.  **Location**: The app scans the root `Documents` directory automatically.

#### **Android Emulator**
1.  Use `adb push`:
    ```bash
    adb push gemma-4-E2B-it-Q4_K_M.gguf /data/user/0/com.example.turboquant_flutter_example/app_flutter/models/
    ```

### Running Benchmarks
1.  Launch the example app (`packages/turboquant_flutter/example`).
2.  **Hardware Probe**: On startup (or via button), the app will detect GPU (Metal/Vulkan) and Total RAM.
3.  **Model Selection**: If provisioned correctly, the model will appear in the "Local Models" list.
4.  **Auto-Run**: If a model is found, a **5-second timer** will automatically trigger generation for benchmarking.
5.  **Metrics**: Results are displayed as **Tokens Per Second (TPS)** at the bottom of the response area.

## 🛠 Platform-Specific Notes

### **iOS Simulator vs. Physical Hardware**
*   **Simulator**: Has strict memory limits (~1.8GB-2GB). The app forces `use_gpu: false` and `n_ctx: 128` on simulators to prevent `SIGABRT` crashes during graph initialization.
*   **Physical Device**: Full Metal support is enabled. `n_ctx` scales up to `8192` depending on RAM.
*   **Linking**: iOS uses static libraries (`.a`). If adding new native dependencies, they must be rebuilt for the correct architecture (`iphonesimulator` or `iphoneos`) and added to `ios/libs/`.

### **Android**
*   **Architecture**: Restricted to `arm64-v8a` in `build.gradle.kts` to avoid legacy NEON/float16 compilation errors.
*   **Logging**: Native debug messages are routed through `__android_log_print` with the tag `TurboQuantNative`.

## ⚠️ Troubleshooting Crashes
*   **White Screen on Startup**: Usually caused by a blocking `llama_backend_init` call during Metal shader compilation. Fixed by moving probing to a post-frame callback.
*   **EXC_BAD_ACCESS (iOS)**: Often related to static library stripping or incorrect `DynamicLibrary.process()` lookups. Ensure symbols are exported and use `-all_load` in Podspec if needed.
*   **Failed to Load Model**: Check file size in logs. If size is ~29 bytes, it's a Hugging Face error message, not a model. Ensure standard filesystem paths are used (strip `file://` prefixes).
