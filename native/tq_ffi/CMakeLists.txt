cmake_minimum_required(VERSION 3.10)
project(tq_ffi)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(LLAMA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/llama_cpp_turboquant)

# Build llama.cpp's common library
set(LLAMA_BUILD_COMMON ON CACHE BOOL "Build common library" FORCE)

# Add llama.cpp as a subdirectory
add_subdirectory(${LLAMA_DIR} ${CMAKE_CURRENT_BINARY_DIR}/llama_cpp)

# Include directories
include_directories(
    include
    ${LLAMA_DIR}/include
    ${LLAMA_DIR}/common
    ${LLAMA_DIR}
)

# Create library
add_library(tq_ffi src/tq_ffi.cpp)

# Link with llama.cpp
target_link_libraries(tq_ffi PRIVATE llama common)

if (ANDROID)
    target_link_libraries(tq_ffi PRIVATE log)
endif()

# Platform specific configurations
if (APPLE)
    # Metal is usually enabled by default in llama.cpp for Apple
    target_compile_definitions(tq_ffi PRIVATE GGML_USE_METAL)
elseif (ANDROID)
    # Vulkan is preferred for Android
    target_compile_definitions(tq_ffi PRIVATE GGML_USE_VULKAN)
endif()

# Install rules if needed, but for Flutter we usually just need the library
