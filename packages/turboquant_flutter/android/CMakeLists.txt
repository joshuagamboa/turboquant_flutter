cmake_minimum_required(VERSION 3.10)
project(turboquant_flutter)

# The tq_ffi CMakeLists.txt is in the root native/tq_ffi directory
# But we want to build it as part of the Android build.
# We'll include the root one.

set(TQ_FFI_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../native/tq_ffi)

# llama.cpp needs to know it's being built for Android
set(ANDROID TRUE)

add_subdirectory(${TQ_FFI_DIR} tq_ffi_build)
