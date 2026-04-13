#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${1:-device}"

case "${TARGET}" in
  device|iphoneos)
    BUILD_DIR="${ROOT_DIR}/native/tq_ffi/build_ios_device"
    TARGET_NAME="iphoneos"
    ;;
  simulator|iphonesimulator)
    BUILD_DIR="${ROOT_DIR}/native/tq_ffi/build_ios"
    TARGET_NAME="iphonesimulator"
    ;;
  *)
    echo "Usage: $0 [device|simulator]" >&2
    exit 1
    ;;
esac

DEST_LIBS_DIR="${ROOT_DIR}/packages/turboquant_flutter/ios/libs"
DEST_HEADER="${ROOT_DIR}/packages/turboquant_flutter/ios/Classes/tq_ffi.h"

mkdir -p "${DEST_LIBS_DIR}" "$(dirname "${DEST_HEADER}")"

copy_archive() {
  local source_path="$1"
  local dest_name="$2"

  if [[ ! -f "${source_path}" ]]; then
    echo "Missing archive: ${source_path}" >&2
    exit 1
  fi

  cp "${source_path}" "${DEST_LIBS_DIR}/${dest_name}"
}

copy_archive "${BUILD_DIR}/libtq_ffi.a" "libtq_ffi.a"
copy_archive "${BUILD_DIR}/llama_cpp/common/libcommon.a" "libcommon.a"
copy_archive "${BUILD_DIR}/llama_cpp/vendor/cpp-httplib/libcpp-httplib.a" "libcpp-httplib.a"
copy_archive "${BUILD_DIR}/llama_cpp/ggml/src/libggml-base.a" "libggml-base.a"
copy_archive "${BUILD_DIR}/llama_cpp/ggml/src/ggml-blas/libggml-blas.a" "libggml-blas.a"
copy_archive "${BUILD_DIR}/llama_cpp/ggml/src/libggml-cpu.a" "libggml-cpu.a"
copy_archive "${BUILD_DIR}/llama_cpp/ggml/src/ggml-metal/libggml-metal.a" "libggml-metal.a"
copy_archive "${BUILD_DIR}/llama_cpp/ggml/src/libggml.a" "libggml.a"
copy_archive "${BUILD_DIR}/llama_cpp/src/libllama.a" "libllama.a"

cp "${ROOT_DIR}/native/tq_ffi/include/tq_ffi.h" "${DEST_HEADER}"

echo "Synced ${TARGET_NAME} artifacts into packages/turboquant_flutter/ios"
