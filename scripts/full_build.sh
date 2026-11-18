#!/usr/bin/env bash
# Build the v0.12.0 fused runtime and run the repository regressions.
set -euo pipefail

usage() {
  cat <<'HELP'
usage: bash scripts/full_build.sh [--legacy] [--no-tests] [--rebuild]

  --legacy    also build and test the deprecated LibTorch/TRT/AOTI profile
  --no-tests  build and run CTest only
  --rebuild   recreate the CMake build directory

Environment:
  GAUSSIFIER_SRC_DIR            training checkout (default ../gaussifier)
  GAUSSIFIER_BUILD_DIR          CMake build directory (default cpp_inference/build)
  CMAKE_CUDA_ARCHITECTURES      CUDA architectures (default 89)
  CMAKE_BUILD_PARALLEL_LEVEL    concurrent build jobs (default 4)
  GAUSSIFIER_MODEL_EXPORT_DIR   legacy TorchScript exports
  GAUSSIFIER_RUNTIME_DIR        legacy TensorRT/AOTI artifacts
HELP
}

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
GAUSSIFIER_SRC_DIR="$(realpath "${GAUSSIFIER_SRC_DIR:-$REPO/../gaussifier}")"
BUILD_DIR="$(realpath -m "${GAUSSIFIER_BUILD_DIR:-cpp_inference/build}")"
RUN_TESTS=1
REBUILD=0
LEGACY=OFF
for arg in "$@"; do
  case "$arg" in
    --legacy) LEGACY=ON ;;
    --no-tests) RUN_TESTS=0 ;;
    --rebuild) REBUILD=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown arg: $arg" >&2; usage >&2; exit 2 ;;
  esac
done
[ -d "$GAUSSIFIER_SRC_DIR/src/gaussifier" ] || {
  echo "Set GAUSSIFIER_SRC_DIR to the Gaussifier checkout root." >&2
  exit 2
}

cmake_args=()
if [ "$LEGACY" = ON ]; then
  uv sync --locked --group dev --extra tensorrt
  export TRT_SDK_DIR="${TRT_SDK_DIR:-$REPO/tensorrt_sdk}"
  bash cpp_inference/scripts/install_trt_sdk.sh --headers-only
  cmake_args+=(
    "-DTRT_INCLUDE_DIR=$TRT_SDK_DIR/include"
    "-DCMAKE_PREFIX_PATH=$(.venv/bin/python -c 'import torch; print(torch.utils.cmake_prefix_path)')"
  )
else
  uv sync --locked --group dev
fi
.venv/bin/python scripts/sync_native_voronoi.py --gaussifier-root "$GAUSSIFIER_SRC_DIR" --check

if [ "$REBUILD" = 1 ] && [ -d "$BUILD_DIR" ]; then
  if [ ! -f "$BUILD_DIR/CMakeCache.txt" ] ||
     ! grep -Fxq "CMAKE_HOME_DIRECTORY:INTERNAL=$REPO/cpp_inference" "$BUILD_DIR/CMakeCache.txt"; then
    echo "Refusing to remove $BUILD_DIR without this checkout's CMake cache." >&2
    exit 2
  fi
  rm -rf -- "$BUILD_DIR"
fi
cmake -S cpp_inference -B "$BUILD_DIR" \
  -DPython3_EXECUTABLE="$REPO/.venv/bin/python" \
  -DCMAKE_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES:-89}" \
  -DGAUSSIFIER_SRC_DIR="$GAUSSIFIER_SRC_DIR" \
  -DGAUSSIFIER_BUILD_FUSED=ON -DGAUSSIFIER_BUILD_LEGACY="$LEGACY" "${cmake_args[@]}"
cmake --build "$BUILD_DIR" --parallel "${CMAKE_BUILD_PARALLEL_LEVEL:-4}"
ctest --test-dir "$BUILD_DIR" --output-on-failure

if [ "$RUN_TESTS" = 1 ]; then
  export GAUSSIFIER_SRC_DIR
  export GFUSED_BIN="$BUILD_DIR/fused/gaussifier_fused_infer"
  export GAUSSIFIER_FUSED_LIBRARY="$BUILD_DIR/fused/libgaussifier_fused.so.1"
  export GAUSSIFIER_REQUIRE_FUSED=1
  export GAUSSIFIER_DENSITY_POINTS_BUILD_DIR="${GAUSSIFIER_DENSITY_POINTS_BUILD_DIR:-$BUILD_DIR/density_points}"
  export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$BUILD_DIR/torch_extensions}"
  if [ "$LEGACY" = ON ]; then
    unset GAUSSIFIER_SKIP_LEGACY
    export GAUSSIFIER_LEGACY_BIN="$BUILD_DIR/gaussifier_infer"
    export GAUSSIFIER_REQUIRE_RUNTIME_ASSETS=1
  else
    export GAUSSIFIER_SKIP_LEGACY=1
  fi
  .venv/bin/python -m pytest tests/ -v
fi
