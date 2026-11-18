# Torch-free fused inference runner: gaussifier_fused_infer links only the CUDA runtime.
#
# Standalone (no libtorch / TensorRT needed):
#   cmake -S cpp_inference/fused -B build-fused -G Ninja \
#         -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES="89;120"
#   cmake --build build-fused --target gaussifier_fused_infer
# Also built by default from cpp_inference/ (see ../CMakeLists.txt).
#
# Options:
#   GAUSSIFIER_SRC_DIR         optional training checkout for the renderer helpers (default: third_party/raster)
#   GFUSED_BUILD_CHECKS=ON     also build sampler_assign_check (sampler exactness checker)
#   GFUSED_ACC16=ON            also build the FP16-accumulate conv variants (off by default)
cmake_minimum_required(VERSION 3.18)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 89)
  endif()
  project(gaussifier_fused LANGUAGES CXX CUDA)
  if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
  endif()
endif()
find_package(CUDAToolkit REQUIRED)

set(GFUSED_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
# Renderer device helpers (config.h, helpers.cuh, glm): the copy in third_party/raster, or the
# simple-sum renderer sources of a training checkout given with -DGAUSSIFIER_SRC_DIR.
set(GFUSED_RASTER_DIR "${GFUSED_DIR}/third_party/raster")
if(DEFINED GAUSSIFIER_SRC_DIR AND EXISTS
   "${GAUSSIFIER_SRC_DIR}/src/gaussifier/native_simple_sum_backend/cuda/csrc/helpers.cuh")
  set(GFUSED_RASTER_DIR
      "${GAUSSIFIER_SRC_DIR}/src/gaussifier/native_simple_sum_backend/cuda/csrc")
endif()

add_library(gaussifier_fused_pipeline STATIC
  "${GFUSED_DIR}/forward_map.cu"
  "${GFUSED_DIR}/correction_head.cu"
  "${GFUSED_DIR}/pipeline_sampler.cu"
  "${GFUSED_DIR}/kloop_render.cu"
  "${GFUSED_DIR}/pipeline.cu"
)
target_include_directories(gaussifier_fused_pipeline PUBLIC "${GFUSED_DIR}")
target_include_directories(gaussifier_fused_pipeline PRIVATE
  "${GFUSED_RASTER_DIR}" "${GFUSED_RASTER_DIR}/third_party/glm")
target_compile_features(gaussifier_fused_pipeline PUBLIC cxx_std_17 cuda_std_17)
target_compile_options(gaussifier_fused_pipeline PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math --expt-relaxed-constexpr -lineinfo>)
set_target_properties(gaussifier_fused_pipeline PROPERTIES
  POSITION_INDEPENDENT_CODE ON CUDA_RUNTIME_LIBRARY Static)
target_link_libraries(gaussifier_fused_pipeline PUBLIC CUDA::cudart_static)
# FP16-accumulate conv kernels (ConvParams::acc16, GFUSED_FM_ACC16 / GFUSED_HEAD_ACC16) for
# GPUs with a faster FP16-accumulate MMA (GeForce); doubles the conv kernels. On by default
# when the build targets sm_120, where the runtime then uses them by default (default_acc16
# in conv_fused.cuh); other devices keep FP32 accumulation either way.
set(GFUSED_ACC16_DEFAULT OFF)
if("${CMAKE_CUDA_ARCHITECTURES}" MATCHES "(^|;)120")
  set(GFUSED_ACC16_DEFAULT ON)
endif()
option(GFUSED_ACC16 "Build the FP16-accumulate conv kernel variants" ${GFUSED_ACC16_DEFAULT})
if(GFUSED_ACC16)
  target_compile_definitions(gaussifier_fused_pipeline PUBLIC GFUSED_ACC16=1)
endif()

# main.cpp includes the pipeline headers (half types, device declarations): compile as CUDA.
set_source_files_properties("${GFUSED_DIR}/main.cpp" PROPERTIES LANGUAGE CUDA)
add_executable(gaussifier_fused_infer "${GFUSED_DIR}/main.cpp")
target_link_libraries(gaussifier_fused_infer PRIVATE gaussifier_fused_pipeline)
target_compile_options(gaussifier_fused_infer PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
set_target_properties(gaussifier_fused_infer PROPERTIES CUDA_RUNTIME_LIBRARY Static)

add_library(gaussifier_fused SHARED "${GFUSED_DIR}/python_api.cu")
target_link_libraries(gaussifier_fused PRIVATE gaussifier_fused_pipeline)
target_compile_options(gaussifier_fused PRIVATE -Xcompiler=-fvisibility=hidden)
target_link_options(gaussifier_fused PRIVATE "LINKER:--exclude-libs,ALL")
set_target_properties(gaussifier_fused PROPERTIES CUDA_RUNTIME_LIBRARY Static SOVERSION 1)
install(TARGETS gaussifier_fused gaussifier_fused_infer LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin)
install(FILES "${GFUSED_DIR}/python_api.h" DESTINATION include/gaussifier_fused)

if(BUILD_TESTING)
  find_package(Python3 COMPONENTS Interpreter REQUIRED)
  foreach(target gaussifier_fused gaussifier_fused_infer)
    add_test(NAME ${target}_dependency_contract
      COMMAND ${Python3_EXECUTABLE} "${GFUSED_DIR}/../../scripts/audit_binary_dependencies.py"
        $<TARGET_FILE:${target}> --contract torch-free)
  endforeach()
endif()

option(GFUSED_BUILD_CHECKS "Build the fused runtime's standalone check tools" OFF)
if(GFUSED_BUILD_CHECKS)
  # Compiles pipeline_sampler.cu itself (with its check hooks); not linked to the pipeline.
  add_executable(sampler_assign_check "${GFUSED_DIR}/sampler_assign_check.cu")
  target_compile_features(sampler_assign_check PRIVATE cuda_std_17)
  target_compile_options(sampler_assign_check PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>)
  set_target_properties(sampler_assign_check PROPERTIES CUDA_RUNTIME_LIBRARY Static)
  # Per-kernel isolated-burst timing / TFLOPS / clock-normalised efficiency of the pipeline.
  add_executable(layer_bench "${GFUSED_DIR}/layer_bench.cu")
  target_link_libraries(layer_bench PRIVATE gaussifier_fused_pipeline)
  target_compile_options(layer_bench PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
  set_target_properties(layer_bench PROPERTIES CUDA_RUNTIME_LIBRARY Static)
endif()
