include_guard(GLOBAL)

if(NOT DEFINED GAUSSIFIER_SAMPLER_ROOT)
  message(FATAL_ERROR "GAUSSIFIER_SAMPLER_ROOT must be set before NativeSources.cmake")
endif()

# Only the rasterizer comes from the training repository. Voronoi is compiled
# from the snapshot shipped in this package, so C++ and Python exercise the
# same sampler implementation.
if(NOT DEFINED GAUSSIFIER_SRC_DIR)
  set(GAUSSIFIER_SRC_DIR "${GAUSSIFIER_SAMPLER_ROOT}/../gaussifier")
endif()

set(
  NATIVE_RASTER_DIR
  "${GAUSSIFIER_SRC_DIR}/src/gaussifier/native_simple_sum_backend/cuda/csrc"
)
set(
  NATIVE_VORONOI_DIR
  "${GAUSSIFIER_SAMPLER_ROOT}/src/gaussifier_sampler/native_voronoi"
)

if(NOT EXISTS "${NATIVE_RASTER_DIR}")
  message(FATAL_ERROR
    "Cannot find Gaussifier rasterizer sources at: ${NATIVE_RASTER_DIR}\n"
    "Check out gaussifier as a sibling repository or pass\n"
    "  -DGAUSSIFIER_SRC_DIR=/path/to/gaussifier\n"
    "to CMake."
  )
endif()

foreach(filename error_diffusion_kernel.cu jfa_kernel.cu voronoi_c_api.h)
  if(NOT EXISTS "${NATIVE_VORONOI_DIR}/${filename}")
    message(FATAL_ERROR "Missing packaged native Voronoi source: ${filename}")
  endif()
endforeach()

file(GLOB RASTER_SOURCES CONFIGURE_DEPENDS
  "${NATIVE_RASTER_DIR}/*.cu"
  "${NATIVE_RASTER_DIR}/*.cpp"
)
list(REMOVE_ITEM RASTER_SOURCES "${NATIVE_RASTER_DIR}/ext.cpp")

set(VORONOI_SOURCES
  "${NATIVE_VORONOI_DIR}/error_diffusion_kernel.cu"
  "${NATIVE_VORONOI_DIR}/jfa_kernel.cu"
)

message(STATUS "Using Gaussifier rasterizer from: ${NATIVE_RASTER_DIR}")
message(STATUS "Using packaged Voronoi sampler from: ${NATIVE_VORONOI_DIR}")
