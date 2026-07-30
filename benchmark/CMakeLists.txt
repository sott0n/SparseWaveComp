add_mlir_library(sparsewave_benchmark_utils
  SHARED
  SparseBenchmarkUtils.cpp

  EXCLUDE_FROM_LIBMLIR
  PARTIAL_SOURCES_INTENDED
  STANDALONE
)

set(SPARSEWAVE_ROCM_PATH "/opt/rocm" CACHE PATH
  "Path to the ROCm toolkit used by benchmark helpers")
find_package(rocsparse QUIET CONFIG PATHS "${SPARSEWAVE_ROCM_PATH}")
if(rocsparse_FOUND)
  add_executable(sparsewave-rocsparse-benchmark
    SparseRocSparseBenchmark.cpp
  )
  target_compile_definitions(sparsewave-rocsparse-benchmark PRIVATE
    __HIP_PLATFORM_AMD__
  )
  target_link_libraries(sparsewave-rocsparse-benchmark PRIVATE
    hip::host
    roc::rocsparse
  )
endif()
