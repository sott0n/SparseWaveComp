set(SPARSEWAVE_MLIR_RUNTIME_LIBRARY_DIR "${LLVM_LIBRARY_DIR}")
set(SPARSEWAVE_IN_TREE_MLIR_RUNTIME_LIBRARY_DIR
    "${SPARSEWAVE_BINARY_DIR}/llvm/lib")
if(NOT EXISTS
       "${SPARSEWAVE_MLIR_RUNTIME_LIBRARY_DIR}/libmlir_rocm_runtime.so"
   AND EXISTS
       "${SPARSEWAVE_IN_TREE_MLIR_RUNTIME_LIBRARY_DIR}/libmlir_rocm_runtime.so")
  set(SPARSEWAVE_MLIR_RUNTIME_LIBRARY_DIR
      "${SPARSEWAVE_IN_TREE_MLIR_RUNTIME_LIBRARY_DIR}")
endif()

configure_lit_site_cfg(
  "${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in"
  "${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py"
  MAIN_CONFIG
  "${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py"
)

set(SPARSEWAVE_TEST_DEPENDS
  FileCheck
  count
  mlir-translate
  not
  sparsewave_benchmark_utils
  sparsewave-opt
  sparsewave-pytorch-opt
)
if(TARGET sparsewave-rocsparse-benchmark)
  list(APPEND SPARSEWAVE_TEST_DEPENDS sparsewave-rocsparse-benchmark)
endif()

add_lit_testsuite(check-sparsewave "Running the SparseWave regression tests"
  "${CMAKE_CURRENT_BINARY_DIR}"
  DEPENDS ${SPARSEWAVE_TEST_DEPENDS}
)
set_target_properties(check-sparsewave PROPERTIES FOLDER "Tests")

add_lit_testsuites(SPARSEWAVE
  "${CMAKE_CURRENT_SOURCE_DIR}"
  DEPENDS ${SPARSEWAVE_TEST_DEPENDS}
)
