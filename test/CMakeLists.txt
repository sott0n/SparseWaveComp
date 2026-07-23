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
  sparsewave-opt
)

add_lit_testsuite(check-sparsewave "Running the SparseWave regression tests"
  "${CMAKE_CURRENT_BINARY_DIR}"
  DEPENDS ${SPARSEWAVE_TEST_DEPENDS}
)
set_target_properties(check-sparsewave PROPERTIES FOLDER "Tests")

add_lit_testsuites(SPARSEWAVE
  "${CMAKE_CURRENT_SOURCE_DIR}"
  DEPENDS ${SPARSEWAVE_TEST_DEPENDS}
)
