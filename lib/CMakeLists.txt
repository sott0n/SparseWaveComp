add_subdirectory(Dialect)
add_subdirectory(Target)

add_mlir_library(SparseWaveInit
  InitAll.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  SparseWaveDialect
  SparseWaveAMDGPUPipelines
)
