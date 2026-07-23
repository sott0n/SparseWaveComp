add_subdirectory(Target)

add_mlir_library(SparseWaveInit
  InitAll.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  SparseWaveAMDGPUPipelines
)
