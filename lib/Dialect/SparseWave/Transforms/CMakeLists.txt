add_mlir_library(SparseWaveTransforms
  SparseWaveToGPU.cpp

  DEPENDS
  SparseWavePassesIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRGPUDialect
  MLIRIR
  MLIRMemRefDialect
  MLIRPass
  MLIRSCFDialect
  MLIRTransforms
  SparseWaveDialect
)
