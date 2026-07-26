add_mlir_library(SparseWaveTransforms
  LinalgSpMVToSparseWave.cpp
  SparseWaveToGPU.cpp

  DEPENDS
  SparseWavePassesIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRBufferizationDialect
  MLIRGPUDialect
  MLIRIR
  MLIRLinalgDialect
  MLIRMemRefDialect
  MLIRPass
  MLIRSCFDialect
  MLIRSparseTensorDialect
  MLIRTransforms
  SparseWaveDialect
)
