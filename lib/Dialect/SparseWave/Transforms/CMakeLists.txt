add_mlir_library(SparseWaveTransforms
  LinalgSparseToSparseWave.cpp
  SparseLoweringUtils.cpp
  SparsePositionToSCF.cpp
  SparseGPUUtils.cpp
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
