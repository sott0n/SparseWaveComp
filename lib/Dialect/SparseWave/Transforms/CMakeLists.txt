add_mlir_library(SparseWaveTransforms
  LinalgSparseToSparseWave.cpp
  PrepareGPUBarePtrABI.cpp
  PositionContractionDecomposition.cpp
  SparsePositionReordering.cpp
  SparseLoweringUtils.cpp
  SparsePositionToSCF.cpp
  SparsePositionScheduling.cpp
  SparseGPUUtils.cpp
  SparseWaveToGPU.cpp
  TorchToSparseWave.cpp

  DEPENDS
  SparseWavePassesIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRBufferizationDialect
  MLIRFuncDialect
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
