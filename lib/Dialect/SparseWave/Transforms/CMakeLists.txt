add_mlir_library(SparseWaveTransforms
  LinalgSparseToSparseWave.cpp
  PrepareGPUBarePtrABI.cpp
  SparseLoweringUtils.cpp
  SparsePositionToSCF.cpp
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
