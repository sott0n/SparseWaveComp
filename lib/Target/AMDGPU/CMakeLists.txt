add_mlir_library(SparseWaveAMDGPUPipelines
  Pipelines.cpp

  ADDITIONAL_HEADER_DIRS
  ${SPARSEWAVE_SOURCE_DIR}/include/sparsewave/Target/AMDGPU

  LINK_LIBS PUBLIC
  MLIRAMDGPUToROCDL
  MLIRAffineToStandard
  MLIRBufferizationTransforms
  MLIRFuncDialect
  MLIRGPUDialect
  MLIRGPUToGPURuntimeTransforms
  MLIRGPUToROCDLTransforms
  MLIRGPUTransforms
  MLIRLinalgTransforms
  MLIRMemRefTransforms
  MLIRPass
  MLIRReconcileUnrealizedCasts
  MLIRROCDLTarget
  MLIRSCFToControlFlow
  MLIRSparseTensorTransforms
  MLIRTransforms
  MLIRVectorToSCF
  SparseWaveTransforms
)
