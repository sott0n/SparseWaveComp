add_mlir_library(SparseWaveAMDGPUPipelines
  Pipelines.cpp

  ADDITIONAL_HEADER_DIRS
  ${SPARSEWAVE_SOURCE_DIR}/include/sparsewave/Target/AMDGPU

  LINK_LIBS PUBLIC
  MLIRAMDGPUToROCDL
  MLIRAffineToStandard
  MLIRGPUDialect
  MLIRGPUToGPURuntimeTransforms
  MLIRGPUToROCDLTransforms
  MLIRGPUTransforms
  MLIRMemRefTransforms
  MLIRPass
  MLIRReconcileUnrealizedCasts
  MLIRROCDLTarget
  MLIRSCFToControlFlow
  MLIRTransforms
  MLIRVectorToSCF
)
