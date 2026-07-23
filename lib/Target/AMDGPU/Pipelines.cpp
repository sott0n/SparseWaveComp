#include "sparsewave/Target/AMDGPU/Pipelines.h"

#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"

void mlir::sparsewave::buildAMDGPUBackendPipeline(
    OpPassManager &pm, const AMDGPUPipelineOptions &options) {
  (void)pm;
  (void)options;
}

void mlir::sparsewave::registerAMDGPUBackendPipeline() {
  PassPipelineRegistration<AMDGPUPipelineOptions>(
      "sparsewave-amdgpu-pipeline",
      "Lower SparseWave programs for an AMD GPU target.",
      buildAMDGPUBackendPipeline);
}
