#include "sparsewave/Target/AMDGPU/Pipelines.h"

#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"

namespace {

struct ValidateAMDTargetPass
    : public mlir::PassWrapper<ValidateAMDTargetPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ValidateAMDTargetPass)

  explicit ValidateAMDTargetPass(llvm::StringRef chip) : chip(chip) {}

  void runOnOperation() override {
    if (chip.empty()) {
      getOperation().emitError("AMDGPU target chip must be specified");
      signalPassFailure();
    }
  }

private:
  std::string chip;
};

} // namespace

void mlir::sparsewave::buildAMDGPUBackendPipeline(
    OpPassManager &pm, const AMDGPUPipelineOptions &options) {
  pm.addPass(std::make_unique<ValidateAMDTargetPass>(options.chip));

  GpuROCDLAttachTargetOptions targetOptions;
  targetOptions.chip = options.chip;
  targetOptions.wave64Flag = options.wavefrontSize == WavefrontSize::Wave64;
  pm.addPass(createGpuROCDLAttachTarget(targetOptions));
}

void mlir::sparsewave::registerAMDGPUBackendPipeline() {
  PassPipelineRegistration<AMDGPUPipelineOptions>(
      "sparsewave-amdgpu-pipeline",
      "Lower SparseWave programs for an AMD GPU target.",
      buildAMDGPUBackendPipeline);
}
