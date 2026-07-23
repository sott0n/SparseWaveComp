#include "sparsewave/Target/AMDGPU/Pipelines.h"

#include "mlir/Conversion/AMDGPUToROCDL/AMDGPUToROCDL.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/GPUToROCDL/GPUToROCDLPass.h"
#include "mlir/Conversion/GPUToROCDL/Runtimes.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

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

  OpPassManager &gpuModulePM = pm.nest<gpu::GPUModuleOp>();
  ConvertAMDGPUToROCDLPassOptions amdgpuToROCDLOptions;
  amdgpuToROCDLOptions.chipset = options.chip;
  gpuModulePM.addPass(createConvertAMDGPUToROCDLPass(amdgpuToROCDLOptions));
  gpuModulePM.addPass(createLowerAffinePass());
  gpuModulePM.addPass(createConvertVectorToSCFPass());
  gpuModulePM.addPass(createSCFToControlFlowPass());
  gpuModulePM.addPass(memref::createExpandStridedMetadataPass());

  ConvertGpuOpsToROCDLOpsOptions rocdlOptions;
  rocdlOptions.chipset = options.chip;
  rocdlOptions.runtime = gpu::amd::Runtime::HIP;
  gpuModulePM.addPass(createConvertGpuOpsToROCDLOps(rocdlOptions));
  gpuModulePM.addPass(createCanonicalizerPass());
  gpuModulePM.addPass(createCSEPass());
  gpuModulePM.addPass(createReconcileUnrealizedCastsPass());
}

void mlir::sparsewave::registerAMDGPUBackendPipeline() {
  PassPipelineRegistration<AMDGPUPipelineOptions>(
      "sparsewave-amdgpu-pipeline",
      "Lower SparseWave programs for an AMD GPU target.",
      buildAMDGPUBackendPipeline);
}
