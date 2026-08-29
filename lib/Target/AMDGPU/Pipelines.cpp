#include "sparsewave/Target/AMDGPU/Pipelines.h"

#include "sparsewave/Dialect/SparseWave/Transforms/Passes.h"

#include "mlir/Conversion/AMDGPUToROCDL/AMDGPUToROCDL.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/GPUCommon/GPUCommonPass.h"
#include "mlir/Conversion/GPUToROCDL/GPUToROCDLPass.h"
#include "mlir/Conversion/GPUToROCDL/Runtimes.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SparseTensor/Transforms/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Target/LLVM/ROCDL/Utils.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

namespace {

struct ValidateAMDTargetPass
    : public mlir::PassWrapper<ValidateAMDTargetPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ValidateAMDTargetPass)

  explicit ValidateAMDTargetPass(
      const mlir::sparsewave::AMDGPUTargetOptions &options)
      : triple(options.triple), chip(options.chip),
        abiVersion(options.abiVersion), optLevel(options.optLevel),
        indexBitWidth(options.indexBitWidth),
        binaryFormat(options.binaryFormat), rocmPath(options.rocmPath) {}

  void runOnOperation() override {
    if (triple.empty()) {
      getOperation().emitError("AMDGPU target triple must be specified");
      signalPassFailure();
      return;
    }
    if (chip.empty()) {
      getOperation().emitError("AMDGPU target chip must be specified");
      signalPassFailure();
      return;
    }
    if (abiVersion != "400" && abiVersion != "500" && abiVersion != "600") {
      getOperation().emitError(
          "AMDHSA code object ABI version must be 400, 500, or 600");
      signalPassFailure();
      return;
    }
    if (optLevel > 3) {
      getOperation().emitError("AMDGPU optimization level must be between 0 "
                               "and 3");
      signalPassFailure();
      return;
    }
    if (indexBitWidth != 32 && indexBitWidth != 64) {
      getOperation().emitError("AMDGPU index bit width must be 32 or 64");
      signalPassFailure();
      return;
    }

    using mlir::sparsewave::AMDGPUCompilationTarget;
    if (binaryFormat != AMDGPUCompilationTarget::Binary &&
        binaryFormat != AMDGPUCompilationTarget::Fatbin)
      return;

    llvm::StringRef toolkitPath =
        rocmPath.empty() ? mlir::ROCDL::getROCMPath() : rocmPath;
    if (!llvm::sys::fs::is_directory(toolkitPath)) {
      getOperation().emitError() << "ROCm toolkit path '" << toolkitPath
                                 << "' does not exist or is not a directory";
      signalPassFailure();
      return;
    }

    llvm::SmallString<128> linkerPath(toolkitPath);
    llvm::sys::path::append(linkerPath, "llvm", "bin", "ld.lld");
    if (!llvm::sys::fs::can_execute(linkerPath)) {
      getOperation().emitError()
          << "ROCm linker '" << linkerPath << "' does not exist or is not "
          << "executable";
      signalPassFailure();
    }
  }

private:
  std::string triple;
  std::string chip;
  std::string abiVersion;
  unsigned optLevel;
  unsigned indexBitWidth;
  mlir::sparsewave::AMDGPUCompilationTarget binaryFormat;
  std::string rocmPath;
};

struct VerifyAMDDeviceLoweringPass
    : public mlir::PassWrapper<VerifyAMDDeviceLoweringPass,
                               mlir::OperationPass<mlir::gpu::GPUModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VerifyAMDDeviceLoweringPass)

  void runOnOperation() override {
    mlir::gpu::GPUModuleOp module = getOperation();
    mlir::WalkResult result = module.walk([&](mlir::Operation *operation) {
      if (operation == module.getOperation())
        return mlir::WalkResult::advance();

      if (mlir::isa<mlir::UnrealizedConversionCastOp>(operation)) {
        operation->emitError(
            "device lowering left an unrealized conversion cast");
        return mlir::WalkResult::interrupt();
      }

      llvm::StringRef dialect = operation->getName().getDialectNamespace();
      if (dialect != "llvm" && dialect != "rocdl") {
        operation->emitError() << "device lowering left illegal operation '"
                               << operation->getName() << "'";
        return mlir::WalkResult::interrupt();
      }
      return mlir::WalkResult::advance();
    });
    if (result.wasInterrupted())
      signalPassFailure();
  }
};

} // namespace

void mlir::sparsewave::buildAMDGPUBackendPipeline(
    OpPassManager &pm, const AMDGPUTargetOptions &options) {
  pm.addPass(std::make_unique<ValidateAMDTargetPass>(options));

  GpuROCDLAttachTargetOptions targetOptions;
  targetOptions.triple = options.triple;
  targetOptions.chip = options.chip;
  targetOptions.features = options.features;
  targetOptions.abiVersion = options.abiVersion;
  targetOptions.optLevel = options.optLevel;
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
  rocdlOptions.indexBitwidth = options.indexBitWidth;
  rocdlOptions.useBarePtrCallConv = options.kernelUseBarePtrCallConv;
  rocdlOptions.runtime = gpu::amd::Runtime::HIP;
  gpuModulePM.addPass(createConvertGpuOpsToROCDLOps(rocdlOptions));
  gpuModulePM.addPass(createCanonicalizerPass());
  gpuModulePM.addPass(createCSEPass());
  gpuModulePM.addPass(createReconcileUnrealizedCastsPass());
  gpuModulePM.addPass(std::make_unique<VerifyAMDDeviceLoweringPass>());

  if (options.binaryFormat != AMDGPUCompilationTarget::None) {
    if (options.lowerHost) {
      GpuToLLVMConversionPassOptions hostOptions;
      hostOptions.hostBarePtrCallConv = options.hostUseBarePtrCallConv;
      hostOptions.kernelBarePtrCallConv = options.kernelUseBarePtrCallConv;
      pm.addPass(createGpuToLLVMConversionPass(hostOptions));
    }

    GpuModuleToBinaryPassOptions binaryOptions;
    switch (options.binaryFormat) {
    case AMDGPUCompilationTarget::None:
      llvm_unreachable("handled before constructing the binary pass");
    case AMDGPUCompilationTarget::LLVM:
      binaryOptions.compilationTarget = "llvm";
      break;
    case AMDGPUCompilationTarget::ISA:
      binaryOptions.compilationTarget = "isa";
      break;
    case AMDGPUCompilationTarget::Binary:
      binaryOptions.compilationTarget = "bin";
      break;
    case AMDGPUCompilationTarget::Fatbin:
      binaryOptions.compilationTarget = "fatbin";
      break;
    }
    binaryOptions.toolkitPath = options.rocmPath;
    pm.addPass(createGpuModuleToBinaryPass(binaryOptions));
  }
}

void mlir::sparsewave::buildSparseWaveToAMDGPUPipeline(
    OpPassManager &pm, const SparseWaveToAMDGPUPipelineOptions &options) {
  // Import supported Torch operators before running the SparseWave pipeline.
  // Modules without Torch operations pass through unchanged.
  pm.addPass(createConvertTorchToSparseWave());

  // Bridge canonical CSR Linalg SpMV/SpMM/SDDMM operations while the
  // SparseTensor encoding and operation-specific output form are still
  // visible.
  pm.addPass(createConvertLinalgSpMVToSparseWave());
  pm.addPass(createConvertLinalgSpMMToSparseWave());
  pm.addPass(createConvertLinalgSDDMMToSparseWave());

  // Generalize named Linalg operations that were not consumed by the bridge,
  // then use the upstream SparseTensor pass to materialize sparse storage and
  // bufferize the remaining tensor program.
  pm.addNestedPass<func::FuncOp>(createLinalgGeneralizeNamedOpsPass());
  pm.addPass(createSparsificationAndBufferizationPass());

  // Normalize bufferized function boundaries. Equivalent results are dropped;
  // other memref results become output parameters accepted by host lowering.
  bufferization::DropEquivalentBufferResultsPassOptions dropResultsOptions;
  dropResultsOptions.modifyPublicFunctions = true;
  pm.addPass(
      bufferization::createDropEquivalentBufferResultsPass(dropResultsOptions));
  bufferization::BufferResultsToOutParamsPassOptions outParamsOptions;
  outParamsOptions.hoistDynamicAllocs = true;
  outParamsOptions.modifyPublicFunctions = true;
  pm.addPass(
      bufferization::createBufferResultsToOutParamsPass(outParamsOptions));

  // Eliminate host-side SparseTensor, Linalg, SCF, and strided-memref metadata
  // left by upstream bufferization before converting GPU launches to LLVM.
  pm.addPass(createStorageSpecifierToLLVMPass());
  pm.addNestedPass<func::FuncOp>(createConvertLinalgToLoopsPass());
  pm.addNestedPass<func::FuncOp>(createSCFToControlFlowPass());
  pm.addPass(memref::createExpandStridedMetadataPass());

  // Decompose position-space SpMV into a generic keyed reduction, then apply
  // an operator-independent position schedule. Row-space mappings are handled
  // directly by the GPU pass.
  bool usesPositionSpMV = options.spmvMapping == "thread-per-position" ||
                          options.spmvMapping == "wave-per-position";
  if (usesPositionSpMV) {
    pm.addPass(createDecomposePositionSpMV());
    ScheduleSparseWavePositionOptions scheduleOptions;
    scheduleOptions.mapping =
        options.spmvMapping == "thread-per-position" ? "thread" : "wave";
    scheduleOptions.blockSize = options.spmvBlockSize;
    scheduleOptions.waveSize =
        static_cast<int64_t>(static_cast<WavefrontSize>(options.wavefrontSize));
    scheduleOptions.threadChunkSize = options.spmvPositionChunkSize;
    scheduleOptions.threadReduction = options.spmvPositionReduction;
    pm.addPass(createScheduleSparseWavePosition(scheduleOptions));
  }

  // Decompose SpMM semantics into a canonical keyed reduction. A separate,
  // operator-independent pass selects the logical axis order before applying
  // the same position scheduler used by SpMV. Position-major preserves the
  // original schedule; RHS-major exposes adjacent equal-key row segments.
  // Wave-per-position-tile assigns the RHS axis to cooperating lanes so that
  // CSR information shared by those columns is loaded once and broadcast.
  bool usesPositionSpMM = options.spmmMapping == "thread-per-position" ||
                          options.spmmMapping == "wave-per-position-tile";
  if (usesPositionSpMM) {
    pm.addPass(createDecomposePositionSpMM());
    ReorderSparseWavePositionOptions reorderOptions;
    if (options.spmmPositionOrder == "position-major")
      reorderOptions.order = {"position", "rhs"};
    else if (options.spmmPositionOrder == "rhs-major")
      reorderOptions.order = {"rhs", "position"};
    else
      reorderOptions.order = {options.spmmPositionOrder};
    pm.addPass(createReorderSparseWavePosition(reorderOptions));
    ScheduleSparseWavePositionOptions scheduleOptions;
    scheduleOptions.mapping =
        options.spmmMapping == "thread-per-position" ? "thread" : "wave";
    scheduleOptions.blockSize = options.spmmBlockSize;
    scheduleOptions.waveSize =
        static_cast<int64_t>(static_cast<WavefrontSize>(options.wavefrontSize));
    if (options.spmmMapping == "wave-per-position-tile")
      scheduleOptions.cooperativeAxis = "rhs";
    scheduleOptions.threadChunkSize = options.spmmPositionChunkSize;
    scheduleOptions.threadReduction = options.spmmPositionReduction;
    pm.addPass(createScheduleSparseWavePosition(scheduleOptions));
  }

  // Map SparseWave operations and explicit logical workers to GPU work. Each
  // operation has an independent block size because its GPU work unit differs.
  ConvertSparseWaveToGPUOptions sparseWaveOptions;
  sparseWaveOptions.mapping =
      usesPositionSpMV ? "thread-per-row" : options.spmvMapping.getValue();
  sparseWaveOptions.blockSize = options.spmvBlockSize;
  sparseWaveOptions.waveSize =
      static_cast<int64_t>(static_cast<WavefrontSize>(options.wavefrontSize));
  sparseWaveOptions.positionBlockSize = options.positionBlockSize;
  sparseWaveOptions.spmmMapping =
      usesPositionSpMM ? "thread-per-output" : options.spmmMapping.getValue();
  sparseWaveOptions.spmmBlockSize = options.spmmBlockSize;
  sparseWaveOptions.spmmTileSize = options.spmmTileSize;
  sparseWaveOptions.sddmmBlockSize = options.sddmmBlockSize;
  sparseWaveOptions.rowReductionBlockSize = options.rowReductionBlockSize;
  sparseWaveOptions.rowwiseMapBlockSize = options.rowwiseMapBlockSize;
  sparseWaveOptions.elementwiseBlockSize = options.elementwiseBlockSize;
  pm.addPass(createConvertSparseWaveToGPU(sparseWaveOptions));

  // Materialize target-independent sparse position partitions and coordinate
  // recovery after GPU worker identities have been assigned.
  pm.addPass(createLowerSparseWavePositionSpace());
  if (usesPositionSpMM) {
    // SpMM decomposition introduces a flattened output view after the earlier
    // host cleanup. Expand that view and its affine address arithmetic before
    // outlining and host LLVM lowering.
    pm.addPass(memref::createExpandStridedMetadataPass());
    pm.addPass(createLowerAffinePass());
  }
  if (options.prepareGPUBarePtrABI)
    pm.addPass(createPrepareGPUBarePtrABI());
  if (options.sinkLaunchIndexComputations)
    pm.addPass(createGpuLaunchSinkIndexComputationsPass());
  pm.addPass(createGpuKernelOutliningPass());
  buildAMDGPUBackendPipeline(pm, options);
}

void mlir::sparsewave::registerAMDGPUBackendPipeline() {
  PassPipelineRegistration<AMDGPUPipelineOptions>(
      "sparsewave-amdgpu-pipeline",
      "Lower outlined GPU kernels for an AMD GPU target.",
      buildAMDGPUBackendPipeline);
}

void mlir::sparsewave::registerSparseWaveToAMDGPUPipeline() {
  PassPipelineRegistration<SparseWaveToAMDGPUPipelineOptions>(
      "sparsewave-to-amdgpu-pipeline",
      "Compile SparseWave programs for an AMD GPU target.",
      buildSparseWaveToAMDGPUPipeline);
}
