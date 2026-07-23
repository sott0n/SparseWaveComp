#ifndef SPARSEWAVE_TARGET_AMDGPU_PIPELINES_H
#define SPARSEWAVE_TARGET_AMDGPU_PIPELINES_H

#include "mlir/Pass/PassOptions.h"

#include <string>

namespace mlir {
class OpPassManager;

namespace sparsewave {

enum class WavefrontSize {
  Wave32 = 32,
  Wave64 = 64,
};

struct AMDGPUPipelineOptions
    : public PassPipelineOptions<AMDGPUPipelineOptions> {
  PassOptions::Option<std::string> triple{
      *this, "triple", llvm::cl::desc("AMDGPU target triple."),
      llvm::cl::init("amdgcn-amd-amdhsa")};

  PassOptions::Option<std::string> chip{
      *this, "chip",
      llvm::cl::desc("AMDGPU target chip (for example, gfx942 or gfx1100)."),
      llvm::cl::init("")};

  PassOptions::Option<std::string> features{
      *this, "features", llvm::cl::desc("AMDGPU target features."),
      llvm::cl::init("")};

  PassOptions::Option<std::string> abiVersion{
      *this, "abi-version", llvm::cl::desc("AMDHSA code object ABI version."),
      llvm::cl::init("600")};

  PassOptions::Option<unsigned> optLevel{
      *this, "opt-level",
      llvm::cl::desc("Optimization level for AMDGPU compilation."),
      llvm::cl::init(2)};

  PassOptions::Option<unsigned> indexBitWidth{
      *this, "index-bitwidth",
      llvm::cl::desc("Bit width used when lowering the index type."),
      llvm::cl::init(64)};

  PassOptions::Option<bool> kernelUseBarePtrCallConv{
      *this, "kernel-bare-ptr-calling-convention",
      llvm::cl::desc(
          "Use the bare pointer calling convention for device kernels."),
      llvm::cl::init(false)};

  PassOptions::Option<WavefrontSize> wavefrontSize{
      *this, "wavefront-size", llvm::cl::desc("AMDGPU wavefront size."),
      llvm::cl::values(
          clEnumValN(WavefrontSize::Wave32, "32", "Use Wave32 mode"),
          clEnumValN(WavefrontSize::Wave64, "64", "Use Wave64 mode")),
      llvm::cl::init(WavefrontSize::Wave64)};
};

void buildAMDGPUBackendPipeline(OpPassManager &pm,
                                const AMDGPUPipelineOptions &options);
void registerAMDGPUBackendPipeline();

} // namespace sparsewave
} // namespace mlir

#endif // SPARSEWAVE_TARGET_AMDGPU_PIPELINES_H
