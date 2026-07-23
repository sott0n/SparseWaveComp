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
  PassOptions::Option<std::string> chip{
      *this, "chip",
      llvm::cl::desc("AMDGPU target chip (for example, gfx942 or gfx1100)."),
      llvm::cl::init("")};

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
