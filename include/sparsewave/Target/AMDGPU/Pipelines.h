#ifndef SPARSEWAVE_TARGET_AMDGPU_PIPELINES_H
#define SPARSEWAVE_TARGET_AMDGPU_PIPELINES_H

#include "mlir/Pass/PassOptions.h"

#include <cstdint>
#include <string>

namespace mlir {
class OpPassManager;

namespace sparsewave {

enum class WavefrontSize {
  Wave32 = 32,
  Wave64 = 64,
};

enum class AMDGPUCompilationTarget {
  None,
  LLVM,
  ISA,
  Binary,
  Fatbin,
};

struct AMDGPUTargetOptions {
  explicit AMDGPUTargetOptions(detail::PassOptions &owner)
      : triple(owner, "triple", llvm::cl::desc("AMDGPU target triple."),
               llvm::cl::init("amdgcn-amd-amdhsa")),
        chip(owner, "chip",
             llvm::cl::desc(
                 "AMDGPU target chip (for example, gfx942 or gfx1100)."),
             llvm::cl::init("")),
        features(owner, "features", llvm::cl::desc("AMDGPU target features."),
                 llvm::cl::init("")),
        abiVersion(owner, "abi-version",
                   llvm::cl::desc("AMDHSA code object ABI version."),
                   llvm::cl::init("600")),
        optLevel(owner, "opt-level",
                 llvm::cl::desc("Optimization level for AMDGPU compilation."),
                 llvm::cl::init(2)),
        indexBitWidth(
            owner, "index-bitwidth",
            llvm::cl::desc("Bit width used when lowering the index type."),
            llvm::cl::init(64)),
        kernelUseBarePtrCallConv(
            owner, "kernel-bare-ptr-calling-convention",
            llvm::cl::desc(
                "Use the bare pointer calling convention for device kernels."),
            llvm::cl::init(false)),
        hostUseBarePtrCallConv(
            owner, "host-bare-ptr-calling-convention",
            llvm::cl::desc(
                "Use the bare pointer calling convention for host functions."),
            llvm::cl::init(false)),
        lowerHost(
            owner, "lower-host",
            llvm::cl::desc("Lower GPU launches to host runtime wrapper calls."),
            llvm::cl::init(true)),
        binaryFormat(
            owner, "binary-format",
            llvm::cl::desc("AMDGPU device compilation output."),
            llvm::cl::values(
                clEnumValN(AMDGPUCompilationTarget::None, "none",
                           "Keep the lowered GPU module"),
                clEnumValN(AMDGPUCompilationTarget::LLVM, "llvm",
                           "Embed LLVM bitcode"),
                clEnumValN(AMDGPUCompilationTarget::ISA, "isa",
                           "Embed AMDGPU assembly"),
                clEnumValN(AMDGPUCompilationTarget::Binary, "bin",
                           "Embed an HSA code object"),
                clEnumValN(AMDGPUCompilationTarget::Fatbin, "fatbin",
                           "Embed an HSA code object with kernel metadata")),
            llvm::cl::init(AMDGPUCompilationTarget::Binary)),
        rocmPath(
            owner, "rocm-path",
            llvm::cl::desc("Path to the ROCm toolkit used for binary linking."),
            llvm::cl::init("")),
        wavefrontSize(
            owner, "wavefront-size", llvm::cl::desc("AMDGPU wavefront size."),
            llvm::cl::values(
                clEnumValN(WavefrontSize::Wave32, "32", "Use Wave32 mode"),
                clEnumValN(WavefrontSize::Wave64, "64", "Use Wave64 mode")),
            llvm::cl::init(WavefrontSize::Wave64)) {}

  detail::PassOptions::Option<std::string> triple;
  detail::PassOptions::Option<std::string> chip;
  detail::PassOptions::Option<std::string> features;
  detail::PassOptions::Option<std::string> abiVersion;
  detail::PassOptions::Option<unsigned> optLevel;
  detail::PassOptions::Option<unsigned> indexBitWidth;
  detail::PassOptions::Option<bool> kernelUseBarePtrCallConv;
  detail::PassOptions::Option<bool> hostUseBarePtrCallConv;
  detail::PassOptions::Option<bool> lowerHost;
  detail::PassOptions::Option<AMDGPUCompilationTarget> binaryFormat;
  detail::PassOptions::Option<std::string> rocmPath;
  detail::PassOptions::Option<WavefrontSize> wavefrontSize;
};

struct AMDGPUPipelineOptions
    : public PassPipelineOptions<AMDGPUPipelineOptions>,
      public AMDGPUTargetOptions {
  AMDGPUPipelineOptions()
      : AMDGPUTargetOptions(static_cast<detail::PassOptions &>(*this)) {}
};

struct SparseWaveToAMDGPUPipelineOptions
    : public PassPipelineOptions<SparseWaveToAMDGPUPipelineOptions>,
      public AMDGPUTargetOptions {
  SparseWaveToAMDGPUPipelineOptions()
      : AMDGPUTargetOptions(static_cast<detail::PassOptions &>(*this)) {}

  PassOptions::Option<std::string> spmvMapping{
      *this, "spmv-mapping", llvm::cl::desc("SpMV work mapping strategy."),
      llvm::cl::init("thread-per-row")};

  PassOptions::Option<int64_t> spmvBlockSize{
      *this, "spmv-block-size",
      llvm::cl::desc("Number of GPU threads in each SpMV block."),
      llvm::cl::init(256)};

  PassOptions::Option<std::string> spmmMapping{
      *this, "spmm-mapping", llvm::cl::desc("SpMM work mapping strategy."),
      llvm::cl::init("thread-per-output")};

  PassOptions::Option<int64_t> spmmBlockSize{
      *this, "spmm-block-size",
      llvm::cl::desc("Number of GPU threads in each SpMM block."),
      llvm::cl::init(256)};

  PassOptions::Option<int64_t> spmmTileSize{
      *this, "spmm-tile-size",
      llvm::cl::desc("Number of output columns in each SpMM wave tile."),
      llvm::cl::init(4)};

  PassOptions::Option<int64_t> sddmmBlockSize{
      *this, "sddmm-block-size",
      llvm::cl::desc("Number of GPU threads in each SDDMM block."),
      llvm::cl::init(256)};

  PassOptions::Option<int64_t> rowReductionBlockSize{
      *this, "row-reduction-block-size",
      llvm::cl::desc("Number of GPU threads in each CSR row-reduction block."),
      llvm::cl::init(256)};

  PassOptions::Option<int64_t> rowwiseMapBlockSize{
      *this, "rowwise-map-block-size",
      llvm::cl::desc("Number of GPU threads in each CSR row-wise map block."),
      llvm::cl::init(256)};

  PassOptions::Option<int64_t> elementwiseBlockSize{
      *this, "elementwise-block-size",
      llvm::cl::desc("Number of GPU threads in each sparse elementwise block."),
      llvm::cl::init(256)};

  PassOptions::Option<bool> sinkLaunchIndexComputations{
      *this, "sink-launch-index-computations",
      llvm::cl::desc(
          "Sink constant index computations into outlined GPU kernels."),
      llvm::cl::init(false)};

  PassOptions::Option<bool> prepareGPUBarePtrABI{
      *this, "prepare-gpu-bare-ptr-abi",
      llvm::cl::desc(
          "Remove unused dynamic memref metadata from GPU kernel captures."),
      llvm::cl::init(false)};
};

void buildAMDGPUBackendPipeline(OpPassManager &pm,
                                const AMDGPUTargetOptions &options);
void buildSparseWaveToAMDGPUPipeline(
    OpPassManager &pm, const SparseWaveToAMDGPUPipelineOptions &options);
void registerAMDGPUBackendPipeline();
void registerSparseWaveToAMDGPUPipeline();

} // namespace sparsewave
} // namespace mlir

#endif // SPARSEWAVE_TARGET_AMDGPU_PIPELINES_H
