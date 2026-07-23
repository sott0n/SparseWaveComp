#include "sparsewave/InitAll.h"

#include "sparsewave/Dialect/SparseWave/IR/SparseWaveDialect.h"
#include "sparsewave/Target/AMDGPU/Pipelines.h"

#include "mlir/IR/DialectRegistry.h"

void mlir::sparsewave::registerAllDialects(DialectRegistry &registry) {
  registry.insert<SparseWaveDialect>();
}

void mlir::sparsewave::registerAllPasses() { registerAMDGPUBackendPipeline(); }
