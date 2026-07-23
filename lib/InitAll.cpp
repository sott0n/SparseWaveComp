#include "sparsewave/InitAll.h"

#include "sparsewave/Target/AMDGPU/Pipelines.h"

#include "mlir/IR/DialectRegistry.h"

void mlir::sparsewave::registerAllDialects(DialectRegistry &registry) {
  (void)registry;
}

void mlir::sparsewave::registerAllPasses() { registerAMDGPUBackendPipeline(); }
