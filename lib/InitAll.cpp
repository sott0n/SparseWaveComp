#include "sparsewave/InitAll.h"

#include "mlir/IR/DialectRegistry.h"

void mlir::sparsewave::registerAllDialects(DialectRegistry &registry) {
  (void)registry;
}

void mlir::sparsewave::registerAllPasses() {}
