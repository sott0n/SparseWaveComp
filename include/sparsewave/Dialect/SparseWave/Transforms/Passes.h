#ifndef SPARSEWAVE_DIALECT_SPARSEWAVE_TRANSFORMS_PASSES_H
#define SPARSEWAVE_DIALECT_SPARSEWAVE_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir::sparsewave {

#define GEN_PASS_DECL
#include "sparsewave/Dialect/SparseWave/Transforms/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "sparsewave/Dialect/SparseWave/Transforms/Passes.h.inc"

} // namespace mlir::sparsewave

#endif // SPARSEWAVE_DIALECT_SPARSEWAVE_TRANSFORMS_PASSES_H
