#ifndef SPARSEWAVE_DIALECT_SPARSEWAVE_IR_SPARSEWAVEOPS_H
#define SPARSEWAVE_DIALECT_SPARSEWAVE_IR_SPARSEWAVEOPS_H

#include "sparsewave/Dialect/SparseWave/IR/SparseWaveDialect.h"

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir::sparsewave {

/// Hardware-independent worker kind used to partition a sparse position
/// space. Target lowering decides how each kind maps to concrete GPU IDs.
enum class PositionMapping {
  Thread,
  Wave,
  Block,
};

FailureOr<PositionMapping> symbolizePositionMapping(llvm::StringRef value);
llvm::StringRef stringifyPositionMapping(PositionMapping mapping);

} // namespace mlir::sparsewave

#define GET_OP_CLASSES
#include "sparsewave/Dialect/SparseWave/IR/SparseWaveOps.h.inc"

#endif // SPARSEWAVE_DIALECT_SPARSEWAVE_IR_SPARSEWAVEOPS_H
