#ifndef SPARSEWAVE_LIB_DIALECT_SPARSEWAVE_TRANSFORMS_SPARSELOWERINGUTILS_H
#define SPARSEWAVE_LIB_DIALECT_SPARSEWAVE_TRANSFORMS_SPARSELOWERINGUTILS_H

#include "mlir/IR/Builders.h"

namespace mlir::sparsewave {

Value castToIndex(OpBuilder &builder, Location loc, Value value);

} // namespace mlir::sparsewave

#endif // SPARSEWAVE_LIB_DIALECT_SPARSEWAVE_TRANSFORMS_SPARSELOWERINGUTILS_H
