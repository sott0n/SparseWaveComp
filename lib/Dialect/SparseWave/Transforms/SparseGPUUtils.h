#ifndef SPARSEWAVE_LIB_DIALECT_SPARSEWAVE_TRANSFORMS_SPARSEGPUUTILS_H
#define SPARSEWAVE_LIB_DIALECT_SPARSEWAVE_TRANSFORMS_SPARSEGPUUTILS_H

#include "mlir/IR/Builders.h"

#include "llvm/ADT/SmallVector.h"

namespace mlir::sparsewave {

Value buildWaveReduction(OpBuilder &builder, Location loc, Value value,
                         int64_t waveSize);

SmallVector<Value> buildWaveReductions(OpBuilder &builder, Location loc,
                                       ValueRange values, int64_t waveSize);

} // namespace mlir::sparsewave

#endif // SPARSEWAVE_LIB_DIALECT_SPARSEWAVE_TRANSFORMS_SPARSEGPUUTILS_H
