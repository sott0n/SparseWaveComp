#include "SparseLoweringUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"

namespace mlir::sparsewave {

Value castToIndex(OpBuilder &builder, Location loc, Value value) {
  if (value.getType().isIndex())
    return value;
  if (cast<IntegerType>(value.getType()).isUnsigned())
    return arith::IndexCastUIOp::create(builder, loc, builder.getIndexType(),
                                        value);
  return arith::IndexCastOp::create(builder, loc, builder.getIndexType(),
                                    value);
}

Value castIndexToType(OpBuilder &builder, Location loc, Value value,
                      Type targetType) {
  if (targetType.isIndex())
    return value;
  if (cast<IntegerType>(targetType).isUnsigned())
    return arith::IndexCastUIOp::create(builder, loc, targetType, value);
  return arith::IndexCastOp::create(builder, loc, targetType, value);
}

} // namespace mlir::sparsewave
