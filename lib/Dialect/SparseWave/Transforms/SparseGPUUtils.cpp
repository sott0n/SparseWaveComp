#include "SparseGPUUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"

namespace mlir::sparsewave {

Value buildWaveReduction(OpBuilder &builder, Location loc, Value value,
                         int64_t waveSize) {
  for (int32_t offset = 1; offset < waveSize; offset <<= 1) {
    Value shuffled = gpu::ShuffleOp::create(builder, loc, value, offset,
                                            waveSize, gpu::ShuffleMode::XOR)
                         .getShuffleResult();
    value = arith::AddFOp::create(builder, loc, value, shuffled);
  }
  return value;
}

SmallVector<Value> buildWaveReductions(OpBuilder &builder, Location loc,
                                       ValueRange values, int64_t waveSize) {
  SmallVector<Value> reducedValues;
  reducedValues.reserve(values.size());
  for (Value value : values)
    reducedValues.push_back(buildWaveReduction(builder, loc, value, waveSize));
  return reducedValues;
}

} // namespace mlir::sparsewave
