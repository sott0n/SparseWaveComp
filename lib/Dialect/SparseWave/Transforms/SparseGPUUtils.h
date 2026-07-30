#ifndef SPARSEWAVE_LIB_DIALECT_SPARSEWAVE_TRANSFORMS_SPARSEGPUUTILS_H
#define SPARSEWAVE_LIB_DIALECT_SPARSEWAVE_TRANSFORMS_SPARSEGPUUTILS_H

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"

#include "llvm/ADT/SmallVector.h"

namespace mlir::sparsewave {

struct WaveWorkDistribution {
  gpu::LaunchOp launch;
  Value waveInBlock;
  Value lane;
  Value workUnit;
  Value workUnitIsActive;
  Value positionStride;
};

struct CSRRowBounds {
  Value start;
  Value end;
};

struct StridedPositionRange {
  Value first;
  Value end;
  Value stride;
};

Value castToIndex(OpBuilder &builder, Location loc, Value value);

WaveWorkDistribution
buildWaveWorkDistribution(PatternRewriter &rewriter, Location loc,
                          Value workUnitCount, Value oneIndex, Value blockSize,
                          Value waveSize, Value wavesPerBlock);

CSRRowBounds buildCSRRowBounds(OpBuilder &builder, Location loc,
                               Value rowOffsets, Value row, Value oneIndex);

StridedPositionRange buildStridedPositionRange(OpBuilder &builder, Location loc,
                                               CSRRowBounds rowBounds,
                                               Value participantOffset,
                                               Value stride);

Value buildWaveReduction(OpBuilder &builder, Location loc, Value value,
                         int64_t waveSize);

SmallVector<Value> buildWaveReductions(OpBuilder &builder, Location loc,
                                       ValueRange values, int64_t waveSize);

} // namespace mlir::sparsewave

#endif // SPARSEWAVE_LIB_DIALECT_SPARSEWAVE_TRANSFORMS_SPARSEGPUUTILS_H
