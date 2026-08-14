#ifndef SPARSEWAVE_LIB_DIALECT_SPARSEWAVE_TRANSFORMS_SPARSEGPUUTILS_H
#define SPARSEWAVE_LIB_DIALECT_SPARSEWAVE_TRANSFORMS_SPARSEGPUUTILS_H

#include "SparseLoweringUtils.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"

#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::sparsewave {

struct LinearThreadWorkDistribution {
  gpu::LaunchOp launch;
  Value workUnit;
  Value workUnitIsActive;
};

struct WaveWorkDistribution {
  gpu::LaunchOp launch;
  Value waveInBlock;
  Value lane;
  Value workUnit;
  Value workUnitIsActive;
  Value positionStride;
};

struct CompressedRowBounds {
  Value start;
  Value end;
};

struct StridedPositionRange {
  Value first;
  Value end;
  Value stride;
};

struct CSRPosition {
  Value position;
  Value column;
  Value value;
};

struct WaveSegmentedReduction {
  Value inclusiveValue;
  Value segmentEnd;
};

enum class CSRCoiterationKind {
  Union,
  Intersection,
};

struct CSRCoiterationEntry {
  Value column;
  Value lhsPosition;
  Value rhsPosition;
  Value lhsPresent;
  Value rhsPresent;
};

using CSRPositionBodyBuilder = llvm::function_ref<SmallVector<Value>(
    OpBuilder &, Location, CSRPosition, ValueRange)>;

using CSRCoiterationBodyBuilder = llvm::function_ref<SmallVector<Value>(
    OpBuilder &, Location, CSRCoiterationEntry, ValueRange)>;

LinearThreadWorkDistribution
buildLinearThreadWorkDistribution(PatternRewriter &rewriter, Location loc,
                                  Value workUnitCount, Value oneIndex,
                                  Value blockSize);

WaveWorkDistribution
buildWaveWorkDistribution(PatternRewriter &rewriter, Location loc,
                          Value workUnitCount, Value oneIndex, Value blockSize,
                          Value waveSize, Value wavesPerBlock);

CompressedRowBounds buildCompressedRowBounds(OpBuilder &builder, Location loc,
                                             Value rowOffsets, Value row,
                                             Value oneIndex);

StridedPositionRange buildStridedPositionRange(OpBuilder &builder, Location loc,
                                               CompressedRowBounds rowBounds,
                                               Value participantOffset,
                                               Value stride);

SmallVector<Value> buildCSRPositionTraversal(OpBuilder &builder, Location loc,
                                             Value columnIndices, Value values,
                                             StridedPositionRange positions,
                                             ValueRange initialValues,
                                             CSRPositionBodyBuilder buildBody);

SmallVector<Value>
buildCSRCoiteration(OpBuilder &builder, Location loc, Value lhsColumnIndices,
                    CompressedRowBounds lhsBounds, Value rhsColumnIndices,
                    CompressedRowBounds rhsBounds, CSRCoiterationKind kind,
                    Value oneIndex, ValueRange initialValues,
                    CSRCoiterationBodyBuilder buildBody);

Value buildWaveReduction(OpBuilder &builder, Location loc, Value value,
                         int64_t waveSize);

/// Builds an inclusive segmented reduction for a prefix of active wave lanes.
/// Active lanes must be contiguous and start at lane zero.
WaveSegmentedReduction
buildWavePrefixSegmentedReduction(OpBuilder &builder, Location loc, Value key,
                                  Value value, Value active, int64_t waveSize);

SmallVector<Value> buildWaveReductions(OpBuilder &builder, Location loc,
                                       ValueRange values, int64_t waveSize);

} // namespace mlir::sparsewave

#endif // SPARSEWAVE_LIB_DIALECT_SPARSEWAVE_TRANSFORMS_SPARSEGPUUTILS_H
