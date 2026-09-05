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

struct CompressedSegmentBounds {
  Value start;
  Value end;
};

struct StridedPositionRange {
  Value first;
  Value end;
  Value stride;
};

struct CompressedPosition {
  Value position;
  Value coordinate;
  Value value;
};

struct WaveSegmentedReduction {
  Value inclusiveValue;
  Value segmentEnd;
};

enum class CompressedCoiterationKind {
  Union,
  Intersection,
};

struct CompressedCoiterationEntry {
  Value coordinate;
  Value lhsPosition;
  Value rhsPosition;
  Value lhsPresent;
  Value rhsPresent;
};

using CompressedPositionBodyBuilder = llvm::function_ref<SmallVector<Value>(
    OpBuilder &, Location, CompressedPosition, ValueRange)>;

using ThreadPerCompressedSegmentBodyBuilder = llvm::function_ref<void(
    OpBuilder &, Location, Value, CompressedSegmentBounds)>;

using CompressedCoiterationBodyBuilder = llvm::function_ref<SmallVector<Value>(
    OpBuilder &, Location, CompressedCoiterationEntry, ValueRange)>;

LinearThreadWorkDistribution
buildLinearThreadWorkDistribution(PatternRewriter &rewriter, Location loc,
                                  Value workUnitCount, Value oneIndex,
                                  Value blockSize);

/// Assigns one compressed segment to each GPU thread and builds the active
/// segment body with its position bounds.
gpu::LaunchOp buildThreadPerCompressedSegment(
    PatternRewriter &rewriter, Location loc, Value segmentCount, Value offsets,
    Value oneIndex, Value blockSize,
    ThreadPerCompressedSegmentBodyBuilder buildBody);

WaveWorkDistribution
buildWaveWorkDistribution(PatternRewriter &rewriter, Location loc,
                          Value workUnitCount, Value oneIndex, Value blockSize,
                          Value waveSize, Value wavesPerBlock);

CompressedSegmentBounds
buildCompressedSegmentBounds(OpBuilder &builder, Location loc, Value offsets,
                             Value segment, Value oneIndex);

StridedPositionRange buildStridedPositionRange(OpBuilder &builder, Location loc,
                                               CompressedSegmentBounds bounds,
                                               Value participantOffset,
                                               Value stride);

/// Traverses positions in a compressed segment and materializes the aligned
/// coordinate and value for each position.
SmallVector<Value> buildCompressedPositionTraversal(
    OpBuilder &builder, Location loc, Value coordinates, Value values,
    StridedPositionRange positions, ValueRange initialValues,
    CompressedPositionBodyBuilder buildBody);

/// Coiterates two sorted, unique coordinate sequences within compressed
/// segments. Union visits coordinates present in either input; intersection
/// visits only coordinates present in both inputs.
SmallVector<Value> buildCompressedCoiteration(
    OpBuilder &builder, Location loc, Value lhsCoordinates,
    CompressedSegmentBounds lhsBounds, Value rhsCoordinates,
    CompressedSegmentBounds rhsBounds, CompressedCoiterationKind kind,
    Value oneIndex, ValueRange initialValues,
    CompressedCoiterationBodyBuilder buildBody);

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
