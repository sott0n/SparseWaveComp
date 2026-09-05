#include "SparseGPUUtils.h"

#include "sparsewave/Dialect/SparseWave/IR/SparseWaveOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

namespace mlir::sparsewave {

LinearThreadWorkDistribution
buildLinearThreadWorkDistribution(PatternRewriter &rewriter, Location loc,
                                  Value workUnitCount, Value oneIndex,
                                  Value blockSize) {
  Value requiredBlocks =
      arith::CeilDivUIOp::create(rewriter, loc, workUnitCount, blockSize);
  Value gridSize =
      arith::MaxUIOp::create(rewriter, loc, requiredBlocks, oneIndex);

  gpu::LaunchOp launch =
      gpu::LaunchOp::create(rewriter, loc, gridSize, oneIndex, oneIndex,
                            blockSize, oneIndex, oneIndex);
  rewriter.setInsertionPointToStart(&launch.getBody().front());

  Value workUnitBase = arith::MulIOp::create(
      rewriter, loc, launch.getBlockIds().x, launch.getBlockSize().x);
  Value workUnit = arith::AddIOp::create(rewriter, loc, workUnitBase,
                                         launch.getThreadIds().x);
  Value workUnitIsActive = arith::CmpIOp::create(
      rewriter, loc, arith::CmpIPredicate::ult, workUnit, workUnitCount);
  return {launch, workUnit, workUnitIsActive};
}

WaveWorkDistribution
buildWaveWorkDistribution(PatternRewriter &rewriter, Location loc,
                          Value workUnitCount, Value oneIndex, Value blockSize,
                          Value waveSize, Value wavesPerBlock) {
  Value requiredBlocks =
      arith::CeilDivUIOp::create(rewriter, loc, workUnitCount, wavesPerBlock);
  Value gridSize =
      arith::MaxUIOp::create(rewriter, loc, requiredBlocks, oneIndex);

  gpu::LaunchOp launch =
      gpu::LaunchOp::create(rewriter, loc, gridSize, oneIndex, oneIndex,
                            blockSize, oneIndex, oneIndex);
  rewriter.setInsertionPointToStart(&launch.getBody().front());

  Value waveInBlock =
      arith::DivUIOp::create(rewriter, loc, launch.getThreadIds().x, waveSize);
  Value lane =
      arith::RemUIOp::create(rewriter, loc, launch.getThreadIds().x, waveSize);
  Value workUnitBase = arith::MulIOp::create(
      rewriter, loc, launch.getBlockIds().x, wavesPerBlock);
  Value workUnit =
      arith::AddIOp::create(rewriter, loc, workUnitBase, waveInBlock);
  Value workUnitIsActive = arith::CmpIOp::create(
      rewriter, loc, arith::CmpIPredicate::ult, workUnit, workUnitCount);
  return {launch, waveInBlock, lane, workUnit, workUnitIsActive, waveSize};
}

CompressedSegmentBounds
buildCompressedSegmentBounds(OpBuilder &builder, Location loc, Value offsets,
                             Value segment, Value oneIndex) {
  Value nextSegment = arith::AddIOp::create(builder, loc, segment, oneIndex);
  Value startValue = memref::LoadOp::create(builder, loc, offsets, segment);
  Value endValue = memref::LoadOp::create(builder, loc, offsets, nextSegment);
  return {castToIndex(builder, loc, startValue),
          castToIndex(builder, loc, endValue)};
}

StridedPositionRange buildStridedPositionRange(OpBuilder &builder, Location loc,
                                               CompressedSegmentBounds bounds,
                                               Value participantOffset,
                                               Value stride) {
  Value first =
      arith::AddIOp::create(builder, loc, bounds.start, participantOffset);
  return {first, bounds.end, stride};
}

SmallVector<Value> buildCompressedPositionTraversal(
    OpBuilder &builder, Location loc, Value coordinates, Value values,
    StridedPositionRange positions, ValueRange initialValues,
    CompressedPositionBodyBuilder buildBody) {
  auto loop = scf::ForOp::create(
      builder, loc, positions.first, positions.end, positions.stride,
      initialValues,
      [&](OpBuilder &loopBuilder, Location loopLoc, Value position,
          ValueRange iterArgs) {
        Value coordinateValue =
            memref::LoadOp::create(loopBuilder, loopLoc, coordinates, position);
        Value coordinate = castToIndex(loopBuilder, loopLoc, coordinateValue);
        Value sparseValue =
            memref::LoadOp::create(loopBuilder, loopLoc, values, position);
        SmallVector<Value> nextValues = buildBody(
            loopBuilder, loopLoc,
            CompressedPosition{position, coordinate, sparseValue}, iterArgs);
        scf::YieldOp::create(loopBuilder, loopLoc, nextValues);
      });
  return SmallVector<Value>(loop.getResults());
}

SmallVector<Value> buildCompressedCoiteration(
    OpBuilder &builder, Location loc, Value lhsCoordinates,
    CompressedSegmentBounds lhsBounds, Value rhsCoordinates,
    CompressedSegmentBounds rhsBounds, CompressedCoiterationKind kind,
    Value oneIndex, ValueRange initialValues,
    CompressedCoiterationBodyBuilder buildBody) {
  SmallVector<Value> initialState{lhsBounds.start, rhsBounds.start};
  llvm::append_range(initialState, initialValues);
  TypeRange stateTypes = ValueRange(initialState).getTypes();
  auto loop = scf::WhileOp::create(builder, loc, stateTypes, initialState);

  OpBuilder::InsertionGuard guard(builder);
  SmallVector<Location> argumentLocations(stateTypes.size(), loc);
  Block *before =
      builder.createBlock(&loop.getBefore(), {}, stateTypes, argumentLocations);
  builder.setInsertionPointToStart(before);
  Value lhsPosition = before->getArgument(0);
  Value rhsPosition = before->getArgument(1);
  Value lhsActive = arith::CmpIOp::create(
      builder, loc, arith::CmpIPredicate::ult, lhsPosition, lhsBounds.end);
  Value rhsActive = arith::CmpIOp::create(
      builder, loc, arith::CmpIPredicate::ult, rhsPosition, rhsBounds.end);
  Value condition;
  if (kind == CompressedCoiterationKind::Union)
    condition = arith::OrIOp::create(builder, loc, lhsActive, rhsActive);
  else
    condition = arith::AndIOp::create(builder, loc, lhsActive, rhsActive);
  scf::ConditionOp::create(builder, loc, condition, before->getArguments());

  Block *after =
      builder.createBlock(&loop.getAfter(), {}, stateTypes, argumentLocations);
  builder.setInsertionPointToStart(after);
  lhsPosition = after->getArgument(0);
  rhsPosition = after->getArgument(1);
  ValueRange iterArgs = after->getArguments().drop_front(2);
  lhsActive = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::ult,
                                    lhsPosition, lhsBounds.end);
  rhsActive = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::ult,
                                    rhsPosition, rhsBounds.end);

  // A union iteration may have exhausted one input. Select the active input as
  // the fallback so neither branch performs an out-of-bounds coordinate load.
  auto lhsCoordinateSelection = scf::IfOp::create(
      builder, loc, lhsActive,
      [&](OpBuilder &thenBuilder, Location thenLoc) {
        Value coordinateValue = memref::LoadOp::create(
            thenBuilder, thenLoc, lhsCoordinates, lhsPosition);
        Value coordinate = castToIndex(thenBuilder, thenLoc, coordinateValue);
        scf::YieldOp::create(thenBuilder, thenLoc, coordinate);
      },
      [&](OpBuilder &elseBuilder, Location elseLoc) {
        Value coordinateValue = memref::LoadOp::create(
            elseBuilder, elseLoc, rhsCoordinates, rhsPosition);
        Value coordinate = castToIndex(elseBuilder, elseLoc, coordinateValue);
        scf::YieldOp::create(elseBuilder, elseLoc, coordinate);
      });
  auto rhsCoordinateSelection = scf::IfOp::create(
      builder, loc, rhsActive,
      [&](OpBuilder &thenBuilder, Location thenLoc) {
        Value coordinateValue = memref::LoadOp::create(
            thenBuilder, thenLoc, rhsCoordinates, rhsPosition);
        Value coordinate = castToIndex(thenBuilder, thenLoc, coordinateValue);
        scf::YieldOp::create(thenBuilder, thenLoc, coordinate);
      },
      [&](OpBuilder &elseBuilder, Location elseLoc) {
        Value coordinateValue = memref::LoadOp::create(
            elseBuilder, elseLoc, lhsCoordinates, lhsPosition);
        Value coordinate = castToIndex(elseBuilder, elseLoc, coordinateValue);
        scf::YieldOp::create(elseBuilder, elseLoc, coordinate);
      });
  Value lhsCoordinate = lhsCoordinateSelection.getResult(0);
  Value rhsCoordinate = rhsCoordinateSelection.getResult(0);

  Value trueValue = arith::ConstantIntOp::create(builder, loc, 1, 1);
  Value lhsOnly = arith::XOrIOp::create(builder, loc, rhsActive, trueValue);
  Value rhsOnly = arith::XOrIOp::create(builder, loc, lhsActive, trueValue);
  Value lhsPrecedes = arith::CmpIOp::create(
      builder, loc, arith::CmpIPredicate::ule, lhsCoordinate, rhsCoordinate);
  Value rhsPrecedes = arith::CmpIOp::create(
      builder, loc, arith::CmpIPredicate::ule, rhsCoordinate, lhsCoordinate);
  Value takeLhs = arith::AndIOp::create(
      builder, loc, lhsActive,
      arith::OrIOp::create(builder, loc, lhsOnly, lhsPrecedes));
  Value takeRhs = arith::AndIOp::create(
      builder, loc, rhsActive,
      arith::OrIOp::create(builder, loc, rhsOnly, rhsPrecedes));
  Value coordinate = arith::SelectOp::create(builder, loc, takeLhs,
                                             lhsCoordinate, rhsCoordinate);

  Value nextLhsPosition = arith::SelectOp::create(
      builder, loc, takeLhs,
      arith::AddIOp::create(builder, loc, lhsPosition, oneIndex), lhsPosition);
  Value nextRhsPosition = arith::SelectOp::create(
      builder, loc, takeRhs,
      arith::AddIOp::create(builder, loc, rhsPosition, oneIndex), rhsPosition);
  Value emit = kind == CompressedCoiterationKind::Union
                   ? trueValue
                   : arith::AndIOp::create(builder, loc, takeLhs, takeRhs);

  auto update = scf::IfOp::create(
      builder, loc, emit,
      [&](OpBuilder &thenBuilder, Location thenLoc) {
        SmallVector<Value> nextValues =
            buildBody(thenBuilder, thenLoc,
                      CompressedCoiterationEntry{coordinate, lhsPosition,
                                                 rhsPosition, takeLhs, takeRhs},
                      iterArgs);
        scf::YieldOp::create(thenBuilder, thenLoc, nextValues);
      },
      [&](OpBuilder &elseBuilder, Location elseLoc) {
        scf::YieldOp::create(elseBuilder, elseLoc, iterArgs);
      });
  SmallVector<Value> nextState{nextLhsPosition, nextRhsPosition};
  llvm::append_range(nextState, update.getResults());
  scf::YieldOp::create(builder, loc, nextState);

  return SmallVector<Value>(loop.getResults().drop_front(2));
}

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

WaveSegmentedReduction
buildWavePrefixSegmentedReduction(OpBuilder &builder, Location loc, Value key,
                                  Value value, Value active, int64_t waveSize) {
  Value activeI32 =
      arith::ExtUIOp::create(builder, loc, builder.getI32Type(), active);
  Value zeroI32 = arith::ConstantIntOp::create(builder, loc, 0, 32);
  Value trueValue = arith::ConstantIntOp::create(builder, loc, 1, 1);

  for (int32_t offset = 1; offset < waveSize; offset <<= 1) {
    auto shuffledKey = gpu::ShuffleOp::create(builder, loc, key, offset,
                                              waveSize, gpu::ShuffleMode::UP);
    auto shuffledValue = gpu::ShuffleOp::create(builder, loc, value, offset,
                                                waveSize, gpu::ShuffleMode::UP);
    Value sameSegment =
        arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::eq, key,
                              shuffledKey.getShuffleResult());
    // Valid source lanes below an active lane are active because the active
    // lanes form a prefix, so no active-state shuffle is required here.
    Value combine = arith::AndIOp::create(
        builder, loc, active,
        arith::AndIOp::create(builder, loc, shuffledKey.getValid(),
                              sameSegment));
    Value accumulated = arith::AddFOp::create(builder, loc, value,
                                              shuffledValue.getShuffleResult());
    value = arith::SelectOp::create(builder, loc, combine, accumulated, value);
  }

  auto nextKey = gpu::ShuffleOp::create(builder, loc, key, 1, waveSize,
                                        gpu::ShuffleMode::DOWN);
  auto nextActive = gpu::ShuffleOp::create(builder, loc, activeI32, 1, waveSize,
                                           gpu::ShuffleMode::DOWN);
  Value nextLaneIsActive =
      arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::ne,
                            nextActive.getShuffleResult(), zeroI32);
  Value nextIsSameSegment = arith::AndIOp::create(
      builder, loc, nextKey.getValid(),
      arith::AndIOp::create(builder, loc, nextLaneIsActive,
                            arith::CmpIOp::create(builder, loc,
                                                  arith::CmpIPredicate::eq, key,
                                                  nextKey.getShuffleResult())));
  Value nextIsDifferent =
      arith::XOrIOp::create(builder, loc, nextIsSameSegment, trueValue);
  Value segmentEnd =
      arith::AndIOp::create(builder, loc, active, nextIsDifferent);
  return {value, segmentEnd};
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
