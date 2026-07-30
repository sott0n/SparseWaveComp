#include "SparseGPUUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

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

CSRRowBounds buildCSRRowBounds(OpBuilder &builder, Location loc,
                               Value rowOffsets, Value row, Value oneIndex) {
  Value nextRow = arith::AddIOp::create(builder, loc, row, oneIndex);
  Value rowStartValue = memref::LoadOp::create(builder, loc, rowOffsets, row);
  Value rowEndValue = memref::LoadOp::create(builder, loc, rowOffsets, nextRow);
  return {castToIndex(builder, loc, rowStartValue),
          castToIndex(builder, loc, rowEndValue)};
}

StridedPositionRange buildStridedPositionRange(OpBuilder &builder, Location loc,
                                               CSRRowBounds rowBounds,
                                               Value participantOffset,
                                               Value stride) {
  Value first =
      arith::AddIOp::create(builder, loc, rowBounds.start, participantOffset);
  return {first, rowBounds.end, stride};
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

SmallVector<Value> buildWaveReductions(OpBuilder &builder, Location loc,
                                       ValueRange values, int64_t waveSize) {
  SmallVector<Value> reducedValues;
  reducedValues.reserve(values.size());
  for (Value value : values)
    reducedValues.push_back(buildWaveReduction(builder, loc, value, waveSize));
  return reducedValues;
}

} // namespace mlir::sparsewave
