#include "sparsewave/Dialect/SparseWave/Transforms/Passes.h"

#include "sparsewave/Dialect/SparseWave/IR/SparseWaveOps.h"

#include "SparseGPUUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/SmallVector.h"

#include <cassert>
#include <limits>

namespace mlir::sparsewave {
#define GEN_PASS_DEF_CONVERTSPARSEWAVETOGPU
#include "sparsewave/Dialect/SparseWave/Transforms/Passes.h.inc"

namespace {

void propagateKernelName(Operation *source, gpu::LaunchOp launch) {
  auto name = source->getAttrOfType<StringAttr>("sparsewave.kernel_name");
  if (!name)
    return;
  auto symbol = FlatSymbolRefAttr::get(source->getContext(), name.getValue());
  launch.setModuleAttr(symbol);
  launch.setFunctionAttr(symbol);
}

Value castIndexToType(OpBuilder &builder, Location loc, Value value,
                      Type targetType) {
  if (targetType.isIndex())
    return value;
  if (cast<IntegerType>(targetType).isUnsigned())
    return arith::IndexCastUIOp::create(builder, loc, targetType, value);
  return arith::IndexCastOp::create(builder, loc, targetType, value);
}

gpu::LaunchOp buildOutputInitialization(PatternRewriter &rewriter, Location loc,
                                        Value output, Value outputSize,
                                        Value zero, Value oneIndex,
                                        Value blockSize) {
  LinearThreadWorkDistribution initialization =
      buildLinearThreadWorkDistribution(rewriter, loc, outputSize, oneIndex,
                                        blockSize);
  scf::IfOp::create(rewriter, loc, initialization.workUnitIsActive,
                    [&](OpBuilder &builder, Location bodyLoc) {
                      memref::StoreOp::create(builder, bodyLoc, zero, output,
                                              initialization.workUnit);
                      scf::YieldOp::create(builder, bodyLoc);
                    },
                    {});
  rewriter.setInsertionPointToEnd(&initialization.launch.getBody().front());
  gpu::TerminatorOp::create(rewriter, loc);
  return initialization.launch;
}

class PositionParallelPattern : public OpRewritePattern<PositionParallelOp> {
public:
  PositionParallelPattern(MLIRContext *context, int64_t blockSize,
                          int64_t waveSize)
      : OpRewritePattern<PositionParallelOp>(context), blockSize(blockSize),
        waveSize(waveSize) {}

  LogicalResult matchAndRewrite(PositionParallelOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value one = arith::ConstantIndexOp::create(rewriter, loc, 1);
    Value blockSizeValue =
        arith::ConstantIndexOp::create(rewriter, loc, blockSize);
    Value workerId;
    Value participantId;
    Value participantCount;
    Value workerIsActive;
    gpu::LaunchOp launch;
    FailureOr<PositionMapping> mapping = op.getMappingKind();
    assert(succeeded(mapping) && "expected a verified position mapping");

    switch (*mapping) {
    case PositionMapping::Thread: {
      LinearThreadWorkDistribution distribution =
          buildLinearThreadWorkDistribution(rewriter, loc, op.getWorkerCount(),
                                            one, blockSizeValue);
      launch = distribution.launch;
      workerId = distribution.workUnit;
      participantId = zero;
      participantCount = one;
      workerIsActive = distribution.workUnitIsActive;
      break;
    }
    case PositionMapping::Wave: {
      Value waveSizeValue =
          arith::ConstantIndexOp::create(rewriter, loc, waveSize);
      Value wavesPerBlockValue =
          arith::ConstantIndexOp::create(rewriter, loc, blockSize / waveSize);
      WaveWorkDistribution distribution = buildWaveWorkDistribution(
          rewriter, loc, op.getWorkerCount(), one, blockSizeValue,
          waveSizeValue, wavesPerBlockValue);
      launch = distribution.launch;
      workerId = distribution.workUnit;
      participantId = distribution.lane;
      participantCount = waveSizeValue;
      workerIsActive = distribution.workUnitIsActive;
      break;
    }
    case PositionMapping::Block: {
      Value gridSize =
          arith::MaxUIOp::create(rewriter, loc, op.getWorkerCount(), one);
      launch = gpu::LaunchOp::create(rewriter, loc, gridSize, one, one,
                                     blockSizeValue, one, one);
      rewriter.setInsertionPointToStart(&launch.getBody().front());
      workerId = launch.getBlockIds().x;
      participantId = launch.getThreadIds().x;
      participantCount = launch.getBlockSize().x;
      workerIsActive =
          arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::ult,
                                workerId, op.getWorkerCount());
      break;
    }
    }

    auto guard = scf::IfOp::create(rewriter, loc, workerIsActive,
                                   /*withElseRegion=*/false);
    Block *thenBody = &guard.getThenRegion().front();

    auto sparseYield = cast<YieldOp>(op.getBody().front().getTerminator());
    rewriter.setInsertionPoint(sparseYield);
    scf::YieldOp::create(rewriter, sparseYield.getLoc());
    rewriter.eraseOp(sparseYield);
    rewriter.eraseOp(thenBody->getTerminator());
    rewriter.mergeBlocks(&op.getBody().front(), thenBody,
                         ValueRange{workerId, participantId, participantCount});

    rewriter.setInsertionPointToEnd(&launch.getBody().front());
    gpu::TerminatorOp::create(rewriter, loc);
    propagateKernelName(op, launch);
    rewriter.eraseOp(op);
    return success();
  }

private:
  int64_t blockSize;
  int64_t waveSize;
};

class ThreadPerNonzeroCOOSpMVPattern : public OpRewritePattern<COOSpMVOp> {
public:
  ThreadPerNonzeroCOOSpMVPattern(MLIRContext *context, int64_t blockSize)
      : OpRewritePattern<COOSpMVOp>(context), blockSize(blockSize) {}

  LogicalResult matchAndRewrite(COOSpMVOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value zeroIndex = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value oneIndex = arith::ConstantIndexOp::create(rewriter, loc, 1);
    Value blockSizeValue =
        arith::ConstantIndexOp::create(rewriter, loc, blockSize);
    Value outputSize =
        memref::DimOp::create(rewriter, loc, op.getOutput(), zeroIndex);
    Value nonzeroCount =
        memref::DimOp::create(rewriter, loc, op.getValues(), zeroIndex);
    auto valueType =
        cast<MemRefType>(op.getValues().getType()).getElementType();
    Value zero = arith::ConstantOp::create(rewriter, loc,
                                           rewriter.getZeroAttr(valueType));

    gpu::LaunchOp initialization =
        buildOutputInitialization(rewriter, loc, op.getOutput(), outputSize,
                                  zero, oneIndex, blockSizeValue);

    rewriter.setInsertionPointAfter(initialization);
    LinearThreadWorkDistribution distribution =
        buildLinearThreadWorkDistribution(rewriter, loc, nonzeroCount, oneIndex,
                                          blockSizeValue);
    Value position = distribution.workUnit;
    scf::IfOp::create(
        rewriter, loc, distribution.workUnitIsActive,
        [&](OpBuilder &builder, Location bodyLoc) {
          Value rowValue = memref::LoadOp::create(builder, bodyLoc,
                                                  op.getRowIndices(), position);
          Value columnValue = memref::LoadOp::create(
              builder, bodyLoc, op.getColumnIndices(), position);
          Value row = castToIndex(builder, bodyLoc, rowValue);
          Value column = castToIndex(builder, bodyLoc, columnValue);
          Value sparseValue = memref::LoadOp::create(builder, bodyLoc,
                                                     op.getValues(), position);
          Value vectorValue =
              memref::LoadOp::create(builder, bodyLoc, op.getVector(), column);
          Value product =
              arith::MulFOp::create(builder, bodyLoc, sparseValue, vectorValue);
          memref::AtomicRMWOp::create(builder, bodyLoc,
                                      arith::AtomicRMWKind::addf, product,
                                      op.getOutput(), ValueRange{row});
          scf::YieldOp::create(builder, bodyLoc);
        },
        {});

    rewriter.setInsertionPointToEnd(&distribution.launch.getBody().front());
    gpu::TerminatorOp::create(rewriter, loc);
    rewriter.eraseOp(op);
    return success();
  }

private:
  int64_t blockSize;
};

class ThreadPerPositionCSRSpMVPattern : public OpRewritePattern<SpMVOp> {
public:
  ThreadPerPositionCSRSpMVPattern(MLIRContext *context, int64_t blockSize)
      : OpRewritePattern<SpMVOp>(context), blockSize(blockSize) {}

  LogicalResult matchAndRewrite(SpMVOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value zeroIndex = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value oneIndex = arith::ConstantIndexOp::create(rewriter, loc, 1);
    Value blockSizeValue =
        arith::ConstantIndexOp::create(rewriter, loc, blockSize);
    Value outputSize =
        memref::DimOp::create(rewriter, loc, op.getOutput(), zeroIndex);
    Value nonzeroCount =
        memref::DimOp::create(rewriter, loc, op.getValues(), zeroIndex);
    auto valueType =
        cast<MemRefType>(op.getValues().getType()).getElementType();
    Value zero = arith::ConstantOp::create(rewriter, loc,
                                           rewriter.getZeroAttr(valueType));

    gpu::LaunchOp initialization =
        buildOutputInitialization(rewriter, loc, op.getOutput(), outputSize,
                                  zero, oneIndex, blockSizeValue);

    rewriter.setInsertionPointAfter(initialization);
    LinearThreadWorkDistribution distribution =
        buildLinearThreadWorkDistribution(rewriter, loc, nonzeroCount, oneIndex,
                                          blockSizeValue);
    Value workerCount = arith::MulIOp::create(
        rewriter, loc, distribution.launch.getGridSize().x,
        distribution.launch.getBlockSize().x);
    auto partition = PositionSpaceOp::create(
        rewriter, loc, rewriter.getIndexType(), rewriter.getIndexType(),
        zeroIndex, nonzeroCount, distribution.workUnit, workerCount, "thread");

    scf::ForOp::create(
        rewriter, loc, partition.getBegin(), partition.getEnd(), oneIndex,
        ValueRange{},
        [&](OpBuilder &builder, Location bodyLoc, Value position, ValueRange) {
          auto coordinates = CSRCoordinatesOp::create(
              builder, bodyLoc, builder.getIndexType(), builder.getIndexType(),
              op.getRowOffsets(), op.getColumnIndices(), position);
          Value sparseValue = memref::LoadOp::create(builder, bodyLoc,
                                                     op.getValues(), position);
          Value vectorValue = memref::LoadOp::create(
              builder, bodyLoc, op.getVector(), coordinates.getColumn());
          Value product =
              arith::MulFOp::create(builder, bodyLoc, sparseValue, vectorValue);
          memref::AtomicRMWOp::create(
              builder, bodyLoc, arith::AtomicRMWKind::addf, product,
              op.getOutput(), ValueRange{coordinates.getRow()});
          scf::YieldOp::create(builder, bodyLoc);
        });

    rewriter.setInsertionPointToEnd(&distribution.launch.getBody().front());
    gpu::TerminatorOp::create(rewriter, loc);
    rewriter.eraseOp(op);
    return success();
  }

private:
  int64_t blockSize;
};

class WavePerPositionCSRSpMVPattern : public OpRewritePattern<SpMVOp> {
public:
  WavePerPositionCSRSpMVPattern(MLIRContext *context, int64_t blockSize,
                                int64_t waveSize)
      : OpRewritePattern<SpMVOp>(context), blockSize(blockSize),
        waveSize(waveSize) {}

  LogicalResult matchAndRewrite(SpMVOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value zeroIndex = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value oneIndex = arith::ConstantIndexOp::create(rewriter, loc, 1);
    Value blockSizeValue =
        arith::ConstantIndexOp::create(rewriter, loc, blockSize);
    Value waveSizeValue =
        arith::ConstantIndexOp::create(rewriter, loc, waveSize);
    Value wavesPerBlockValue =
        arith::ConstantIndexOp::create(rewriter, loc, blockSize / waveSize);
    Value outputSize =
        memref::DimOp::create(rewriter, loc, op.getOutput(), zeroIndex);
    Value nonzeroCount =
        memref::DimOp::create(rewriter, loc, op.getValues(), zeroIndex);
    auto valueType =
        cast<MemRefType>(op.getValues().getType()).getElementType();
    Value zero = arith::ConstantOp::create(rewriter, loc,
                                           rewriter.getZeroAttr(valueType));

    gpu::LaunchOp initialization =
        buildOutputInitialization(rewriter, loc, op.getOutput(), outputSize,
                                  zero, oneIndex, blockSizeValue);

    rewriter.setInsertionPointAfter(initialization);
    LinearThreadWorkDistribution distribution =
        buildLinearThreadWorkDistribution(rewriter, loc, nonzeroCount, oneIndex,
                                          blockSizeValue);
    Value thread = distribution.launch.getThreadIds().x;
    Value waveInBlock =
        arith::DivUIOp::create(rewriter, loc, thread, waveSizeValue);
    Value lane = arith::RemUIOp::create(rewriter, loc, thread, waveSizeValue);
    Value waveBase = arith::MulIOp::create(
        rewriter, loc, distribution.launch.getBlockIds().x, wavesPerBlockValue);
    Value wave = arith::AddIOp::create(rewriter, loc, waveBase, waveInBlock);
    Value waveCount = arith::MulIOp::create(
        rewriter, loc, distribution.launch.getGridSize().x, wavesPerBlockValue);
    auto partition = PositionSpaceOp::create(
        rewriter, loc, rewriter.getIndexType(), rewriter.getIndexType(),
        zeroIndex, nonzeroCount, wave, waveCount, "wave");
    Value position =
        arith::AddIOp::create(rewriter, loc, partition.getBegin(), lane);
    Value active = arith::CmpIOp::create(
        rewriter, loc, arith::CmpIPredicate::ult, position, partition.getEnd());

    auto entry = scf::IfOp::create(
        rewriter, loc, active,
        [&](OpBuilder &builder, Location bodyLoc) {
          auto coordinates = CSRCoordinatesOp::create(
              builder, bodyLoc, builder.getIndexType(), builder.getIndexType(),
              op.getRowOffsets(), op.getColumnIndices(), position);
          Value sparseValue = memref::LoadOp::create(builder, bodyLoc,
                                                     op.getValues(), position);
          Value vectorValue = memref::LoadOp::create(
              builder, bodyLoc, op.getVector(), coordinates.getColumn());
          Value product =
              arith::MulFOp::create(builder, bodyLoc, sparseValue, vectorValue);
          scf::YieldOp::create(builder, bodyLoc,
                               ValueRange{coordinates.getRow(), product});
        },
        [&](OpBuilder &builder, Location bodyLoc) {
          scf::YieldOp::create(builder, bodyLoc, ValueRange{zeroIndex, zero});
        });
    Value row = entry.getResult(0);
    Value rowKey =
        arith::IndexCastOp::create(rewriter, loc, rewriter.getI64Type(), row);
    WaveSegmentedReduction reduction = buildWavePrefixSegmentedReduction(
        rewriter, loc, rowKey, entry.getResult(1), active, waveSize);
    scf::IfOp::create(rewriter, loc, reduction.segmentEnd,
                      [&](OpBuilder &builder, Location bodyLoc) {
                        memref::AtomicRMWOp::create(
                            builder, bodyLoc, arith::AtomicRMWKind::addf,
                            reduction.inclusiveValue, op.getOutput(),
                            ValueRange{row});
                        scf::YieldOp::create(builder, bodyLoc);
                      });

    rewriter.setInsertionPointToEnd(&distribution.launch.getBody().front());
    gpu::TerminatorOp::create(rewriter, loc);
    rewriter.eraseOp(op);
    return success();
  }

private:
  int64_t blockSize;
  int64_t waveSize;
};

class ThreadPerRowSpMVPattern : public OpRewritePattern<SpMVOp> {
public:
  ThreadPerRowSpMVPattern(MLIRContext *context, int64_t blockSize)
      : OpRewritePattern<SpMVOp>(context), blockSize(blockSize) {}

  LogicalResult matchAndRewrite(SpMVOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value zeroIndex = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value oneIndex = arith::ConstantIndexOp::create(rewriter, loc, 1);
    Value blockSizeValue =
        arith::ConstantIndexOp::create(rewriter, loc, blockSize);
    Value rowCount =
        memref::DimOp::create(rewriter, loc, op.getOutput(), zeroIndex);
    LinearThreadWorkDistribution distribution =
        buildLinearThreadWorkDistribution(rewriter, loc, rowCount, oneIndex,
                                          blockSizeValue);
    Value row = distribution.workUnit;

    scf::IfOp::create(
        rewriter, loc, distribution.workUnitIsActive,
        [&](OpBuilder &builder, Location bodyLoc) {
          CompressedRowBounds rowBounds = buildCompressedRowBounds(
              builder, bodyLoc, op.getRowOffsets(), row, oneIndex);

          auto valueType =
              cast<MemRefType>(op.getValues().getType()).getElementType();
          Value zero = arith::ConstantOp::create(
              builder, bodyLoc, builder.getZeroAttr(valueType));
          StridedPositionRange positions{rowBounds.start, rowBounds.end,
                                         oneIndex};
          SmallVector<Value> reduction = buildCSRPositionTraversal(
              builder, bodyLoc, op.getColumnIndices(), op.getValues(),
              positions, ValueRange{zero},
              [&](OpBuilder &loopBuilder, Location loopLoc,
                  CSRPosition position, ValueRange iterArgs) {
                Value vectorValue = memref::LoadOp::create(
                    loopBuilder, loopLoc, op.getVector(), position.column);
                Value product = arith::MulFOp::create(
                    loopBuilder, loopLoc, position.value, vectorValue);
                Value sum = arith::AddFOp::create(loopBuilder, loopLoc,
                                                  iterArgs.front(), product);
                return SmallVector<Value>{sum};
              });
          memref::StoreOp::create(builder, bodyLoc, reduction.front(),
                                  op.getOutput(), row);
          scf::YieldOp::create(builder, bodyLoc);
        },
        {});

    rewriter.setInsertionPointToEnd(&distribution.launch.getBody().front());
    gpu::TerminatorOp::create(rewriter, loc);
    rewriter.eraseOp(op);
    return success();
  }

private:
  int64_t blockSize;
};

class WavePerRowSpMVPattern : public OpRewritePattern<SpMVOp> {
public:
  WavePerRowSpMVPattern(MLIRContext *context, int64_t blockSize,
                        int64_t waveSize)
      : OpRewritePattern<SpMVOp>(context), blockSize(blockSize),
        waveSize(waveSize) {}

  LogicalResult matchAndRewrite(SpMVOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value zeroIndex = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value oneIndex = arith::ConstantIndexOp::create(rewriter, loc, 1);
    Value blockSizeValue =
        arith::ConstantIndexOp::create(rewriter, loc, blockSize);
    Value waveSizeValue =
        arith::ConstantIndexOp::create(rewriter, loc, waveSize);
    Value wavesPerBlockValue =
        arith::ConstantIndexOp::create(rewriter, loc, blockSize / waveSize);
    Value rowCount =
        memref::DimOp::create(rewriter, loc, op.getOutput(), zeroIndex);
    WaveWorkDistribution distribution = buildWaveWorkDistribution(
        rewriter, loc, rowCount, oneIndex, blockSizeValue, waveSizeValue,
        wavesPerBlockValue);
    Value row = distribution.workUnit;

    scf::IfOp::create(
        rewriter, loc, distribution.workUnitIsActive,
        [&](OpBuilder &builder, Location bodyLoc) {
          CompressedRowBounds rowBounds = buildCompressedRowBounds(
              builder, bodyLoc, op.getRowOffsets(), row, oneIndex);
          StridedPositionRange positions = buildStridedPositionRange(
              builder, bodyLoc, rowBounds, distribution.lane,
              distribution.positionStride);

          auto valueType =
              cast<MemRefType>(op.getValues().getType()).getElementType();
          Value zero = arith::ConstantOp::create(
              builder, bodyLoc, builder.getZeroAttr(valueType));
          SmallVector<Value> partialReduction = buildCSRPositionTraversal(
              builder, bodyLoc, op.getColumnIndices(), op.getValues(),
              positions, ValueRange{zero},
              [&](OpBuilder &loopBuilder, Location loopLoc,
                  CSRPosition position, ValueRange iterArgs) {
                Value vectorValue = memref::LoadOp::create(
                    loopBuilder, loopLoc, op.getVector(), position.column);
                Value product = arith::MulFOp::create(
                    loopBuilder, loopLoc, position.value, vectorValue);
                Value sum = arith::AddFOp::create(loopBuilder, loopLoc,
                                                  iterArgs.front(), product);
                return SmallVector<Value>{sum};
              });

          Value waveSum = buildWaveReduction(
              builder, bodyLoc, partialReduction.front(), waveSize);

          Value laneIsZero =
              arith::CmpIOp::create(builder, bodyLoc, arith::CmpIPredicate::eq,
                                    distribution.lane, zeroIndex);
          scf::IfOp::create(builder, bodyLoc, laneIsZero,
                            [&](OpBuilder &storeBuilder, Location storeLoc) {
                              memref::StoreOp::create(storeBuilder, storeLoc,
                                                      waveSum, op.getOutput(),
                                                      row);
                              scf::YieldOp::create(storeBuilder, storeLoc);
                            });
          scf::YieldOp::create(builder, bodyLoc);
        },
        {});

    rewriter.setInsertionPointToEnd(&distribution.launch.getBody().front());
    gpu::TerminatorOp::create(rewriter, loc);
    rewriter.eraseOp(op);
    return success();
  }

private:
  int64_t blockSize;
  int64_t waveSize;
};

class BlockPerRowSpMVPattern : public OpRewritePattern<SpMVOp> {
public:
  BlockPerRowSpMVPattern(MLIRContext *context, int64_t blockSize,
                         int64_t waveSize)
      : OpRewritePattern<SpMVOp>(context), blockSize(blockSize),
        waveSize(waveSize) {}

  LogicalResult matchAndRewrite(SpMVOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value zeroIndex = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value oneIndex = arith::ConstantIndexOp::create(rewriter, loc, 1);
    Value blockSizeValue =
        arith::ConstantIndexOp::create(rewriter, loc, blockSize);
    Value waveSizeValue =
        arith::ConstantIndexOp::create(rewriter, loc, waveSize);
    int64_t wavesPerBlock = blockSize / waveSize;
    Value wavesPerBlockValue =
        arith::ConstantIndexOp::create(rewriter, loc, wavesPerBlock);
    Value rowCount =
        memref::DimOp::create(rewriter, loc, op.getOutput(), zeroIndex);
    Value gridSize = arith::MaxUIOp::create(rewriter, loc, rowCount, oneIndex);

    gpu::LaunchOp launch =
        gpu::LaunchOp::create(rewriter, loc, gridSize, oneIndex, oneIndex,
                              blockSizeValue, oneIndex, oneIndex);
    auto valueType =
        cast<MemRefType>(op.getValues().getType()).getElementType();
    auto workgroupAddressSpace = gpu::AddressSpaceAttr::get(
        rewriter.getContext(), gpu::AddressSpace::Workgroup);
    auto waveSumsType =
        MemRefType::get({wavesPerBlock}, valueType, MemRefLayoutAttrInterface{},
                        Attribute(workgroupAddressSpace));
    Value waveSums = launch.addWorkgroupAttribution(waveSumsType, loc);

    rewriter.setInsertionPointToStart(&launch.getBody().front());
    Value thread = launch.getThreadIds().x;
    Value row = launch.getBlockIds().x;
    Value waveInBlock =
        arith::DivUIOp::create(rewriter, loc, thread, waveSizeValue);
    Value lane = arith::RemUIOp::create(rewriter, loc, thread, waveSizeValue);
    Value rowIsActive = arith::CmpIOp::create(
        rewriter, loc, arith::CmpIPredicate::ult, row, rowCount);

    scf::IfOp::create(
        rewriter, loc, rowIsActive,
        [&](OpBuilder &builder, Location bodyLoc) {
          CompressedRowBounds rowBounds = buildCompressedRowBounds(
              builder, bodyLoc, op.getRowOffsets(), row, oneIndex);
          StridedPositionRange positions = buildStridedPositionRange(
              builder, bodyLoc, rowBounds, thread, blockSizeValue);
          Value zero = arith::ConstantOp::create(
              builder, bodyLoc, builder.getZeroAttr(valueType));

          SmallVector<Value> partialReduction = buildCSRPositionTraversal(
              builder, bodyLoc, op.getColumnIndices(), op.getValues(),
              positions, ValueRange{zero},
              [&](OpBuilder &loopBuilder, Location loopLoc,
                  CSRPosition position, ValueRange iterArgs) {
                Value vectorValue = memref::LoadOp::create(
                    loopBuilder, loopLoc, op.getVector(), position.column);
                Value product = arith::MulFOp::create(
                    loopBuilder, loopLoc, position.value, vectorValue);
                Value sum = arith::AddFOp::create(loopBuilder, loopLoc,
                                                  iterArgs.front(), product);
                return SmallVector<Value>{sum};
              });

          Value waveSum = buildWaveReduction(
              builder, bodyLoc, partialReduction.front(), waveSize);
          Value laneIsZero = arith::CmpIOp::create(
              builder, bodyLoc, arith::CmpIPredicate::eq, lane, zeroIndex);
          scf::IfOp::create(builder, bodyLoc, laneIsZero,
                            [&](OpBuilder &storeBuilder, Location storeLoc) {
                              memref::StoreOp::create(storeBuilder, storeLoc,
                                                      waveSum, waveSums,
                                                      waveInBlock);
                              scf::YieldOp::create(storeBuilder, storeLoc);
                            });

          gpu::BarrierOp::create(builder, bodyLoc);

          Value waveIsZero =
              arith::CmpIOp::create(builder, bodyLoc, arith::CmpIPredicate::eq,
                                    waveInBlock, zeroIndex);
          scf::IfOp::create(
              builder, bodyLoc, waveIsZero,
              [&](OpBuilder &finalBuilder, Location finalLoc) {
                Value laneHasWave = arith::CmpIOp::create(
                    finalBuilder, finalLoc, arith::CmpIPredicate::ult, lane,
                    wavesPerBlockValue);
                auto initialWaveSum = scf::IfOp::create(
                    finalBuilder, finalLoc, TypeRange{valueType}, laneHasWave,
                    /*withElseRegion=*/true);
                finalBuilder.setInsertionPointToStart(
                    &initialWaveSum.getThenRegion().front());
                Value value = memref::LoadOp::create(finalBuilder, finalLoc,
                                                     waveSums, lane);
                scf::YieldOp::create(finalBuilder, finalLoc, value);
                finalBuilder.setInsertionPointToStart(
                    &initialWaveSum.getElseRegion().front());
                scf::YieldOp::create(finalBuilder, finalLoc, zero);
                finalBuilder.setInsertionPointAfter(initialWaveSum);
                Value blockSum =
                    buildWaveReduction(finalBuilder, finalLoc,
                                       initialWaveSum.getResult(0), waveSize);
                Value laneIsZero = arith::CmpIOp::create(
                    finalBuilder, finalLoc, arith::CmpIPredicate::eq, lane,
                    zeroIndex);
                scf::IfOp::create(
                    finalBuilder, finalLoc, laneIsZero,
                    [&](OpBuilder &storeBuilder, Location storeLoc) {
                      memref::StoreOp::create(storeBuilder, storeLoc, blockSum,
                                              op.getOutput(), row);
                      scf::YieldOp::create(storeBuilder, storeLoc);
                    });
                scf::YieldOp::create(finalBuilder, finalLoc);
              });
          scf::YieldOp::create(builder, bodyLoc);
        },
        {});

    rewriter.setInsertionPointToEnd(&launch.getBody().front());
    gpu::TerminatorOp::create(rewriter, loc);
    rewriter.eraseOp(op);
    return success();
  }

private:
  int64_t blockSize;
  int64_t waveSize;
};

class ThreadPerOutputSpMMPattern : public OpRewritePattern<SpMMOp> {
public:
  ThreadPerOutputSpMMPattern(MLIRContext *context, int64_t blockSize)
      : OpRewritePattern<SpMMOp>(context), blockSize(blockSize) {}

  LogicalResult matchAndRewrite(SpMMOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value zeroIndex = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value oneIndex = arith::ConstantIndexOp::create(rewriter, loc, 1);
    Value blockSizeValue =
        arith::ConstantIndexOp::create(rewriter, loc, blockSize);
    Value rowCount =
        memref::DimOp::create(rewriter, loc, op.getOutput(), zeroIndex);
    Value columnCount =
        memref::DimOp::create(rewriter, loc, op.getOutput(), oneIndex);
    Value outputElementCount =
        arith::MulIOp::create(rewriter, loc, rowCount, columnCount);
    LinearThreadWorkDistribution distribution =
        buildLinearThreadWorkDistribution(rewriter, loc, outputElementCount,
                                          oneIndex, blockSizeValue);
    Value element = distribution.workUnit;

    scf::IfOp::create(
        rewriter, loc, distribution.workUnitIsActive,
        [&](OpBuilder &builder, Location bodyLoc) {
          Value row =
              arith::DivUIOp::create(builder, bodyLoc, element, columnCount);
          Value outputColumn =
              arith::RemUIOp::create(builder, bodyLoc, element, columnCount);
          CompressedRowBounds rowBounds = buildCompressedRowBounds(
              builder, bodyLoc, op.getRowOffsets(), row, oneIndex);

          auto valueType =
              cast<MemRefType>(op.getValues().getType()).getElementType();
          Value zero = arith::ConstantOp::create(
              builder, bodyLoc, builder.getZeroAttr(valueType));
          StridedPositionRange positions{rowBounds.start, rowBounds.end,
                                         oneIndex};
          SmallVector<Value> reduction = buildCSRPositionTraversal(
              builder, bodyLoc, op.getColumnIndices(), op.getValues(),
              positions, ValueRange{zero},
              [&](OpBuilder &loopBuilder, Location loopLoc,
                  CSRPosition position, ValueRange iterArgs) {
                Value rhsValue = memref::LoadOp::create(
                    loopBuilder, loopLoc, op.getRhs(),
                    ValueRange{position.column, outputColumn});
                Value product = arith::MulFOp::create(loopBuilder, loopLoc,
                                                      position.value, rhsValue);
                Value sum = arith::AddFOp::create(loopBuilder, loopLoc,
                                                  iterArgs.front(), product);
                return SmallVector<Value>{sum};
              });
          memref::StoreOp::create(builder, bodyLoc, reduction.front(),
                                  op.getOutput(),
                                  ValueRange{row, outputColumn});
          scf::YieldOp::create(builder, bodyLoc);
        },
        {});

    rewriter.setInsertionPointToEnd(&distribution.launch.getBody().front());
    gpu::TerminatorOp::create(rewriter, loc);
    propagateKernelName(op, distribution.launch);
    rewriter.eraseOp(op);
    return success();
  }

private:
  int64_t blockSize;
};

class ThreadPerOutputBSRSpMMPattern : public OpRewritePattern<BSRSpMMOp> {
public:
  ThreadPerOutputBSRSpMMPattern(MLIRContext *context, int64_t gpuBlockSize)
      : OpRewritePattern<BSRSpMMOp>(context), gpuBlockSize(gpuBlockSize) {}

  LogicalResult matchAndRewrite(BSRSpMMOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value zeroIndex = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value oneIndex = arith::ConstantIndexOp::create(rewriter, loc, 1);
    Value gpuBlockSizeValue =
        arith::ConstantIndexOp::create(rewriter, loc, gpuBlockSize);
    Value bsrBlockSize = arith::ConstantIndexOp::create(
        rewriter, loc, op.getBlockSizeAttr().getInt());
    Value valuesPerBlock =
        arith::MulIOp::create(rewriter, loc, bsrBlockSize, bsrBlockSize);
    Value rowCount =
        memref::DimOp::create(rewriter, loc, op.getOutput(), zeroIndex);
    Value columnCount =
        memref::DimOp::create(rewriter, loc, op.getOutput(), oneIndex);
    Value outputElementCount =
        arith::MulIOp::create(rewriter, loc, rowCount, columnCount);
    LinearThreadWorkDistribution distribution =
        buildLinearThreadWorkDistribution(rewriter, loc, outputElementCount,
                                          oneIndex, gpuBlockSizeValue);
    Value element = distribution.workUnit;

    scf::IfOp::create(
        rewriter, loc, distribution.workUnitIsActive,
        [&](OpBuilder &builder, Location bodyLoc) {
          Value row =
              arith::DivUIOp::create(builder, bodyLoc, element, columnCount);
          Value outputColumn =
              arith::RemUIOp::create(builder, bodyLoc, element, columnCount);
          Value blockRow =
              arith::DivUIOp::create(builder, bodyLoc, row, bsrBlockSize);
          Value localRow =
              arith::RemUIOp::create(builder, bodyLoc, row, bsrBlockSize);
          CompressedRowBounds blockRowBounds = buildCompressedRowBounds(
              builder, bodyLoc, op.getBlockRowOffsets(), blockRow, oneIndex);

          auto valueType =
              cast<MemRefType>(op.getBlockValues().getType()).getElementType();
          Value zero = arith::ConstantOp::create(
              builder, bodyLoc, builder.getZeroAttr(valueType));
          auto blockTraversal = scf::ForOp::create(
              builder, bodyLoc, blockRowBounds.start, blockRowBounds.end,
              oneIndex, ValueRange{zero},
              [&](OpBuilder &blockBuilder, Location blockLoc,
                  Value blockPosition, ValueRange blockSums) {
                Value blockColumnValue = memref::LoadOp::create(
                    blockBuilder, blockLoc, op.getBlockColumnIndices(),
                    blockPosition);
                Value blockColumn =
                    castToIndex(blockBuilder, blockLoc, blockColumnValue);
                Value blockValueBase = arith::MulIOp::create(
                    blockBuilder, blockLoc, blockPosition, valuesPerBlock);
                Value localRowBase = arith::MulIOp::create(
                    blockBuilder, blockLoc, localRow, bsrBlockSize);
                blockValueBase = arith::AddIOp::create(
                    blockBuilder, blockLoc, blockValueBase, localRowBase);
                Value rhsRowBase = arith::MulIOp::create(
                    blockBuilder, blockLoc, blockColumn, bsrBlockSize);

                auto blockRowReduction = scf::ForOp::create(
                    blockBuilder, blockLoc, zeroIndex, bsrBlockSize, oneIndex,
                    blockSums,
                    [&](OpBuilder &elementBuilder, Location elementLoc,
                        Value localColumn, ValueRange iterArgs) {
                      Value blockValuePosition =
                          arith::AddIOp::create(elementBuilder, elementLoc,
                                                blockValueBase, localColumn);
                      Value blockValue = memref::LoadOp::create(
                          elementBuilder, elementLoc, op.getBlockValues(),
                          blockValuePosition);
                      Value rhsRow = arith::AddIOp::create(
                          elementBuilder, elementLoc, rhsRowBase, localColumn);
                      Value rhsValue = memref::LoadOp::create(
                          elementBuilder, elementLoc, op.getRhs(),
                          ValueRange{rhsRow, outputColumn});
                      Value product = arith::MulFOp::create(
                          elementBuilder, elementLoc, blockValue, rhsValue);
                      Value sum =
                          arith::AddFOp::create(elementBuilder, elementLoc,
                                                iterArgs.front(), product);
                      scf::YieldOp::create(elementBuilder, elementLoc, sum);
                    });
                scf::YieldOp::create(blockBuilder, blockLoc,
                                     blockRowReduction.getResult(0));
              });
          memref::StoreOp::create(builder, bodyLoc, blockTraversal.getResult(0),
                                  op.getOutput(),
                                  ValueRange{row, outputColumn});
          scf::YieldOp::create(builder, bodyLoc);
        },
        {});

    rewriter.setInsertionPointToEnd(&distribution.launch.getBody().front());
    gpu::TerminatorOp::create(rewriter, loc);
    rewriter.eraseOp(op);
    return success();
  }

private:
  int64_t gpuBlockSize;
};

class ThreadPerRowSDDMMPattern : public OpRewritePattern<SDDMMOp> {
public:
  ThreadPerRowSDDMMPattern(MLIRContext *context, int64_t blockSize)
      : OpRewritePattern<SDDMMOp>(context), blockSize(blockSize) {}

  LogicalResult matchAndRewrite(SDDMMOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value zeroIndex = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value oneIndex = arith::ConstantIndexOp::create(rewriter, loc, 1);
    Value blockSizeValue =
        arith::ConstantIndexOp::create(rewriter, loc, blockSize);
    Value rowCount =
        memref::DimOp::create(rewriter, loc, op.getLhs(), zeroIndex);
    Value reductionSize =
        memref::DimOp::create(rewriter, loc, op.getLhs(), oneIndex);
    LinearThreadWorkDistribution distribution =
        buildLinearThreadWorkDistribution(rewriter, loc, rowCount, oneIndex,
                                          blockSizeValue);
    Value row = distribution.workUnit;

    scf::IfOp::create(
        rewriter, loc, distribution.workUnitIsActive,
        [&](OpBuilder &builder, Location bodyLoc) {
          CompressedRowBounds rowBounds = buildCompressedRowBounds(
              builder, bodyLoc, op.getRowOffsets(), row, oneIndex);
          StridedPositionRange positions{rowBounds.start, rowBounds.end,
                                         oneIndex};
          auto valueType =
              cast<MemRefType>(op.getValues().getType()).getElementType();
          Value zero = arith::ConstantOp::create(
              builder, bodyLoc, builder.getZeroAttr(valueType));

          buildCSRPositionTraversal(
              builder, bodyLoc, op.getColumnIndices(), op.getValues(),
              positions, ValueRange{},
              [&](OpBuilder &positionBuilder, Location positionLoc,
                  CSRPosition position, ValueRange) {
                auto dot = scf::ForOp::create(
                    positionBuilder, positionLoc, zeroIndex, reductionSize,
                    oneIndex, ValueRange{zero},
                    [&](OpBuilder &reductionBuilder, Location reductionLoc,
                        Value reductionIndex, ValueRange iterArgs) {
                      Value lhsValue = memref::LoadOp::create(
                          reductionBuilder, reductionLoc, op.getLhs(),
                          ValueRange{row, reductionIndex});
                      Value rhsValue = memref::LoadOp::create(
                          reductionBuilder, reductionLoc, op.getRhs(),
                          ValueRange{reductionIndex, position.column});
                      Value product = arith::MulFOp::create(
                          reductionBuilder, reductionLoc, lhsValue, rhsValue);
                      Value sum =
                          arith::AddFOp::create(reductionBuilder, reductionLoc,
                                                iterArgs.front(), product);
                      scf::YieldOp::create(reductionBuilder, reductionLoc, sum);
                    });
                Value weighted = dot.getResult(0);
                Block &combineBody = op.getBody().front();
                auto yield = cast<YieldOp>(combineBody.getTerminator());
                IRMapping mapping;
                mapping.map(combineBody.getArgument(0), position.value);
                mapping.map(combineBody.getArgument(1), weighted);
                for (Operation &operation : combineBody.without_terminator())
                  positionBuilder.clone(operation, mapping);
                Value combined = mapping.lookup(yield.getResults().front());
                memref::StoreOp::create(positionBuilder, positionLoc, combined,
                                        op.getOutputValues(),
                                        position.position);
                return SmallVector<Value>{};
              });
          scf::YieldOp::create(builder, bodyLoc);
        },
        {});

    rewriter.setInsertionPointToEnd(&distribution.launch.getBody().front());
    gpu::TerminatorOp::create(rewriter, loc);
    propagateKernelName(op, distribution.launch);
    rewriter.eraseOp(op);
    return success();
  }

private:
  int64_t blockSize;
};

class ThreadPerRowCSRRowReducePattern
    : public OpRewritePattern<CSRRowReduceOp> {
public:
  ThreadPerRowCSRRowReducePattern(MLIRContext *context, int64_t blockSize)
      : OpRewritePattern<CSRRowReduceOp>(context), blockSize(blockSize) {}

  LogicalResult matchAndRewrite(CSRRowReduceOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value zeroIndex = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value oneIndex = arith::ConstantIndexOp::create(rewriter, loc, 1);
    Value blockSizeValue =
        arith::ConstantIndexOp::create(rewriter, loc, blockSize);
    Value rowCount =
        memref::DimOp::create(rewriter, loc, op.getOutput(), zeroIndex);
    LinearThreadWorkDistribution distribution =
        buildLinearThreadWorkDistribution(rewriter, loc, rowCount, oneIndex,
                                          blockSizeValue);
    Value row = distribution.workUnit;

    scf::IfOp::create(
        rewriter, loc, distribution.workUnitIsActive,
        [&](OpBuilder &builder, Location bodyLoc) {
          CompressedRowBounds rowBounds = buildCompressedRowBounds(
              builder, bodyLoc, op.getRowOffsets(), row, oneIndex);
          auto valueType = cast<FloatType>(
              cast<MemRefType>(op.getValues().getType()).getElementType());
          FloatAttr identityAttr =
              op.getKind() == "sum"
                  ? builder.getFloatAttr(valueType, 0.0)
                  : builder.getFloatAttr(
                        valueType, -std::numeric_limits<double>::infinity());
          Value identity =
              arith::ConstantOp::create(builder, bodyLoc, identityAttr);
          auto reduction = scf::ForOp::create(
              builder, bodyLoc, rowBounds.start, rowBounds.end, oneIndex,
              ValueRange{identity},
              [&](OpBuilder &reductionBuilder, Location reductionLoc,
                  Value position, ValueRange iterArgs) {
                Value value = memref::LoadOp::create(
                    reductionBuilder, reductionLoc, op.getValues(), position);
                Value next =
                    op.getKind() == "sum"
                        ? arith::AddFOp::create(reductionBuilder, reductionLoc,
                                                iterArgs.front(), value)
                              .getResult()
                        : arith::MaximumFOp::create(reductionBuilder,
                                                    reductionLoc,
                                                    iterArgs.front(), value)
                              .getResult();
                scf::YieldOp::create(reductionBuilder, reductionLoc, next);
              });
          memref::StoreOp::create(builder, bodyLoc, reduction.getResult(0),
                                  op.getOutput(), row);
          scf::YieldOp::create(builder, bodyLoc);
        });

    rewriter.setInsertionPointToEnd(&distribution.launch.getBody().front());
    gpu::TerminatorOp::create(rewriter, loc);
    propagateKernelName(op, distribution.launch);
    rewriter.eraseOp(op);
    return success();
  }

private:
  int64_t blockSize;
};

class ThreadPerRowCSRRowwiseMapPattern
    : public OpRewritePattern<CSRRowwiseMapOp> {
public:
  ThreadPerRowCSRRowwiseMapPattern(MLIRContext *context, int64_t blockSize)
      : OpRewritePattern<CSRRowwiseMapOp>(context), blockSize(blockSize) {}

  LogicalResult matchAndRewrite(CSRRowwiseMapOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value zeroIndex = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value oneIndex = arith::ConstantIndexOp::create(rewriter, loc, 1);
    Value blockSizeValue =
        arith::ConstantIndexOp::create(rewriter, loc, blockSize);
    Value rowCount =
        memref::DimOp::create(rewriter, loc, op.getRowValues(), zeroIndex);
    LinearThreadWorkDistribution distribution =
        buildLinearThreadWorkDistribution(rewriter, loc, rowCount, oneIndex,
                                          blockSizeValue);
    Value row = distribution.workUnit;

    scf::IfOp::create(
        rewriter, loc, distribution.workUnitIsActive,
        [&](OpBuilder &builder, Location bodyLoc) {
          CompressedRowBounds rowBounds = buildCompressedRowBounds(
              builder, bodyLoc, op.getRowOffsets(), row, oneIndex);
          Value rowValue =
              memref::LoadOp::create(builder, bodyLoc, op.getRowValues(), row);
          Block &mapBody = op.getBody().front();
          auto yield = cast<YieldOp>(mapBody.getTerminator());
          scf::ForOp::create(
              builder, bodyLoc, rowBounds.start, rowBounds.end, oneIndex,
              ValueRange{},
              [&](OpBuilder &mapBuilder, Location mapLoc, Value position,
                  ValueRange) {
                Value value = memref::LoadOp::create(mapBuilder, mapLoc,
                                                     op.getValues(), position);
                IRMapping mapping;
                mapping.map(mapBody.getArgument(0), value);
                mapping.map(mapBody.getArgument(1), rowValue);
                for (Operation &operation : mapBody.without_terminator())
                  mapBuilder.clone(operation, mapping);
                Value mapped = mapping.lookup(yield.getResults().front());
                memref::StoreOp::create(mapBuilder, mapLoc, mapped,
                                        op.getOutputValues(), position);
                scf::YieldOp::create(mapBuilder, mapLoc);
              });
          scf::YieldOp::create(builder, bodyLoc);
        });

    rewriter.setInsertionPointToEnd(&distribution.launch.getBody().front());
    gpu::TerminatorOp::create(rewriter, loc);
    propagateKernelName(op, distribution.launch);
    rewriter.eraseOp(op);
    return success();
  }

private:
  int64_t blockSize;
};

class ThreadPerRowCSRElementwisePattern
    : public OpRewritePattern<CSRElementwiseOp> {
public:
  ThreadPerRowCSRElementwisePattern(MLIRContext *context, int64_t blockSize)
      : OpRewritePattern<CSRElementwiseOp>(context), blockSize(blockSize) {}

  LogicalResult matchAndRewrite(CSRElementwiseOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value zeroIndex = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value oneIndex = arith::ConstantIndexOp::create(rewriter, loc, 1);
    Value blockSizeValue =
        arith::ConstantIndexOp::create(rewriter, loc, blockSize);
    Value rowOffsetCount = memref::DimOp::create(
        rewriter, loc, op.getOutputRowOffsets(), zeroIndex);
    Value rowCount =
        arith::SubIOp::create(rewriter, loc, rowOffsetCount, oneIndex);
    CSRCoiterationKind coiterationKind = op.getKind() == "add"
                                             ? CSRCoiterationKind::Union
                                             : CSRCoiterationKind::Intersection;

    // Symbolic phase: count output coordinates independently for each row.
    // Counts are temporarily stored at outputRowOffsets[row + 1].
    LinearThreadWorkDistribution symbolic = buildLinearThreadWorkDistribution(
        rewriter, loc, rowCount, oneIndex, blockSizeValue);
    Value row = symbolic.workUnit;
    scf::IfOp::create(
        rewriter, loc, symbolic.workUnitIsActive,
        [&](OpBuilder &builder, Location bodyLoc) {
          CompressedRowBounds lhsBounds = buildCompressedRowBounds(
              builder, bodyLoc, op.getLhsRowOffsets(), row, oneIndex);
          CompressedRowBounds rhsBounds = buildCompressedRowBounds(
              builder, bodyLoc, op.getRhsRowOffsets(), row, oneIndex);
          SmallVector<Value> result = buildCSRCoiteration(
              builder, bodyLoc, op.getLhsColumnIndices(), lhsBounds,
              op.getRhsColumnIndices(), rhsBounds, coiterationKind, oneIndex,
              ValueRange{zeroIndex},
              [&](OpBuilder &entryBuilder, Location entryLoc,
                  CSRCoiterationEntry, ValueRange iterArgs) {
                Value nextCount = arith::AddIOp::create(
                    entryBuilder, entryLoc, iterArgs.front(), oneIndex);
                return SmallVector<Value>{nextCount};
              });
          Value nextRow =
              arith::AddIOp::create(builder, bodyLoc, row, oneIndex);
          Type offsetType = cast<MemRefType>(op.getOutputRowOffsets().getType())
                                .getElementType();
          Value count =
              castIndexToType(builder, bodyLoc, result.front(), offsetType);
          memref::StoreOp::create(builder, bodyLoc, count,
                                  op.getOutputRowOffsets(), nextRow);
          scf::YieldOp::create(builder, bodyLoc);
        });
    rewriter.setInsertionPointToEnd(&symbolic.launch.getBody().front());
    gpu::TerminatorOp::create(rewriter, loc);

    // Prefix phase: convert row counts into CSR offsets and publish total NNZ.
    // This correctness baseline deliberately isolates a sequential scan so it
    // can later be replaced by a parallel scan without changing coiteration.
    rewriter.setInsertionPointAfter(symbolic.launch);
    gpu::LaunchOp prefix =
        gpu::LaunchOp::create(rewriter, loc, oneIndex, oneIndex, oneIndex,
                              oneIndex, oneIndex, oneIndex);
    rewriter.setInsertionPointToStart(&prefix.getBody().front());
    Type offsetType =
        cast<MemRefType>(op.getOutputRowOffsets().getType()).getElementType();
    Value zeroOffset = arith::ConstantOp::create(
        rewriter, loc, rewriter.getZeroAttr(offsetType));
    memref::StoreOp::create(rewriter, loc, zeroOffset, op.getOutputRowOffsets(),
                            zeroIndex);
    auto scan = scf::ForOp::create(
        rewriter, loc, zeroIndex, rowCount, oneIndex, ValueRange{zeroOffset},
        [&](OpBuilder &builder, Location bodyLoc, Value scanRow,
            ValueRange iterArgs) {
          Value nextRow =
              arith::AddIOp::create(builder, bodyLoc, scanRow, oneIndex);
          Value rowNnz = memref::LoadOp::create(
              builder, bodyLoc, op.getOutputRowOffsets(), nextRow);
          Value total =
              arith::AddIOp::create(builder, bodyLoc, iterArgs.front(), rowNnz);
          memref::StoreOp::create(builder, bodyLoc, total,
                                  op.getOutputRowOffsets(), nextRow);
          scf::YieldOp::create(builder, bodyLoc, total);
        });
    memref::StoreOp::create(rewriter, loc, scan.getResult(0), op.getOutputNnz(),
                            zeroIndex);
    gpu::TerminatorOp::create(rewriter, loc);

    // Numeric phase: replay the same coiteration and assemble values into the
    // positions assigned by the prefix phase.
    rewriter.setInsertionPointAfter(prefix);
    LinearThreadWorkDistribution numeric = buildLinearThreadWorkDistribution(
        rewriter, loc, rowCount, oneIndex, blockSizeValue);
    row = numeric.workUnit;
    scf::IfOp::create(
        rewriter, loc, numeric.workUnitIsActive,
        [&](OpBuilder &builder, Location bodyLoc) {
          CompressedRowBounds lhsBounds = buildCompressedRowBounds(
              builder, bodyLoc, op.getLhsRowOffsets(), row, oneIndex);
          CompressedRowBounds rhsBounds = buildCompressedRowBounds(
              builder, bodyLoc, op.getRhsRowOffsets(), row, oneIndex);
          Value outputStartValue = memref::LoadOp::create(
              builder, bodyLoc, op.getOutputRowOffsets(), row);
          Value outputStart = castToIndex(builder, bodyLoc, outputStartValue);
          auto valueType =
              cast<MemRefType>(op.getOutputValues().getType()).getElementType();
          Value zeroValue = arith::ConstantOp::create(
              builder, bodyLoc, builder.getZeroAttr(valueType));

          buildCSRCoiteration(
              builder, bodyLoc, op.getLhsColumnIndices(), lhsBounds,
              op.getRhsColumnIndices(), rhsBounds, coiterationKind, oneIndex,
              ValueRange{outputStart},
              [&](OpBuilder &entryBuilder, Location entryLoc,
                  CSRCoiterationEntry entry, ValueRange iterArgs) {
                Value outputPosition = iterArgs.front();
                Value outputValue;
                if (coiterationKind == CSRCoiterationKind::Union) {
                  auto lhsValue = scf::IfOp::create(
                      entryBuilder, entryLoc, entry.lhsPresent,
                      [&](OpBuilder &thenBuilder, Location thenLoc) {
                        Value value = memref::LoadOp::create(
                            thenBuilder, thenLoc, op.getLhsValues(),
                            entry.lhsPosition);
                        scf::YieldOp::create(thenBuilder, thenLoc, value);
                      },
                      [&](OpBuilder &elseBuilder, Location elseLoc) {
                        scf::YieldOp::create(elseBuilder, elseLoc, zeroValue);
                      });
                  auto rhsValue = scf::IfOp::create(
                      entryBuilder, entryLoc, entry.rhsPresent,
                      [&](OpBuilder &thenBuilder, Location thenLoc) {
                        Value value = memref::LoadOp::create(
                            thenBuilder, thenLoc, op.getRhsValues(),
                            entry.rhsPosition);
                        scf::YieldOp::create(thenBuilder, thenLoc, value);
                      },
                      [&](OpBuilder &elseBuilder, Location elseLoc) {
                        scf::YieldOp::create(elseBuilder, elseLoc, zeroValue);
                      });
                  outputValue = arith::AddFOp::create(entryBuilder, entryLoc,
                                                      lhsValue.getResult(0),
                                                      rhsValue.getResult(0));
                } else {
                  Value lhsValue = memref::LoadOp::create(
                      entryBuilder, entryLoc, op.getLhsValues(),
                      entry.lhsPosition);
                  Value rhsValue = memref::LoadOp::create(
                      entryBuilder, entryLoc, op.getRhsValues(),
                      entry.rhsPosition);
                  outputValue = arith::MulFOp::create(entryBuilder, entryLoc,
                                                      lhsValue, rhsValue);
                }

                Value outputColumn = castIndexToType(entryBuilder, entryLoc,
                                                     entry.column, offsetType);
                memref::StoreOp::create(entryBuilder, entryLoc, outputColumn,
                                        op.getOutputColumnIndices(),
                                        outputPosition);
                memref::StoreOp::create(entryBuilder, entryLoc, outputValue,
                                        op.getOutputValues(), outputPosition);
                Value nextOutputPosition = arith::AddIOp::create(
                    entryBuilder, entryLoc, outputPosition, oneIndex);
                return SmallVector<Value>{nextOutputPosition};
              });
          scf::YieldOp::create(builder, bodyLoc);
        });
    rewriter.setInsertionPointToEnd(&numeric.launch.getBody().front());
    gpu::TerminatorOp::create(rewriter, loc);
    rewriter.eraseOp(op);
    return success();
  }

private:
  int64_t blockSize;
};

class WavePerRowTileSpMMPattern : public OpRewritePattern<SpMMOp> {
public:
  WavePerRowTileSpMMPattern(MLIRContext *context, int64_t blockSize,
                            int64_t waveSize, int64_t tileSize)
      : OpRewritePattern<SpMMOp>(context), blockSize(blockSize),
        waveSize(waveSize), tileSize(tileSize) {}

  LogicalResult matchAndRewrite(SpMMOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value zeroIndex = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value oneIndex = arith::ConstantIndexOp::create(rewriter, loc, 1);
    Value blockSizeValue =
        arith::ConstantIndexOp::create(rewriter, loc, blockSize);
    Value waveSizeValue =
        arith::ConstantIndexOp::create(rewriter, loc, waveSize);
    Value tileSizeValue =
        arith::ConstantIndexOp::create(rewriter, loc, tileSize);
    Value wavesPerBlockValue =
        arith::ConstantIndexOp::create(rewriter, loc, blockSize / waveSize);
    Value rowCount =
        memref::DimOp::create(rewriter, loc, op.getOutput(), zeroIndex);
    Value columnCount =
        memref::DimOp::create(rewriter, loc, op.getOutput(), oneIndex);
    Value tilesPerRow =
        arith::CeilDivUIOp::create(rewriter, loc, columnCount, tileSizeValue);
    Value workUnitCount =
        arith::MulIOp::create(rewriter, loc, rowCount, tilesPerRow);
    WaveWorkDistribution distribution = buildWaveWorkDistribution(
        rewriter, loc, workUnitCount, oneIndex, blockSizeValue, waveSizeValue,
        wavesPerBlockValue);
    Value workUnit = distribution.workUnit;

    scf::IfOp::create(
        rewriter, loc, distribution.workUnitIsActive,
        [&](OpBuilder &builder, Location bodyLoc) {
          Value row =
              arith::DivUIOp::create(builder, bodyLoc, workUnit, tilesPerRow);
          Value tile =
              arith::RemUIOp::create(builder, bodyLoc, workUnit, tilesPerRow);
          Value firstOutputColumn =
              arith::MulIOp::create(builder, bodyLoc, tile, tileSizeValue);
          CompressedRowBounds rowBounds = buildCompressedRowBounds(
              builder, bodyLoc, op.getRowOffsets(), row, oneIndex);
          StridedPositionRange positions = buildStridedPositionRange(
              builder, bodyLoc, rowBounds, distribution.lane,
              distribution.positionStride);

          auto valueType =
              cast<MemRefType>(op.getValues().getType()).getElementType();
          Value zero = arith::ConstantOp::create(
              builder, bodyLoc, builder.getZeroAttr(valueType));
          SmallVector<Value> outputColumns;
          outputColumns.reserve(tileSize);
          for (int64_t tileColumn = 0; tileColumn < tileSize; ++tileColumn) {
            Value tileColumnValue =
                arith::ConstantIndexOp::create(builder, bodyLoc, tileColumn);
            Value outputColumn = arith::AddIOp::create(
                builder, bodyLoc, firstOutputColumn, tileColumnValue);
            outputColumns.push_back(outputColumn);
          }

          auto buildPartialReductions =
              [&](OpBuilder &tileBuilder, Location tileLoc,
                  bool guardColumns) -> SmallVector<Value> {
            SmallVector<Value> activeColumns;
            if (guardColumns) {
              activeColumns.reserve(tileSize);
              for (Value outputColumn : outputColumns) {
                activeColumns.push_back(arith::CmpIOp::create(
                    tileBuilder, tileLoc, arith::CmpIPredicate::ult,
                    outputColumn, columnCount));
              }
            }

            SmallVector<Value> initialSums(tileSize, zero);
            return buildCSRPositionTraversal(
                tileBuilder, tileLoc, op.getColumnIndices(), op.getValues(),
                positions, initialSums,
                [&](OpBuilder &loopBuilder, Location loopLoc,
                    CSRPosition position, ValueRange iterArgs) {
                  SmallVector<Value> nextSums;
                  nextSums.reserve(tileSize);
                  for (int64_t tileColumn = 0; tileColumn < tileSize;
                       ++tileColumn) {
                    if (!guardColumns) {
                      Value rhsValue = memref::LoadOp::create(
                          loopBuilder, loopLoc, op.getRhs(),
                          ValueRange{position.column,
                                     outputColumns[tileColumn]});
                      Value product = arith::MulFOp::create(
                          loopBuilder, loopLoc, position.value, rhsValue);
                      nextSums.push_back(arith::AddFOp::create(
                          loopBuilder, loopLoc, iterArgs[tileColumn], product));
                      continue;
                    }

                    auto update = scf::IfOp::create(
                        loopBuilder, loopLoc, TypeRange{valueType},
                        activeColumns[tileColumn], /*withElseRegion=*/true);
                    loopBuilder.setInsertionPointToStart(
                        &update.getThenRegion().front());
                    Value rhsValue = memref::LoadOp::create(
                        loopBuilder, loopLoc, op.getRhs(),
                        ValueRange{position.column, outputColumns[tileColumn]});
                    Value product = arith::MulFOp::create(
                        loopBuilder, loopLoc, position.value, rhsValue);
                    Value sum = arith::AddFOp::create(
                        loopBuilder, loopLoc, iterArgs[tileColumn], product);
                    scf::YieldOp::create(loopBuilder, loopLoc, sum);
                    loopBuilder.setInsertionPointToStart(
                        &update.getElseRegion().front());
                    scf::YieldOp::create(loopBuilder, loopLoc,
                                         iterArgs[tileColumn]);
                    loopBuilder.setInsertionPointAfter(update);
                    nextSums.push_back(update.getResult(0));
                  }
                  return nextSums;
                });
          };

          Value tileEnd = arith::AddIOp::create(
              builder, bodyLoc, firstOutputColumn, tileSizeValue);
          Value isFullTile =
              arith::CmpIOp::create(builder, bodyLoc, arith::CmpIPredicate::ule,
                                    tileEnd, columnCount);
          SmallVector<Type> reductionTypes(tileSize, valueType);
          auto tileReductions = scf::IfOp::create(
              builder, bodyLoc, TypeRange(reductionTypes), isFullTile,
              /*withElseRegion=*/true);
          builder.setInsertionPointToStart(
              &tileReductions.getThenRegion().front());
          SmallVector<Value> fullTileReductions =
              buildPartialReductions(builder, bodyLoc, /*guardColumns=*/false);
          scf::YieldOp::create(builder, bodyLoc, fullTileReductions);
          builder.setInsertionPointToStart(
              &tileReductions.getElseRegion().front());
          SmallVector<Value> tailTileReductions =
              buildPartialReductions(builder, bodyLoc, /*guardColumns=*/true);
          scf::YieldOp::create(builder, bodyLoc, tailTileReductions);
          builder.setInsertionPointAfter(tileReductions);

          SmallVector<Value> waveSums = buildWaveReductions(
              builder, bodyLoc, tileReductions.getResults(), waveSize);

          Value laneIsZero =
              arith::CmpIOp::create(builder, bodyLoc, arith::CmpIPredicate::eq,
                                    distribution.lane, zeroIndex);
          scf::IfOp::create(
              builder, bodyLoc, laneIsZero,
              [&](OpBuilder &laneBuilder, Location laneLoc) {
                scf::IfOp::create(
                    laneBuilder, laneLoc, isFullTile,
                    [&](OpBuilder &fullTileBuilder, Location fullTileLoc) {
                      for (int64_t tileColumn = 0; tileColumn < tileSize;
                           ++tileColumn) {
                        memref::StoreOp::create(
                            fullTileBuilder, fullTileLoc, waveSums[tileColumn],
                            op.getOutput(),
                            ValueRange{row, outputColumns[tileColumn]});
                      }
                      scf::YieldOp::create(fullTileBuilder, fullTileLoc);
                    },
                    [&](OpBuilder &tailTileBuilder, Location tailTileLoc) {
                      for (int64_t tileColumn = 0; tileColumn < tileSize;
                           ++tileColumn) {
                        Value columnIsActive = arith::CmpIOp::create(
                            tailTileBuilder, tailTileLoc,
                            arith::CmpIPredicate::ult,
                            outputColumns[tileColumn], columnCount);
                        scf::IfOp::create(
                            tailTileBuilder, tailTileLoc, columnIsActive,
                            [&](OpBuilder &storeBuilder, Location storeLoc) {
                              memref::StoreOp::create(
                                  storeBuilder, storeLoc, waveSums[tileColumn],
                                  op.getOutput(),
                                  ValueRange{row, outputColumns[tileColumn]});
                              scf::YieldOp::create(storeBuilder, storeLoc);
                            });
                      }
                      scf::YieldOp::create(tailTileBuilder, tailTileLoc);
                    });
                scf::YieldOp::create(laneBuilder, laneLoc);
              });
          scf::YieldOp::create(builder, bodyLoc);
        },
        {});

    rewriter.setInsertionPointToEnd(&distribution.launch.getBody().front());
    gpu::TerminatorOp::create(rewriter, loc);
    propagateKernelName(op, distribution.launch);
    rewriter.eraseOp(op);
    return success();
  }

private:
  int64_t blockSize;
  int64_t waveSize;
  int64_t tileSize;
};

class ConvertSparseWaveToGPU
    : public impl::ConvertSparseWaveToGPUBase<ConvertSparseWaveToGPU> {
public:
  using impl::ConvertSparseWaveToGPUBase<
      ConvertSparseWaveToGPU>::ConvertSparseWaveToGPUBase;

  void runOnOperation() override {
    if (mapping != "thread-per-row" && mapping != "thread-per-position" &&
        mapping != "wave-per-position" && mapping != "wave-per-row" &&
        mapping != "block-per-row") {
      getOperation().emitError()
          << "unsupported SpMV mapping strategy '" << mapping
          << "'; expected 'thread-per-row', 'thread-per-position', "
             "'wave-per-position', 'wave-per-row', or 'block-per-row'";
      signalPassFailure();
      return;
    }
    if (spmmMapping != "thread-per-output" &&
        spmmMapping != "wave-per-row-tile") {
      getOperation().emitError()
          << "unsupported SpMM mapping strategy '" << spmmMapping
          << "'; expected 'thread-per-output' or 'wave-per-row-tile'";
      signalPassFailure();
      return;
    }
    if (blockSize < 1 || blockSize > 1024) {
      getOperation().emitError()
          << "SpMV block size must be between 1 and 1024, but got "
          << blockSize.getValue();
      signalPassFailure();
      return;
    }
    if (positionBlockSize < 1 || positionBlockSize > 1024) {
      getOperation().emitError()
          << "position-parallel block size must be between 1 and 1024, but "
             "got "
          << positionBlockSize.getValue();
      signalPassFailure();
      return;
    }
    if (spmmBlockSize < 1 || spmmBlockSize > 1024) {
      getOperation().emitError()
          << "SpMM block size must be between 1 and 1024, but got "
          << spmmBlockSize.getValue();
      signalPassFailure();
      return;
    }
    if (spmmTileSize < 1 || spmmTileSize > 32) {
      getOperation().emitError()
          << "SpMM tile size must be between 1 and 32, but got "
          << spmmTileSize.getValue();
      signalPassFailure();
      return;
    }
    if (sddmmBlockSize < 1 || sddmmBlockSize > 1024) {
      getOperation().emitError()
          << "SDDMM block size must be between 1 and 1024, but got "
          << sddmmBlockSize.getValue();
      signalPassFailure();
      return;
    }
    if (rowReductionBlockSize < 1 || rowReductionBlockSize > 1024) {
      getOperation().emitError()
          << "CSR row-reduction block size must be between 1 and 1024, but got "
          << rowReductionBlockSize.getValue();
      signalPassFailure();
      return;
    }
    if (rowwiseMapBlockSize < 1 || rowwiseMapBlockSize > 1024) {
      getOperation().emitError()
          << "CSR row-wise map block size must be between 1 and 1024, but got "
          << rowwiseMapBlockSize.getValue();
      signalPassFailure();
      return;
    }
    if (elementwiseBlockSize < 1 || elementwiseBlockSize > 1024) {
      getOperation().emitError()
          << "elementwise block size must be between 1 and 1024, but got "
          << elementwiseBlockSize.getValue();
      signalPassFailure();
      return;
    }
    if (spmmMapping == "wave-per-row-tile" && waveSize != 32) {
      getOperation().emitError()
          << "wave-per-row-tile currently requires Wave32, but got wave size "
          << waveSize.getValue();
      signalPassFailure();
      return;
    }
    if (spmmMapping == "wave-per-row-tile" && spmmBlockSize % waveSize != 0) {
      getOperation().emitError()
          << "wave-per-row-tile requires the SpMM block size to be a multiple "
             "of "
          << waveSize.getValue() << ", but got " << spmmBlockSize.getValue();
      signalPassFailure();
      return;
    }
    bool usesWaveCooperation = mapping == "wave-per-position" ||
                               mapping == "wave-per-row" ||
                               mapping == "block-per-row";
    if (usesWaveCooperation && waveSize != 32) {
      getOperation().emitError()
          << mapping << " currently requires Wave32, but got wave size "
          << waveSize.getValue();
      signalPassFailure();
      return;
    }
    if (usesWaveCooperation && blockSize % waveSize != 0) {
      getOperation().emitError()
          << mapping << " requires the SpMV block size to be a multiple of "
          << waveSize.getValue() << ", but got " << blockSize.getValue();
      signalPassFailure();
      return;
    }

    bool hasPositionWaveMapping = false;
    getOperation().walk([&](PositionParallelOp op) {
      FailureOr<PositionMapping> mapping = op.getMappingKind();
      hasPositionWaveMapping |=
          succeeded(mapping) && *mapping == PositionMapping::Wave;
    });
    if (hasPositionWaveMapping && waveSize != 32 && waveSize != 64) {
      getOperation().emitError()
          << "wave position mapping requires a wave size of 32 or 64, but got "
          << waveSize.getValue();
      signalPassFailure();
      return;
    }
    if (hasPositionWaveMapping && positionBlockSize % waveSize != 0) {
      getOperation().emitError()
          << "wave position mapping requires the position block size to be a "
             "multiple of the wave size, but got "
          << positionBlockSize.getValue() << " and " << waveSize.getValue();
      signalPassFailure();
      return;
    }

    RewritePatternSet patterns(&getContext());
    patterns.add<PositionParallelPattern>(&getContext(), positionBlockSize,
                                          waveSize);
    patterns.add<ThreadPerNonzeroCOOSpMVPattern>(&getContext(), blockSize);
    patterns.add<ThreadPerRowSDDMMPattern>(&getContext(), sddmmBlockSize);
    patterns.add<ThreadPerRowCSRRowReducePattern>(&getContext(),
                                                  rowReductionBlockSize);
    patterns.add<ThreadPerRowCSRRowwiseMapPattern>(&getContext(),
                                                   rowwiseMapBlockSize);
    patterns.add<ThreadPerRowCSRElementwisePattern>(&getContext(),
                                                    elementwiseBlockSize);
    patterns.add<ThreadPerOutputBSRSpMMPattern>(&getContext(), spmmBlockSize);
    if (spmmMapping == "thread-per-output")
      patterns.add<ThreadPerOutputSpMMPattern>(&getContext(), spmmBlockSize);
    else
      patterns.add<WavePerRowTileSpMMPattern>(&getContext(), spmmBlockSize,
                                              waveSize, spmmTileSize);
    if (mapping == "thread-per-row")
      patterns.add<ThreadPerRowSpMVPattern>(&getContext(), blockSize);
    else if (mapping == "thread-per-position")
      patterns.add<ThreadPerPositionCSRSpMVPattern>(&getContext(), blockSize);
    else if (mapping == "wave-per-position")
      patterns.add<WavePerPositionCSRSpMVPattern>(&getContext(), blockSize,
                                                  waveSize);
    else if (mapping == "wave-per-row")
      patterns.add<WavePerRowSpMVPattern>(&getContext(), blockSize, waveSize);
    else
      patterns.add<BlockPerRowSpMVPattern>(&getContext(), blockSize, waveSize);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
} // namespace mlir::sparsewave
