#include "sparsewave/Dialect/SparseWave/Transforms/Passes.h"

#include "sparsewave/Dialect/SparseWave/IR/SparseWaveOps.h"

#include "SparseGPUUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/SmallVector.h"

namespace mlir::sparsewave {
#define GEN_PASS_DEF_CONVERTSPARSEWAVETOGPU
#include "sparsewave/Dialect/SparseWave/Transforms/Passes.h.inc"

namespace {

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

    LinearThreadWorkDistribution initialization =
        buildLinearThreadWorkDistribution(rewriter, loc, outputSize, oneIndex,
                                          blockSizeValue);
    scf::IfOp::create(rewriter, loc, initialization.workUnitIsActive,
                      [&](OpBuilder &builder, Location bodyLoc) {
                        memref::StoreOp::create(builder, bodyLoc, zero,
                                                op.getOutput(),
                                                initialization.workUnit);
                        scf::YieldOp::create(builder, bodyLoc);
                      },
                      {});
    rewriter.setInsertionPointToEnd(&initialization.launch.getBody().front());
    gpu::TerminatorOp::create(rewriter, loc);

    rewriter.setInsertionPointAfter(initialization.launch);
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
                Value weighted =
                    arith::MulFOp::create(positionBuilder, positionLoc,
                                          position.value, dot.getResult(0));
                memref::StoreOp::create(positionBuilder, positionLoc, weighted,
                                        op.getOutputValues(),
                                        position.position);
                return SmallVector<Value>{};
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
    if (mapping != "thread-per-row" && mapping != "wave-per-row" &&
        mapping != "block-per-row") {
      getOperation().emitError()
          << "unsupported SpMV mapping strategy '" << mapping
          << "'; expected 'thread-per-row', 'wave-per-row', or "
             "'block-per-row'";
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
    if (mapping != "thread-per-row" && waveSize != 32) {
      getOperation().emitError()
          << mapping << " currently requires Wave32, but got wave size "
          << waveSize.getValue();
      signalPassFailure();
      return;
    }
    if (mapping != "thread-per-row" && blockSize % waveSize != 0) {
      getOperation().emitError()
          << mapping << " requires the SpMV block size to be a multiple of "
          << waveSize.getValue() << ", but got " << blockSize.getValue();
      signalPassFailure();
      return;
    }

    RewritePatternSet patterns(&getContext());
    patterns.add<ThreadPerNonzeroCOOSpMVPattern>(&getContext(), blockSize);
    patterns.add<ThreadPerRowSDDMMPattern>(&getContext(), sddmmBlockSize);
    patterns.add<ThreadPerOutputBSRSpMMPattern>(&getContext(), spmmBlockSize);
    if (spmmMapping == "thread-per-output")
      patterns.add<ThreadPerOutputSpMMPattern>(&getContext(), spmmBlockSize);
    else
      patterns.add<WavePerRowTileSpMMPattern>(&getContext(), spmmBlockSize,
                                              waveSize, spmmTileSize);
    if (mapping == "thread-per-row")
      patterns.add<ThreadPerRowSpMVPattern>(&getContext(), blockSize);
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
