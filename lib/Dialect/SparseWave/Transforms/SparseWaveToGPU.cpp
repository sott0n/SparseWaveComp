#include "sparsewave/Dialect/SparseWave/Transforms/Passes.h"

#include "sparsewave/Dialect/SparseWave/IR/SparseWaveOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::sparsewave {
#define GEN_PASS_DEF_CONVERTSPARSEWAVETOGPU
#include "sparsewave/Dialect/SparseWave/Transforms/Passes.h.inc"

namespace {

Value castToIndex(OpBuilder &builder, Location loc, Value value) {
  if (value.getType().isIndex())
    return value;
  if (cast<IntegerType>(value.getType()).isUnsigned())
    return arith::IndexCastUIOp::create(builder, loc, builder.getIndexType(),
                                        value);
  return arith::IndexCastOp::create(builder, loc, builder.getIndexType(),
                                    value);
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
    Value requiredBlocks =
        arith::CeilDivUIOp::create(rewriter, loc, rowCount, blockSizeValue);
    Value gridSize =
        arith::MaxUIOp::create(rewriter, loc, requiredBlocks, oneIndex);

    gpu::LaunchOp launch =
        gpu::LaunchOp::create(rewriter, loc, gridSize, oneIndex, oneIndex,
                              blockSizeValue, oneIndex, oneIndex);
    rewriter.setInsertionPointToStart(&launch.getBody().front());

    Value rowBase = arith::MulIOp::create(rewriter, loc, launch.getBlockIds().x,
                                          launch.getBlockSize().x);
    Value row =
        arith::AddIOp::create(rewriter, loc, rowBase, launch.getThreadIds().x);
    Value rowIsActive = arith::CmpIOp::create(
        rewriter, loc, arith::CmpIPredicate::ult, row, rowCount);

    scf::IfOp::create(
        rewriter, loc, rowIsActive,
        [&](OpBuilder &builder, Location bodyLoc) {
          Value nextRow =
              arith::AddIOp::create(builder, bodyLoc, row, oneIndex);
          Value rowStartValue =
              memref::LoadOp::create(builder, bodyLoc, op.getRowOffsets(), row);
          Value rowEndValue = memref::LoadOp::create(
              builder, bodyLoc, op.getRowOffsets(), nextRow);
          Value rowStart = castToIndex(builder, bodyLoc, rowStartValue);
          Value rowEnd = castToIndex(builder, bodyLoc, rowEndValue);

          auto valueType =
              cast<MemRefType>(op.getValues().getType()).getElementType();
          Value zero = arith::ConstantOp::create(
              builder, bodyLoc, builder.getZeroAttr(valueType));
          auto reduction = scf::ForOp::create(
              builder, bodyLoc, rowStart, rowEnd, oneIndex, ValueRange{zero},
              [&](OpBuilder &loopBuilder, Location loopLoc, Value position,
                  ValueRange iterArgs) {
                Value columnValue = memref::LoadOp::create(
                    loopBuilder, loopLoc, op.getColumnIndices(), position);
                Value column = castToIndex(loopBuilder, loopLoc, columnValue);
                Value matrixValue = memref::LoadOp::create(
                    loopBuilder, loopLoc, op.getValues(), position);
                Value vectorValue = memref::LoadOp::create(
                    loopBuilder, loopLoc, op.getVector(), column);
                Value product = arith::MulFOp::create(loopBuilder, loopLoc,
                                                      matrixValue, vectorValue);
                Value sum = arith::AddFOp::create(loopBuilder, loopLoc,
                                                  iterArgs.front(), product);
                scf::YieldOp::create(loopBuilder, loopLoc, sum);
              });
          memref::StoreOp::create(builder, bodyLoc, reduction.getResult(0),
                                  op.getOutput(), row);
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
    Value requiredBlocks =
        arith::CeilDivUIOp::create(rewriter, loc, rowCount, wavesPerBlockValue);
    Value gridSize =
        arith::MaxUIOp::create(rewriter, loc, requiredBlocks, oneIndex);

    gpu::LaunchOp launch =
        gpu::LaunchOp::create(rewriter, loc, gridSize, oneIndex, oneIndex,
                              blockSizeValue, oneIndex, oneIndex);
    rewriter.setInsertionPointToStart(&launch.getBody().front());

    Value waveInBlock = arith::DivUIOp::create(
        rewriter, loc, launch.getThreadIds().x, waveSizeValue);
    Value lane = arith::RemUIOp::create(rewriter, loc, launch.getThreadIds().x,
                                        waveSizeValue);
    Value rowBase = arith::MulIOp::create(rewriter, loc, launch.getBlockIds().x,
                                          wavesPerBlockValue);
    Value row = arith::AddIOp::create(rewriter, loc, rowBase, waveInBlock);
    Value rowIsActive = arith::CmpIOp::create(
        rewriter, loc, arith::CmpIPredicate::ult, row, rowCount);

    scf::IfOp::create(
        rewriter, loc, rowIsActive,
        [&](OpBuilder &builder, Location bodyLoc) {
          Value nextRow =
              arith::AddIOp::create(builder, bodyLoc, row, oneIndex);
          Value rowStartValue =
              memref::LoadOp::create(builder, bodyLoc, op.getRowOffsets(), row);
          Value rowEndValue = memref::LoadOp::create(
              builder, bodyLoc, op.getRowOffsets(), nextRow);
          Value rowStart = castToIndex(builder, bodyLoc, rowStartValue);
          Value rowEnd = castToIndex(builder, bodyLoc, rowEndValue);
          Value firstPosition =
              arith::AddIOp::create(builder, bodyLoc, rowStart, lane);

          auto valueType =
              cast<MemRefType>(op.getValues().getType()).getElementType();
          Value zero = arith::ConstantOp::create(
              builder, bodyLoc, builder.getZeroAttr(valueType));
          auto partialReduction = scf::ForOp::create(
              builder, bodyLoc, firstPosition, rowEnd, waveSizeValue,
              ValueRange{zero},
              [&](OpBuilder &loopBuilder, Location loopLoc, Value position,
                  ValueRange iterArgs) {
                Value columnValue = memref::LoadOp::create(
                    loopBuilder, loopLoc, op.getColumnIndices(), position);
                Value column = castToIndex(loopBuilder, loopLoc, columnValue);
                Value matrixValue = memref::LoadOp::create(
                    loopBuilder, loopLoc, op.getValues(), position);
                Value vectorValue = memref::LoadOp::create(
                    loopBuilder, loopLoc, op.getVector(), column);
                Value product = arith::MulFOp::create(loopBuilder, loopLoc,
                                                      matrixValue, vectorValue);
                Value sum = arith::AddFOp::create(loopBuilder, loopLoc,
                                                  iterArgs.front(), product);
                scf::YieldOp::create(loopBuilder, loopLoc, sum);
              });

          Value waveSum = buildWaveReduction(
              builder, bodyLoc, partialReduction.getResult(0), waveSize);

          Value laneIsZero = arith::CmpIOp::create(
              builder, bodyLoc, arith::CmpIPredicate::eq, lane, zeroIndex);
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

    rewriter.setInsertionPointToEnd(&launch.getBody().front());
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
          Value nextRow =
              arith::AddIOp::create(builder, bodyLoc, row, oneIndex);
          Value rowStartValue =
              memref::LoadOp::create(builder, bodyLoc, op.getRowOffsets(), row);
          Value rowEndValue = memref::LoadOp::create(
              builder, bodyLoc, op.getRowOffsets(), nextRow);
          Value rowStart = castToIndex(builder, bodyLoc, rowStartValue);
          Value rowEnd = castToIndex(builder, bodyLoc, rowEndValue);
          Value firstPosition =
              arith::AddIOp::create(builder, bodyLoc, rowStart, thread);
          Value zero = arith::ConstantOp::create(
              builder, bodyLoc, builder.getZeroAttr(valueType));

          auto partialReduction = scf::ForOp::create(
              builder, bodyLoc, firstPosition, rowEnd, blockSizeValue,
              ValueRange{zero},
              [&](OpBuilder &loopBuilder, Location loopLoc, Value position,
                  ValueRange iterArgs) {
                Value columnValue = memref::LoadOp::create(
                    loopBuilder, loopLoc, op.getColumnIndices(), position);
                Value column = castToIndex(loopBuilder, loopLoc, columnValue);
                Value matrixValue = memref::LoadOp::create(
                    loopBuilder, loopLoc, op.getValues(), position);
                Value vectorValue = memref::LoadOp::create(
                    loopBuilder, loopLoc, op.getVector(), column);
                Value product = arith::MulFOp::create(loopBuilder, loopLoc,
                                                      matrixValue, vectorValue);
                Value sum = arith::AddFOp::create(loopBuilder, loopLoc,
                                                  iterArgs.front(), product);
                scf::YieldOp::create(loopBuilder, loopLoc, sum);
              });

          Value waveSum = buildWaveReduction(
              builder, bodyLoc, partialReduction.getResult(0), waveSize);
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
    Value requiredBlocks = arith::CeilDivUIOp::create(
        rewriter, loc, outputElementCount, blockSizeValue);
    Value gridSize =
        arith::MaxUIOp::create(rewriter, loc, requiredBlocks, oneIndex);

    gpu::LaunchOp launch =
        gpu::LaunchOp::create(rewriter, loc, gridSize, oneIndex, oneIndex,
                              blockSizeValue, oneIndex, oneIndex);
    rewriter.setInsertionPointToStart(&launch.getBody().front());

    Value elementBase = arith::MulIOp::create(
        rewriter, loc, launch.getBlockIds().x, launch.getBlockSize().x);
    Value element = arith::AddIOp::create(rewriter, loc, elementBase,
                                          launch.getThreadIds().x);
    Value elementIsActive = arith::CmpIOp::create(
        rewriter, loc, arith::CmpIPredicate::ult, element, outputElementCount);

    scf::IfOp::create(
        rewriter, loc, elementIsActive,
        [&](OpBuilder &builder, Location bodyLoc) {
          Value row =
              arith::DivUIOp::create(builder, bodyLoc, element, columnCount);
          Value outputColumn =
              arith::RemUIOp::create(builder, bodyLoc, element, columnCount);
          Value nextRow =
              arith::AddIOp::create(builder, bodyLoc, row, oneIndex);
          Value rowStartValue =
              memref::LoadOp::create(builder, bodyLoc, op.getRowOffsets(), row);
          Value rowEndValue = memref::LoadOp::create(
              builder, bodyLoc, op.getRowOffsets(), nextRow);
          Value rowStart = castToIndex(builder, bodyLoc, rowStartValue);
          Value rowEnd = castToIndex(builder, bodyLoc, rowEndValue);

          auto valueType =
              cast<MemRefType>(op.getValues().getType()).getElementType();
          Value zero = arith::ConstantOp::create(
              builder, bodyLoc, builder.getZeroAttr(valueType));
          auto reduction = scf::ForOp::create(
              builder, bodyLoc, rowStart, rowEnd, oneIndex, ValueRange{zero},
              [&](OpBuilder &loopBuilder, Location loopLoc, Value position,
                  ValueRange iterArgs) {
                Value reductionIndexValue = memref::LoadOp::create(
                    loopBuilder, loopLoc, op.getColumnIndices(), position);
                Value reductionIndex =
                    castToIndex(loopBuilder, loopLoc, reductionIndexValue);
                Value sparseValue = memref::LoadOp::create(
                    loopBuilder, loopLoc, op.getValues(), position);
                Value rhsValue = memref::LoadOp::create(
                    loopBuilder, loopLoc, op.getRhs(),
                    ValueRange{reductionIndex, outputColumn});
                Value product = arith::MulFOp::create(loopBuilder, loopLoc,
                                                      sparseValue, rhsValue);
                Value sum = arith::AddFOp::create(loopBuilder, loopLoc,
                                                  iterArgs.front(), product);
                scf::YieldOp::create(loopBuilder, loopLoc, sum);
              });
          memref::StoreOp::create(builder, bodyLoc, reduction.getResult(0),
                                  op.getOutput(),
                                  ValueRange{row, outputColumn});
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
    if (spmmMapping != "thread-per-output") {
      getOperation().emitError()
          << "unsupported SpMM mapping strategy '" << spmmMapping
          << "'; expected 'thread-per-output'";
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
    if (spmmMapping == "thread-per-output")
      patterns.add<ThreadPerOutputSpMMPattern>(&getContext(), spmmBlockSize);
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
