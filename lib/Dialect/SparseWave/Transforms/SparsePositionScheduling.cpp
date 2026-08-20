#include "sparsewave/Dialect/SparseWave/Transforms/Passes.h"

#include "sparsewave/Dialect/SparseWave/IR/SparseWaveOps.h"

#include "SparseGPUUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::sparsewave {
#define GEN_PASS_DEF_SCHEDULESPARSEWAVEPOSITION
#include "sparsewave/Dialect/SparseWave/Transforms/Passes.h.inc"

namespace {

struct KeyedContribution {
  Value key;
  Value value;
};

PositionParallelOp buildOutputInitialization(PatternRewriter &rewriter,
                                             Location loc, Value output,
                                             Value outputSize, Value zero,
                                             int64_t blockSize) {
  auto parallel =
      PositionParallelOp::create(rewriter, loc, outputSize, "thread",
                                 rewriter.getI64IntegerAttr(blockSize));
  Block *body =
      rewriter.createBlock(&parallel.getBody(), parallel.getBody().end(),
                           {rewriter.getIndexType(), rewriter.getIndexType(),
                            rewriter.getIndexType()},
                           {loc, loc, loc});
  rewriter.setInsertionPointToStart(body);
  memref::StoreOp::create(rewriter, loc, zero, output, body->getArgument(0));
  YieldOp::create(rewriter, loc);
  return parallel;
}

KeyedContribution cloneContributionBody(OpBuilder &builder,
                                        PositionReduceOp reduction,
                                        Value position) {
  Block &source = reduction.getBody().front();
  IRMapping mapping;
  mapping.map(source.getArgument(0), position);
  for (Operation &operation : source.without_terminator())
    builder.clone(operation, mapping);
  auto yield = cast<YieldOp>(source.getTerminator());
  return {mapping.lookup(yield.getResults()[0]),
          mapping.lookup(yield.getResults()[1])};
}

class ThreadPositionReducePattern : public OpRewritePattern<PositionReduceOp> {
public:
  ThreadPositionReducePattern(MLIRContext *context, int64_t blockSize)
      : OpRewritePattern<PositionReduceOp>(context), blockSize(blockSize) {}

  LogicalResult matchAndRewrite(PositionReduceOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value zeroIndex = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value oneIndex = arith::ConstantIndexOp::create(rewriter, loc, 1);
    Value blockSizeValue =
        arith::ConstantIndexOp::create(rewriter, loc, blockSize);
    Value outputSize =
        memref::DimOp::create(rewriter, loc, op.getOutput(), zeroIndex);
    Type valueType =
        cast<MemRefType>(op.getOutput().getType()).getElementType();
    Value zero = arith::ConstantOp::create(rewriter, loc,
                                           rewriter.getZeroAttr(valueType));
    PositionParallelOp initialization = buildOutputInitialization(
        rewriter, loc, op.getOutput(), outputSize, zero, blockSize);

    rewriter.setInsertionPointAfter(initialization);
    Value positionCount =
        arith::SubIOp::create(rewriter, loc, op.getUpper(), op.getLower());
    Value requiredBlocks = arith::CeilDivUIOp::create(
        rewriter, loc, positionCount, blockSizeValue);
    Value gridSize =
        arith::MaxUIOp::create(rewriter, loc, requiredBlocks, oneIndex);
    Value workerCount =
        arith::MulIOp::create(rewriter, loc, gridSize, blockSizeValue);
    auto parallel =
        PositionParallelOp::create(rewriter, loc, workerCount, "thread",
                                   rewriter.getI64IntegerAttr(blockSize));
    Block *body =
        rewriter.createBlock(&parallel.getBody(), parallel.getBody().end(),
                             {rewriter.getIndexType(), rewriter.getIndexType(),
                              rewriter.getIndexType()},
                             {loc, loc, loc});
    rewriter.setInsertionPointToStart(body);
    Value worker = body->getArgument(0);
    auto partition = PositionSpaceOp::create(
        rewriter, loc, rewriter.getIndexType(), rewriter.getIndexType(),
        op.getLower(), op.getUpper(), worker, workerCount);
    scf::ForOp::create(
        rewriter, loc, partition.getBegin(), partition.getEnd(), oneIndex,
        ValueRange{},
        [&](OpBuilder &builder, Location bodyLoc, Value position, ValueRange) {
          KeyedContribution contribution =
              cloneContributionBody(builder, op, position);
          memref::AtomicRMWOp::create(
              builder, bodyLoc, arith::AtomicRMWKind::addf, contribution.value,
              op.getOutput(), ValueRange{contribution.key});
          scf::YieldOp::create(builder, bodyLoc);
        });
    YieldOp::create(rewriter, loc);
    rewriter.eraseOp(op);
    return success();
  }

private:
  int64_t blockSize;
};

class WavePositionReducePattern : public OpRewritePattern<PositionReduceOp> {
public:
  WavePositionReducePattern(MLIRContext *context, int64_t blockSize,
                            int64_t waveSize)
      : OpRewritePattern<PositionReduceOp>(context), blockSize(blockSize),
        waveSize(waveSize) {}

  LogicalResult matchAndRewrite(PositionReduceOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value zeroIndex = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value oneIndex = arith::ConstantIndexOp::create(rewriter, loc, 1);
    Value blockSizeValue =
        arith::ConstantIndexOp::create(rewriter, loc, blockSize);
    Value wavesPerBlockValue =
        arith::ConstantIndexOp::create(rewriter, loc, blockSize / waveSize);
    Value outputSize =
        memref::DimOp::create(rewriter, loc, op.getOutput(), zeroIndex);
    Type valueType =
        cast<MemRefType>(op.getOutput().getType()).getElementType();
    Value zero = arith::ConstantOp::create(rewriter, loc,
                                           rewriter.getZeroAttr(valueType));
    PositionParallelOp initialization = buildOutputInitialization(
        rewriter, loc, op.getOutput(), outputSize, zero, blockSize);

    rewriter.setInsertionPointAfter(initialization);
    Value positionCount =
        arith::SubIOp::create(rewriter, loc, op.getUpper(), op.getLower());
    Value requiredBlocks = arith::CeilDivUIOp::create(
        rewriter, loc, positionCount, blockSizeValue);
    Value gridSize =
        arith::MaxUIOp::create(rewriter, loc, requiredBlocks, oneIndex);
    Value waveCount =
        arith::MulIOp::create(rewriter, loc, gridSize, wavesPerBlockValue);
    auto parallel =
        PositionParallelOp::create(rewriter, loc, waveCount, "wave",
                                   rewriter.getI64IntegerAttr(blockSize));
    Block *body =
        rewriter.createBlock(&parallel.getBody(), parallel.getBody().end(),
                             {rewriter.getIndexType(), rewriter.getIndexType(),
                              rewriter.getIndexType()},
                             {loc, loc, loc});
    rewriter.setInsertionPointToStart(body);
    Value wave = body->getArgument(0);
    Value lane = body->getArgument(1);
    auto partition = PositionSpaceOp::create(
        rewriter, loc, rewriter.getIndexType(), rewriter.getIndexType(),
        op.getLower(), op.getUpper(), wave, waveCount);
    Value position =
        arith::AddIOp::create(rewriter, loc, partition.getBegin(), lane);
    Value active = arith::CmpIOp::create(
        rewriter, loc, arith::CmpIPredicate::ult, position, partition.getEnd());
    auto entry = scf::IfOp::create(
        rewriter, loc, active,
        [&](OpBuilder &builder, Location bodyLoc) {
          KeyedContribution contribution =
              cloneContributionBody(builder, op, position);
          scf::YieldOp::create(
              builder, bodyLoc,
              ValueRange{contribution.key, contribution.value});
        },
        [&](OpBuilder &builder, Location bodyLoc) {
          scf::YieldOp::create(builder, bodyLoc, ValueRange{zeroIndex, zero});
        });
    Value key = entry.getResult(0);
    Value keyI64 =
        arith::IndexCastOp::create(rewriter, loc, rewriter.getI64Type(), key);
    WaveSegmentedReduction reduction = buildWavePrefixSegmentedReduction(
        rewriter, loc, keyI64, entry.getResult(1), active, waveSize);
    scf::IfOp::create(rewriter, loc, reduction.segmentEnd,
                      [&](OpBuilder &builder, Location bodyLoc) {
                        memref::AtomicRMWOp::create(
                            builder, bodyLoc, arith::AtomicRMWKind::addf,
                            reduction.inclusiveValue, op.getOutput(),
                            ValueRange{key});
                        scf::YieldOp::create(builder, bodyLoc);
                      });
    YieldOp::create(rewriter, loc);
    rewriter.eraseOp(op);
    return success();
  }

private:
  int64_t blockSize;
  int64_t waveSize;
};

class ScheduleSparseWavePosition
    : public impl::ScheduleSparseWavePositionBase<ScheduleSparseWavePosition> {
public:
  using impl::ScheduleSparseWavePositionBase<
      ScheduleSparseWavePosition>::ScheduleSparseWavePositionBase;

  void runOnOperation() override {
    if (mapping != "thread" && mapping != "wave") {
      getOperation().emitError() << "unsupported position mapping '" << mapping
                                 << "'; expected 'thread' or 'wave'";
      signalPassFailure();
      return;
    }
    if (blockSize < 1 || blockSize > 1024) {
      getOperation().emitError()
          << "position block size must be between 1 and 1024, but got "
          << blockSize.getValue();
      signalPassFailure();
      return;
    }
    if (mapping == "wave" && waveSize != 32) {
      getOperation().emitError()
          << "wave position mapping currently requires Wave32, but got "
          << waveSize.getValue();
      signalPassFailure();
      return;
    }
    if (mapping == "wave" && blockSize % waveSize != 0) {
      getOperation().emitError()
          << "wave position mapping requires the block size to be a multiple "
             "of "
          << waveSize.getValue() << ", but got " << blockSize.getValue();
      signalPassFailure();
      return;
    }

    RewritePatternSet patterns(&getContext());
    if (mapping == "thread")
      patterns.add<ThreadPositionReducePattern>(&getContext(), blockSize);
    else
      patterns.add<WavePositionReducePattern>(&getContext(), blockSize,
                                              waveSize);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
} // namespace mlir::sparsewave
