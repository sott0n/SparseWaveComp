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

std::optional<CSRRowAtPositionOp>
matchCSRRowRecovery(PositionReduceOp reduction) {
  Block &body = reduction.getBody().front();
  auto yield = cast<YieldOp>(body.getTerminator());
  for (Operation &operation : body.without_terminator()) {
    auto recovery = dyn_cast<CSRRowAtPositionOp>(operation);
    if (recovery && recovery.getPosition() == body.getArgument(0) &&
        yield.getResults()[0] == recovery.getRow())
      return recovery;
  }
  return std::nullopt;
}

KeyedContribution cloneContributionBodyWithCSRRow(OpBuilder &builder,
                                                  PositionReduceOp reduction,
                                                  Value position,
                                                  CSRRowAtPositionOp recovery,
                                                  Value row) {
  Block &source = reduction.getBody().front();
  IRMapping mapping;
  mapping.map(source.getArgument(0), position);
  mapping.map(recovery.getRow(), row);
  for (Operation &operation : source.without_terminator()) {
    if (&operation == recovery.getOperation())
      continue;
    builder.clone(operation, mapping);
  }
  auto yield = cast<YieldOp>(source.getTerminator());
  return {mapping.lookup(yield.getResults()[0]),
          mapping.lookup(yield.getResults()[1])};
}

Value buildNextCSRRow(OpBuilder &builder, Location loc, Value rowOffsets,
                      Value row, Value position) {
  Value one = arith::ConstantIndexOp::create(builder, loc, 1);
  auto advance = scf::WhileOp::create(builder, loc,
                                      TypeRange{builder.getIndexType()}, row);
  SmallVector<Location> argumentLocations(1, loc);
  Block *before = builder.createBlock(
      &advance.getBefore(), {}, advance.getResultTypes(), argumentLocations);
  builder.setInsertionPointToStart(before);
  Value currentRow = before->getArgument(0);
  Value nextRow = arith::AddIOp::create(builder, loc, currentRow, one);
  Value boundaryValue =
      memref::LoadOp::create(builder, loc, rowOffsets, nextRow);
  Value boundary = castToIndex(builder, loc, boundaryValue);
  Value crossesBoundary = arith::CmpIOp::create(
      builder, loc, arith::CmpIPredicate::ule, boundary, position);
  scf::ConditionOp::create(builder, loc, crossesBoundary, currentRow);

  Block *after = builder.createBlock(
      &advance.getAfter(), {}, advance.getResultTypes(), argumentLocations);
  builder.setInsertionPointToStart(after);
  currentRow = after->getArgument(0);
  Value advancedRow = arith::AddIOp::create(builder, loc, currentRow, one);
  scf::YieldOp::create(builder, loc, advancedRow);

  builder.setInsertionPointAfter(advance);
  return advance.getResult(0);
}

class ThreadPositionReducePattern : public OpRewritePattern<PositionReduceOp> {
public:
  ThreadPositionReducePattern(MLIRContext *context, int64_t blockSize,
                              int64_t chunkSize)
      : OpRewritePattern<PositionReduceOp>(context), blockSize(blockSize),
        chunkSize(chunkSize) {}

  LogicalResult matchAndRewrite(PositionReduceOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value zeroIndex = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value chunkSizeValue =
        arith::ConstantIndexOp::create(rewriter, loc, chunkSize);
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
    Value workerCount = arith::CeilDivUIOp::create(rewriter, loc, positionCount,
                                                   chunkSizeValue);
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
    auto chunk =
        PositionForOp::create(rewriter, loc, op.getLower(), op.getUpper(),
                              worker, rewriter.getI64IntegerAttr(chunkSize));
    Block *chunkBody = rewriter.createBlock(
        &chunk.getBody(), chunk.getBody().end(),
        {rewriter.getIndexType(), rewriter.getIndexType()}, {loc, loc});
    rewriter.setInsertionPointToStart(chunkBody);
    KeyedContribution contribution =
        cloneContributionBody(rewriter, op, chunkBody->getArgument(0));
    memref::AtomicRMWOp::create(rewriter, loc, arith::AtomicRMWKind::addf,
                                contribution.value, op.getOutput(),
                                ValueRange{contribution.key});
    YieldOp::create(rewriter, loc);
    rewriter.setInsertionPointAfter(chunk);
    YieldOp::create(rewriter, loc);
    rewriter.eraseOp(op);
    return success();
  }

private:
  int64_t blockSize;
  int64_t chunkSize;
};

class ThreadSegmentedPositionReducePattern
    : public OpRewritePattern<PositionReduceOp> {
public:
  ThreadSegmentedPositionReducePattern(MLIRContext *context, int64_t blockSize,
                                       int64_t chunkSize)
      : OpRewritePattern<PositionReduceOp>(context), blockSize(blockSize),
        chunkSize(chunkSize) {}

  LogicalResult matchAndRewrite(PositionReduceOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value zeroIndex = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value oneIndex = arith::ConstantIndexOp::create(rewriter, loc, 1);
    Value chunkSizeValue =
        arith::ConstantIndexOp::create(rewriter, loc, chunkSize);
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
    Value workerCount = arith::CeilDivUIOp::create(rewriter, loc, positionCount,
                                                   chunkSizeValue);
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
    Value chunkOffset =
        arith::MulIOp::create(rewriter, loc, worker, chunkSizeValue);
    Value begin =
        arith::AddIOp::create(rewriter, loc, op.getLower(), chunkOffset);
    Value remaining =
        arith::SubIOp::create(rewriter, loc, op.getUpper(), begin);
    Value boundedSize =
        arith::MinUIOp::create(rewriter, loc, remaining, chunkSizeValue);
    Value end = arith::AddIOp::create(rewriter, loc, begin, boundedSize);

    std::optional<CSRRowAtPositionOp> csrRecovery = matchCSRRowRecovery(op);
    KeyedContribution first;
    if (csrRecovery) {
      Value firstRow =
          CSRRowAtPositionOp::create(rewriter, loc, rewriter.getIndexType(),
                                     csrRecovery->getRowOffsets(), begin);
      first = cloneContributionBodyWithCSRRow(rewriter, op, begin, *csrRecovery,
                                              firstRow);
    } else {
      first = cloneContributionBody(rewriter, op, begin);
    }

    Value nextPosition = arith::AddIOp::create(rewriter, loc, begin, oneIndex);
    auto reduction = scf::ForOp::create(
        rewriter, loc, nextPosition, end, oneIndex,
        ValueRange{first.key, first.value},
        [&](OpBuilder &builder, Location bodyLoc, Value position,
            ValueRange iterArgs) {
          Value currentKey = iterArgs[0];
          Value currentValue = iterArgs[1];
          KeyedContribution contribution;
          if (csrRecovery) {
            Value row =
                buildNextCSRRow(builder, bodyLoc, csrRecovery->getRowOffsets(),
                                currentKey, position);
            contribution = cloneContributionBodyWithCSRRow(
                builder, op, position, *csrRecovery, row);
          } else {
            contribution = cloneContributionBody(builder, op, position);
          }
          Value sameKey =
              arith::CmpIOp::create(builder, bodyLoc, arith::CmpIPredicate::eq,
                                    currentKey, contribution.key);
          Value combined = arith::AddFOp::create(builder, bodyLoc, currentValue,
                                                 contribution.value);
          scf::IfOp::create(
              builder, bodyLoc, sameKey,
              [&](OpBuilder &thenBuilder, Location thenLoc) {
                scf::YieldOp::create(thenBuilder, thenLoc);
              },
              [&](OpBuilder &elseBuilder, Location elseLoc) {
                memref::AtomicRMWOp::create(
                    elseBuilder, elseLoc, arith::AtomicRMWKind::addf,
                    currentValue, op.getOutput(), ValueRange{currentKey});
                scf::YieldOp::create(elseBuilder, elseLoc);
              });
          Value nextValue = arith::SelectOp::create(
              builder, bodyLoc, sameKey, combined, contribution.value);
          scf::YieldOp::create(builder, bodyLoc,
                               ValueRange{contribution.key, nextValue});
        });
    memref::AtomicRMWOp::create(rewriter, loc, arith::AtomicRMWKind::addf,
                                reduction.getResult(1), op.getOutput(),
                                ValueRange{reduction.getResult(0)});
    YieldOp::create(rewriter, loc);
    rewriter.eraseOp(op);
    return success();
  }

private:
  int64_t blockSize;
  int64_t chunkSize;
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
    if (threadChunkSize < 1) {
      getOperation().emitError()
          << "thread position chunk size must be positive, but got "
          << threadChunkSize.getValue();
      signalPassFailure();
      return;
    }
    if (threadReduction != "atomic" && threadReduction != "segmented") {
      getOperation().emitError()
          << "unsupported thread position reduction '" << threadReduction
          << "'; expected 'atomic' or 'segmented'";
      signalPassFailure();
      return;
    }
    if (mapping == "wave" && threadChunkSize != 1) {
      getOperation().emitError()
          << "thread position chunk size applies only to thread mapping";
      signalPassFailure();
      return;
    }
    if (mapping == "wave" && threadReduction != "atomic") {
      getOperation().emitError()
          << "thread position reduction applies only to thread mapping";
      signalPassFailure();
      return;
    }

    RewritePatternSet patterns(&getContext());
    if (mapping == "thread") {
      if (threadReduction == "segmented")
        patterns.add<ThreadSegmentedPositionReducePattern>(
            &getContext(), blockSize, threadChunkSize);
      else
        patterns.add<ThreadPositionReducePattern>(&getContext(), blockSize,
                                                  threadChunkSize);
    } else {
      patterns.add<WavePositionReducePattern>(&getContext(), blockSize,
                                              waveSize);
    }
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
} // namespace mlir::sparsewave
