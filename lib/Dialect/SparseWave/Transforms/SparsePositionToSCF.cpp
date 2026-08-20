#include "sparsewave/Dialect/SparseWave/Transforms/Passes.h"

#include "sparsewave/Dialect/SparseWave/IR/SparseWaveOps.h"

#include "SparseLoweringUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::sparsewave {
#define GEN_PASS_DEF_LOWERSPARSEWAVEPOSITIONSPACE
#include "sparsewave/Dialect/SparseWave/Transforms/Passes.h.inc"

namespace {

class LowerPositionSplitPattern : public OpRewritePattern<PositionSplitOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(PositionSplitOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value factor = arith::ConstantIndexOp::create(rewriter, loc,
                                                  op.getFactorAttr().getInt());
    Value outer =
        arith::DivUIOp::create(rewriter, loc, op.getPosition(), factor);
    Value inner =
        arith::RemUIOp::create(rewriter, loc, op.getPosition(), factor);
    rewriter.replaceOp(op, ValueRange{outer, inner});
    return success();
  }
};

class LowerPositionForPattern : public OpRewritePattern<PositionForOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(PositionForOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value one = arith::ConstantIndexOp::create(rewriter, loc, 1);
    Value factor = arith::ConstantIndexOp::create(rewriter, loc,
                                                  op.getFactorAttr().getInt());
    Value span =
        arith::SubIOp::create(rewriter, loc, op.getUpper(), op.getLower());

    // Compute ceil(span / factor) without forming span + factor - 1, which
    // could overflow at the upper end of the index range.
    Value fullChunks = arith::DivUIOp::create(rewriter, loc, span, factor);
    Value remainder = arith::RemUIOp::create(rewriter, loc, span, factor);
    Value hasRemainder = arith::CmpIOp::create(
        rewriter, loc, arith::CmpIPredicate::ne, remainder, zero);
    Value extraChunk =
        arith::SelectOp::create(rewriter, loc, hasRemainder, one, zero);
    Value chunkCount =
        arith::AddIOp::create(rewriter, loc, fullChunks, extraChunk);
    Value active = arith::CmpIOp::create(
        rewriter, loc, arith::CmpIPredicate::ult, op.getWorkerId(), chunkCount);

    // Substitute zero before multiplication for an inactive worker. This
    // avoids overflowing worker_id * factor even when a rounded-up launch
    // supplies an arbitrarily large extra worker ID.
    Value safeWorker =
        arith::SelectOp::create(rewriter, loc, active, op.getWorkerId(), zero);
    Value chunkOffset =
        arith::MulIOp::create(rewriter, loc, safeWorker, factor);
    Value begin =
        arith::AddIOp::create(rewriter, loc, op.getLower(), chunkOffset);
    Value remaining =
        arith::SubIOp::create(rewriter, loc, op.getUpper(), begin);
    Value boundedSize =
        arith::MinUIOp::create(rewriter, loc, remaining, factor);
    Value size =
        arith::SelectOp::create(rewriter, loc, active, boundedSize, zero);
    Value end = arith::AddIOp::create(rewriter, loc, begin, size);

    auto loop = scf::ForOp::create(rewriter, loc, begin, end, one);
    Block *loopBody = loop.getBody();
    rewriter.setInsertionPointToStart(loopBody);
    Value inner =
        arith::SubIOp::create(rewriter, loc, loop.getInductionVar(), begin);

    auto sparseYield = cast<YieldOp>(op.getBody().front().getTerminator());
    rewriter.setInsertionPoint(sparseYield);
    scf::YieldOp::create(rewriter, sparseYield.getLoc());
    rewriter.eraseOp(sparseYield);
    rewriter.eraseOp(loopBody->getTerminator());
    rewriter.mergeBlocks(&op.getBody().front(), loopBody,
                         ValueRange{loop.getInductionVar(), inner});
    rewriter.eraseOp(op);
    return success();
  }
};

class LowerPositionReorderPattern : public OpRewritePattern<PositionReorderOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(PositionReorderOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value one = arith::ConstantIndexOp::create(rewriter, loc, 1);
    SmallVector<Value> inductionVariables(op.getLower().size());
    Block *innermostBody = nullptr;

    // Build loops from outermost to innermost according to the scheduling
    // permutation. Keep induction variables indexed by logical axis so the
    // region interface does not change when execution order changes.
    for (int64_t axis : op.getOrder()) {
      auto loop = scf::ForOp::create(rewriter, loc, op.getLower()[axis],
                                     op.getUpper()[axis], one);
      inductionVariables[axis] = loop.getInductionVar();
      innermostBody = loop.getBody();
      rewriter.setInsertionPointToStart(innermostBody);
    }

    auto sparseYield = cast<YieldOp>(op.getBody().front().getTerminator());
    rewriter.setInsertionPoint(sparseYield);
    scf::YieldOp::create(rewriter, sparseYield.getLoc());
    rewriter.eraseOp(sparseYield);
    rewriter.eraseOp(innermostBody->getTerminator());
    rewriter.mergeBlocks(&op.getBody().front(), innermostBody,
                         inductionVariables);
    rewriter.eraseOp(op);
    return success();
  }
};

class LowerPositionCollapsePattern
    : public OpRewritePattern<PositionCollapseOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(PositionCollapseOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value one = arith::ConstantIndexOp::create(rewriter, loc, 1);
    Value nonEmpty =
        arith::ConstantOp::create(rewriter, loc, rewriter.getBoolAttr(true));
    Value remaining = op.getWorkerId();
    SmallVector<Value> coordinates(op.getLower().size());
    ArrayRef<int64_t> order = op.getOrder();

    // Recover coordinates from innermost to outermost. Repeated division
    // avoids forming the product of all extents, which could overflow index.
    for (size_t orderIndex = order.size(); orderIndex > 0; --orderIndex) {
      int64_t axis = order[orderIndex - 1];
      Value extent = arith::SubIOp::create(rewriter, loc, op.getUpper()[axis],
                                           op.getLower()[axis]);
      Value hasExtent = arith::CmpIOp::create(
          rewriter, loc, arith::CmpIPredicate::ne, extent, zero);
      nonEmpty = arith::AndIOp::create(rewriter, loc, nonEmpty, hasExtent);

      // Use one as a safe divisor for an empty axis. The accumulated
      // nonEmpty predicate prevents the recovered coordinates from escaping.
      Value safeExtent =
          arith::SelectOp::create(rewriter, loc, hasExtent, extent, one);
      Value offset =
          arith::RemUIOp::create(rewriter, loc, remaining, safeExtent);
      remaining = arith::DivUIOp::create(rewriter, loc, remaining, safeExtent);
      coordinates[axis] =
          arith::AddIOp::create(rewriter, loc, op.getLower()[axis], offset);
    }

    Value workerInRange = arith::CmpIOp::create(
        rewriter, loc, arith::CmpIPredicate::eq, remaining, zero);
    Value active =
        arith::AndIOp::create(rewriter, loc, nonEmpty, workerInRange);
    auto guard = scf::IfOp::create(rewriter, loc, active,
                                   /*withElseRegion=*/false);
    Block *thenBody = &guard.getThenRegion().front();

    auto sparseYield = cast<YieldOp>(op.getBody().front().getTerminator());
    rewriter.setInsertionPoint(sparseYield);
    scf::YieldOp::create(rewriter, sparseYield.getLoc());
    rewriter.eraseOp(sparseYield);
    rewriter.eraseOp(thenBody->getTerminator());
    rewriter.mergeBlocks(&op.getBody().front(), thenBody, coordinates);
    rewriter.eraseOp(op);
    return success();
  }
};

class LowerPositionSpacePattern : public OpRewritePattern<PositionSpaceOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(PositionSpaceOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value one = arith::ConstantIndexOp::create(rewriter, loc, 1);
    Value span =
        arith::SubIOp::create(rewriter, loc, op.getUpper(), op.getLower());
    Value base =
        arith::DivUIOp::create(rewriter, loc, span, op.getWorkerCount());
    Value remainder =
        arith::RemUIOp::create(rewriter, loc, span, op.getWorkerCount());
    Value receivesRemainder = arith::CmpIOp::create(
        rewriter, loc, arith::CmpIPredicate::ult, op.getWorkerId(), remainder);
    Value positionsBefore = arith::SelectOp::create(
        rewriter, loc, receivesRemainder, op.getWorkerId(), remainder);
    Value baseOffset =
        arith::MulIOp::create(rewriter, loc, op.getWorkerId(), base);
    Value begin = arith::AddIOp::create(
        rewriter, loc, op.getLower(),
        arith::AddIOp::create(rewriter, loc, baseOffset, positionsBefore));
    Value extra =
        arith::SelectOp::create(rewriter, loc, receivesRemainder, one, zero);
    Value end = arith::AddIOp::create(
        rewriter, loc, begin,
        arith::AddIOp::create(rewriter, loc, base, extra));

    rewriter.replaceOp(op, ValueRange{begin, end});
    return success();
  }
};

Value buildCSRRowSearch(OpBuilder &builder, Location loc, Value rowOffsets,
                        Value position) {
  Value zero = arith::ConstantIndexOp::create(builder, loc, 0);
  Value one = arith::ConstantIndexOp::create(builder, loc, 1);
  Value two = arith::ConstantIndexOp::create(builder, loc, 2);
  Value rowOffsetsSize = memref::DimOp::create(builder, loc, rowOffsets, zero);

  // Search [1, rowOffsetsSize) for the first offset greater than position.
  // Starting at one makes the final predecessor a valid CSR row index.
  auto search = scf::WhileOp::create(
      builder, loc, TypeRange{builder.getIndexType(), builder.getIndexType()},
      ValueRange{one, rowOffsetsSize});

  SmallVector<Location> argumentLocations(2, loc);
  Block *before = builder.createBlock(
      &search.getBefore(), {}, search.getResultTypes(), argumentLocations);
  builder.setInsertionPointToStart(before);
  Value lower = before->getArgument(0);
  Value upper = before->getArgument(1);
  Value continueSearch = arith::CmpIOp::create(
      builder, loc, arith::CmpIPredicate::ult, lower, upper);
  scf::ConditionOp::create(builder, loc, continueSearch,
                           before->getArguments());

  Block *after = builder.createBlock(
      &search.getAfter(), {}, search.getResultTypes(), argumentLocations);
  builder.setInsertionPointToStart(after);
  lower = after->getArgument(0);
  upper = after->getArgument(1);
  Value distance = arith::SubIOp::create(builder, loc, upper, lower);
  Value midpoint = arith::AddIOp::create(
      builder, loc, lower, arith::DivUIOp::create(builder, loc, distance, two));
  Value offsetValue =
      memref::LoadOp::create(builder, loc, rowOffsets, ValueRange{midpoint});
  Value offset = castToIndex(builder, loc, offsetValue);
  Value offsetAtOrBeforePosition = arith::CmpIOp::create(
      builder, loc, arith::CmpIPredicate::ule, offset, position);
  Value nextLower = arith::SelectOp::create(
      builder, loc, offsetAtOrBeforePosition,
      arith::AddIOp::create(builder, loc, midpoint, one), lower);
  Value nextUpper = arith::SelectOp::create(
      builder, loc, offsetAtOrBeforePosition, upper, midpoint);
  scf::YieldOp::create(builder, loc, ValueRange{nextLower, nextUpper});

  builder.setInsertionPointAfter(search);
  return arith::SubIOp::create(builder, loc, search.getResult(0), one);
}

class LowerCSRRowAtPositionPattern
    : public OpRewritePattern<CSRRowAtPositionOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(CSRRowAtPositionOp op,
                                PatternRewriter &rewriter) const override {
    Value row = buildCSRRowSearch(rewriter, op.getLoc(), op.getRowOffsets(),
                                  op.getPosition());
    rewriter.replaceOp(op, row);
    return success();
  }
};

class LowerCSRCoordinatesPattern : public OpRewritePattern<CSRCoordinatesOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(CSRCoordinatesOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value row =
        buildCSRRowSearch(rewriter, loc, op.getRowOffsets(), op.getPosition());
    Value columnValue = memref::LoadOp::create(
        rewriter, loc, op.getColumnIndices(), ValueRange{op.getPosition()});
    Value column = castToIndex(rewriter, loc, columnValue);
    rewriter.replaceOp(op, ValueRange{row, column});
    return success();
  }
};

class LowerSparseWavePositionSpace
    : public impl::LowerSparseWavePositionSpaceBase<
          LowerSparseWavePositionSpace> {
public:
  using impl::LowerSparseWavePositionSpaceBase<
      LowerSparseWavePositionSpace>::LowerSparseWavePositionSpaceBase;

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<LowerPositionSplitPattern, LowerPositionForPattern,
                 LowerPositionReorderPattern, LowerPositionCollapsePattern,
                 LowerPositionSpacePattern, LowerCSRRowAtPositionPattern,
                 LowerCSRCoordinatesPattern>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
} // namespace mlir::sparsewave
