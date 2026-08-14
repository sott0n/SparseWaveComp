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

class LowerCSRCoordinatesPattern : public OpRewritePattern<CSRCoordinatesOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(CSRCoordinatesOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value one = arith::ConstantIndexOp::create(rewriter, loc, 1);
    Value two = arith::ConstantIndexOp::create(rewriter, loc, 2);
    Value rowOffsetsSize =
        memref::DimOp::create(rewriter, loc, op.getRowOffsets(), zero);

    // Search [1, rowOffsetsSize) for the first offset greater than position.
    // Starting at one makes the final predecessor a valid CSR row index.
    auto search = scf::WhileOp::create(
        rewriter, loc,
        TypeRange{rewriter.getIndexType(), rewriter.getIndexType()},
        ValueRange{one, rowOffsetsSize});

    OpBuilder::InsertionGuard guard(rewriter);
    SmallVector<Location> argumentLocations(2, loc);
    Block *before = rewriter.createBlock(
        &search.getBefore(), {}, search.getResultTypes(), argumentLocations);
    rewriter.setInsertionPointToStart(before);
    Value lower = before->getArgument(0);
    Value upper = before->getArgument(1);
    Value continueSearch = arith::CmpIOp::create(
        rewriter, loc, arith::CmpIPredicate::ult, lower, upper);
    scf::ConditionOp::create(rewriter, loc, continueSearch,
                             before->getArguments());

    Block *after = rewriter.createBlock(
        &search.getAfter(), {}, search.getResultTypes(), argumentLocations);
    rewriter.setInsertionPointToStart(after);
    lower = after->getArgument(0);
    upper = after->getArgument(1);
    Value distance = arith::SubIOp::create(rewriter, loc, upper, lower);
    Value midpoint = arith::AddIOp::create(
        rewriter, loc, lower,
        arith::DivUIOp::create(rewriter, loc, distance, two));
    Value offsetValue = memref::LoadOp::create(
        rewriter, loc, op.getRowOffsets(), ValueRange{midpoint});
    Value offset = castToIndex(rewriter, loc, offsetValue);
    Value offsetAtOrBeforePosition = arith::CmpIOp::create(
        rewriter, loc, arith::CmpIPredicate::ule, offset, op.getPosition());
    Value nextLower = arith::SelectOp::create(
        rewriter, loc, offsetAtOrBeforePosition,
        arith::AddIOp::create(rewriter, loc, midpoint, one), lower);
    Value nextUpper = arith::SelectOp::create(
        rewriter, loc, offsetAtOrBeforePosition, upper, midpoint);
    scf::YieldOp::create(rewriter, loc, ValueRange{nextLower, nextUpper});

    rewriter.setInsertionPointAfter(search);
    Value row = arith::SubIOp::create(rewriter, loc, search.getResult(0), one);
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
    patterns.add<LowerPositionSplitPattern, LowerPositionSpacePattern,
                 LowerCSRCoordinatesPattern>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
} // namespace mlir::sparsewave
