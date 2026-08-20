#include "sparsewave/Dialect/SparseWave/Transforms/Passes.h"

#include "sparsewave/Dialect/SparseWave/IR/SparseWaveOps.h"

#include "SparseGPUUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::sparsewave {
#define GEN_PASS_DEF_DECOMPOSEPOSITIONSPMV
#include "sparsewave/Dialect/SparseWave/Transforms/Passes.h.inc"

namespace {

class DecomposePositionSpMVPattern : public OpRewritePattern<SpMVOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(SpMVOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value nonzeroCount =
        memref::DimOp::create(rewriter, loc, op.getValues(), zero);
    auto reduction = PositionReduceOp::create(rewriter, loc, zero, nonzeroCount,
                                              op.getOutput(), "sum");
    Block *body =
        rewriter.createBlock(&reduction.getBody(), reduction.getBody().end(),
                             {rewriter.getIndexType()}, {loc});
    rewriter.setInsertionPointToStart(body);
    CSRSpMVProduct element = buildCSRSpMVProduct(
        rewriter, loc, op.getRowOffsets(), op.getColumnIndices(),
        op.getValues(), op.getVector(), body->getArgument(0));
    YieldOp::create(rewriter, loc, ValueRange{element.row, element.product});
    rewriter.eraseOp(op);
    return success();
  }
};

class DecomposePositionSpMV
    : public impl::DecomposePositionSpMVBase<DecomposePositionSpMV> {
public:
  using impl::DecomposePositionSpMVBase<
      DecomposePositionSpMV>::DecomposePositionSpMVBase;

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<DecomposePositionSpMVPattern>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
} // namespace mlir::sparsewave
