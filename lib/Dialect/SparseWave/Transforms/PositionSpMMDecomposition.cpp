#include "sparsewave/Dialect/SparseWave/Transforms/Passes.h"

#include "sparsewave/Dialect/SparseWave/IR/SparseWaveOps.h"

#include "SparseGPUUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::sparsewave {
#define GEN_PASS_DEF_DECOMPOSEPOSITIONSPMM
#include "sparsewave/Dialect/SparseWave/Transforms/Passes.h.inc"

namespace {

class DecomposePositionSpMMPattern : public OpRewritePattern<SpMMOp> {
public:
  DecomposePositionSpMMPattern(MLIRContext *context, StringRef iterationOrder)
      : OpRewritePattern<SpMMOp>(context),
        order(iterationOrder == "rhs-major" ? SmallVector<int64_t>{1, 0}
                                            : SmallVector<int64_t>{0, 1}) {}

  LogicalResult matchAndRewrite(SpMMOp op,
                                PatternRewriter &rewriter) const override {
    MemRefType outputType = op.getOutput().getType();
    if (!outputType.getLayout().isIdentity())
      return rewriter.notifyMatchFailure(
          op, "position-space SpMM requires an identity-layout output");

    Location loc = op.getLoc();
    Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value one = arith::ConstantIndexOp::create(rewriter, loc, 1);
    Value nonzeroCount =
        memref::DimOp::create(rewriter, loc, op.getValues(), zero);
    Value rhsColumnCount =
        memref::DimOp::create(rewriter, loc, op.getRhs(), one);
    SmallVector<ReassociationIndices> reassociation{{0, 1}};
    Value flattenedOutput = memref::CollapseShapeOp::create(
        rewriter, loc, op.getOutput(), reassociation);
    auto reduction = PositionReduceOp::create(
        rewriter, loc, ValueRange{zero, zero},
        ValueRange{nonzeroCount, rhsColumnCount},
        rewriter.getDenseI64ArrayAttr(order), flattenedOutput, "sum");
    Block *body = rewriter.createBlock(
        &reduction.getBody(), reduction.getBody().end(),
        {rewriter.getIndexType(), rewriter.getIndexType()}, {loc, loc});
    rewriter.setInsertionPointToStart(body);

    // The region always uses logical axis order. The generic position
    // scheduler interprets the order permutation when collapsing the domain.
    Value position = body->getArgument(0);
    Value outputColumn = body->getArgument(1);
    Value row = CSRRowAtPositionOp::create(
        rewriter, loc, rewriter.getIndexType(), op.getRowOffsets(), position);
    Value columnValue =
        memref::LoadOp::create(rewriter, loc, op.getColumnIndices(), position);
    Value column = castToIndex(rewriter, loc, columnValue);
    Value sparseValue =
        memref::LoadOp::create(rewriter, loc, op.getValues(), position);
    Value rhsValue = memref::LoadOp::create(rewriter, loc, op.getRhs(),
                                            ValueRange{column, outputColumn});
    Value product = arith::MulFOp::create(rewriter, loc, sparseValue, rhsValue);
    Value outputKey = arith::AddIOp::create(
        rewriter, loc,
        arith::MulIOp::create(rewriter, loc, row, rhsColumnCount),
        outputColumn);
    YieldOp::create(rewriter, loc, ValueRange{outputKey, product});
    rewriter.eraseOp(op);
    return success();
  }

private:
  SmallVector<int64_t> order;
};

class DecomposePositionSpMM
    : public impl::DecomposePositionSpMMBase<DecomposePositionSpMM> {
public:
  using impl::DecomposePositionSpMMBase<
      DecomposePositionSpMM>::DecomposePositionSpMMBase;

  void runOnOperation() override {
    if (iterationOrder != "position-major" && iterationOrder != "rhs-major") {
      getOperation().emitError()
          << "unsupported position-space SpMM iteration order '"
          << iterationOrder << "'; expected 'position-major' or 'rhs-major'";
      signalPassFailure();
      return;
    }

    RewritePatternSet patterns(&getContext());
    patterns.add<DecomposePositionSpMMPattern>(&getContext(), iterationOrder);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
      return;
    }

    bool hasUnsupportedSpMM = false;
    getOperation().walk([&](SpMMOp op) {
      op.emitError(
          "position-space SpMM requires an identity-layout output memref");
      hasUnsupportedSpMM = true;
    });
    if (hasUnsupportedSpMM)
      signalPassFailure();
  }
};

} // namespace
} // namespace mlir::sparsewave
