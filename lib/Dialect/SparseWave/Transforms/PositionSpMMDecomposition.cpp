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

struct PositionSpMMElement {
  Value outputRow;
  Value reductionCoordinate;
  Value value;
};

// This adapter isolates the format-specific SparseWave bridge operation.
// Removing this compatibility boundary requires upstream SparseTensor
// iteration to support the mixed sparse/dense contractions consumed here.
struct CSRPositionSpMMAdapter {
  using Op = SpMMOp;

  static Value getRhs(Op op) { return op.getRhs(); }
  static Value getOutput(Op op) { return op.getOutput(); }

  static Value getPositionCount(OpBuilder &builder, Location loc, Op op) {
    Value zero = arith::ConstantIndexOp::create(builder, loc, 0);
    return memref::DimOp::create(builder, loc, op.getValues(), zero);
  }

  static PositionSpMMElement buildElement(OpBuilder &builder, Location loc,
                                          Op op, Value position) {
    Value row = CompressedSegmentAtPositionOp::create(
        builder, loc, builder.getIndexType(), op.getRowOffsets(), position);
    Value columnValue =
        memref::LoadOp::create(builder, loc, op.getColumnIndices(), position);
    Value column = castToIndex(builder, loc, columnValue);
    Value value =
        memref::LoadOp::create(builder, loc, op.getValues(), position);
    return {row, column, value};
  }
};

template <typename Adapter>
LogicalResult createPositionSpMMReduction(PatternRewriter &rewriter,
                                          typename Adapter::Op op) {
  Value output = Adapter::getOutput(op);
  MemRefType outputType = cast<MemRefType>(output.getType());
  if (!outputType.getLayout().isIdentity())
    return rewriter.notifyMatchFailure(
        op, "position-space SpMM requires an identity-layout output");

  Location loc = op.getLoc();
  Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
  Value one = arith::ConstantIndexOp::create(rewriter, loc, 1);
  Value positionCount = Adapter::getPositionCount(rewriter, loc, op);
  Value rhs = Adapter::getRhs(op);
  Value rhsColumnCount = memref::DimOp::create(rewriter, loc, rhs, one);
  SmallVector<ReassociationIndices> reassociation{{0, 1}};
  Value flattenedOutput =
      memref::CollapseShapeOp::create(rewriter, loc, output, reassociation);
  auto reduction = PositionReduceOp::create(
      rewriter, loc, ValueRange{zero, zero},
      ValueRange{positionCount, rhsColumnCount},
      rewriter.getStrArrayAttr({"position", "rhs"}),
      rewriter.getDenseI64ArrayAttr({0, 1}), flattenedOutput, "sum");
  Block *body = rewriter.createBlock(
      &reduction.getBody(), reduction.getBody().end(),
      {rewriter.getIndexType(), rewriter.getIndexType()}, {loc, loc});
  rewriter.setInsertionPointToStart(body);

  // The region always uses logical axis order. The generic position scheduler
  // interprets the order permutation when collapsing the domain.
  Value position = body->getArgument(0);
  Value outputColumn = body->getArgument(1);
  PositionSpMMElement element =
      Adapter::buildElement(rewriter, loc, op, position);
  Value rhsValue = memref::LoadOp::create(
      rewriter, loc, rhs,
      ValueRange{element.reductionCoordinate, outputColumn});
  Value product = arith::MulFOp::create(rewriter, loc, element.value, rhsValue);
  Value outputKey = arith::AddIOp::create(
      rewriter, loc,
      arith::MulIOp::create(rewriter, loc, element.outputRow, rhsColumnCount),
      outputColumn);
  YieldOp::create(rewriter, loc, ValueRange{outputKey, product});
  return success();
}

template <typename Adapter>
class DecomposePositionSpMMPattern
    : public OpRewritePattern<typename Adapter::Op> {
public:
  using Op = typename Adapter::Op;
  using OpRewritePattern<Op>::OpRewritePattern;

  LogicalResult matchAndRewrite(Op op,
                                PatternRewriter &rewriter) const override {
    if (failed(createPositionSpMMReduction<Adapter>(rewriter, op)))
      return failure();
    rewriter.eraseOp(op);
    return success();
  }
};

class DecomposePositionSpMM
    : public impl::DecomposePositionSpMMBase<DecomposePositionSpMM> {
public:
  using impl::DecomposePositionSpMMBase<
      DecomposePositionSpMM>::DecomposePositionSpMMBase;

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<DecomposePositionSpMMPattern<CSRPositionSpMMAdapter>>(
        &getContext());
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
