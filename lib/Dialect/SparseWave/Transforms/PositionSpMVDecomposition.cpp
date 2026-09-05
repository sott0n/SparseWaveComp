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

struct PositionSpMVElement {
  Value row;
  Value column;
  Value value;
};

// These adapters isolate the format-specific SparseWave bridge operations.
// Removing this compatibility boundary requires upstream SparseTensor
// iteration to support the mixed sparse/dense contractions consumed here.
struct CSRPositionSpMVAdapter {
  using Op = SpMVOp;

  static Value getValues(Op op) { return op.getValues(); }
  static Value getVector(Op op) { return op.getVector(); }
  static Value getOutput(Op op) { return op.getOutput(); }

  static PositionSpMVElement buildElement(OpBuilder &builder, Location loc,
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

struct COOPositionSpMVAdapter {
  using Op = COOSpMVOp;

  static Value getValues(Op op) { return op.getValues(); }
  static Value getVector(Op op) { return op.getVector(); }
  static Value getOutput(Op op) { return op.getOutput(); }

  static PositionSpMVElement buildElement(OpBuilder &builder, Location loc,
                                          Op op, Value position) {
    Value rowValue =
        memref::LoadOp::create(builder, loc, op.getRowIndices(), position);
    Value columnValue =
        memref::LoadOp::create(builder, loc, op.getColumnIndices(), position);
    Value row = castToIndex(builder, loc, rowValue);
    Value column = castToIndex(builder, loc, columnValue);
    Value value =
        memref::LoadOp::create(builder, loc, op.getValues(), position);
    return {row, column, value};
  }
};

template <typename Adapter>
void createPositionSpMVReduction(PatternRewriter &rewriter,
                                 typename Adapter::Op op) {
  Location loc = op.getLoc();
  Value values = Adapter::getValues(op);
  Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
  Value nonzeroCount = memref::DimOp::create(rewriter, loc, values, zero);
  auto reduction = PositionReduceOp::create(
      rewriter, loc, ValueRange{zero}, ValueRange{nonzeroCount},
      rewriter.getStrArrayAttr({"position"}),
      rewriter.getDenseI64ArrayAttr({0}), Adapter::getOutput(op), "sum");
  Block *body =
      rewriter.createBlock(&reduction.getBody(), reduction.getBody().end(),
                           {rewriter.getIndexType()}, {loc});
  rewriter.setInsertionPointToStart(body);
  PositionSpMVElement element =
      Adapter::buildElement(rewriter, loc, op, body->getArgument(0));
  Value vectorValue = memref::LoadOp::create(
      rewriter, loc, Adapter::getVector(op), element.column);
  Value product =
      arith::MulFOp::create(rewriter, loc, element.value, vectorValue);
  YieldOp::create(rewriter, loc, ValueRange{element.row, product});
}

template <typename Adapter>
class DecomposePositionSpMVPattern
    : public OpRewritePattern<typename Adapter::Op> {
public:
  using Op = typename Adapter::Op;
  using OpRewritePattern<Op>::OpRewritePattern;

  LogicalResult matchAndRewrite(Op op,
                                PatternRewriter &rewriter) const override {
    createPositionSpMVReduction<Adapter>(rewriter, op);
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
    patterns.add<DecomposePositionSpMVPattern<COOPositionSpMVAdapter>>(
        &getContext());
    if (!preserveDirectMapping)
      patterns.add<DecomposePositionSpMVPattern<CSRPositionSpMVAdapter>>(
          &getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
} // namespace mlir::sparsewave
