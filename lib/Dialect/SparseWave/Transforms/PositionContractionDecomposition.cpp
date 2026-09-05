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
#define GEN_PASS_DEF_DECOMPOSEPOSITIONSPMV
#include "sparsewave/Dialect/SparseWave/Transforms/Passes.h.inc"

namespace {

struct SparsePositionElement {
  Value outputCoordinate;
  Value reductionCoordinate;
  Value value;
};

// These adapters isolate the format-specific SparseWave bridge operations.
// Removing this compatibility boundary requires upstream SparseTensor
// iteration to support the mixed sparse/dense contractions consumed here.
template <typename OpTy> struct CSRPositionStorageAdapter {
  using Op = OpTy;

  static Value getPositionCount(OpBuilder &builder, Location loc, Op op) {
    Value zero = arith::ConstantIndexOp::create(builder, loc, 0);
    return memref::DimOp::create(builder, loc, op.getValues(), zero);
  }

  static SparsePositionElement buildElement(OpBuilder &builder, Location loc,
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

template <typename OpTy> struct COOPositionStorageAdapter {
  using Op = OpTy;

  static Value getPositionCount(OpBuilder &builder, Location loc, Op op) {
    Value zero = arith::ConstantIndexOp::create(builder, loc, 0);
    return memref::DimOp::create(builder, loc, op.getValues(), zero);
  }

  static SparsePositionElement buildElement(OpBuilder &builder, Location loc,
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

struct PositionContribution {
  Value outputKey;
  Value value;
};

template <typename StorageAdapter> struct VectorContractionAdapter {
  using Op = typename StorageAdapter::Op;

  static LogicalResult validate(Op, PatternRewriter &) { return success(); }

  static SmallVector<Value> buildUpperBounds(OpBuilder &builder, Location loc,
                                             Op op) {
    return {StorageAdapter::getPositionCount(builder, loc, op)};
  }

  static ArrayAttr getAxes(OpBuilder &builder) {
    return builder.getStrArrayAttr({"position"});
  }

  static DenseI64ArrayAttr getOrder(OpBuilder &builder) {
    return builder.getDenseI64ArrayAttr({0});
  }

  static Value buildReductionOutput(OpBuilder &, Location, Op op) {
    return op.getOutput();
  }

  static PositionContribution buildContribution(OpBuilder &builder,
                                                Location loc, Op op,
                                                ValueRange coordinates,
                                                ValueRange) {
    SparsePositionElement element =
        StorageAdapter::buildElement(builder, loc, op, coordinates.front());
    Value vectorValue = memref::LoadOp::create(builder, loc, op.getVector(),
                                               element.reductionCoordinate);
    Value product =
        arith::MulFOp::create(builder, loc, element.value, vectorValue);
    return {element.outputCoordinate, product};
  }
};

template <typename StorageAdapter> struct MatrixContractionAdapter {
  using Op = typename StorageAdapter::Op;

  static LogicalResult validate(Op op, PatternRewriter &rewriter) {
    auto outputType = cast<MemRefType>(op.getOutput().getType());
    if (!outputType.getLayout().isIdentity())
      return rewriter.notifyMatchFailure(
          op, "position-space matrix contraction requires an identity-layout "
              "output");
    return success();
  }

  static SmallVector<Value> buildUpperBounds(OpBuilder &builder, Location loc,
                                             Op op) {
    Value positionCount = StorageAdapter::getPositionCount(builder, loc, op);
    Value one = arith::ConstantIndexOp::create(builder, loc, 1);
    Value rhsColumnCount =
        memref::DimOp::create(builder, loc, op.getRhs(), one);
    return {positionCount, rhsColumnCount};
  }

  static ArrayAttr getAxes(OpBuilder &builder) {
    return builder.getStrArrayAttr({"position", "rhs"});
  }

  static DenseI64ArrayAttr getOrder(OpBuilder &builder) {
    return builder.getDenseI64ArrayAttr({0, 1});
  }

  static Value buildReductionOutput(OpBuilder &builder, Location loc, Op op) {
    SmallVector<ReassociationIndices> reassociation{{0, 1}};
    return memref::CollapseShapeOp::create(builder, loc, op.getOutput(),
                                           reassociation);
  }

  static PositionContribution buildContribution(OpBuilder &builder,
                                                Location loc, Op op,
                                                ValueRange coordinates,
                                                ValueRange upperBounds) {
    Value position = coordinates[0];
    Value outputColumn = coordinates[1];
    SparsePositionElement element =
        StorageAdapter::buildElement(builder, loc, op, position);
    Value rhsValue = memref::LoadOp::create(
        builder, loc, op.getRhs(),
        ValueRange{element.reductionCoordinate, outputColumn});
    Value product =
        arith::MulFOp::create(builder, loc, element.value, rhsValue);
    Value rhsColumnCount = upperBounds[1];
    Value outputKey = arith::AddIOp::create(
        builder, loc,
        arith::MulIOp::create(builder, loc, element.outputCoordinate,
                              rhsColumnCount),
        outputColumn);
    return {outputKey, product};
  }
};

template <typename ContractionAdapter>
LogicalResult
createPositionContractionReduction(PatternRewriter &rewriter,
                                   typename ContractionAdapter::Op op) {
  if (failed(ContractionAdapter::validate(op, rewriter)))
    return failure();

  Location loc = op.getLoc();
  SmallVector<Value> upperBounds =
      ContractionAdapter::buildUpperBounds(rewriter, loc, op);
  Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
  SmallVector<Value> lowerBounds(upperBounds.size(), zero);
  Value output = ContractionAdapter::buildReductionOutput(rewriter, loc, op);
  auto reduction = PositionReduceOp::create(
      rewriter, loc, lowerBounds, upperBounds,
      ContractionAdapter::getAxes(rewriter),
      ContractionAdapter::getOrder(rewriter), output, "sum");
  SmallVector<Type> coordinateTypes(upperBounds.size(),
                                    rewriter.getIndexType());
  SmallVector<Location> coordinateLocations(upperBounds.size(), loc);
  Block *body =
      rewriter.createBlock(&reduction.getBody(), reduction.getBody().end(),
                           coordinateTypes, coordinateLocations);
  rewriter.setInsertionPointToStart(body);
  PositionContribution contribution = ContractionAdapter::buildContribution(
      rewriter, loc, op, body->getArguments(), upperBounds);
  YieldOp::create(rewriter, loc,
                  ValueRange{contribution.outputKey, contribution.value});
  return success();
}

template <typename ContractionAdapter>
class DecomposePositionContractionPattern
    : public OpRewritePattern<typename ContractionAdapter::Op> {
public:
  using Op = typename ContractionAdapter::Op;
  using OpRewritePattern<Op>::OpRewritePattern;

  LogicalResult matchAndRewrite(Op op,
                                PatternRewriter &rewriter) const override {
    if (failed(createPositionContractionReduction<ContractionAdapter>(rewriter,
                                                                      op)))
      return failure();
    rewriter.eraseOp(op);
    return success();
  }
};

using CSRVectorContraction =
    VectorContractionAdapter<CSRPositionStorageAdapter<SpMVOp>>;
using COOVectorContraction =
    VectorContractionAdapter<COOPositionStorageAdapter<COOSpMVOp>>;
using CSRMatrixContraction =
    MatrixContractionAdapter<CSRPositionStorageAdapter<SpMMOp>>;

class DecomposePositionSpMV
    : public impl::DecomposePositionSpMVBase<DecomposePositionSpMV> {
public:
  using impl::DecomposePositionSpMVBase<
      DecomposePositionSpMV>::DecomposePositionSpMVBase;

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<DecomposePositionContractionPattern<COOVectorContraction>>(
        &getContext());
    if (!preserveDirectMapping)
      patterns.add<DecomposePositionContractionPattern<CSRVectorContraction>>(
          &getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

class DecomposePositionSpMM
    : public impl::DecomposePositionSpMMBase<DecomposePositionSpMM> {
public:
  using impl::DecomposePositionSpMMBase<
      DecomposePositionSpMM>::DecomposePositionSpMMBase;

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<DecomposePositionContractionPattern<CSRMatrixContraction>>(
        &getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
      return;
    }

    bool hasUnsupportedContraction = false;
    getOperation().walk([&](SpMMOp op) {
      op.emitError("position-space matrix contraction requires an "
                   "identity-layout output memref");
      hasUnsupportedContraction = true;
    });
    if (hasUnsupportedContraction)
      signalPassFailure();
  }
};

} // namespace
} // namespace mlir::sparsewave
