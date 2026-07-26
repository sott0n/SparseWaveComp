#include "sparsewave/Dialect/SparseWave/Transforms/Passes.h"

#include "sparsewave/Dialect/SparseWave/IR/SparseWaveOps.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensorType.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::sparsewave {
#define GEN_PASS_DEF_CONVERTLINALGSPMVTOSPARSEWAVE
#include "sparsewave/Dialect/SparseWave/Transforms/Passes.h.inc"

namespace {

bool hasCanonicalSpMVIndexingMaps(linalg::LinalgOp op) {
  SmallVector<AffineMap> maps = op.getIndexingMapsArray();
  if (maps.size() != 3)
    return false;

  MLIRContext *context = op.getContext();
  AffineExpr row = getAffineDimExpr(0, context);
  AffineExpr column = getAffineDimExpr(1, context);
  return maps[0] == AffineMap::get(2, 0, {row, column}, context) &&
         maps[1] == AffineMap::get(2, 0, column, context) &&
         maps[2] == AffineMap::get(2, 0, row, context);
}

linalg::FillOp getSingleUseZeroFill(Value value) {
  auto fill = value.getDefiningOp<linalg::FillOp>();
  if (!fill || !value.hasOneUse() ||
      !matchPattern(fill.getDpsInputOperand(0)->get(), m_AnyZeroFloat()))
    return {};
  return fill;
}

LogicalResult rewriteLinalgSpMV(linalg::LinalgOp op,
                                PatternRewriter &rewriter) {
  if (op.getNumDpsInputs() != 2 || op.getNumDpsInits() != 1 ||
      op->getNumResults() != 1)
    return rewriter.notifyMatchFailure(
        op, "expected two inputs, one output, and one result");
  if (!linalg::isaContractionOpInterface(op))
    return rewriter.notifyMatchFailure(op, "expected an add/multiply "
                                           "Linalg contraction");
  if (!hasCanonicalSpMVIndexingMaps(op))
    return rewriter.notifyMatchFailure(
        op, "expected canonical matrix-vector indexing maps");

  Value matrix = op.getDpsInputOperand(0)->get();
  Value vector = op.getDpsInputOperand(1)->get();
  Value output = op.getDpsInitOperand(0)->get();
  auto matrixType = dyn_cast<RankedTensorType>(matrix.getType());
  auto vectorType = dyn_cast<RankedTensorType>(vector.getType());
  auto outputType = dyn_cast<RankedTensorType>(output.getType());
  if (!matrixType || !vectorType || !outputType || matrixType.getRank() != 2 ||
      vectorType.getRank() != 1 || outputType.getRank() != 1)
    return rewriter.notifyMatchFailure(
        op, "expected rank-2 matrix and rank-1 vector tensors");

  sparse_tensor::SparseTensorType sparseMatrix(matrixType);
  if (!sparseMatrix.hasEncoding() || !sparseMatrix.isIdentity() ||
      sparseMatrix.getLvlRank() != 2 || !sparseMatrix.isDenseLvl(0) ||
      !sparseMatrix.isCompressedLvl(1))
    return rewriter.notifyMatchFailure(
        op, "expected an identity-mapped CSR SparseTensor matrix");
  if (sparse_tensor::getSparseTensorEncoding(vectorType) ||
      sparse_tensor::getSparseTensorEncoding(outputType))
    return rewriter.notifyMatchFailure(
        op, "expected dense vector and output tensors");
  if (matrixType.getElementType() != vectorType.getElementType() ||
      matrixType.getElementType() != outputType.getElementType() ||
      !isa<FloatType>(matrixType.getElementType()))
    return rewriter.notifyMatchFailure(
        op, "expected matching floating-point element types");
  linalg::FillOp zeroFill = getSingleUseZeroFill(output);
  if (!zeroFill)
    return rewriter.notifyMatchFailure(
        op, "expected a single-use, statically zero-filled output tensor");
  if (sparseMatrix.getPosWidth() != sparseMatrix.getCrdWidth())
    return rewriter.notifyMatchFailure(
        op, "expected matching CSR position and coordinate types");

  Location loc = op.getLoc();
  Value rowOffsets =
      sparse_tensor::ToPositionsOp::create(rewriter, loc, matrix, 1);
  Value columnIndices =
      sparse_tensor::ToCoordinatesOp::create(rewriter, loc, matrix, 1);
  Value values = sparse_tensor::ToValuesOp::create(rewriter, loc, matrix);

  auto vectorBufferType =
      MemRefType::get(vectorType.getShape(), vectorType.getElementType());
  auto outputBufferType =
      MemRefType::get(outputType.getShape(), outputType.getElementType());
  Value vectorBuffer = bufferization::ToBufferOp::create(
      rewriter, loc, vectorBufferType, vector, /*readOnly=*/true);
  Value outputStorage = zeroFill.getDpsInitOperand(0)->get();
  Value outputBuffer = bufferization::ToBufferOp::create(
      rewriter, loc, outputBufferType, outputStorage, /*readOnly=*/false);

  SpMVOp::create(rewriter, loc, rowOffsets, columnIndices, values, vectorBuffer,
                 outputBuffer);
  Value result = bufferization::ToTensorOp::create(
      rewriter, loc, outputType, outputBuffer, /*restrict=*/true,
      /*writable=*/true);
  rewriter.replaceOp(op, result);
  rewriter.eraseOp(zeroFill);
  return success();
}

template <typename OpTy>
class LinalgSpMVPattern : public OpRewritePattern<OpTy> {
public:
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    return rewriteLinalgSpMV(cast<linalg::LinalgOp>(op.getOperation()),
                             rewriter);
  }
};

class ConvertLinalgSpMVToSparseWave
    : public impl::ConvertLinalgSpMVToSparseWaveBase<
          ConvertLinalgSpMVToSparseWave> {
public:
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<LinalgSpMVPattern<linalg::GenericOp>,
                 LinalgSpMVPattern<linalg::MatvecOp>>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
} // namespace mlir::sparsewave
