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
#define GEN_PASS_DEF_CONVERTLINALGSPMMTOSPARSEWAVE
#include "sparsewave/Dialect/SparseWave/Transforms/Passes.h.inc"

namespace {

struct SparseContractionMatch {
  Value matrix;
  Value denseInput;
  Value output;
  RankedTensorType matrixType;
  RankedTensorType denseInputType;
  RankedTensorType outputType;
  linalg::FillOp zeroFill;
};

struct CSRStorage {
  Value rowOffsets;
  Value columnIndices;
  Value values;
};

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

bool hasCanonicalSpMMIndexingMaps(linalg::LinalgOp op) {
  SmallVector<AffineMap> maps = op.getIndexingMapsArray();
  if (maps.size() != 3)
    return false;

  MLIRContext *context = op.getContext();
  AffineExpr row = getAffineDimExpr(0, context);
  AffineExpr column = getAffineDimExpr(1, context);
  AffineExpr reduction = getAffineDimExpr(2, context);
  return maps[0] == AffineMap::get(3, 0, {row, reduction}, context) &&
         maps[1] == AffineMap::get(3, 0, {reduction, column}, context) &&
         maps[2] == AffineMap::get(3, 0, {row, column}, context);
}

linalg::FillOp getSingleUseZeroFill(Value value) {
  auto fill = value.getDefiningOp<linalg::FillOp>();
  if (!fill || !value.hasOneUse() ||
      !matchPattern(fill.getDpsInputOperand(0)->get(), m_AnyZeroFloat()))
    return {};
  return fill;
}

LogicalResult matchSparseContraction(linalg::LinalgOp op,
                                     int64_t denseInputRank, int64_t outputRank,
                                     SparseContractionMatch &match,
                                     PatternRewriter &rewriter) {
  if (op.getNumDpsInputs() != 2 || op.getNumDpsInits() != 1 ||
      op->getNumResults() != 1)
    return rewriter.notifyMatchFailure(
        op, "expected two inputs, one output, and one result");
  if (!linalg::isaContractionOpInterface(op))
    return rewriter.notifyMatchFailure(op, "expected an add/multiply "
                                           "Linalg contraction");

  match.matrix = op.getDpsInputOperand(0)->get();
  match.denseInput = op.getDpsInputOperand(1)->get();
  match.output = op.getDpsInitOperand(0)->get();
  match.matrixType = dyn_cast<RankedTensorType>(match.matrix.getType());
  match.denseInputType = dyn_cast<RankedTensorType>(match.denseInput.getType());
  match.outputType = dyn_cast<RankedTensorType>(match.output.getType());
  if (!match.matrixType || !match.denseInputType || !match.outputType ||
      match.matrixType.getRank() != 2 ||
      match.denseInputType.getRank() != denseInputRank ||
      match.outputType.getRank() != outputRank)
    return rewriter.notifyMatchFailure(
        op, "unexpected sparse contraction operand ranks");

  sparse_tensor::SparseTensorType sparseMatrix(match.matrixType);
  if (!sparseMatrix.hasEncoding() || !sparseMatrix.isIdentity() ||
      sparseMatrix.getLvlRank() != 2 || !sparseMatrix.isDenseLvl(0) ||
      !sparseMatrix.isCompressedLvl(1))
    return rewriter.notifyMatchFailure(
        op, "expected an identity-mapped CSR SparseTensor matrix");
  if (sparse_tensor::getSparseTensorEncoding(match.denseInputType) ||
      sparse_tensor::getSparseTensorEncoding(match.outputType))
    return rewriter.notifyMatchFailure(
        op, "expected dense right-hand side and output tensors");
  if (match.matrixType.getElementType() !=
          match.denseInputType.getElementType() ||
      match.matrixType.getElementType() != match.outputType.getElementType() ||
      !isa<FloatType>(match.matrixType.getElementType()))
    return rewriter.notifyMatchFailure(
        op, "expected matching floating-point element types");
  match.zeroFill = getSingleUseZeroFill(match.output);
  if (!match.zeroFill)
    return rewriter.notifyMatchFailure(
        op, "expected a single-use, statically zero-filled output tensor");
  if (sparseMatrix.getPosWidth() != sparseMatrix.getCrdWidth())
    return rewriter.notifyMatchFailure(
        op, "expected matching CSR position and coordinate types");
  return success();
}

CSRStorage extractCSRStorage(PatternRewriter &rewriter, Location loc,
                             Value matrix) {
  return {
      sparse_tensor::ToPositionsOp::create(rewriter, loc, matrix, 1),
      sparse_tensor::ToCoordinatesOp::create(rewriter, loc, matrix, 1),
      sparse_tensor::ToValuesOp::create(rewriter, loc, matrix),
  };
}

Value bufferizeDenseTensor(PatternRewriter &rewriter, Location loc,
                           Value tensor, RankedTensorType tensorType,
                           bool readOnly) {
  auto bufferType =
      MemRefType::get(tensorType.getShape(), tensorType.getElementType());
  return bufferization::ToBufferOp::create(rewriter, loc, bufferType, tensor,
                                           readOnly);
}

void replaceContraction(linalg::LinalgOp op, PatternRewriter &rewriter,
                        const SparseContractionMatch &match,
                        Value outputBuffer) {
  Location loc = op.getLoc();
  Value result = bufferization::ToTensorOp::create(
      rewriter, loc, match.outputType, outputBuffer, /*restrict=*/true,
      /*writable=*/true);
  rewriter.replaceOp(op, result);
  rewriter.eraseOp(match.zeroFill);
}

LogicalResult rewriteLinalgSpMV(linalg::LinalgOp op,
                                PatternRewriter &rewriter) {
  if (!hasCanonicalSpMVIndexingMaps(op))
    return rewriter.notifyMatchFailure(
        op, "expected canonical matrix-vector indexing maps");

  SparseContractionMatch match;
  if (failed(matchSparseContraction(op, /*denseInputRank=*/1,
                                    /*outputRank=*/1, match, rewriter)))
    return failure();

  Location loc = op.getLoc();
  CSRStorage csr = extractCSRStorage(rewriter, loc, match.matrix);
  Value vectorBuffer = bufferizeDenseTensor(
      rewriter, loc, match.denseInput, match.denseInputType, /*readOnly=*/true);
  Value outputStorage = match.zeroFill.getDpsInitOperand(0)->get();
  Value outputBuffer =
      bufferizeDenseTensor(rewriter, loc, outputStorage, match.outputType,
                           /*readOnly=*/false);
  SpMVOp::create(rewriter, loc, csr.rowOffsets, csr.columnIndices, csr.values,
                 vectorBuffer, outputBuffer);
  replaceContraction(op, rewriter, match, outputBuffer);
  return success();
}

LogicalResult rewriteLinalgSpMM(linalg::LinalgOp op,
                                PatternRewriter &rewriter) {
  if (!hasCanonicalSpMMIndexingMaps(op))
    return rewriter.notifyMatchFailure(
        op, "expected canonical matrix-matrix indexing maps");

  SparseContractionMatch match;
  if (failed(matchSparseContraction(op, /*denseInputRank=*/2,
                                    /*outputRank=*/2, match, rewriter)))
    return failure();

  Location loc = op.getLoc();
  CSRStorage csr = extractCSRStorage(rewriter, loc, match.matrix);
  Value rhsBuffer = bufferizeDenseTensor(
      rewriter, loc, match.denseInput, match.denseInputType, /*readOnly=*/true);
  Value outputStorage = match.zeroFill.getDpsInitOperand(0)->get();
  Value outputBuffer =
      bufferizeDenseTensor(rewriter, loc, outputStorage, match.outputType,
                           /*readOnly=*/false);
  SpMMOp::create(rewriter, loc, csr.rowOffsets, csr.columnIndices, csr.values,
                 rhsBuffer, outputBuffer);
  replaceContraction(op, rewriter, match, outputBuffer);
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

template <typename OpTy>
class LinalgSpMMPattern : public OpRewritePattern<OpTy> {
public:
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    return rewriteLinalgSpMM(cast<linalg::LinalgOp>(op.getOperation()),
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

class ConvertLinalgSpMMToSparseWave
    : public impl::ConvertLinalgSpMMToSparseWaveBase<
          ConvertLinalgSpMMToSparseWave> {
public:
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<LinalgSpMMPattern<linalg::GenericOp>,
                 LinalgSpMMPattern<linalg::MatmulOp>>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
} // namespace mlir::sparsewave
