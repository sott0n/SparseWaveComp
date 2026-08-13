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
#define GEN_PASS_DEF_CONVERTLINALGSDDMMTOSPARSEWAVE
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
  int64_t blockSize = 0;
};

struct CSRStorage {
  Value rowOffsets;
  Value columnIndices;
  Value values;
};

struct COOStorage {
  Value rowIndices;
  Value columnIndices;
  Value values;
};

struct BSRStorage {
  Value blockRowOffsets;
  Value blockColumnIndices;
  Value blockValues;
};

struct SDDMMMatch {
  Value lhs;
  Value rhs;
  Value sample;
  Value output;
  RankedTensorType lhsType;
  RankedTensorType rhsType;
  RankedTensorType sampleType;
  RankedTensorType outputType;
  linalg::FillOp zeroFill;
};

enum class SparseMatrixFormat {
  CSR,
  COO,
  BSR,
};

bool matchCanonicalSquareBSR(sparse_tensor::SparseTensorType sparseMatrix,
                             int64_t &blockSize) {
  if (sparseMatrix.getDimRank() != 2 || sparseMatrix.getLvlRank() != 4 ||
      !sparseMatrix.isDenseLvl(0) || !sparseMatrix.isCompressedLvl(1) ||
      !sparseMatrix.isOrderedLvl(1) || !sparseMatrix.isUniqueLvl(1) ||
      !sparseMatrix.isDenseLvl(2) || !sparseMatrix.isDenseLvl(3))
    return false;

  AffineMap dimToLvl = sparseMatrix.getDimToLvl();
  if (!sparse_tensor::isBlockSparsity(dimToLvl))
    return false;
  SmallVector<unsigned> blockSizes = sparse_tensor::getBlockSize(dimToLvl);
  if (blockSizes.size() != 2 || blockSizes[0] <= 1 ||
      blockSizes[0] != blockSizes[1])
    return false;

  blockSize = blockSizes[0];
  MLIRContext *context = sparseMatrix.getElementType().getContext();
  AffineExpr row = getAffineDimExpr(0, context);
  AffineExpr column = getAffineDimExpr(1, context);
  AffineMap expected =
      AffineMap::get(2, 0,
                     {row.floorDiv(blockSize), column.floorDiv(blockSize),
                      row % blockSize, column % blockSize},
                     context);
  return dimToLvl == expected;
}

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

bool hasCanonicalSDDMMIndexingMaps(linalg::LinalgOp op) {
  SmallVector<AffineMap> maps = op.getIndexingMapsArray();
  if (maps.size() != 4)
    return false;

  MLIRContext *context = op.getContext();
  AffineExpr row = getAffineDimExpr(0, context);
  AffineExpr column = getAffineDimExpr(1, context);
  AffineExpr reduction = getAffineDimExpr(2, context);
  return maps[0] == AffineMap::get(3, 0, {row, reduction}, context) &&
         maps[1] == AffineMap::get(3, 0, {reduction, column}, context) &&
         maps[2] == AffineMap::get(3, 0, {row, column}, context) &&
         maps[3] == AffineMap::get(3, 0, {row, column}, context);
}

void collectMultiplyLeaves(Value value, SmallVectorImpl<Value> &leaves) {
  if (auto multiply = value.getDefiningOp<arith::MulFOp>()) {
    collectMultiplyLeaves(multiply.getLhs(), leaves);
    collectMultiplyLeaves(multiply.getRhs(), leaves);
    return;
  }
  leaves.push_back(value);
}

bool matchTripleProduct(Value value, Value lhs, Value rhs, Value sample) {
  SmallVector<Value> leaves;
  collectMultiplyLeaves(value, leaves);
  if (leaves.size() != 3)
    return false;

  SmallVector<Value> expected{lhs, rhs, sample};
  for (Value leaf : leaves) {
    auto it = llvm::find(expected, leaf);
    if (it == expected.end())
      return false;
    expected.erase(it);
  }
  return expected.empty();
}

bool hasCanonicalSDDMMBody(linalg::GenericOp op) {
  Block &body = op.getRegion().front();
  auto yield = dyn_cast<linalg::YieldOp>(body.getTerminator());
  if (!yield || yield.getNumOperands() != 1 || body.getNumArguments() != 4)
    return false;

  auto add = yield.getOperand(0).getDefiningOp<arith::AddFOp>();
  if (!add)
    return false;
  Value lhs = body.getArgument(0);
  Value rhs = body.getArgument(1);
  Value sample = body.getArgument(2);
  Value output = body.getArgument(3);
  return (add.getLhs() == output &&
          matchTripleProduct(add.getRhs(), lhs, rhs, sample)) ||
         (add.getRhs() == output &&
          matchTripleProduct(add.getLhs(), lhs, rhs, sample));
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
                                     SparseMatrixFormat format,
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
  if (!sparseMatrix.hasEncoding())
    return rewriter.notifyMatchFailure(op, "expected a SparseTensor matrix");
  switch (format) {
  case SparseMatrixFormat::CSR:
    if (!sparseMatrix.isIdentity() || sparseMatrix.getLvlRank() != 2 ||
        !sparseMatrix.isDenseLvl(0) || !sparseMatrix.isCompressedLvl(1))
      return rewriter.notifyMatchFailure(op,
                                         "expected a CSR SparseTensor matrix");
    break;
  case SparseMatrixFormat::COO:
    if (!sparseMatrix.isIdentity() || sparseMatrix.getLvlRank() != 2 ||
        !sparseMatrix.isCOOType())
      return rewriter.notifyMatchFailure(op,
                                         "expected a COO SparseTensor matrix");
    break;
  case SparseMatrixFormat::BSR:
    if (!matchCanonicalSquareBSR(sparseMatrix, match.blockSize))
      return rewriter.notifyMatchFailure(
          op, "expected a canonical square BSR SparseTensor matrix");
    break;
  }
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
  if ((format == SparseMatrixFormat::CSR ||
       format == SparseMatrixFormat::BSR) &&
      sparseMatrix.getPosWidth() != sparseMatrix.getCrdWidth())
    return rewriter.notifyMatchFailure(
        op, "expected matching sparse position and coordinate types");
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

COOStorage extractCOOStorage(PatternRewriter &rewriter, Location loc,
                             Value matrix) {
  return {
      sparse_tensor::ToCoordinatesOp::create(rewriter, loc, matrix, 0),
      sparse_tensor::ToCoordinatesOp::create(rewriter, loc, matrix, 1),
      sparse_tensor::ToValuesOp::create(rewriter, loc, matrix),
  };
}

BSRStorage extractBSRStorage(PatternRewriter &rewriter, Location loc,
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

LogicalResult matchSDDMM(linalg::GenericOp op, SDDMMMatch &match,
                         PatternRewriter &rewriter) {
  if (op.getNumDpsInputs() != 3 || op.getNumDpsInits() != 1 ||
      op->getNumResults() != 1)
    return rewriter.notifyMatchFailure(
        op, "expected three inputs, one output, and one result");
  if (!hasCanonicalSDDMMIndexingMaps(op))
    return rewriter.notifyMatchFailure(
        op, "expected canonical SDDMM indexing maps");
  SmallVector<utils::IteratorType> iteratorTypes = op.getIteratorTypesArray();
  if (iteratorTypes.size() != 3 ||
      iteratorTypes[0] != utils::IteratorType::parallel ||
      iteratorTypes[1] != utils::IteratorType::parallel ||
      iteratorTypes[2] != utils::IteratorType::reduction)
    return rewriter.notifyMatchFailure(
        op, "expected parallel, parallel, reduction iterators");
  if (!hasCanonicalSDDMMBody(op))
    return rewriter.notifyMatchFailure(
        op, "expected a sampled multiply/add reduction");

  match.lhs = op.getDpsInputOperand(0)->get();
  match.rhs = op.getDpsInputOperand(1)->get();
  match.sample = op.getDpsInputOperand(2)->get();
  match.output = op.getDpsInitOperand(0)->get();
  match.lhsType = dyn_cast<RankedTensorType>(match.lhs.getType());
  match.rhsType = dyn_cast<RankedTensorType>(match.rhs.getType());
  match.sampleType = dyn_cast<RankedTensorType>(match.sample.getType());
  match.outputType = dyn_cast<RankedTensorType>(match.output.getType());
  if (!match.lhsType || !match.rhsType || !match.sampleType ||
      !match.outputType || match.lhsType.getRank() != 2 ||
      match.rhsType.getRank() != 2 || match.sampleType.getRank() != 2 ||
      match.outputType.getRank() != 2)
    return rewriter.notifyMatchFailure(op, "expected rank-2 tensor operands");

  sparse_tensor::SparseTensorType sparseSample(match.sampleType);
  sparse_tensor::SparseTensorType sparseOutput(match.outputType);
  if (!sparseSample.hasEncoding() || !sparseSample.isIdentity() ||
      sparseSample.getLvlRank() != 2 || !sparseSample.isDenseLvl(0) ||
      !sparseSample.isCompressedLvl(1))
    return rewriter.notifyMatchFailure(
        op, "expected an identity-mapped CSR SparseTensor sample");
  if (!sparseOutput.hasEncoding() || !sparseOutput.isIdentity() ||
      sparseOutput.getLvlRank() != 2 || !sparseOutput.isDenseLvl(0) ||
      !sparseOutput.isCompressedLvl(1) ||
      sparseOutput.getEncoding() != sparseSample.getEncoding())
    return rewriter.notifyMatchFailure(
        op, "expected a matching CSR SparseTensor output");
  if (sparse_tensor::getSparseTensorEncoding(match.lhsType) ||
      sparse_tensor::getSparseTensorEncoding(match.rhsType))
    return rewriter.notifyMatchFailure(op, "expected dense input tensors");
  if (sparseSample.getPosWidth() != sparseSample.getCrdWidth() ||
      sparseOutput.getPosWidth() != sparseOutput.getCrdWidth())
    return rewriter.notifyMatchFailure(
        op, "expected matching CSR position and coordinate types");

  Type elementType = match.sampleType.getElementType();
  if (match.lhsType.getElementType() != elementType ||
      match.rhsType.getElementType() != elementType ||
      match.outputType.getElementType() != elementType ||
      !isa<FloatType>(elementType))
    return rewriter.notifyMatchFailure(
        op, "expected matching floating-point element types");
  match.zeroFill = getSingleUseZeroFill(match.output);
  if (!match.zeroFill)
    return rewriter.notifyMatchFailure(
        op, "expected a single-use, statically zero-filled sparse output");
  if (match.zeroFill.getDpsInitOperand(0)->get() != match.sample)
    return rewriter.notifyMatchFailure(
        op, "expected the zero-filled output to reuse the sample structure");
  return success();
}

LogicalResult rewriteLinalgSDDMM(linalg::GenericOp op,
                                 PatternRewriter &rewriter) {
  SDDMMMatch match;
  if (failed(matchSDDMM(op, match, rewriter)))
    return failure();

  Location loc = op.getLoc();
  CSRStorage sample = extractCSRStorage(rewriter, loc, match.sample);
  Value lhsBuffer = bufferizeDenseTensor(rewriter, loc, match.lhs,
                                         match.lhsType, /*readOnly=*/true);
  Value rhsBuffer = bufferizeDenseTensor(rewriter, loc, match.rhs,
                                         match.rhsType, /*readOnly=*/true);

  // The zero-filled destination is derived from the sample itself, proving
  // that the input and result share one CSR structure. Each GPU work item
  // reads a sample value before replacing that same position.
  auto sddmm =
      SDDMMOp::create(rewriter, loc, sample.rowOffsets, sample.columnIndices,
                      sample.values, lhsBuffer, rhsBuffer, sample.values);
  Block *body = new Block();
  sddmm.getBody().push_back(body);
  Type valueType = match.sampleType.getElementType();
  BlockArgument sampleValue = body->addArgument(valueType, loc);
  BlockArgument dotProduct = body->addArgument(valueType, loc);
  OpBuilder bodyBuilder = OpBuilder::atBlockEnd(body);
  Value weighted =
      arith::MulFOp::create(bodyBuilder, loc, sampleValue, dotProduct);
  YieldOp::create(bodyBuilder, loc, weighted);
  rewriter.replaceOpWithNewOp<sparse_tensor::LoadOp>(op, match.sample);
  rewriter.eraseOp(match.zeroFill);
  return success();
}

LogicalResult rewriteLinalgSpMV(linalg::LinalgOp op,
                                PatternRewriter &rewriter) {
  if (!hasCanonicalSpMVIndexingMaps(op))
    return rewriter.notifyMatchFailure(
        op, "expected canonical matrix-vector indexing maps");

  SparseContractionMatch match;
  if (succeeded(
          matchSparseContraction(op, /*denseInputRank=*/1, /*outputRank=*/1,
                                 SparseMatrixFormat::CSR, match, rewriter))) {
    // CSR exposes row bounds and lowers to row-oriented work distribution and
    // reduction strategies.
    Location loc = op.getLoc();
    CSRStorage csr = extractCSRStorage(rewriter, loc, match.matrix);
    Value vectorBuffer =
        bufferizeDenseTensor(rewriter, loc, match.denseInput,
                             match.denseInputType, /*readOnly=*/true);
    Value outputStorage = match.zeroFill.getDpsInitOperand(0)->get();
    Value outputBuffer =
        bufferizeDenseTensor(rewriter, loc, outputStorage, match.outputType,
                             /*readOnly=*/false);
    SpMVOp::create(rewriter, loc, csr.rowOffsets, csr.columnIndices, csr.values,
                   vectorBuffer, outputBuffer);
    replaceContraction(op, rewriter, match, outputBuffer);
    return success();
  }

  if (failed(matchSparseContraction(op, /*denseInputRank=*/1, /*outputRank=*/1,
                                    SparseMatrixFormat::COO, match, rewriter)))
    return failure();

  // COO exposes one row/column pair per nonzero and lowers to independent
  // products followed by atomic output accumulation.
  Location loc = op.getLoc();
  COOStorage coo = extractCOOStorage(rewriter, loc, match.matrix);
  Value vectorBuffer = bufferizeDenseTensor(
      rewriter, loc, match.denseInput, match.denseInputType, /*readOnly=*/true);
  Value outputStorage = match.zeroFill.getDpsInitOperand(0)->get();
  Value outputBuffer =
      bufferizeDenseTensor(rewriter, loc, outputStorage, match.outputType,
                           /*readOnly=*/false);
  COOSpMVOp::create(rewriter, loc, coo.rowIndices, coo.columnIndices,
                    coo.values, vectorBuffer, outputBuffer);
  replaceContraction(op, rewriter, match, outputBuffer);
  return success();
}

LogicalResult rewriteLinalgSpMM(linalg::LinalgOp op,
                                PatternRewriter &rewriter) {
  if (!hasCanonicalSpMMIndexingMaps(op))
    return rewriter.notifyMatchFailure(
        op, "expected canonical matrix-matrix indexing maps");

  SparseContractionMatch match;
  if (succeeded(
          matchSparseContraction(op, /*denseInputRank=*/2, /*outputRank=*/2,
                                 SparseMatrixFormat::CSR, match, rewriter))) {
    // CSR exposes scalar row bounds and one value per compressed coordinate.
    Location loc = op.getLoc();
    CSRStorage csr = extractCSRStorage(rewriter, loc, match.matrix);
    Value rhsBuffer =
        bufferizeDenseTensor(rewriter, loc, match.denseInput,
                             match.denseInputType, /*readOnly=*/true);
    Value outputStorage = match.zeroFill.getDpsInitOperand(0)->get();
    Value outputBuffer =
        bufferizeDenseTensor(rewriter, loc, outputStorage, match.outputType,
                             /*readOnly=*/false);
    SpMMOp::create(rewriter, loc, csr.rowOffsets, csr.columnIndices, csr.values,
                   rhsBuffer, outputBuffer);
    replaceContraction(op, rewriter, match, outputBuffer);
    return success();
  }

  if (failed(matchSparseContraction(op, /*denseInputRank=*/2, /*outputRank=*/2,
                                    SparseMatrixFormat::BSR, match, rewriter)))
    return failure();

  // Canonical BSR exposes block-row bounds and block columns at level 1. Its
  // dense local-row/local-column levels flatten values in row-major order.
  Location loc = op.getLoc();
  BSRStorage bsr = extractBSRStorage(rewriter, loc, match.matrix);
  Value rhsBuffer = bufferizeDenseTensor(
      rewriter, loc, match.denseInput, match.denseInputType, /*readOnly=*/true);
  Value outputStorage = match.zeroFill.getDpsInitOperand(0)->get();
  Value outputBuffer =
      bufferizeDenseTensor(rewriter, loc, outputStorage, match.outputType,
                           /*readOnly=*/false);
  BSRSpMMOp::create(rewriter, loc, bsr.blockRowOffsets, bsr.blockColumnIndices,
                    bsr.blockValues, rhsBuffer, outputBuffer, match.blockSize);
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

class ConvertLinalgSDDMMToSparseWave
    : public impl::ConvertLinalgSDDMMToSparseWaveBase<
          ConvertLinalgSDDMMToSparseWave> {
public:
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<LinalgSDDMMPattern>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }

private:
  class LinalgSDDMMPattern : public OpRewritePattern<linalg::GenericOp> {
  public:
    using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(linalg::GenericOp op,
                                  PatternRewriter &rewriter) const override {
      return rewriteLinalgSDDMM(op, rewriter);
    }
  };
};

} // namespace
} // namespace mlir::sparsewave
