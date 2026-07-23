#include "sparsewave/Dialect/SparseWave/Transforms/Passes.h"

#include "sparsewave/Dialect/SparseWave/IR/SparseWaveOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::sparsewave {
#define GEN_PASS_DEF_CONVERTSPARSEWAVETOGPU
#include "sparsewave/Dialect/SparseWave/Transforms/Passes.h.inc"

namespace {

constexpr int64_t kBlockSize = 256;

Value castToIndex(OpBuilder &builder, Location loc, Value value) {
  if (value.getType().isIndex())
    return value;
  if (cast<IntegerType>(value.getType()).isUnsigned())
    return arith::IndexCastUIOp::create(builder, loc, builder.getIndexType(),
                                        value);
  return arith::IndexCastOp::create(builder, loc, builder.getIndexType(),
                                    value);
}

class SpMVToGPUPattern : public OpRewritePattern<SpMVOp> {
public:
  using OpRewritePattern<SpMVOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SpMVOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value zeroIndex = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value oneIndex = arith::ConstantIndexOp::create(rewriter, loc, 1);
    Value blockSize = arith::ConstantIndexOp::create(rewriter, loc, kBlockSize);
    Value rowCount =
        memref::DimOp::create(rewriter, loc, op.getOutput(), zeroIndex);
    Value requiredBlocks =
        arith::CeilDivUIOp::create(rewriter, loc, rowCount, blockSize);
    Value gridSize =
        arith::MaxUIOp::create(rewriter, loc, requiredBlocks, oneIndex);

    gpu::LaunchOp launch =
        gpu::LaunchOp::create(rewriter, loc, gridSize, oneIndex, oneIndex,
                              blockSize, oneIndex, oneIndex);
    rewriter.setInsertionPointToStart(&launch.getBody().front());

    Value rowBase = arith::MulIOp::create(rewriter, loc, launch.getBlockIds().x,
                                          launch.getBlockSize().x);
    Value row =
        arith::AddIOp::create(rewriter, loc, rowBase, launch.getThreadIds().x);
    Value rowIsActive = arith::CmpIOp::create(
        rewriter, loc, arith::CmpIPredicate::ult, row, rowCount);

    scf::IfOp::create(
        rewriter, loc, rowIsActive,
        [&](OpBuilder &builder, Location bodyLoc) {
          Value nextRow =
              arith::AddIOp::create(builder, bodyLoc, row, oneIndex);
          Value rowStartValue =
              memref::LoadOp::create(builder, bodyLoc, op.getRowOffsets(), row);
          Value rowEndValue = memref::LoadOp::create(
              builder, bodyLoc, op.getRowOffsets(), nextRow);
          Value rowStart = castToIndex(builder, bodyLoc, rowStartValue);
          Value rowEnd = castToIndex(builder, bodyLoc, rowEndValue);

          auto valueType =
              cast<MemRefType>(op.getValues().getType()).getElementType();
          Value zero = arith::ConstantOp::create(
              builder, bodyLoc, builder.getZeroAttr(valueType));
          auto reduction = scf::ForOp::create(
              builder, bodyLoc, rowStart, rowEnd, oneIndex, ValueRange{zero},
              [&](OpBuilder &loopBuilder, Location loopLoc, Value position,
                  ValueRange iterArgs) {
                Value columnValue = memref::LoadOp::create(
                    loopBuilder, loopLoc, op.getColumnIndices(), position);
                Value column = castToIndex(loopBuilder, loopLoc, columnValue);
                Value matrixValue = memref::LoadOp::create(
                    loopBuilder, loopLoc, op.getValues(), position);
                Value vectorValue = memref::LoadOp::create(
                    loopBuilder, loopLoc, op.getVector(), column);
                Value product = arith::MulFOp::create(loopBuilder, loopLoc,
                                                      matrixValue, vectorValue);
                Value sum = arith::AddFOp::create(loopBuilder, loopLoc,
                                                  iterArgs.front(), product);
                scf::YieldOp::create(loopBuilder, loopLoc, sum);
              });
          memref::StoreOp::create(builder, bodyLoc, reduction.getResult(0),
                                  op.getOutput(), row);
          scf::YieldOp::create(builder, bodyLoc);
        },
        {});

    rewriter.setInsertionPointToEnd(&launch.getBody().front());
    gpu::TerminatorOp::create(rewriter, loc);
    rewriter.eraseOp(op);
    return success();
  }
};

class ConvertSparseWaveToGPU
    : public impl::ConvertSparseWaveToGPUBase<ConvertSparseWaveToGPU> {
public:
  using impl::ConvertSparseWaveToGPUBase<
      ConvertSparseWaveToGPU>::ConvertSparseWaveToGPUBase;

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<SpMVToGPUPattern>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
} // namespace mlir::sparsewave
