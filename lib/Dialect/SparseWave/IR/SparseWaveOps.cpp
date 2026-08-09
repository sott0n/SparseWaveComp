#include "sparsewave/Dialect/SparseWave/IR/SparseWaveOps.h"

#include "mlir/IR/Diagnostics.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>

using namespace mlir;
using namespace mlir::sparsewave;

namespace {

LogicalResult verifyRank(Operation *op, MemRefType type, int64_t rank,
                         StringRef name) {
  if (type.getRank() != rank)
    return op->emitOpError()
           << name << " must be a rank-" << rank << " memref, but got " << type;
  return success();
}

bool areCompatibleStaticDimensions(int64_t lhs, int64_t rhs) {
  return ShapedType::isDynamic(lhs) || ShapedType::isDynamic(rhs) || lhs == rhs;
}

LogicalResult verifyCSRComponents(Operation *op, MemRefType rowOffsetsType,
                                  MemRefType columnIndicesType,
                                  MemRefType valuesType) {
  if (failed(verifyRank(op, rowOffsetsType, 1, "row offsets")) ||
      failed(verifyRank(op, columnIndicesType, 1, "column indices")) ||
      failed(verifyRank(op, valuesType, 1, "values")))
    return failure();

  Type indexType = rowOffsetsType.getElementType();
  if (!indexType.isIntOrIndex())
    return op->emitOpError()
           << "row offsets must have integer or index elements, but got "
           << indexType;
  if (columnIndicesType.getElementType() != indexType)
    return op->emitOpError()
           << "row offsets and column indices must have the same element type";

  Type valueType = valuesType.getElementType();
  if (!isa<FloatType>(valueType))
    return op->emitOpError()
           << "values must have floating-point elements, but got " << valueType;

  int64_t columnIndicesSize = columnIndicesType.getDimSize(0);
  int64_t valuesSize = valuesType.getDimSize(0);
  if (!areCompatibleStaticDimensions(columnIndicesSize, valuesSize))
    return op->emitOpError()
           << "column indices and values must have the same size, but got "
           << columnIndicesSize << " and " << valuesSize;
  return success();
}

LogicalResult verifyCSRStorage(Operation *op, MemRefType rowOffsetsType,
                               MemRefType columnIndicesType,
                               MemRefType valuesType, MemRefType outputType,
                               StringRef outputDimensionName) {
  if (failed(verifyCSRComponents(op, rowOffsetsType, columnIndicesType,
                                 valuesType)))
    return failure();

  int64_t rowOffsetsSize = rowOffsetsType.getDimSize(0);
  int64_t outputRows = outputType.getDimSize(0);
  if (!ShapedType::isDynamic(rowOffsetsSize) &&
      !ShapedType::isDynamic(outputRows) && rowOffsetsSize != outputRows + 1)
    return op->emitOpError()
           << "row offsets size must equal " << outputDimensionName
           << " plus one, but got " << rowOffsetsSize << " and " << outputRows;
  return success();
}

LogicalResult verifyCOOStorage(Operation *op, MemRefType rowIndicesType,
                               MemRefType columnIndicesType,
                               MemRefType valuesType) {
  if (failed(verifyRank(op, rowIndicesType, 1, "row indices")) ||
      failed(verifyRank(op, columnIndicesType, 1, "column indices")) ||
      failed(verifyRank(op, valuesType, 1, "values")))
    return failure();

  Type indexType = rowIndicesType.getElementType();
  if (!indexType.isIntOrIndex())
    return op->emitOpError()
           << "row indices must have integer or index elements, but got "
           << indexType;
  if (columnIndicesType.getElementType() != indexType)
    return op->emitOpError()
           << "row and column indices must have the same element type";

  Type valueType = valuesType.getElementType();
  if (!isa<FloatType>(valueType))
    return op->emitOpError()
           << "values must have floating-point elements, but got " << valueType;

  int64_t rowIndicesSize = rowIndicesType.getDimSize(0);
  int64_t columnIndicesSize = columnIndicesType.getDimSize(0);
  int64_t valuesSize = valuesType.getDimSize(0);
  if (!areCompatibleStaticDimensions(rowIndicesSize, columnIndicesSize) ||
      !areCompatibleStaticDimensions(rowIndicesSize, valuesSize) ||
      !areCompatibleStaticDimensions(columnIndicesSize, valuesSize))
    return op->emitOpError()
           << "row indices, column indices, and values must have the same size";
  return success();
}

LogicalResult verifyBSRStorage(Operation *op, MemRefType blockRowOffsetsType,
                               MemRefType blockColumnIndicesType,
                               MemRefType blockValuesType,
                               MemRefType outputType, int64_t blockSize) {
  if (failed(verifyRank(op, blockRowOffsetsType, 1, "block-row offsets")) ||
      failed(
          verifyRank(op, blockColumnIndicesType, 1, "block-column indices")) ||
      failed(verifyRank(op, blockValuesType, 1, "block values")))
    return failure();

  Type indexType = blockRowOffsetsType.getElementType();
  if (!indexType.isIntOrIndex())
    return op->emitOpError()
           << "block-row offsets must have integer or index elements, but got "
           << indexType;
  if (blockColumnIndicesType.getElementType() != indexType)
    return op->emitOpError()
           << "block-row offsets and block-column indices must have the same "
              "element type";

  Type valueType = blockValuesType.getElementType();
  if (!isa<FloatType>(valueType))
    return op->emitOpError()
           << "block values must have floating-point elements, but got "
           << valueType;
  if (blockSize <= 0)
    return op->emitOpError()
           << "block size must be positive, but got " << blockSize;

  int64_t outputRows = outputType.getDimSize(0);
  if (!ShapedType::isDynamic(outputRows)) {
    if (outputRows % blockSize != 0)
      return op->emitOpError()
             << "output rows must be divisible by block size, but got "
             << outputRows << " and " << blockSize;

    int64_t blockRowOffsetsSize = blockRowOffsetsType.getDimSize(0);
    int64_t expectedOffsetsSize = outputRows / blockSize + 1;
    if (!ShapedType::isDynamic(blockRowOffsetsSize) &&
        blockRowOffsetsSize != expectedOffsetsSize)
      return op->emitOpError()
             << "block-row offsets size must equal the number of block rows "
                "plus one, but got "
             << blockRowOffsetsSize << " and " << outputRows / blockSize;
  }

  int64_t blockCount = blockColumnIndicesType.getDimSize(0);
  int64_t blockValuesSize = blockValuesType.getDimSize(0);
  if (!ShapedType::isDynamic(blockCount) &&
      !ShapedType::isDynamic(blockValuesSize)) {
    int64_t valuesPerBlock;
    int64_t expectedValuesSize;
    if (llvm::MulOverflow(blockSize, blockSize, valuesPerBlock) ||
        llvm::MulOverflow(blockCount, valuesPerBlock, expectedValuesSize))
      return op->emitOpError()
             << "block size and block count exceed the supported index range";
    if (blockValuesSize != expectedValuesSize)
      return op->emitOpError()
             << "block values size must equal the number of blocks times "
                "block_size squared, but got "
             << blockValuesSize << " and " << blockCount << " blocks";
  }
  return success();
}

} // namespace

LogicalResult SpMVOp::verify() {
  MemRefType rowOffsetsType = getRowOffsets().getType();
  MemRefType columnIndicesType = getColumnIndices().getType();
  MemRefType valuesType = getValues().getType();
  MemRefType vectorType = getVector().getType();
  MemRefType outputType = getOutput().getType();

  if (failed(verifyRank(*this, vectorType, 1, "vector")) ||
      failed(verifyRank(*this, outputType, 1, "output")) ||
      failed(verifyCSRStorage(*this, rowOffsetsType, columnIndicesType,
                              valuesType, outputType, "output size")))
    return failure();

  Type valueType = valuesType.getElementType();
  if (vectorType.getElementType() != valueType ||
      outputType.getElementType() != valueType)
    return emitOpError()
           << "values, vector, and output must have the same element type";
  return success();
}

LogicalResult COOSpMVOp::verify() {
  MemRefType rowIndicesType = getRowIndices().getType();
  MemRefType columnIndicesType = getColumnIndices().getType();
  MemRefType valuesType = getValues().getType();
  MemRefType vectorType = getVector().getType();
  MemRefType outputType = getOutput().getType();

  if (failed(verifyRank(*this, vectorType, 1, "vector")) ||
      failed(verifyRank(*this, outputType, 1, "output")) ||
      failed(verifyCOOStorage(*this, rowIndicesType, columnIndicesType,
                              valuesType)))
    return failure();

  Type valueType = valuesType.getElementType();
  if (vectorType.getElementType() != valueType ||
      outputType.getElementType() != valueType)
    return emitOpError()
           << "values, vector, and output must have the same element type";
  return success();
}

LogicalResult SpMMOp::verify() {
  MemRefType rowOffsetsType = getRowOffsets().getType();
  MemRefType columnIndicesType = getColumnIndices().getType();
  MemRefType valuesType = getValues().getType();
  MemRefType rhsType = getRhs().getType();
  MemRefType outputType = getOutput().getType();

  if (failed(verifyRank(*this, rhsType, 2, "right-hand side")) ||
      failed(verifyRank(*this, outputType, 2, "output")) ||
      failed(verifyCSRStorage(*this, rowOffsetsType, columnIndicesType,
                              valuesType, outputType, "output rows")))
    return failure();

  Type valueType = valuesType.getElementType();
  if (rhsType.getElementType() != valueType ||
      outputType.getElementType() != valueType)
    return emitOpError()
           << "values, right-hand side, and output must have the same element "
              "type";
  if (!areCompatibleStaticDimensions(rhsType.getDimSize(1),
                                     outputType.getDimSize(1)))
    return emitOpError()
           << "right-hand side and output must have the same number of columns";

  return success();
}

LogicalResult BSRSpMMOp::verify() {
  MemRefType blockRowOffsetsType = getBlockRowOffsets().getType();
  MemRefType blockColumnIndicesType = getBlockColumnIndices().getType();
  MemRefType blockValuesType = getBlockValues().getType();
  MemRefType rhsType = getRhs().getType();
  MemRefType outputType = getOutput().getType();
  int64_t blockSize = getBlockSizeAttr().getInt();

  if (failed(verifyRank(*this, rhsType, 2, "right-hand side")) ||
      failed(verifyRank(*this, outputType, 2, "output")) ||
      failed(verifyBSRStorage(*this, blockRowOffsetsType,
                              blockColumnIndicesType, blockValuesType,
                              outputType, blockSize)))
    return failure();

  Type valueType = blockValuesType.getElementType();
  if (rhsType.getElementType() != valueType ||
      outputType.getElementType() != valueType)
    return emitOpError()
           << "block values, right-hand side, and output must have the same "
              "element type";
  if (!areCompatibleStaticDimensions(rhsType.getDimSize(1),
                                     outputType.getDimSize(1)))
    return emitOpError()
           << "right-hand side and output must have the same number of columns";

  int64_t rhsRows = rhsType.getDimSize(0);
  if (!ShapedType::isDynamic(rhsRows) && rhsRows % blockSize != 0)
    return emitOpError()
           << "right-hand-side rows must be divisible by block size, but got "
           << rhsRows << " and " << blockSize;
  return success();
}

LogicalResult SDDMMOp::verify() {
  MemRefType rowOffsetsType = getRowOffsets().getType();
  MemRefType columnIndicesType = getColumnIndices().getType();
  MemRefType valuesType = getValues().getType();
  MemRefType lhsType = getLhs().getType();
  MemRefType rhsType = getRhs().getType();
  MemRefType outputValuesType = getOutputValues().getType();

  if (failed(verifyRank(*this, lhsType, 2, "left-hand side")) ||
      failed(verifyRank(*this, rhsType, 2, "right-hand side")) ||
      failed(verifyRank(*this, outputValuesType, 1, "output values")) ||
      failed(verifyCSRStorage(*this, rowOffsetsType, columnIndicesType,
                              valuesType, lhsType, "left-hand-side rows")))
    return failure();

  Type valueType = valuesType.getElementType();
  if (lhsType.getElementType() != valueType ||
      rhsType.getElementType() != valueType ||
      outputValuesType.getElementType() != valueType)
    return emitOpError()
           << "values, dense operands, and output values must have the same "
              "element type";
  if (!areCompatibleStaticDimensions(lhsType.getDimSize(1),
                                     rhsType.getDimSize(0)))
    return emitOpError()
           << "left-hand-side columns and right-hand-side rows must match";
  if (!areCompatibleStaticDimensions(valuesType.getDimSize(0),
                                     outputValuesType.getDimSize(0)))
    return emitOpError()
           << "values and output values must have the same size, but got "
           << valuesType.getDimSize(0) << " and "
           << outputValuesType.getDimSize(0);

  return success();
}

LogicalResult CSRElementwiseOp::verify() {
  MemRefType lhsRowOffsetsType = getLhsRowOffsets().getType();
  MemRefType lhsColumnIndicesType = getLhsColumnIndices().getType();
  MemRefType lhsValuesType = getLhsValues().getType();
  MemRefType rhsRowOffsetsType = getRhsRowOffsets().getType();
  MemRefType rhsColumnIndicesType = getRhsColumnIndices().getType();
  MemRefType rhsValuesType = getRhsValues().getType();
  MemRefType outputRowOffsetsType = getOutputRowOffsets().getType();
  MemRefType outputColumnIndicesType = getOutputColumnIndices().getType();
  MemRefType outputValuesType = getOutputValues().getType();
  MemRefType outputNnzType = getOutputNnz().getType();

  if (getKind() != "add" && getKind() != "multiply")
    return emitOpError() << "kind must be 'add' or 'multiply', but got '"
                         << getKind() << "'";

  if (failed(verifyCSRComponents(*this, lhsRowOffsetsType, lhsColumnIndicesType,
                                 lhsValuesType)) ||
      failed(verifyCSRComponents(*this, rhsRowOffsetsType, rhsColumnIndicesType,
                                 rhsValuesType)) ||
      failed(
          verifyRank(*this, outputRowOffsetsType, 1, "output row offsets")) ||
      failed(verifyRank(*this, outputColumnIndicesType, 1,
                        "output column indices")) ||
      failed(verifyRank(*this, outputValuesType, 1, "output values")) ||
      failed(verifyRank(*this, outputNnzType, 1, "output NNZ")))
    return failure();

  if (!areCompatibleStaticDimensions(lhsRowOffsetsType.getDimSize(0),
                                     rhsRowOffsetsType.getDimSize(0)) ||
      !areCompatibleStaticDimensions(lhsRowOffsetsType.getDimSize(0),
                                     outputRowOffsetsType.getDimSize(0)) ||
      !areCompatibleStaticDimensions(rhsRowOffsetsType.getDimSize(0),
                                     outputRowOffsetsType.getDimSize(0)))
    return emitOpError()
           << "left, right, and output row offsets must have the same size";

  Type indexType = lhsRowOffsetsType.getElementType();
  if (rhsRowOffsetsType.getElementType() != indexType ||
      outputRowOffsetsType.getElementType() != indexType ||
      outputColumnIndicesType.getElementType() != indexType ||
      outputNnzType.getElementType() != indexType)
    return emitOpError()
           << "all row offsets, column indices, and output NNZ must have the "
              "same element type";

  Type valueType = lhsValuesType.getElementType();
  if (rhsValuesType.getElementType() != valueType ||
      outputValuesType.getElementType() != valueType)
    return emitOpError()
           << "left, right, and output values must have the same element type";

  if (!areCompatibleStaticDimensions(outputColumnIndicesType.getDimSize(0),
                                     outputValuesType.getDimSize(0)))
    return emitOpError()
           << "output column indices and values must have the same capacity, "
              "but got "
           << outputColumnIndicesType.getDimSize(0) << " and "
           << outputValuesType.getDimSize(0);

  int64_t lhsNnzCapacity = lhsColumnIndicesType.getDimSize(0);
  int64_t rhsNnzCapacity = rhsColumnIndicesType.getDimSize(0);
  int64_t outputCapacity = outputColumnIndicesType.getDimSize(0);
  if (!ShapedType::isDynamic(lhsNnzCapacity) &&
      !ShapedType::isDynamic(rhsNnzCapacity) &&
      !ShapedType::isDynamic(outputCapacity)) {
    int64_t requiredCapacity;
    if (getKind() == "add") {
      if (llvm::AddOverflow(lhsNnzCapacity, rhsNnzCapacity, requiredCapacity))
        return emitOpError()
               << "input capacities exceed the supported index range";
    } else {
      requiredCapacity = std::min(lhsNnzCapacity, rhsNnzCapacity);
    }
    if (outputCapacity < requiredCapacity)
      return emitOpError() << "output capacity for '" << getKind()
                           << "' must be at least " << requiredCapacity
                           << ", but got " << outputCapacity;
  }

  int64_t outputNnzSize = outputNnzType.getDimSize(0);
  if (!ShapedType::isDynamic(outputNnzSize) && outputNnzSize != 1)
    return emitOpError() << "output NNZ must contain one element, but got "
                         << outputNnzSize;

  return success();
}

#define GET_OP_CLASSES
#include "sparsewave/Dialect/SparseWave/IR/SparseWaveOps.cpp.inc"
