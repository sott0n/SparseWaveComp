#include "sparsewave/Dialect/SparseWave/IR/SparseWaveOps.h"

#include "mlir/IR/Diagnostics.h"

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

LogicalResult verifyCSRStorage(Operation *op, MemRefType rowOffsetsType,
                               MemRefType columnIndicesType,
                               MemRefType valuesType, MemRefType outputType,
                               StringRef outputDimensionName) {
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

  int64_t rowOffsetsSize = rowOffsetsType.getDimSize(0);
  int64_t outputRows = outputType.getDimSize(0);
  if (!ShapedType::isDynamic(rowOffsetsSize) &&
      !ShapedType::isDynamic(outputRows) && rowOffsetsSize != outputRows + 1)
    return op->emitOpError()
           << "row offsets size must equal " << outputDimensionName
           << " plus one, but got " << rowOffsetsSize << " and " << outputRows;

  int64_t columnIndicesSize = columnIndicesType.getDimSize(0);
  int64_t valuesSize = valuesType.getDimSize(0);
  if (!areCompatibleStaticDimensions(columnIndicesSize, valuesSize))
    return op->emitOpError()
           << "column indices and values must have the same size, but got "
           << columnIndicesSize << " and " << valuesSize;
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

#define GET_OP_CLASSES
#include "sparsewave/Dialect/SparseWave/IR/SparseWaveOps.cpp.inc"
