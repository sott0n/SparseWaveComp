#include "sparsewave/Dialect/SparseWave/IR/SparseWaveOps.h"

#include "mlir/IR/Diagnostics.h"

using namespace mlir;
using namespace mlir::sparsewave;

namespace {

LogicalResult verifyRankOne(SpMVOp op, MemRefType type, StringRef name) {
  if (type.getRank() != 1)
    return op.emitOpError()
           << name << " must be a rank-1 memref, but got " << type;
  return success();
}

bool areCompatibleStaticDimensions(int64_t lhs, int64_t rhs) {
  return ShapedType::isDynamic(lhs) || ShapedType::isDynamic(rhs) || lhs == rhs;
}

} // namespace

LogicalResult SpMVOp::verify() {
  MemRefType rowOffsetsType = getRowOffsets().getType();
  MemRefType columnIndicesType = getColumnIndices().getType();
  MemRefType valuesType = getValues().getType();
  MemRefType vectorType = getVector().getType();
  MemRefType outputType = getOutput().getType();

  if (failed(verifyRankOne(*this, rowOffsetsType, "row offsets")) ||
      failed(verifyRankOne(*this, columnIndicesType, "column indices")) ||
      failed(verifyRankOne(*this, valuesType, "values")) ||
      failed(verifyRankOne(*this, vectorType, "vector")) ||
      failed(verifyRankOne(*this, outputType, "output")))
    return failure();

  Type indexType = rowOffsetsType.getElementType();
  if (!indexType.isIntOrIndex())
    return emitOpError()
           << "row offsets must have integer or index elements, but got "
           << indexType;
  if (columnIndicesType.getElementType() != indexType)
    return emitOpError()
           << "row offsets and column indices must have the same element type";

  Type valueType = valuesType.getElementType();
  if (!isa<FloatType>(valueType))
    return emitOpError() << "values must have floating-point elements, but got "
                         << valueType;
  if (vectorType.getElementType() != valueType ||
      outputType.getElementType() != valueType)
    return emitOpError()
           << "values, vector, and output must have the same element type";

  int64_t rowOffsetsSize = rowOffsetsType.getDimSize(0);
  int64_t outputSize = outputType.getDimSize(0);
  if (!ShapedType::isDynamic(rowOffsetsSize) &&
      !ShapedType::isDynamic(outputSize) && rowOffsetsSize != outputSize + 1)
    return emitOpError()
           << "row offsets size must equal output size plus one, but got "
           << rowOffsetsSize << " and " << outputSize;

  int64_t columnIndicesSize = columnIndicesType.getDimSize(0);
  int64_t valuesSize = valuesType.getDimSize(0);
  if (!areCompatibleStaticDimensions(columnIndicesSize, valuesSize))
    return emitOpError()
           << "column indices and values must have the same size, but got "
           << columnIndicesSize << " and " << valuesSize;

  return success();
}

#define GET_OP_CLASSES
#include "sparsewave/Dialect/SparseWave/IR/SparseWaveOps.cpp.inc"
