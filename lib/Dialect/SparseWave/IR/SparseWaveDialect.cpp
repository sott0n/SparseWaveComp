#include "sparsewave/Dialect/SparseWave/IR/SparseWaveDialect.h"

#include "sparsewave/Dialect/SparseWave/IR/SparseWaveOps.h"

using namespace mlir;
using namespace mlir::sparsewave;

#include "sparsewave/Dialect/SparseWave/IR/SparseWaveOpsDialect.cpp.inc"

void SparseWaveDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "sparsewave/Dialect/SparseWave/IR/SparseWaveOps.cpp.inc"
      >();
}
