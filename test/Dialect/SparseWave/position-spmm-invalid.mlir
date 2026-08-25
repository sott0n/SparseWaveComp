// RUN: not sparsewave-opt %s --decompose-position-spmm 2>&1 \
// RUN:   | FileCheck %s
// RUN: not sparsewave-opt %s \
// RUN:   --decompose-position-spmm='iteration-order=diagonal' 2>&1 \
// RUN:   | FileCheck %s --check-prefix=ORDER

// CHECK: position-space SpMM requires an identity-layout output memref
// ORDER: unsupported position-space SpMM iteration order 'diagonal'; expected 'position-major' or 'rhs-major'

func.func @strided_output(
    %rowOffsets: memref<?xi32>,
    %columnIndices: memref<?xi32>,
    %values: memref<?xf32>,
    %rhs: memref<?x?xf32>,
    %output: memref<?x?xf32, strided<[?, ?], offset: ?>>) {
  sparsewave.spmm %rowOffsets, %columnIndices, %values, %rhs, %output
      : memref<?xi32>, memref<?xi32>, memref<?xf32>,
        memref<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>
  return
}
