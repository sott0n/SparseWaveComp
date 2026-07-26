// RUN: sparsewave-opt %s \
// RUN:   --convert-sparsewave-to-gpu='block-size=128' \
// RUN:   | FileCheck %s

// CHECK-LABEL: func.func @spmm(
// CHECK: %[[ROWS:.*]] = memref.dim %{{.*}}, %{{.*}} : memref<?x?xf32>
// CHECK: %[[COLUMNS:.*]] = memref.dim %{{.*}}, %{{.*}} : memref<?x?xf32>
// CHECK: %[[ELEMENTS:.*]] = arith.muli %[[ROWS]], %[[COLUMNS]] : index
// CHECK: gpu.launch blocks
// CHECK: %[[ROW:.*]] = arith.divui %{{.*}}, %[[COLUMNS]] : index
// CHECK: %[[COLUMN:.*]] = arith.remui %{{.*}}, %[[COLUMNS]] : index
// CHECK: %[[START_VALUE:.*]] = memref.load %{{.*}}[%[[ROW]]]
// CHECK: scf.for
// CHECK: %[[RHS_VALUE:.*]] = memref.load %{{.*}}[%{{.*}}, %[[COLUMN]]]
// CHECK: arith.mulf
// CHECK: memref.store %{{.*}}, %{{.*}}[%[[ROW]], %[[COLUMN]]]
// CHECK-NOT: sparsewave.spmm
func.func @spmm(
    %rowOffsets: memref<?xi32>,
    %columnIndices: memref<?xi32>,
    %values: memref<?xf32>,
    %rhs: memref<?x?xf32>,
    %output: memref<?x?xf32>) {
  sparsewave.spmm %rowOffsets, %columnIndices, %values, %rhs, %output
      : memref<?xi32>, memref<?xi32>, memref<?xf32>,
        memref<?x?xf32>, memref<?x?xf32>
  return
}
