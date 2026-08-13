// RUN: sparsewave-opt %s \
// RUN:   --convert-sparsewave-to-gpu='sddmm-block-size=64' \
// RUN:   | FileCheck %s

// CHECK-LABEL: func.func @sddmm(
// CHECK: %[[ALPHA:.*]] = arith.constant 5.000000e-01 : f32
// CHECK: %[[BETA:.*]] = arith.constant 2.000000e+00 : f32
// CHECK: %[[ZERO:.*]] = arith.constant 0 : index
// CHECK: %[[ONE:.*]] = arith.constant 1 : index
// CHECK: %[[BLOCK_SIZE:.*]] = arith.constant 64 : index
// CHECK: %[[ROWS:.*]] = memref.dim %[[LHS:.*]], %[[ZERO]]
// CHECK: %[[K_SIZE:.*]] = memref.dim %[[LHS]], %[[ONE]]
// CHECK: gpu.launch blocks
// CHECK-SAME: threads(
// CHECK-SAME: = %[[BLOCK_SIZE]],
// CHECK: %[[ROW_START_VALUE:.*]] = memref.load %{{.*}}[%{{.*}}]
// CHECK: %[[ROW_START:.*]] = arith.index_cast %[[ROW_START_VALUE]]
// CHECK: scf.for %[[POSITION:.*]] = %[[ROW_START]]
// CHECK: %[[COLUMN_VALUE:.*]] = memref.load %{{.*}}[%[[POSITION]]]
// CHECK: %[[COLUMN:.*]] = arith.index_cast %[[COLUMN_VALUE]]
// CHECK: %[[SPARSE_VALUE:.*]] = memref.load %{{.*}}[%[[POSITION]]]
// CHECK: %[[DOT:.*]] = scf.for %[[K:.*]] = %[[ZERO]] to %[[K_SIZE]]
// CHECK: %[[LHS_VALUE:.*]] = memref.load %[[LHS]][%{{.*}}, %[[K]]]
// CHECK: %[[RHS_VALUE:.*]] = memref.load %{{.*}}[%[[K]], %[[COLUMN]]]
// CHECK: %[[PRODUCT:.*]] = arith.mulf %[[LHS_VALUE]], %[[RHS_VALUE]]
// CHECK: arith.addf
// CHECK: %[[SCALED_SAMPLE:.*]] = arith.mulf %[[SPARSE_VALUE]], %[[BETA]]
// CHECK: %[[SCALED_DOT:.*]] = arith.mulf %[[DOT]], %[[ALPHA]]
// CHECK: %[[COMBINED:.*]] = arith.addf %[[SCALED_SAMPLE]], %[[SCALED_DOT]]
// CHECK: memref.store %[[COMBINED]], %{{.*}}[%[[POSITION]]]
// CHECK-NOT: sparsewave.sddmm
func.func @sddmm(
    %rowOffsets: memref<?xi32>,
    %columnIndices: memref<?xi32>,
    %values: memref<?xf32>,
    %lhs: memref<?x?xf32>,
    %rhs: memref<?x?xf32>,
    %outputValues: memref<?xf32>) {
  sparsewave.sddmm %rowOffsets, %columnIndices, %values, %lhs, %rhs,
      %outputValues {
    ^bb0(%sample: f32, %dot: f32):
      %beta = arith.constant 2.0 : f32
      %alpha = arith.constant 0.5 : f32
      %scaledSample = arith.mulf %sample, %beta : f32
      %scaledDot = arith.mulf %dot, %alpha : f32
      %combined = arith.addf %scaledSample, %scaledDot : f32
      sparsewave.yield %combined : f32
  }
      : memref<?xi32>, memref<?xi32>, memref<?xf32>,
        memref<?x?xf32>, memref<?x?xf32>, memref<?xf32>
  return
}
