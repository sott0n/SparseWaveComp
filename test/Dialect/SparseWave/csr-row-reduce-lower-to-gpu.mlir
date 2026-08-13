// RUN: sparsewave-opt %s \
// RUN:   --convert-sparsewave-to-gpu='row-reduction-block-size=64' \
// RUN:   | FileCheck %s

// CHECK-LABEL: func.func @sum(
// CHECK: %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[BLOCK_SIZE:.*]] = arith.constant 64 : index
// CHECK: %[[ROWS:.*]] = memref.dim %[[OUTPUT:.*]], %{{.*}}
// CHECK: gpu.launch blocks
// CHECK-SAME: threads(
// CHECK-SAME: = %[[BLOCK_SIZE]],
// CHECK: %[[REDUCTION:.*]] = scf.for
// CHECK-SAME: iter_args({{.*}} = %[[ZERO]])
// CHECK: %[[VALUE:.*]] = memref.load %{{.*}}[%{{.*}}]
// CHECK: arith.addf %{{.*}}, %[[VALUE]]
// CHECK: memref.store %[[REDUCTION]], %[[OUTPUT]]
// CHECK-NOT: sparsewave.csr_row_reduce
func.func @sum(
    %rowOffsets: memref<?xi32>, %columnIndices: memref<?xi32>,
    %values: memref<?xf32>, %output: memref<?xf32>) {
  sparsewave.csr_row_reduce %rowOffsets, %columnIndices, %values, %output
      kind = "sum"
      : memref<?xi32>, memref<?xi32>, memref<?xf32>, memref<?xf32>
  return
}

// CHECK-LABEL: func.func @max(
// CHECK: %[[NEG_INF:.*]] = arith.constant 0xFF800000 : f32
// CHECK: %[[MAX_REDUCTION:.*]] = scf.for
// CHECK-SAME: iter_args({{.*}} = %[[NEG_INF]])
// CHECK: arith.maximumf
// CHECK: memref.store %[[MAX_REDUCTION]], %{{.*}}
// CHECK-NOT: sparsewave.csr_row_reduce
func.func @max(
    %rowOffsets: memref<?xi32>, %columnIndices: memref<?xi32>,
    %values: memref<?xf32>, %output: memref<?xf32>) {
  sparsewave.csr_row_reduce %rowOffsets, %columnIndices, %values, %output
      kind = "max"
      : memref<?xi32>, memref<?xi32>, memref<?xf32>, memref<?xf32>
  return
}
