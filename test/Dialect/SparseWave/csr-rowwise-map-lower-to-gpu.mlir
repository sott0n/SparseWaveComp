// RUN: sparsewave-opt %s \
// RUN:   --convert-sparsewave-to-gpu='rowwise-map-block-size=64' \
// RUN:   | FileCheck %s

// CHECK-LABEL: func.func @subtract_exp(
// CHECK: %[[BLOCK_SIZE:.*]] = arith.constant 64 : index
// CHECK: %[[ROWS:.*]] = memref.dim %[[ROW_VALUES:.*]], %{{.*}}
// CHECK: gpu.launch blocks
// CHECK-SAME: threads(
// CHECK-SAME: = %[[BLOCK_SIZE]],
// CHECK: %[[ROW_VALUE:.*]] = memref.load %[[ROW_VALUES]][%{{.*}}]
// CHECK: scf.for
// CHECK: %[[VALUE:.*]] = memref.load %{{.*}}[%{{.*}}]
// CHECK: %[[SHIFTED:.*]] = arith.subf %[[VALUE]], %[[ROW_VALUE]]
// CHECK: %[[MAPPED:.*]] = math.exp %[[SHIFTED]]
// CHECK: memref.store %[[MAPPED]], %{{.*}}[%{{.*}}]
// CHECK-NOT: sparsewave.csr_rowwise_map
func.func @subtract_exp(
    %rowOffsets: memref<?xi32>, %columnIndices: memref<?xi32>,
    %values: memref<?xf32>, %rowValues: memref<?xf32>,
    %outputValues: memref<?xf32>) {
  sparsewave.csr_rowwise_map %rowOffsets, %columnIndices, %values, %rowValues,
      %outputValues {
    ^bb0(%value: f32, %rowValue: f32):
      %shifted = arith.subf %value, %rowValue : f32
      %mapped = math.exp %shifted : f32
      sparsewave.yield %mapped : f32
  } : memref<?xi32>, memref<?xi32>, memref<?xf32>, memref<?xf32>,
      memref<?xf32>
  return
}
