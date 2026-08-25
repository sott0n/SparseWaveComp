// RUN: sparsewave-opt %s \
// RUN:   --reorder-sparsewave-position='order=rhs,position' \
// RUN:   | FileCheck %s

// This reduction deliberately has no SpMM operation. It verifies that reorder
// changes only the generic domain permutation and preserves logical arguments.

// CHECK-LABEL: func.func @keyed_sum(
// CHECK: sparsewave.position_reduce lower({{.*}}) upper({{.*}})
// CHECK-SAME: axes = ["position", "rhs"] order = [1, 0]
// CHECK: ^bb0(%[[POSITION:.*]]: index, %[[COLUMN:.*]]: index):
// CHECK: memref.load %{{.*}}[%[[POSITION]], %[[COLUMN]]]

func.func @keyed_sum(%values: memref<?x?xf32>, %output: memref<?xf32>) {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %positions = memref.dim %values, %zero : memref<?x?xf32>
  %columns = memref.dim %values, %one : memref<?x?xf32>
  sparsewave.position_reduce lower (%zero, %zero)
      upper (%positions, %columns) axes = ["position", "rhs"] order = [0, 1]
      into %output kind = "sum" {
  ^bb0(%position: index, %column: index):
    %key = arith.addi %position, %column : index
    %value = memref.load %values[%position, %column] : memref<?x?xf32>
    sparsewave.yield %key, %value : index, f32
  } : memref<?xf32>
  return
}
