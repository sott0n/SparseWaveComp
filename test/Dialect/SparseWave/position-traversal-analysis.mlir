// RUN: sparsewave-opt %s --report-sparsewave-position-traversal 2>&1 \
// RUN:   | FileCheck %s

// The analysis depends only on the logical axes and their collapse order.

// CHECK: remark: axis 'position' traversal is nondecreasing
// CHECK: remark: axis 'rhs' traversal is may-reset
func.func @position_major(%values: memref<?x?xf32>,
                          %output: memref<?xf32>) {
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

// CHECK: remark: axis 'position' traversal is may-reset
// CHECK: remark: axis 'rhs' traversal is nondecreasing
func.func @rhs_major(%values: memref<?x?xf32>, %output: memref<?xf32>) {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %positions = memref.dim %values, %zero : memref<?x?xf32>
  %columns = memref.dim %values, %one : memref<?x?xf32>
  sparsewave.position_reduce lower (%zero, %zero)
      upper (%positions, %columns) axes = ["position", "rhs"] order = [1, 0]
      into %output kind = "sum" {
  ^bb0(%position: index, %column: index):
    %key = arith.addi %position, %column : index
    %value = memref.load %values[%position, %column] : memref<?x?xf32>
    sparsewave.yield %key, %value : index, f32
  } : memref<?xf32>
  return
}
