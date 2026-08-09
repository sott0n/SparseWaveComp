// RUN: sparsewave-opt %s | FileCheck %s

// CHECK-LABEL: func.func @position_space(
func.func @position_space(
    %lower: index, %upper: index, %workerId: index, %workerCount: index,
    %rowOffsets: memref<?xi32>, %columnIndices: memref<?xi32>) {
  // CHECK: %[[BEGIN:.*]], %[[END:.*]] = sparsewave.position_space
  // CHECK-SAME: mapping = "wave" : index
  %begin, %end = sparsewave.position_space %lower to %upper
      partition %workerId of %workerCount mapping = "wave" : index

  // CHECK: sparsewave.csr_coordinates
  %row, %column = sparsewave.csr_coordinates
      %rowOffsets, %columnIndices at %begin : memref<?xi32>, memref<?xi32>
  return
}

// CHECK-LABEL: func.func @static_valid_partition(
func.func @static_valid_partition() {
  %lower = arith.constant 0 : index
  %upper = arith.constant 17 : index
  %workerId = arith.constant 3 : index
  %workerCount = arith.constant 8 : index
  // CHECK: sparsewave.position_space
  // CHECK-SAME: mapping = "thread" : index
  %begin, %end = sparsewave.position_space %lower to %upper
      partition %workerId of %workerCount mapping = "thread" : index
  return
}
