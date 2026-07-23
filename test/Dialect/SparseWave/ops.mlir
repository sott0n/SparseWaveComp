// RUN: sparsewave-opt %s | FileCheck %s

// CHECK-LABEL: func.func @static_spmv(
func.func @static_spmv(
    %rowOffsets: memref<5xi32>, %columnIndices: memref<8xi32>,
    %values: memref<8xf32>, %vector: memref<4xf32>,
    %output: memref<4xf32>) {
  // CHECK: sparsewave.spmv %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : memref<5xi32>, memref<8xi32>, memref<8xf32>, memref<4xf32>, memref<4xf32>
  sparsewave.spmv %rowOffsets, %columnIndices, %values, %vector, %output
      : memref<5xi32>, memref<8xi32>, memref<8xf32>,
        memref<4xf32>, memref<4xf32>
  return
}

// CHECK-LABEL: func.func @dynamic_spmv(
func.func @dynamic_spmv(
    %rowOffsets: memref<?xindex>, %columnIndices: memref<?xindex>,
    %values: memref<?xf64>, %vector: memref<?xf64>,
    %output: memref<?xf64>) {
  // CHECK: sparsewave.spmv
  sparsewave.spmv %rowOffsets, %columnIndices, %values, %vector, %output
      : memref<?xindex>, memref<?xindex>, memref<?xf64>,
        memref<?xf64>, memref<?xf64>
  return
}
