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

// CHECK-LABEL: func.func @static_coo_spmv(
func.func @static_coo_spmv(
    %rowIndices: memref<8xi32>, %columnIndices: memref<8xi32>,
    %values: memref<8xf32>, %vector: memref<4xf32>,
    %output: memref<4xf32>) {
  // CHECK: sparsewave.coo_spmv
  sparsewave.coo_spmv %rowIndices, %columnIndices, %values, %vector, %output
      : memref<8xi32>, memref<8xi32>, memref<8xf32>,
        memref<4xf32>, memref<4xf32>
  return
}

// CHECK-LABEL: func.func @static_spmm(
func.func @static_spmm(
    %rowOffsets: memref<5xi32>, %columnIndices: memref<8xi32>,
    %values: memref<8xf32>, %rhs: memref<4x3xf32>,
    %output: memref<4x3xf32>) {
  // CHECK: sparsewave.spmm
  sparsewave.spmm %rowOffsets, %columnIndices, %values, %rhs, %output
      : memref<5xi32>, memref<8xi32>, memref<8xf32>,
        memref<4x3xf32>, memref<4x3xf32>
  return
}

// CHECK-LABEL: func.func @static_bsr_spmm(
func.func @static_bsr_spmm(
    %blockRowOffsets: memref<3xi32>, %blockColumnIndices: memref<3xi32>,
    %blockValues: memref<12xf32>, %rhs: memref<4x3xf32>,
    %output: memref<4x3xf32>) {
  // CHECK: sparsewave.bsr_spmm
  // CHECK-SAME: block_size = 2
  sparsewave.bsr_spmm %blockRowOffsets, %blockColumnIndices, %blockValues,
      %rhs, %output block_size = 2
      : memref<3xi32>, memref<3xi32>, memref<12xf32>,
        memref<4x3xf32>, memref<4x3xf32>
  return
}

// CHECK-LABEL: func.func @dynamic_bsr_spmm(
func.func @dynamic_bsr_spmm(
    %blockRowOffsets: memref<?xindex>, %blockColumnIndices: memref<?xindex>,
    %blockValues: memref<?xf64>, %rhs: memref<?x?xf64>,
    %output: memref<?x?xf64>) {
  // CHECK: sparsewave.bsr_spmm
  // CHECK-SAME: block_size = 4
  sparsewave.bsr_spmm %blockRowOffsets, %blockColumnIndices, %blockValues,
      %rhs, %output block_size = 4
      : memref<?xindex>, memref<?xindex>, memref<?xf64>,
        memref<?x?xf64>, memref<?x?xf64>
  return
}

// CHECK-LABEL: func.func @static_sddmm(
func.func @static_sddmm(
    %rowOffsets: memref<5xi32>, %columnIndices: memref<8xi32>,
    %values: memref<8xf32>, %lhs: memref<4x3xf32>,
    %rhs: memref<3x4xf32>, %outputValues: memref<8xf32>) {
  // CHECK: sparsewave.sddmm
  sparsewave.sddmm %rowOffsets, %columnIndices, %values, %lhs, %rhs,
      %outputValues
      : memref<5xi32>, memref<8xi32>, memref<8xf32>,
        memref<4x3xf32>, memref<3x4xf32>, memref<8xf32>
  return
}

// CHECK-LABEL: func.func @static_csr_row_reduce(
func.func @static_csr_row_reduce(
    %rowOffsets: memref<5xi32>, %columnIndices: memref<8xi32>,
    %values: memref<8xf32>, %output: memref<4xf32>) {
  // CHECK: sparsewave.csr_row_reduce
  // CHECK-SAME: kind = "sum"
  sparsewave.csr_row_reduce %rowOffsets, %columnIndices, %values, %output
      kind = "sum"
      : memref<5xi32>, memref<8xi32>, memref<8xf32>, memref<4xf32>
  return
}
