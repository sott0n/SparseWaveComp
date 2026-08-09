// RUN: sparsewave-opt %s | FileCheck %s

// CHECK-LABEL: func.func @csr_add(
func.func @csr_add(
    %lhsOffsets: memref<5xi32>, %lhsColumns: memref<8xi32>,
    %lhsValues: memref<8xf32>, %rhsOffsets: memref<5xi32>,
    %rhsColumns: memref<7xi32>, %rhsValues: memref<7xf32>,
    %outputOffsets: memref<5xi32>, %outputColumns: memref<15xi32>,
    %outputValues: memref<15xf32>, %outputNnz: memref<1xi32>) {
  // CHECK: sparsewave.csr_elementwise
  // CHECK-SAME: kind = "add"
  sparsewave.csr_elementwise
      %lhsOffsets, %lhsColumns, %lhsValues,
      %rhsOffsets, %rhsColumns, %rhsValues,
      %outputOffsets, %outputColumns, %outputValues, %outputNnz kind = "add"
      : memref<5xi32>, memref<8xi32>, memref<8xf32>,
        memref<5xi32>, memref<7xi32>, memref<7xf32>,
        memref<5xi32>, memref<15xi32>, memref<15xf32>, memref<1xi32>
  return
}

// CHECK-LABEL: func.func @csr_multiply_dynamic(
func.func @csr_multiply_dynamic(
    %lhsOffsets: memref<?xindex>, %lhsColumns: memref<?xindex>,
    %lhsValues: memref<?xf64>, %rhsOffsets: memref<?xindex>,
    %rhsColumns: memref<?xindex>, %rhsValues: memref<?xf64>,
    %outputOffsets: memref<?xindex>, %outputColumns: memref<?xindex>,
    %outputValues: memref<?xf64>, %outputNnz: memref<1xindex>) {
  // CHECK: sparsewave.csr_elementwise
  // CHECK-SAME: kind = "multiply"
  sparsewave.csr_elementwise
      %lhsOffsets, %lhsColumns, %lhsValues,
      %rhsOffsets, %rhsColumns, %rhsValues,
      %outputOffsets, %outputColumns, %outputValues, %outputNnz
      kind = "multiply"
      : memref<?xindex>, memref<?xindex>, memref<?xf64>,
        memref<?xindex>, memref<?xindex>, memref<?xf64>,
        memref<?xindex>, memref<?xindex>, memref<?xf64>, memref<1xindex>
  return
}
