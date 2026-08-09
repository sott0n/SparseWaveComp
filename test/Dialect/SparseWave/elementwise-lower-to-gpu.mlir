// RUN: sparsewave-opt %s \
// RUN:   --convert-sparsewave-to-gpu='elementwise-block-size=64' \
// RUN:   | FileCheck %s

// CHECK-LABEL: func.func @csr_add(
// CHECK: %[[ZERO:.*]] = arith.constant 0 : index
// CHECK: %[[ONE:.*]] = arith.constant 1 : index
// CHECK: %[[BLOCK_SIZE:.*]] = arith.constant 64 : index
// CHECK: %[[OFFSET_COUNT:.*]] = memref.dim %[[OUTPUT_OFFSETS:.*]], %[[ZERO]]
// CHECK: %[[ROW_COUNT:.*]] = arith.subi %[[OFFSET_COUNT]], %[[ONE]]
// CHECK: gpu.launch
// CHECK-SAME: threads(
// CHECK-SAME: = %[[BLOCK_SIZE]],
// CHECK: scf.while
// CHECK: arith.ori
// CHECK: memref.store {{.*}}, %[[OUTPUT_OFFSETS]]
// CHECK: gpu.launch
// CHECK: scf.for
// CHECK: memref.store {{.*}}, %[[OUTPUT_OFFSETS]]
// CHECK: memref.store {{.*}}, %{{.*}}[%[[ZERO]]
// CHECK: gpu.launch
// CHECK: scf.while
// CHECK: arith.addf
// CHECK: memref.store {{.*}}, %{{.*}}[%{{.*}}]
// CHECK-NOT: sparsewave.csr_elementwise
func.func @csr_add(
    %lhsOffsets: memref<?xi32>, %lhsColumns: memref<?xi32>,
    %lhsValues: memref<?xf32>, %rhsOffsets: memref<?xi32>,
    %rhsColumns: memref<?xi32>, %rhsValues: memref<?xf32>,
    %outputOffsets: memref<?xi32>, %outputColumns: memref<?xi32>,
    %outputValues: memref<?xf32>, %outputNnz: memref<1xi32>) {
  sparsewave.csr_elementwise
      %lhsOffsets, %lhsColumns, %lhsValues,
      %rhsOffsets, %rhsColumns, %rhsValues,
      %outputOffsets, %outputColumns, %outputValues, %outputNnz kind = "add"
      : memref<?xi32>, memref<?xi32>, memref<?xf32>,
        memref<?xi32>, memref<?xi32>, memref<?xf32>,
        memref<?xi32>, memref<?xi32>, memref<?xf32>, memref<1xi32>
  return
}

// CHECK-LABEL: func.func @csr_multiply(
// CHECK: gpu.launch
// CHECK: scf.while
// CHECK: arith.andi
// CHECK: gpu.launch
// CHECK: scf.for
// CHECK: gpu.launch
// CHECK: scf.while
// CHECK: arith.mulf
// CHECK-NOT: sparsewave.csr_elementwise
func.func @csr_multiply(
    %lhsOffsets: memref<?xi32>, %lhsColumns: memref<?xi32>,
    %lhsValues: memref<?xf32>, %rhsOffsets: memref<?xi32>,
    %rhsColumns: memref<?xi32>, %rhsValues: memref<?xf32>,
    %outputOffsets: memref<?xi32>, %outputColumns: memref<?xi32>,
    %outputValues: memref<?xf32>, %outputNnz: memref<1xi32>) {
  sparsewave.csr_elementwise
      %lhsOffsets, %lhsColumns, %lhsValues,
      %rhsOffsets, %rhsColumns, %rhsValues,
      %outputOffsets, %outputColumns, %outputValues, %outputNnz
      kind = "multiply"
      : memref<?xi32>, memref<?xi32>, memref<?xf32>,
        memref<?xi32>, memref<?xi32>, memref<?xf32>,
        memref<?xi32>, memref<?xi32>, memref<?xf32>, memref<1xi32>
  return
}
