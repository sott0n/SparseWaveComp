// RUN: sparsewave-opt %s \
// RUN:   --convert-sparsewave-to-gpu='spmm-block-size=64' \
// RUN:   | FileCheck %s

// CHECK-LABEL: func.func @bsr_spmm(
// CHECK: %[[GPU_BLOCK_SIZE:.*]] = arith.constant 64 : index
// CHECK: %[[BSR_BLOCK_SIZE:.*]] = arith.constant 2 : index
// CHECK: %[[VALUES_PER_BLOCK:.*]] = arith.constant 4 : index
// CHECK: %[[ROWS:.*]] = memref.dim %{{.*}}, %{{.*}} : memref<?x?xf32>
// CHECK: %[[COLUMNS:.*]] = memref.dim %{{.*}}, %{{.*}} : memref<?x?xf32>
// CHECK: %[[ELEMENTS:.*]] = arith.muli %[[ROWS]], %[[COLUMNS]] : index
// CHECK: gpu.launch blocks
// CHECK-SAME: = %[[GPU_BLOCK_SIZE]],
// CHECK: %[[ROW:.*]] = arith.divui %{{.*}}, %[[COLUMNS]] : index
// CHECK: %[[OUTPUT_COLUMN:.*]] = arith.remui %{{.*}}, %[[COLUMNS]] : index
// CHECK: %[[BLOCK_ROW:.*]] = arith.divui %[[ROW]], %[[BSR_BLOCK_SIZE]]
// CHECK: %[[LOCAL_ROW:.*]] = arith.remui %[[ROW]], %[[BSR_BLOCK_SIZE]]
// CHECK: %[[BLOCK_START_VALUE:.*]] = memref.load %{{.*}}[%[[BLOCK_ROW]]]
// CHECK: scf.for %[[BLOCK_POSITION:.*]] = %{{.*}} to %{{.*}} step %{{.*}}
// CHECK: %[[BLOCK_COLUMN_VALUE:.*]] = memref.load %{{.*}}[%[[BLOCK_POSITION]]]
// CHECK: scf.for %[[LOCAL_COLUMN:.*]] = %{{.*}} to %[[BSR_BLOCK_SIZE]] step %{{.*}}
// CHECK: %[[BLOCK_VALUE:.*]] = memref.load %{{.*}}[%{{.*}}]
// CHECK: %[[RHS_VALUE:.*]] = memref.load %{{.*}}[%{{.*}}, %[[OUTPUT_COLUMN]]]
// CHECK: %[[PRODUCT:.*]] = arith.mulf %[[BLOCK_VALUE]], %[[RHS_VALUE]]
// CHECK: memref.store %{{.*}}, %{{.*}}[%[[ROW]], %[[OUTPUT_COLUMN]]]
// CHECK-NOT: sparsewave.bsr_spmm
func.func @bsr_spmm(
    %blockRowOffsets: memref<?xi32>,
    %blockColumnIndices: memref<?xi32>,
    %blockValues: memref<?xf32>,
    %rhs: memref<?x?xf32>,
    %output: memref<?x?xf32>) {
  sparsewave.bsr_spmm %blockRowOffsets, %blockColumnIndices, %blockValues,
      %rhs, %output block_size = 2
      : memref<?xi32>, memref<?xi32>, memref<?xf32>,
        memref<?x?xf32>, memref<?x?xf32>
  return
}
