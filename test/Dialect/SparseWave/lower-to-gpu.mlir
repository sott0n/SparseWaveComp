// RUN: sparsewave-opt %s --convert-sparsewave-to-gpu | FileCheck %s
// RUN: sparsewave-opt %s \
// RUN:   --convert-sparsewave-to-gpu='mapping=thread-per-row block-size=128' \
// RUN:   | FileCheck %s --check-prefix=CUSTOM

// CHECK-LABEL: func.func @spmv(
// CHECK: %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[C1:.*]] = arith.constant 1 : index
// CHECK: %[[BLOCK_SIZE:.*]] = arith.constant 256 : index
// CHECK: %[[ROWS:.*]] = memref.dim %{{.*}}, %[[C0]]
// CHECK: %[[REQUIRED_BLOCKS:.*]] = arith.ceildivui %[[ROWS]], %[[BLOCK_SIZE]]
// CHECK: %[[GRID:.*]] = arith.maxui %[[REQUIRED_BLOCKS]], %[[C1]]
// CHECK: gpu.launch blocks(%[[BLOCK_X:.*]], %{{.*}}, %{{.*}}) in (%{{.*}} = %[[GRID]], %{{.*}} = %[[C1]], %{{.*}} = %[[C1]]) threads(%[[THREAD_X:.*]], %{{.*}}, %{{.*}}) in (%[[LAUNCH_BLOCK_SIZE:.*]] = %[[BLOCK_SIZE]],
// CHECK: %[[ROW_BASE:.*]] = arith.muli %[[BLOCK_X]], %[[LAUNCH_BLOCK_SIZE]]
// CHECK: %[[ROW:.*]] = arith.addi %[[ROW_BASE]], %[[THREAD_X]]
// CHECK: %[[ACTIVE:.*]] = arith.cmpi ult, %[[ROW]], %[[ROWS]]
// CHECK: scf.if %[[ACTIVE]]
// CHECK: %[[NEXT_ROW:.*]] = arith.addi %[[ROW]], %[[C1]]
// CHECK: %[[START_I32:.*]] = memref.load %{{.*}}[%[[ROW]]]
// CHECK: %[[END_I32:.*]] = memref.load %{{.*}}[%[[NEXT_ROW]]]
// CHECK: %[[START:.*]] = arith.index_cast %[[START_I32]] : i32 to index
// CHECK: %[[END:.*]] = arith.index_cast %[[END_I32]] : i32 to index
// CHECK: %[[SUM:.*]] = scf.for %[[POSITION:.*]] = %[[START]] to %[[END]]
// CHECK-SAME: step %[[C1]] iter_args(%[[ACC:.*]] = %[[ZERO]]) -> (f32)
// CHECK: %[[COLUMN_I32:.*]] = memref.load %{{.*}}[%[[POSITION]]]
// CHECK: %[[COLUMN:.*]] = arith.index_cast %[[COLUMN_I32]] : i32 to index
// CHECK: %[[MATRIX_VALUE:.*]] = memref.load %{{.*}}[%[[POSITION]]]
// CHECK: %[[VECTOR_VALUE:.*]] = memref.load %{{.*}}[%[[COLUMN]]]
// CHECK: %[[PRODUCT:.*]] = arith.mulf %[[MATRIX_VALUE]], %[[VECTOR_VALUE]]
// CHECK: %[[NEXT_SUM:.*]] = arith.addf %[[ACC]], %[[PRODUCT]]
// CHECK: scf.yield %[[NEXT_SUM]]
// CHECK: memref.store %[[SUM]], %{{.*}}[%[[ROW]]]
// CHECK-NOT: sparsewave.spmv

// CUSTOM: %[[BLOCK_SIZE:.*]] = arith.constant 128 : index
// CUSTOM: gpu.launch
// CUSTOM-SAME: threads(%{{.*}}, %{{.*}}, %{{.*}}) in
// CUSTOM-SAME: (%{{.*}} = %[[BLOCK_SIZE]],

func.func @spmv(
    %rowOffsets: memref<?xi32>,
    %columnIndices: memref<?xi32>,
    %values: memref<?xf32>,
    %vector: memref<?xf32>,
    %output: memref<?xf32>) {
  sparsewave.spmv %rowOffsets, %columnIndices, %values, %vector, %output
      : memref<?xi32>, memref<?xi32>, memref<?xf32>, memref<?xf32>,
        memref<?xf32>
  return
}
