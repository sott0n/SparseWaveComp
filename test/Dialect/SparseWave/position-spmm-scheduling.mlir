// RUN: sparsewave-opt %s \
// RUN:   --decompose-position-spmm \
// RUN:   | FileCheck %s --check-prefix=DECOMPOSE
// RUN: sparsewave-opt %s \
// RUN:   --pass-pipeline='builtin.module(decompose-position-spmm,schedule-sparsewave-position{mapping=thread block-size=128 thread-chunk-size=4})' \
// RUN:   | FileCheck %s --check-prefix=THREAD

// DECOMPOSE-LABEL: func.func @spmm(
// DECOMPOSE: %[[NNZ:.*]] = memref.dim %{{.*}}, %{{.*}} : memref<?xf32>
// DECOMPOSE: %[[RHS_COLS:.*]] = memref.dim %{{.*}}, %{{.*}} : memref<?x?xf32>
// DECOMPOSE: %[[ITERATIONS:.*]] = arith.muli %[[NNZ]], %[[RHS_COLS]] : index
// DECOMPOSE: %[[FLAT_OUTPUT:.*]] = memref.collapse_shape
// DECOMPOSE: sparsewave.position_reduce %{{.*}} to %[[ITERATIONS]] into %[[FLAT_OUTPUT]] kind = "sum" {
// DECOMPOSE: ^bb0(%[[ITERATION:.*]]: index):
// DECOMPOSE: %[[POSITION:.*]] = arith.divui %[[ITERATION]], %[[RHS_COLS]] : index
// DECOMPOSE: %[[OUTPUT_COL:.*]] = arith.remui %[[ITERATION]], %[[RHS_COLS]] : index
// DECOMPOSE: %[[ROW:.*]] = sparsewave.csr_row_at_position %{{.*}} at %[[POSITION]]
// DECOMPOSE: memref.load %{{.*}}[%[[POSITION]]]
// DECOMPOSE: memref.load %{{.*}}[%{{.*}}, %[[OUTPUT_COL]]]
// DECOMPOSE: %[[PRODUCT:.*]] = arith.mulf
// DECOMPOSE: %[[ROW_BASE:.*]] = arith.muli %[[ROW]], %[[RHS_COLS]] : index
// DECOMPOSE: %[[KEY:.*]] = arith.addi %[[ROW_BASE]], %[[OUTPUT_COL]] : index
// DECOMPOSE: sparsewave.yield %[[KEY]], %[[PRODUCT]] : index, f32
// DECOMPOSE-NOT: sparsewave.spmm

// THREAD-LABEL: func.func @spmm(
// THREAD: sparsewave.position_parallel %{{.*}} mapping = "thread" block_size = 128 {
// THREAD: memref.store
// THREAD: sparsewave.position_parallel %{{.*}} mapping = "thread" block_size = 128 {
// THREAD: sparsewave.position_for %{{.*}} in %{{.*}} to %{{.*}} by 4
// THREAD: arith.divui
// THREAD: arith.remui
// THREAD: sparsewave.csr_row_at_position
// THREAD: memref.atomic_rmw addf
// THREAD-NOT: sparsewave.spmm

func.func @spmm(
    %rowOffsets: memref<?xi32>,
    %columnIndices: memref<?xi32>,
    %values: memref<?xf32>,
    %rhs: memref<?x?xf32>,
    %output: memref<?x?xf32>) {
  sparsewave.spmm %rowOffsets, %columnIndices, %values, %rhs, %output
      : memref<?xi32>, memref<?xi32>, memref<?xf32>,
        memref<?x?xf32>, memref<?x?xf32>
  return
}
