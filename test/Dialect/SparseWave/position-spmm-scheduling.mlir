// RUN: sparsewave-opt %s \
// RUN:   --decompose-position-spmm \
// RUN:   | FileCheck %s --check-prefix=DECOMPOSE
// RUN: sparsewave-opt %s \
// RUN:   --pass-pipeline='builtin.module(decompose-position-spmm,reorder-sparsewave-position{order=rhs,position})' \
// RUN:   | FileCheck %s --check-prefix=RHS-MAJOR
// RUN: sparsewave-opt %s \
// RUN:   --pass-pipeline='builtin.module(decompose-position-spmm,schedule-sparsewave-position{mapping=thread block-size=128 thread-chunk-size=4})' \
// RUN:   | FileCheck %s --check-prefix=THREAD
// RUN: sparsewave-opt %s \
// RUN:   --pass-pipeline='builtin.module(decompose-position-spmm,reorder-sparsewave-position{order=rhs,position},schedule-sparsewave-position{mapping=thread block-size=128 thread-chunk-size=4 thread-reduction=segmented})' \
// RUN:   | FileCheck %s --check-prefix=RHS-SEGMENTED
// RUN: sparsewave-opt %s \
// RUN:   --pass-pipeline='builtin.module(decompose-position-spmm,schedule-sparsewave-position{mapping=wave block-size=64 wave-size=32 cooperative-axis=rhs})' \
// RUN:   | FileCheck %s --check-prefix=WAVE-COOPERATIVE
// RUN: sparsewave-opt %s \
// RUN:   --pass-pipeline='builtin.module(decompose-position-spmm,schedule-sparsewave-position{mapping=wave block-size=64 wave-size=32 cooperative-axis=rhs cooperative-chunk-size=4})' \
// RUN:   | FileCheck %s --check-prefix=WAVE-COOPERATIVE-CHUNK

// DECOMPOSE-LABEL: func.func @spmm(
// DECOMPOSE: %[[NNZ:.*]] = memref.dim %{{.*}}, %{{.*}} : memref<?xf32>
// DECOMPOSE: %[[RHS_COLS:.*]] = memref.dim %{{.*}}, %{{.*}} : memref<?x?xf32>
// DECOMPOSE: %[[FLAT_OUTPUT:.*]] = memref.collapse_shape
// DECOMPOSE: sparsewave.position_reduce lower(%{{.*}}, %{{.*}}) upper(%[[NNZ]], %[[RHS_COLS]]) axes = ["position", "rhs"] order = [0, 1] into %[[FLAT_OUTPUT]] kind = "sum" {
// DECOMPOSE: ^bb0(%[[POSITION:.*]]: index, %[[OUTPUT_COL:.*]]: index):
// DECOMPOSE: %[[ROW:.*]] = sparsewave.compressed_segment_at_position %{{.*}} at %[[POSITION]]
// DECOMPOSE: memref.load %{{.*}}[%[[POSITION]]]
// DECOMPOSE: memref.load %{{.*}}[%{{.*}}, %[[OUTPUT_COL]]]
// DECOMPOSE: %[[PRODUCT:.*]] = arith.mulf
// DECOMPOSE: %[[ROW_BASE:.*]] = arith.muli %[[ROW]], %[[RHS_COLS]] : index
// DECOMPOSE: %[[KEY:.*]] = arith.addi %[[ROW_BASE]], %[[OUTPUT_COL]] : index
// DECOMPOSE: sparsewave.yield %[[KEY]], %[[PRODUCT]] : index, f32
// DECOMPOSE-NOT: sparsewave.spmm

// RHS-MAJOR-LABEL: func.func @spmm(
// RHS-MAJOR: %[[NNZ:.*]] = memref.dim %{{.*}}, %{{.*}} : memref<?xf32>
// RHS-MAJOR: %[[RHS_COLS:.*]] = memref.dim %{{.*}}, %{{.*}} : memref<?x?xf32>
// RHS-MAJOR: sparsewave.position_reduce lower(%{{.*}}, %{{.*}}) upper(%[[NNZ]], %[[RHS_COLS]]) axes = ["position", "rhs"] order = [1, 0]
// RHS-MAJOR: ^bb0(%[[POSITION:.*]]: index, %[[OUTPUT_COL:.*]]: index):
// RHS-MAJOR: sparsewave.compressed_segment_at_position %{{.*}} at %[[POSITION]]
// RHS-MAJOR: memref.load %{{.*}}[%{{.*}}, %[[OUTPUT_COL]]]
// RHS-MAJOR: %[[KEY:.*]] = arith.addi %{{.*}}, %[[OUTPUT_COL]] : index
// RHS-MAJOR: sparsewave.yield %[[KEY]], %{{.*}} : index, f32

// THREAD-LABEL: func.func @spmm(
// THREAD: sparsewave.position_parallel %{{.*}} mapping = "thread" block_size = 128 {
// THREAD: memref.store
// THREAD: sparsewave.position_parallel %{{.*}} mapping = "thread" block_size = 128 {
// THREAD: sparsewave.position_for %{{.*}} in %{{.*}} to %{{.*}} by 4
// THREAD: arith.remui
// THREAD: arith.divui
// THREAD: sparsewave.compressed_segment_at_position
// THREAD: memref.atomic_rmw addf
// THREAD-NOT: sparsewave.spmm

// RHS-SEGMENTED-LABEL: func.func @spmm(
// RHS-SEGMENTED: sparsewave.position_parallel
// RHS-SEGMENTED: scf.for
// RHS-SEGMENTED: arith.remui
// RHS-SEGMENTED: arith.divui
// RHS-SEGMENTED: arith.cmpi eq
// RHS-SEGMENTED: arith.addf
// RHS-SEGMENTED: scf.if
// RHS-SEGMENTED: memref.atomic_rmw addf
// RHS-SEGMENTED: memref.atomic_rmw addf
// RHS-SEGMENTED-NOT: sparsewave.spmm

// WAVE-COOPERATIVE-LABEL: func.func @spmm(
// WAVE-COOPERATIVE: sparsewave.position_parallel %{{.*}} mapping = "wave" block_size = 64 {
// WAVE-COOPERATIVE: scf.if %{{.*}} -> (index) {
// WAVE-COOPERATIVE:   sparsewave.compressed_segment_at_position
// WAVE-COOPERATIVE: gpu.shuffle idx %{{.*}} : i64
// WAVE-COOPERATIVE: scf.if %{{.*}} -> (i32) {
// WAVE-COOPERATIVE:   memref.load %{{.*}}[%{{.*}}] : memref<?xi32>
// WAVE-COOPERATIVE: gpu.shuffle idx %{{.*}} : i32
// WAVE-COOPERATIVE: scf.if %{{.*}} -> (f32) {
// WAVE-COOPERATIVE:   memref.load %{{.*}}[%{{.*}}] : memref<?xf32>
// WAVE-COOPERATIVE: gpu.shuffle idx %{{.*}} : f32
// WAVE-COOPERATIVE: scf.if %{{.*}} {
// WAVE-COOPERATIVE:   memref.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<?x?xf32>
// WAVE-COOPERATIVE:   memref.atomic_rmw addf
// WAVE-COOPERATIVE-NOT: sparsewave.spmm

// WAVE-COOPERATIVE-CHUNK-LABEL: func.func @spmm(
// WAVE-COOPERATIVE-CHUNK: arith.ceildivui %{{.*}}, %{{.*}} : index
// WAVE-COOPERATIVE-CHUNK: sparsewave.position_parallel %{{.*}} mapping = "wave" block_size = 64 {
// WAVE-COOPERATIVE-CHUNK: scf.for
// WAVE-COOPERATIVE-CHUNK: sparsewave.compressed_segment_at_position
// WAVE-COOPERATIVE-CHUNK: arith.cmpi eq
// WAVE-COOPERATIVE-CHUNK: memref.atomic_rmw addf
// WAVE-COOPERATIVE-CHUNK: arith.addf
// WAVE-COOPERATIVE-CHUNK: memref.atomic_rmw addf
// WAVE-COOPERATIVE-CHUNK-NOT: sparsewave.spmm

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
