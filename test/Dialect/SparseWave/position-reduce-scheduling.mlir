// RUN: sparsewave-opt %s \
// RUN:   --schedule-sparsewave-position='mapping=thread block-size=64' \
// RUN:   | FileCheck %s --check-prefix=THREAD
// RUN: sparsewave-opt %s \
// RUN:   --schedule-sparsewave-position='mapping=wave block-size=64 wave-size=32' \
// RUN:   | FileCheck %s --check-prefix=WAVE
// RUN: sparsewave-opt %s \
// RUN:   --schedule-sparsewave-position='mapping=thread block-size=64 thread-chunk-size=4 thread-reduction=segmented' \
// RUN:   | FileCheck %s --check-prefix=SEGMENTED
// RUN: sparsewave-opt %s \
// RUN:   --schedule-sparsewave-position='mapping=thread block-size=64' \
// RUN:   | FileCheck %s --check-prefix=MULTI-AXIS

// This keyed reduction deliberately has no CSR or SpMV operations. It verifies
// that position scheduling depends only on position_reduce semantics.

// THREAD-LABEL: func.func @keyed_sum(
// THREAD-NOT: sparsewave.position_reduce
// THREAD: sparsewave.position_parallel %{{.*}} mapping = "thread" block_size = 64 {
// THREAD: memref.store
// THREAD: sparsewave.position_parallel %{{.*}} mapping = "thread" block_size = 64 {
// THREAD: sparsewave.position_for %{{.*}} in %{{.*}} to %{{.*}} by 1
// THREAD: arith.remui
// THREAD: memref.load
// THREAD: memref.atomic_rmw addf
// THREAD-NOT: gpu.shuffle

// WAVE-LABEL: func.func @keyed_sum(
// WAVE-NOT: sparsewave.position_reduce
// WAVE: sparsewave.position_parallel %{{.*}} mapping = "thread" block_size = 64 {
// WAVE: memref.store
// WAVE: sparsewave.position_parallel %{{.*}} mapping = "wave" block_size = 64 {
// WAVE: sparsewave.position_space
// WAVE: arith.remui
// WAVE: memref.load
// WAVE: gpu.shuffle up
// WAVE: memref.atomic_rmw addf

// SEGMENTED-LABEL: func.func @keyed_sum(
// SEGMENTED-NOT: sparsewave.position_reduce
// SEGMENTED: sparsewave.position_parallel %{{.*}} mapping = "thread" block_size = 64 {
// SEGMENTED: scf.for
// SEGMENTED: arith.remui
// SEGMENTED: arith.addf
// SEGMENTED: memref.atomic_rmw addf
// SEGMENTED: memref.atomic_rmw addf
// SEGMENTED-NOT: sparsewave.csr_row_at_position

func.func @keyed_sum(%values: memref<?xf32>, %output: memref<?xf32>) {
  %zero = arith.constant 0 : index
  %keys = arith.constant 4 : index
  %upper = memref.dim %values, %zero : memref<?xf32>
  sparsewave.position_reduce lower (%zero) upper (%upper)
      axes = ["position"] order = [0]
      into %output kind = "sum" {
  ^bb0(%position: index):
    %key = arith.remui %position, %keys : index
    %value = memref.load %values[%position] : memref<?xf32>
    sparsewave.yield %key, %value : index, f32
  } : memref<?xf32>
  return
}

// This rank-2 reduction verifies that collapse order and coordinate recovery
// belong to the generic scheduler rather than an SpMM decomposition pattern.

// MULTI-AXIS-LABEL: func.func @multi_axis_keyed_sum(
// MULTI-AXIS-NOT: sparsewave.position_reduce
// MULTI-AXIS: %[[COUNT:.*]] = arith.muli
// MULTI-AXIS: sparsewave.position_for
// MULTI-AXIS: arith.remui
// MULTI-AXIS: arith.divui
// MULTI-AXIS: memref.load %{{.*}}[%{{.*}}, %{{.*}}]
// MULTI-AXIS: memref.atomic_rmw addf

func.func @multi_axis_keyed_sum(%values: memref<?x?xf32>,
                                %output: memref<?xf32>) {
  %zero = arith.constant 0 : index
  %positions = memref.dim %values, %zero : memref<?x?xf32>
  %one = arith.constant 1 : index
  %columns = memref.dim %values, %one : memref<?x?xf32>
  sparsewave.position_reduce lower (%zero, %zero)
      upper (%positions, %columns) axes = ["position", "rhs"] order = [1, 0]
      into %output kind = "sum" {
  ^bb0(%position: index, %column: index):
    %rowBase = arith.muli %position, %columns : index
    %key = arith.addi %rowBase, %column : index
    %value = memref.load %values[%position, %column] : memref<?x?xf32>
    sparsewave.yield %key, %value : index, f32
  } : memref<?xf32>
  return
}
