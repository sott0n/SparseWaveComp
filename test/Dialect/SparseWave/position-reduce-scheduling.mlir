// RUN: sparsewave-opt %s \
// RUN:   --schedule-sparsewave-position='mapping=thread block-size=64' \
// RUN:   | FileCheck %s --check-prefix=THREAD
// RUN: sparsewave-opt %s \
// RUN:   --schedule-sparsewave-position='mapping=wave block-size=64 wave-size=32' \
// RUN:   | FileCheck %s --check-prefix=WAVE
// RUN: sparsewave-opt %s \
// RUN:   --schedule-sparsewave-position='mapping=thread block-size=64 thread-chunk-size=4 thread-reduction=segmented' \
// RUN:   | FileCheck %s --check-prefix=SEGMENTED

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
  sparsewave.position_reduce %zero to %upper into %output kind = "sum" {
  ^bb0(%position: index):
    %key = arith.remui %position, %keys : index
    %value = memref.load %values[%position] : memref<?xf32>
    sparsewave.yield %key, %value : index, f32
  } : index, memref<?xf32>
  return
}
