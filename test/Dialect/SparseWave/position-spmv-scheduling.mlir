// RUN: sparsewave-opt %s \
// RUN:   --decompose-position-spmv \
// RUN:   | FileCheck %s --check-prefix=DECOMPOSE
// RUN: sparsewave-opt %s \
// RUN:   --pass-pipeline='builtin.module(decompose-position-spmv,schedule-sparsewave-position{mapping=thread block-size=128})' \
// RUN:   | FileCheck %s --check-prefix=THREAD
// RUN: sparsewave-opt %s \
// RUN:   --pass-pipeline='builtin.module(decompose-position-spmv,schedule-sparsewave-position{mapping=thread block-size=128 thread-chunk-size=4})' \
// RUN:   | FileCheck %s --check-prefix=CHUNK
// RUN: sparsewave-opt %s \
// RUN:   --pass-pipeline='builtin.module(decompose-position-spmv,schedule-sparsewave-position{mapping=wave block-size=128 wave-size=32})' \
// RUN:   | FileCheck %s --check-prefix=WAVE

// DECOMPOSE-LABEL: func.func @spmv(
// DECOMPOSE-NOT: gpu.launch
// DECOMPOSE-NOT: sparsewave.position_parallel
// DECOMPOSE: sparsewave.position_reduce %{{.*}} to %{{.*}} into %{{.*}} kind = "sum" {
// DECOMPOSE: ^bb0(%[[POSITION:.*]]: index):
// DECOMPOSE: %[[ROW:.*]], %{{.*}} = sparsewave.csr_coordinates %{{.*}}, %{{.*}} at %[[POSITION]]
// DECOMPOSE: %[[PRODUCT:.*]] = arith.mulf
// DECOMPOSE: sparsewave.yield %[[ROW]], %[[PRODUCT]] : index, f32
// DECOMPOSE-NOT: sparsewave.spmv

// THREAD-LABEL: func.func @spmv(
// THREAD-NOT: gpu.launch
// THREAD: sparsewave.position_parallel %{{.*}} mapping = "thread" block_size = 128 {
// THREAD: memref.store
// THREAD: sparsewave.position_parallel %[[WORKERS:.*]] mapping = "thread" block_size = 128 {
// THREAD-NEXT: ^bb0(%[[WORKER:[^,]+]]: index
// THREAD: sparsewave.position_for %[[WORKER]] in %{{.*}} to %{{.*}} by 1
// THREAD-NEXT: ^bb0(%[[POSITION:.*]]: index, %{{.*}}: index):
// THREAD: sparsewave.csr_coordinates %{{.*}}, %{{.*}} at %[[POSITION]]
// THREAD: memref.atomic_rmw addf
// THREAD-NOT: sparsewave.spmv

// CHUNK-LABEL: func.func @spmv(
// CHUNK: %[[COUNT:.*]] = arith.ceildivui %{{.*}}, %{{.*}} : index
// CHUNK: sparsewave.position_parallel %[[COUNT]] mapping = "thread" block_size = 128 {
// CHUNK-NEXT: ^bb0(%[[WORKER:[^,]+]]: index
// CHUNK: sparsewave.position_for %[[WORKER]] in %{{.*}} to %{{.*}} by 4
// CHUNK-NEXT: ^bb0(%[[POSITION:.*]]: index, %{{.*}}: index):
// CHUNK: sparsewave.csr_coordinates %{{.*}}, %{{.*}} at %[[POSITION]]
// CHUNK: memref.atomic_rmw addf

// WAVE-LABEL: func.func @spmv(
// WAVE-NOT: gpu.launch
// WAVE: sparsewave.position_parallel %{{.*}} mapping = "thread" block_size = 128 {
// WAVE: memref.store
// WAVE: sparsewave.position_parallel %[[WAVES:.*]] mapping = "wave" block_size = 128 {
// WAVE-NEXT: ^bb0(%[[WAVE:[^,]+]]: index, %[[LANE:[^,]+]]: index
// WAVE: %[[BEGIN:.*]], %[[END:.*]] = sparsewave.position_space
// WAVE-SAME: partition %[[WAVE]] of %[[WAVES]]
// WAVE: %[[POSITION:.*]] = arith.addi %[[BEGIN]], %[[LANE]]
// WAVE: sparsewave.csr_coordinates %{{.*}}, %{{.*}} at %[[POSITION]]
// WAVE: gpu.shuffle up
// WAVE: memref.atomic_rmw addf
// WAVE-NOT: sparsewave.spmv

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
