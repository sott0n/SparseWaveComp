// RUN: sparsewave-opt %s \
// RUN:   --decompose-position-spmv \
// RUN:   | FileCheck %s --check-prefix=DECOMPOSE
// RUN: sparsewave-opt %s \
// RUN:   --decompose-position-spmv='preserve-direct-mapping=true' \
// RUN:   | FileCheck %s --check-prefix=DIRECT
// RUN: sparsewave-opt %s \
// RUN:   --pass-pipeline='builtin.module(decompose-position-spmv,schedule-sparsewave-position{mapping=thread block-size=128})' \
// RUN:   | FileCheck %s --check-prefix=THREAD
// RUN: sparsewave-opt %s \
// RUN:   --pass-pipeline='builtin.module(decompose-position-spmv,schedule-sparsewave-position{mapping=thread block-size=128 thread-chunk-size=4})' \
// RUN:   | FileCheck %s --check-prefix=CHUNK
// RUN: sparsewave-opt %s \
// RUN:   --pass-pipeline='builtin.module(decompose-position-spmv,schedule-sparsewave-position{mapping=thread block-size=128 thread-chunk-size=4 thread-reduction=segmented})' \
// RUN:   | FileCheck %s --check-prefix=SEGMENTED
// RUN: sparsewave-opt %s \
// RUN:   --pass-pipeline='builtin.module(decompose-position-spmv,schedule-sparsewave-position{mapping=wave block-size=128 wave-size=32})' \
// RUN:   | FileCheck %s --check-prefix=WAVE

// DECOMPOSE-LABEL: func.func @spmv(
// DECOMPOSE-NOT: gpu.launch
// DECOMPOSE-NOT: sparsewave.position_parallel
// DECOMPOSE: sparsewave.position_reduce lower(%{{.*}}) upper(%{{.*}}) axes = ["position"] order = [0] into %{{.*}} kind = "sum" {
// DECOMPOSE: ^bb0(%[[POSITION:.*]]: index):
// DECOMPOSE: %[[ROW:.*]] = sparsewave.csr_row_at_position %{{.*}} at %[[POSITION]]
// DECOMPOSE: memref.load %{{.*}}[%[[POSITION]]]
// DECOMPOSE: %[[PRODUCT:.*]] = arith.mulf
// DECOMPOSE: sparsewave.yield %[[ROW]], %[[PRODUCT]] : index, f32
// DECOMPOSE-NOT: sparsewave.spmv

// DIRECT-LABEL: func.func @spmv(
// DIRECT: sparsewave.spmv
// DIRECT-NOT: sparsewave.position_reduce

// THREAD-LABEL: func.func @spmv(
// THREAD-NOT: gpu.launch
// THREAD: sparsewave.position_parallel %{{.*}} mapping = "thread" block_size = 128 {
// THREAD: memref.store
// THREAD: sparsewave.position_parallel %[[WORKERS:.*]] mapping = "thread" block_size = 128 {
// THREAD-NEXT: ^bb0(%[[WORKER:[^,]+]]: index
// THREAD: sparsewave.position_for %[[WORKER]] in %{{.*}} to %{{.*}} by 1
// THREAD-NEXT: ^bb0(%[[POSITION:.*]]: index, %{{.*}}: index):
// THREAD: sparsewave.csr_row_at_position %{{.*}} at %[[POSITION]]
// THREAD: memref.atomic_rmw addf
// THREAD-NOT: sparsewave.spmv

// CHUNK-LABEL: func.func @spmv(
// CHUNK: %[[COUNT:.*]] = arith.ceildivui %{{.*}}, %{{.*}} : index
// CHUNK: sparsewave.position_parallel %[[COUNT]] mapping = "thread" block_size = 128 {
// CHUNK-NEXT: ^bb0(%[[WORKER:[^,]+]]: index
// CHUNK: sparsewave.position_for %[[WORKER]] in %{{.*}} to %{{.*}} by 4
// CHUNK-NEXT: ^bb0(%[[POSITION:.*]]: index, %{{.*}}: index):
// CHUNK: sparsewave.csr_row_at_position %{{.*}} at %[[POSITION]]
// CHUNK: memref.atomic_rmw addf

// SEGMENTED-LABEL: func.func @spmv(
// SEGMENTED-SAME: %[[OFFSETS:[^ :,]+]]: memref<?xi32>
// SEGMENTED-COUNT-1: %[[FIRST_ROW:.*]] = sparsewave.csr_row_at_position %[[OFFSETS]]
// SEGMENTED: %[[BOUNDARY_INDEX:.*]] = arith.addi %[[FIRST_ROW]], %{{.*}} : index
// SEGMENTED: %[[BOUNDARY_RAW:.*]] = memref.load %[[OFFSETS]][%[[BOUNDARY_INDEX]]]
// SEGMENTED: %[[BOUNDARY:.*]] = arith.index_cast %[[BOUNDARY_RAW]]
// SEGMENTED: %{{.*}}:3 = scf.for
// SEGMENTED-SAME: iter_args(%{{.*}} = %[[FIRST_ROW]], %{{.*}} = %{{.*}}, %{{.*}} = %[[BOUNDARY]])
// SEGMENTED: scf.while (%{{.*}} = %{{.*}}, %[[CURRENT_BOUNDARY:.*]] = %{{.*}})
// SEGMENTED-NOT: memref.load %[[OFFSETS]]
// SEGMENTED: arith.cmpi ule, %[[CURRENT_BOUNDARY]], %{{.*}} : index
// SEGMENTED: } do {
// SEGMENTED: memref.load %[[OFFSETS]]
// SEGMENTED: arith.addf
// SEGMENTED: scf.if
// SEGMENTED: memref.atomic_rmw addf
// SEGMENTED: memref.atomic_rmw addf

// WAVE-LABEL: func.func @spmv(
// WAVE-NOT: gpu.launch
// WAVE: sparsewave.position_parallel %{{.*}} mapping = "thread" block_size = 128 {
// WAVE: memref.store
// WAVE: sparsewave.position_parallel %[[WAVES:.*]] mapping = "wave" block_size = 128 {
// WAVE-NEXT: ^bb0(%[[WAVE:[^,]+]]: index, %[[LANE:[^,]+]]: index
// WAVE: %[[BEGIN:.*]], %[[END:.*]] = sparsewave.position_space
// WAVE-SAME: partition %[[WAVE]] of %[[WAVES]]
// WAVE: %[[POSITION:.*]] = arith.addi %[[BEGIN]], %[[LANE]]
// WAVE: sparsewave.csr_row_at_position %{{.*}} at %[[POSITION]]
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
