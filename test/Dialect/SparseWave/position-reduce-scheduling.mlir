// RUN: sparsewave-opt %s \
// RUN:   --schedule-sparsewave-position='mapping=thread block-size=64' \
// RUN:   | FileCheck %s --check-prefix=THREAD
// RUN: sparsewave-opt %s \
// RUN:   --schedule-sparsewave-position='mapping=wave block-size=64 wave-size=32' \
// RUN:   | FileCheck %s --check-prefix=WAVE
// RUN: sparsewave-opt %s \
// RUN:   --schedule-sparsewave-position='mapping=block block-size=64' \
// RUN:   | FileCheck %s --check-prefix=BLOCK
// RUN: sparsewave-opt %s \
// RUN:   --schedule-sparsewave-position='mapping=block block-size=64 block-reduction=atomic' \
// RUN:   | FileCheck %s --check-prefix=BLOCK-ATOMIC
// RUN: sparsewave-opt %s \
// RUN:   --pass-pipeline='builtin.module(schedule-sparsewave-position{mapping=block block-size=64},convert-sparsewave-to-gpu{position-block-size=64},lower-sparsewave-position-space)' \
// RUN:   | FileCheck %s --check-prefix=BLOCK-GPU
// RUN: sparsewave-opt %s \
// RUN:   --schedule-sparsewave-position='mapping=wave block-size=64 wave-size=32' \
// RUN:   | FileCheck %s --check-prefixes=WAVE-COMPRESSED,WAVE-RESET
// RUN: sparsewave-opt %s \
// RUN:   --schedule-sparsewave-position='mapping=wave block-size=64 wave-size=32 wave-reduction=workspace' \
// RUN:   | FileCheck %s --check-prefix=WORKSPACE
// RUN: sparsewave-opt %s \
// RUN:   --schedule-sparsewave-position='mapping=wave block-size=64 wave-size=32 wave-chunk-size=4 wave-reduction=workspace' \
// RUN:   | FileCheck %s --check-prefix=WAVE-CHUNK
// RUN: sparsewave-opt %s \
// RUN:   --schedule-sparsewave-position='mapping=wave block-size=64 wave-size=32 wave-chunk-size=4' \
// RUN:   | FileCheck %s --check-prefix=WAVE-CHUNK-ATOMIC
// RUN: sparsewave-opt %s \
// RUN:   --schedule-sparsewave-position='mapping=thread block-size=64 thread-chunk-size=4 thread-reduction=segmented' \
// RUN:   | FileCheck %s --check-prefix=SEGMENTED
// RUN: sparsewave-opt %s \
// RUN:   --schedule-sparsewave-position='mapping=thread block-size=64' \
// RUN:   | FileCheck %s --check-prefix=MULTI-AXIS
// RUN: sparsewave-opt %s \
// RUN:   --schedule-sparsewave-position='mapping=thread block-size=64 thread-chunk-size=4 thread-reduction=segmented' \
// RUN:   | FileCheck %s --check-prefix=COMPRESSED
// RUN: sparsewave-opt %s \
// RUN:   --schedule-sparsewave-position='mapping=thread block-size=64 thread-chunk-size=4 thread-reduction=segmented' \
// RUN:   | FileCheck %s --check-prefix=COMPRESSED-MULTI

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
// WAVE: arith.addf
// WAVE: memref.atomic_rmw addf

// BLOCK-LABEL: func.func @keyed_sum(
// BLOCK-NOT: sparsewave.position_reduce
// BLOCK-DAG: %[[COUNT:.*]] = memref.dim
// BLOCK-DAG: %[[BLOCK_SIZE:.*]] = arith.constant 64 : index
// BLOCK: sparsewave.position_parallel %{{.*}} mapping = "thread" block_size = 64 {
// BLOCK: memref.store
// BLOCK: %[[BLOCK_COUNT:.*]] = arith.ceildivui %[[COUNT]], %[[BLOCK_SIZE]]
// BLOCK: sparsewave.position_parallel %[[BLOCK_COUNT]] mapping = "block" block_size = 64 {
// BLOCK: ^bb0(%[[BLOCK:.*]]: index, %[[PARTICIPANT:.*]]: index, %[[PARTICIPANT_COUNT:.*]]: index):
// BLOCK: %[[BEGIN:.*]], %[[END:.*]] = sparsewave.position_space {{.*}} to %[[COUNT]] partition %[[BLOCK]] of %[[BLOCK_COUNT]]
// BLOCK: %[[FIRST:.*]] = arith.addi %[[BEGIN]], %[[PARTICIPANT]]
// BLOCK: %[[KEYS:.*]] = memref.alloca() : memref<4xindex, #gpu.address_space<workgroup>>
// BLOCK: %[[VALUES:.*]] = memref.alloca() : memref<4xf32, #gpu.address_space<workgroup>>
// BLOCK: %[[COUNTS:.*]] = memref.alloca() : memref<2xi32, #gpu.address_space<workgroup>>
// BLOCK: arith.remui %[[FIRST]]
// BLOCK: memref.load
// BLOCK: gpu.shuffle up
// BLOCK: arith.addf
// BLOCK: memref.store {{.*}}, %[[KEYS]]
// BLOCK: memref.store {{.*}}, %[[VALUES]]
// BLOCK: gpu.shuffle idx
// BLOCK: memref.store {{.*}}, %[[COUNTS]]
// BLOCK: gpu.barrier
// BLOCK: scf.for
// BLOCK: memref.load %[[COUNTS]]
// BLOCK: memref.load %[[KEYS]]
// BLOCK: memref.load %[[VALUES]]
// BLOCK: memref.atomic_rmw addf

// BLOCK-ATOMIC-LABEL: func.func @keyed_sum(
// BLOCK-ATOMIC-NOT: memref.alloca
// BLOCK-ATOMIC: sparsewave.position_parallel %{{.*}} mapping = "block" block_size = 64 {
// BLOCK-ATOMIC: scf.for
// BLOCK-ATOMIC: memref.atomic_rmw addf
// BLOCK-ATOMIC-NOT: memref.alloca

// BLOCK-GPU-LABEL: func.func @keyed_sum(
// BLOCK-GPU-NOT: sparsewave.position_
// BLOCK-GPU: gpu.launch
// BLOCK-GPU: memref.store
// BLOCK-GPU: gpu.launch
// BLOCK-GPU-SAME: workgroup(%{{.*}} : memref<4xindex, #gpu.address_space<workgroup>>, %{{.*}} : memref<4xf32, #gpu.address_space<workgroup>>, %{{.*}} : memref<2xi32, #gpu.address_space<workgroup>>)
// BLOCK-GPU: gpu.shuffle up
// BLOCK-GPU: gpu.barrier
// BLOCK-GPU: arith.addf
// BLOCK-GPU: memref.atomic_rmw addf
// BLOCK-GPU-NOT: sparsewave.position_

// WORKSPACE-LABEL: func.func @keyed_sum(
// WORKSPACE: %[[KEYS:.*]] = gpu.alloc (%[[WORKSPACE_SIZE:.*]]) : memref<?xindex>
// WORKSPACE: %[[VALUES:.*]] = gpu.alloc (%[[WORKSPACE_SIZE]]) : memref<?xf32>
// WORKSPACE: sparsewave.position_parallel %[[WORKSPACE_SIZE]] mapping = "thread" block_size = 64 {
// WORKSPACE: memref.store {{.*}}, %[[KEYS]]
// WORKSPACE: memref.store {{.*}}, %[[VALUES]]
// WORKSPACE: sparsewave.position_parallel {{.*}} mapping = "wave" block_size = 64 {
// WORKSPACE: gpu.shuffle up
// WORKSPACE: arith.cmpi eq
// WORKSPACE: memref.store {{.*}}, %[[KEYS]]
// WORKSPACE: memref.store {{.*}}, %[[VALUES]]
// WORKSPACE: memref.atomic_rmw addf
// WORKSPACE: sparsewave.position_parallel {{.*}} mapping = "wave" block_size = 64 {
// WORKSPACE: memref.load %[[KEYS]]
// WORKSPACE: memref.load %[[VALUES]]
// WORKSPACE: memref.atomic_rmw addf
// WORKSPACE: gpu.dealloc %[[KEYS]]
// WORKSPACE: gpu.dealloc %[[VALUES]]

// SEGMENTED-LABEL: func.func @keyed_sum(
// SEGMENTED-NOT: sparsewave.position_reduce
// SEGMENTED: sparsewave.position_parallel %{{.*}} mapping = "thread" block_size = 64 {
// SEGMENTED: sparsewave.position_for
// SEGMENTED-SAME: iter_args(%{{.*}}, %{{.*}} : index, f32) -> (index, f32)
// SEGMENTED: arith.remui
// SEGMENTED: arith.addf
// SEGMENTED: memref.atomic_rmw addf
// SEGMENTED: memref.atomic_rmw addf

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

// This reduction uses a generic compressed-level key, independently of any
// matrix format or named sparse operator. The segmented schedule performs one
// initial lookup and then advances the segment at offset boundaries.

// WAVE-COMPRESSED-LABEL: func.func @compressed_segment_sum(
// WAVE-COMPRESSED-NOT: sparsewave.compressed_segment_at_position
// WAVE-COMPRESSED: gpu.shuffle idx
// WAVE-COMPRESSED: scf.while
// WAVE-COMPRESSED: arith.shrui
// WAVE-COMPRESSED: gpu.shuffle xor
// WAVE-COMPRESSED: memref.load %{{.*}}[%{{.*}}] : memref<?xf32>
// WAVE-COMPRESSED: return

// WAVE-CHUNK-LABEL: func.func @compressed_segment_sum(
// WAVE-CHUNK: sparsewave.position_parallel %{{.*}} mapping = "wave" block_size = 64 {
// WAVE-CHUNK: %{{.*}}:2 = sparsewave.position_for
// WAVE-CHUNK-SAME: by 4
// WAVE-CHUNK-SAME: iter_args(%{{.*}}, %{{.*}} : index, index)
// WAVE-CHUNK: scf.if %{{.*}} -> (index, index) {
// WAVE-CHUNK-COUNT-1: gpu.shuffle idx
// WAVE-CHUNK: } else {
// WAVE-CHUNK: scf.while
// WAVE-CHUNK: sparsewave.yield %{{.*}}, %{{.*}} : index, index
// WAVE-CHUNK-NOT: sparsewave.compressed_segment_at_position
// WAVE-CHUNK: return

// WAVE-CHUNK-LABEL: func.func @compressed_multi_axis_sum(
// WAVE-CHUNK: sparsewave.position_for
// WAVE-CHUNK-SAME: by 4
// WAVE-CHUNK: sparsewave.compressed_segment_at_position
// WAVE-CHUNK-NOT: gpu.shuffle idx
// WAVE-CHUNK: return

// WAVE-CHUNK-ATOMIC-LABEL: func.func @compressed_segment_sum(
// WAVE-CHUNK-ATOMIC-NOT: gpu.alloc
// WAVE-CHUNK-ATOMIC: sparsewave.position_for
// WAVE-CHUNK-ATOMIC-SAME: by 4
// WAVE-CHUNK-ATOMIC: memref.atomic_rmw addf
// WAVE-CHUNK-ATOMIC-NOT: gpu.alloc
// WAVE-CHUNK-ATOMIC: return

// COMPRESSED-LABEL: func.func @compressed_segment_sum(
// COMPRESSED-SAME: %[[OFFSETS:[^ :,]+]]: memref<?xi32>
// COMPRESSED-COUNT-1: %[[FIRST:.*]] = sparsewave.compressed_segment_at_position %[[OFFSETS]]
// COMPRESSED: %[[BOUNDARY_INDEX:.*]] = arith.addi %[[FIRST]], %{{.*}} : index
// COMPRESSED: %[[BOUNDARY_RAW:.*]] = memref.load %[[OFFSETS]][%[[BOUNDARY_INDEX]]]
// COMPRESSED: %{{.*}}:4 = sparsewave.position_for
// COMPRESSED-SAME: iter_args(%[[FIRST]], %{{.*}}, %[[FIRST]], %{{.*}} : index, f32, index, index)
// COMPRESSED-SAME: -> (index, f32, index, index)
// COMPRESSED: scf.while
// COMPRESSED-NOT: sparsewave.compressed_segment_at_position
// COMPRESSED: return

func.func @compressed_segment_sum(%offsets: memref<?xi32>,
                                  %values: memref<?xf32>,
                                  %output: memref<?xf32>) {
  %zero = arith.constant 0 : index
  %upper = memref.dim %values, %zero : memref<?xf32>
  sparsewave.position_reduce lower (%zero) upper (%upper)
      axes = ["position"] order = [0]
      into %output kind = "sum" {
  ^bb0(%position: index):
    %segment = sparsewave.compressed_segment_at_position %offsets at %position
        : memref<?xi32>
    %value = memref.load %values[%position] : memref<?xf32>
    sparsewave.yield %segment, %value : index, f32
  } : memref<?xf32>
  return
}

// This rank-2 reduction verifies that collapse order and coordinate recovery
// belong to the generic scheduler rather than an SpMM decomposition pattern.

// MULTI-AXIS-LABEL: func.func @multi_axis_keyed_sum(
// MULTI-AXIS-NOT: sparsewave.position_reduce
// MULTI-AXIS: %[[COUNT:.*]] = arith.muli
// MULTI-AXIS: sparsewave.position_for
// MULTI-AXIS: sparsewave.position_collapse %{{.*}} in lower(%{{.*}}, %{{.*}}) upper(%{{.*}}, %{{.*}}) order = [1, 0]
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

// The recovered segment is independent of the composite output key. Since
// position is not the outermost axis, the scheduler also detects when it
// restarts and performs a new segment lookup only at that boundary.

// WAVE-RESET-LABEL: func.func @compressed_multi_axis_sum(
// WAVE-RESET: sparsewave.compressed_segment_at_position
// WAVE-RESET-NOT: gpu.shuffle idx
// WAVE-RESET: gpu.shuffle up
// WAVE-RESET: return

// COMPRESSED-MULTI-LABEL: func.func @compressed_multi_axis_sum(
// COMPRESSED-MULTI: %{{.*}}:4 = sparsewave.position_collapse
// COMPRESSED-MULTI-SAME: order = [1, 0]
// COMPRESSED-MULTI: sparsewave.compressed_segment_at_position
// COMPRESSED-MULTI: %{{.*}}:5 = sparsewave.position_for
// COMPRESSED-MULTI-SAME: iter_args(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : index, f32, index, index, index)
// COMPRESSED-MULTI: arith.cmpi ult
// COMPRESSED-MULTI: scf.if
// COMPRESSED-MULTI: sparsewave.compressed_segment_at_position
// COMPRESSED-MULTI: scf.while
// COMPRESSED-MULTI: arith.muli
// COMPRESSED-MULTI: arith.addi
// COMPRESSED-MULTI-NOT: sparsewave.position_reduce

func.func @compressed_multi_axis_sum(%offsets: memref<?xi32>,
                                     %values: memref<?x?xf32>,
                                     %output: memref<?xf32>) {
  %zero = arith.constant 0 : index
  %positions = memref.dim %values, %zero : memref<?x?xf32>
  %one = arith.constant 1 : index
  %columns = memref.dim %values, %one : memref<?x?xf32>
  sparsewave.position_reduce lower (%zero, %zero)
      upper (%positions, %columns) axes = ["position", "column"] order = [1, 0]
      into %output kind = "sum" {
  ^bb0(%position: index, %column: index):
    %segment = sparsewave.compressed_segment_at_position %offsets at %position
        : memref<?xi32>
    %segmentBase = arith.muli %segment, %columns : index
    %key = arith.addi %segmentBase, %column : index
    %value = memref.load %values[%position, %column] : memref<?x?xf32>
    sparsewave.yield %key, %value : index, f32
  } : memref<?xf32>
  return
}
