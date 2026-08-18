// RUN: sparsewave-opt %s \
// RUN:   --convert-sparsewave-to-gpu='position-block-size=64 wave-size=32' \
// RUN:   | FileCheck %s

// CHECK-LABEL: func.func @thread_mapping(
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[C1:.*]] = arith.constant 1 : index
// CHECK: %[[BLOCK_SIZE:.*]] = arith.constant 64 : index
// CHECK: %[[BLOCKS:.*]] = arith.ceildivui %[[WORKERS:.*]], %[[BLOCK_SIZE]]
// CHECK: gpu.launch blocks(%[[BLOCK_ID_X:.*]], %{{.*}}, %{{.*}}) in (%{{.*}} = %{{.*}}, %{{.*}} = %[[C1]], %{{.*}} = %[[C1]]) threads(%[[THREAD_ID_X:.*]], %{{.*}}, %{{.*}}) in (%[[BLOCK_SIZE_ARG:.*]] = %[[BLOCK_SIZE]], %{{.*}} = %[[C1]], %{{.*}} = %[[C1]])
// CHECK: %[[BASE:.*]] = arith.muli %[[BLOCK_ID_X]], %[[BLOCK_SIZE_ARG]]
// CHECK: %[[WORKER:.*]] = arith.addi %[[BASE]], %[[THREAD_ID_X]]
// CHECK: %[[ACTIVE:.*]] = arith.cmpi ult, %[[WORKER]], %[[WORKERS]]
// CHECK: scf.if %[[ACTIVE]] {
// CHECK: memref.store %[[C0]], %{{.*}}[%[[WORKER]]]
// CHECK: gpu.terminator
// CHECK-NOT: sparsewave.position_parallel
func.func @thread_mapping(%workers: index, %output: memref<?xindex>) {
  sparsewave.position_parallel %workers mapping = "thread" {
  ^bb0(%worker: index, %participant: index, %participantCount: index):
    memref.store %participant, %output[%worker] : memref<?xindex>
    sparsewave.yield
  }
  return
}

// CHECK-LABEL: func.func @wave_mapping(
// CHECK: %[[C1:.*]] = arith.constant 1 : index
// CHECK: %[[BLOCK_SIZE:.*]] = arith.constant 64 : index
// CHECK: %[[WAVE_SIZE:.*]] = arith.constant 32 : index
// CHECK: %[[WAVES_PER_BLOCK:.*]] = arith.constant 2 : index
// CHECK: %[[BLOCKS:.*]] = arith.ceildivui %[[WORKERS:.*]], %[[WAVES_PER_BLOCK]]
// CHECK: gpu.launch blocks(%[[BLOCK_ID_X:.*]], %{{.*}}, %{{.*}}) in (%{{.*}} = %{{.*}}, %{{.*}} = %[[C1]], %{{.*}} = %[[C1]]) threads(%[[THREAD_ID_X:.*]], %{{.*}}, %{{.*}}) in (%{{.*}} = %[[BLOCK_SIZE]], %{{.*}} = %[[C1]], %{{.*}} = %[[C1]])
// CHECK: %[[WAVE_IN_BLOCK:.*]] = arith.divui %[[THREAD_ID_X]], %[[WAVE_SIZE]]
// CHECK: %[[LANE:.*]] = arith.remui %[[THREAD_ID_X]], %[[WAVE_SIZE]]
// CHECK: %[[WORKER_BASE:.*]] = arith.muli %[[BLOCK_ID_X]], %[[WAVES_PER_BLOCK]]
// CHECK: %[[WORKER:.*]] = arith.addi %[[WORKER_BASE]], %[[WAVE_IN_BLOCK]]
// CHECK: %[[ACTIVE:.*]] = arith.cmpi ult, %[[WORKER]], %[[WORKERS]]
// CHECK: scf.if %[[ACTIVE]] {
// CHECK: memref.store %[[LANE]], %{{.*}}[%[[WORKER]], %[[WAVE_SIZE]]]
// CHECK: gpu.terminator
// CHECK-NOT: sparsewave.position_parallel
func.func @wave_mapping(%workers: index, %output: memref<?x?xindex>) {
  sparsewave.position_parallel %workers mapping = "wave" {
  ^bb0(%worker: index, %lane: index, %waveSize: index):
    memref.store %lane, %output[%worker, %waveSize] : memref<?x?xindex>
    sparsewave.yield
  }
  return
}

// CHECK-LABEL: func.func @block_mapping(
// CHECK: %[[C1:.*]] = arith.constant 1 : index
// CHECK: %[[BLOCK_SIZE:.*]] = arith.constant 64 : index
// CHECK: %[[GRID_SIZE:.*]] = arith.maxui %[[WORKERS:.*]], %[[C1]]
// CHECK: gpu.launch blocks(%[[BLOCK_ID_X:.*]], %{{.*}}, %{{.*}}) in (%{{.*}} = %[[GRID_SIZE]], %{{.*}} = %[[C1]], %{{.*}} = %[[C1]]) threads(%[[THREAD_ID_X:.*]], %{{.*}}, %{{.*}}) in (%[[BLOCK_SIZE_ARG:.*]] = %[[BLOCK_SIZE]], %{{.*}} = %[[C1]], %{{.*}} = %[[C1]])
// CHECK: %[[ACTIVE:.*]] = arith.cmpi ult, %[[BLOCK_ID_X]], %[[WORKERS]]
// CHECK: scf.if %[[ACTIVE]] {
// CHECK: memref.store %[[THREAD_ID_X]], %{{.*}}[%[[BLOCK_ID_X]], %[[BLOCK_SIZE_ARG]]]
// CHECK: gpu.terminator
// CHECK-NOT: sparsewave.position_parallel
func.func @block_mapping(%workers: index, %output: memref<?x?xindex>) {
  sparsewave.position_parallel %workers mapping = "block" {
  ^bb0(%worker: index, %thread: index, %blockSize: index):
    memref.store %thread, %output[%worker, %blockSize] : memref<?x?xindex>
    sparsewave.yield
  }
  return
}
