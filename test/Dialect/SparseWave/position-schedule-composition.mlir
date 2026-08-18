// RUN: sparsewave-opt %s \
// RUN:   --convert-sparsewave-to-gpu='position-block-size=64 wave-size=32' \
// RUN:   --lower-sparsewave-position-space \
// RUN:   | FileCheck %s

// A logical thread worker is first decoded into (row, chunk), then the chunk
// is materialized as a bounded sequential position loop. This exercises the
// complete parallel -> collapse -> for scheduling composition.

// CHECK-LABEL: func.func @parallel_collapsed_chunks(
// CHECK: %[[WORKERS:.*]] = arith.constant 6 : index
// CHECK: gpu.launch
// CHECK: %[[WORKER:.*]] = arith.addi
// CHECK: %[[GPU_ACTIVE:.*]] = arith.cmpi ult, %[[WORKER]], %[[WORKERS]]
// CHECK: scf.if %[[GPU_ACTIVE]] {
// CHECK: %[[CHUNK_OFFSET:.*]] = arith.remui %{{.*}}, %{{.*}} : index
// CHECK: %[[ROW_OFFSET:.*]] = arith.remui %{{.*}}, %{{.*}} : index
// CHECK: %[[COLLAPSE_ACTIVE:.*]] = arith.cmpi eq
// CHECK: scf.if %[[COLLAPSE_ACTIVE]] {
// CHECK: %[[CHUNK_ACTIVE:.*]] = arith.cmpi ult, %[[CHUNK_OFFSET]], %{{.*}}
// CHECK: %[[SAFE_CHUNK:.*]] = arith.select %[[CHUNK_ACTIVE]], %[[CHUNK_OFFSET]],
// CHECK: %[[BEGIN:.*]] = arith.muli %[[SAFE_CHUNK]],
// CHECK: %[[END:.*]] = arith.addi %[[BEGIN]],
// CHECK: scf.for %[[POSITION:.*]] = %[[BEGIN]] to %[[END]] step
// CHECK: memref.store %{{.*}}, %{{.*}}[%{{.*}}, %[[POSITION]]]
// CHECK-NOT: sparsewave.position_
func.func @parallel_collapsed_chunks(%output: memref<?x?xindex>) {
  %zero = arith.constant 0 : index
  %rows = arith.constant 2 : index
  %chunks = arith.constant 3 : index
  %workers = arith.constant 6 : index
  %positionsPerRow = arith.constant 10 : index

  sparsewave.position_parallel %workers mapping = "thread" {
  ^bb0(%worker: index, %participant: index, %participantCount: index):
    sparsewave.position_collapse %worker in lower (%zero, %zero)
        upper (%rows, %chunks) order = [0, 1] {
    ^bb1(%row: index, %chunk: index):
      sparsewave.position_for %chunk in %zero to %positionsPerRow by 4 : index {
      ^bb2(%position: index, %inner: index):
        memref.store %inner, %output[%row, %position] : memref<?x?xindex>
        sparsewave.yield
      }
      sparsewave.yield
    }
    sparsewave.yield
  }
  return
}

// Wave mapping keeps the logical worker separate from its cooperating lane.

// CHECK-LABEL: func.func @wave_collapsed_worker(
// CHECK: %[[WAVE_WORKERS:.*]] = arith.constant 2 : index
// CHECK: %[[WAVE_SIZE:.*]] = arith.constant 32 : index
// CHECK: gpu.launch
// CHECK: %[[LANE:.*]] = arith.remui %{{.*}}, %[[WAVE_SIZE]]
// CHECK: %[[WAVE_WORKER:.*]] = arith.addi
// CHECK: %[[WAVE_ACTIVE:.*]] = arith.cmpi ult, %[[WAVE_WORKER]], %[[WAVE_WORKERS]]
// CHECK: scf.if %[[WAVE_ACTIVE]] {
// CHECK: %[[LOGICAL_WAVE:.*]] = arith.remui %[[WAVE_WORKER]], %[[WAVE_WORKERS]]
// CHECK: scf.if
// CHECK: memref.store %[[LANE]], %{{.*}}[%[[LOGICAL_WAVE]], %[[LANE]]]
// CHECK-NOT: sparsewave.position_
func.func @wave_collapsed_worker(%output: memref<?x?xindex>) {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %workers = arith.constant 2 : index
  sparsewave.position_parallel %workers mapping = "wave" {
  ^bb0(%worker: index, %lane: index, %waveSize: index):
    sparsewave.position_collapse %worker in lower (%zero, %zero)
        upper (%one, %workers) order = [0, 1] {
    ^bb1(%unit: index, %logicalWorker: index):
      memref.store %lane, %output[%logicalWorker, %lane] : memref<?x?xindex>
      sparsewave.yield
    }
    sparsewave.yield
  }
  return
}

// Block mapping likewise exposes the cooperating thread without changing the
// logical coordinate recovered by the target-independent collapse.

// CHECK-LABEL: func.func @block_collapsed_worker(
// CHECK: %[[BLOCK_WORKERS:.*]] = arith.constant 2 : index
// CHECK: %[[BLOCK_SIZE:.*]] = arith.constant 64 : index
// CHECK: gpu.launch blocks(%[[BLOCK_ID:[^,]+]],
// CHECK-SAME: threads(%[[THREAD_ID:[^,]+]],
// CHECK: %[[BLOCK_ACTIVE:.*]] = arith.cmpi ult, %[[BLOCK_ID]], %[[BLOCK_WORKERS]]
// CHECK: scf.if %[[BLOCK_ACTIVE]] {
// CHECK: %[[LOGICAL_BLOCK:.*]] = arith.remui %[[BLOCK_ID]], %[[BLOCK_WORKERS]]
// CHECK: scf.if
// CHECK: memref.store %[[THREAD_ID]], %{{.*}}[%[[LOGICAL_BLOCK]], %[[THREAD_ID]]]
// CHECK-NOT: sparsewave.position_
func.func @block_collapsed_worker(%output: memref<?x?xindex>) {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %workers = arith.constant 2 : index
  sparsewave.position_parallel %workers mapping = "block" {
  ^bb0(%worker: index, %thread: index, %blockSize: index):
    sparsewave.position_collapse %worker in lower (%zero, %zero)
        upper (%one, %workers) order = [0, 1] {
    ^bb1(%unit: index, %logicalWorker: index):
      memref.store %thread, %output[%logicalWorker, %thread]
          : memref<?x?xindex>
      sparsewave.yield
    }
    sparsewave.yield
  }
  return
}

// Even though lowering launches one physical block, zero logical workers keep
// the composed body behind a false worker-range guard.

// CHECK-LABEL: func.func @zero_parallel_work(
// CHECK: %[[ZERO:.*]] = arith.constant 0 : index
// CHECK: %[[ONE:.*]] = arith.constant 1 : index
// CHECK: gpu.launch blocks{{.*}} in (%{{.*}} = %[[ONE]],
// CHECK: %[[ACTIVE:.*]] = arith.cmpi ult, %{{.*}}, %[[ZERO]]
// CHECK: scf.if %[[ACTIVE]] {
// CHECK: memref.store
// CHECK-NOT: sparsewave.position_
func.func @zero_parallel_work(%output: memref<?xindex>) {
  %zero = arith.constant 0 : index
  sparsewave.position_parallel %zero mapping = "thread" {
  ^bb0(%worker: index, %participant: index, %participantCount: index):
    memref.store %participant, %output[%worker] : memref<?xindex>
    sparsewave.yield
  }
  return
}
