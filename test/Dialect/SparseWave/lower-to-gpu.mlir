// RUN: sparsewave-opt %s --convert-sparsewave-to-gpu | FileCheck %s
// RUN: sparsewave-opt %s \
// RUN:   --convert-sparsewave-to-gpu='mapping=thread-per-row block-size=128' \
// RUN:   | FileCheck %s --check-prefix=CUSTOM
// RUN: sparsewave-opt %s \
// RUN:   --convert-sparsewave-to-gpu='mapping=wave-per-row block-size=128 wave-size=32' \
// RUN:   | FileCheck %s --check-prefix=WAVE
// RUN: sparsewave-opt %s \
// RUN:   --convert-sparsewave-to-gpu='mapping=block-per-row block-size=128 wave-size=32' \
// RUN:   | FileCheck %s --check-prefix=BLOCK
// RUN: sparsewave-opt %s \
// RUN:   --convert-sparsewave-to-gpu='mapping=thread-per-position block-size=128' \
// RUN:   | FileCheck %s --check-prefix=POS
// RUN: sparsewave-opt %s \
// RUN:   --pass-pipeline='builtin.module(convert-sparsewave-to-gpu{mapping=thread-per-position block-size=128},lower-sparsewave-position-space)' \
// RUN:   | FileCheck %s --check-prefix=POS-LOWER
// RUN: sparsewave-opt %s \
// RUN:   --convert-sparsewave-to-gpu='mapping=wave-per-position block-size=128 wave-size=32' \
// RUN:   | FileCheck %s --check-prefix=SEGMENT

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

// WAVE-LABEL: func.func @spmv(
// WAVE: %[[ZERO_INDEX:.*]] = arith.constant 0 : index
// WAVE: %[[ONE_INDEX:.*]] = arith.constant 1 : index
// WAVE: %[[BLOCK_SIZE:.*]] = arith.constant 128 : index
// WAVE: %[[WAVE_SIZE:.*]] = arith.constant 32 : index
// WAVE: %[[WAVES_PER_BLOCK:.*]] = arith.constant 4 : index
// WAVE: %[[ROWS:.*]] = memref.dim %{{.*}}, %[[ZERO_INDEX]]
// WAVE: %[[REQUIRED_BLOCKS:.*]] = arith.ceildivui %[[ROWS]], %[[WAVES_PER_BLOCK]]
// WAVE: %[[GRID:.*]] = arith.maxui %[[REQUIRED_BLOCKS]], %[[ONE_INDEX]]
// WAVE: gpu.launch blocks(%[[BLOCK_X:.*]], %{{.*}}, %{{.*}}) in (%{{.*}} = %[[GRID]], %{{.*}} = %[[ONE_INDEX]], %{{.*}} = %[[ONE_INDEX]]) threads(%[[THREAD_X:.*]], %{{.*}}, %{{.*}}) in (%{{.*}} = %[[BLOCK_SIZE]],
// WAVE: %[[WAVE_ID:.*]] = arith.divui %[[THREAD_X]], %[[WAVE_SIZE]]
// WAVE: %[[LANE_ID:.*]] = arith.remui %[[THREAD_X]], %[[WAVE_SIZE]]
// WAVE: %[[ROW_BASE:.*]] = arith.muli %[[BLOCK_X]], %[[WAVES_PER_BLOCK]]
// WAVE: %[[ROW:.*]] = arith.addi %[[ROW_BASE]], %[[WAVE_ID]]
// WAVE: %[[ACTIVE:.*]] = arith.cmpi ult, %[[ROW]], %[[ROWS]]
// WAVE: scf.if %[[ACTIVE]]
// WAVE: %[[START:.*]] = arith.index_cast
// WAVE: %[[END:.*]] = arith.index_cast
// WAVE: %[[FIRST_POSITION:.*]] = arith.addi %[[START]], %[[LANE_ID]]
// WAVE: %[[PARTIAL:.*]] = scf.for %[[POSITION:.*]] = %[[FIRST_POSITION]] to %[[END]] step %[[WAVE_SIZE]]
// WAVE-COUNT-5: gpu.shuffle xor
// WAVE: %[[LANE_ZERO:.*]] = arith.cmpi eq, %[[LANE_ID]], %[[ZERO_INDEX]]
// WAVE: scf.if %[[LANE_ZERO]]
// WAVE: memref.store %{{.*}}, %{{.*}}[%[[ROW]]]
// WAVE-NOT: sparsewave.spmv

// BLOCK-LABEL: func.func @spmv(
// BLOCK: %[[ZERO_INDEX:.*]] = arith.constant 0 : index
// BLOCK: %[[ONE_INDEX:.*]] = arith.constant 1 : index
// BLOCK: %[[BLOCK_SIZE:.*]] = arith.constant 128 : index
// BLOCK: %[[WAVE_SIZE:.*]] = arith.constant 32 : index
// BLOCK: %[[WAVES_PER_BLOCK:.*]] = arith.constant 4 : index
// BLOCK: %[[ROWS:.*]] = memref.dim %{{.*}}, %[[ZERO_INDEX]]
// BLOCK: %[[GRID:.*]] = arith.maxui %[[ROWS]], %[[ONE_INDEX]]
// BLOCK: gpu.launch blocks(%[[ROW:.*]], %{{.*}}, %{{.*}}) in (%{{.*}} = %[[GRID]], %{{.*}} = %[[ONE_INDEX]], %{{.*}} = %[[ONE_INDEX]]) threads(%[[THREAD:.*]], %{{.*}}, %{{.*}}) in (%{{.*}} = %[[BLOCK_SIZE]],
// BLOCK-SAME: workgroup(%[[WAVE_SUMS:.*]] : memref<4xf32, #gpu.address_space<workgroup>>)
// BLOCK: %[[WAVE_ID:.*]] = arith.divui %[[THREAD]], %[[WAVE_SIZE]]
// BLOCK: %[[LANE_ID:.*]] = arith.remui %[[THREAD]], %[[WAVE_SIZE]]
// BLOCK: %[[ACTIVE:.*]] = arith.cmpi ult, %[[ROW]], %[[ROWS]]
// BLOCK: scf.if %[[ACTIVE]]
// BLOCK: %[[START:.*]] = arith.index_cast
// BLOCK: %[[END:.*]] = arith.index_cast
// BLOCK: %[[FIRST_POSITION:.*]] = arith.addi %[[START]], %[[THREAD]]
// BLOCK: %[[PARTIAL:.*]] = scf.for %[[POSITION:.*]] = %[[FIRST_POSITION]] to %[[END]] step %[[BLOCK_SIZE]]
// BLOCK-COUNT-5: gpu.shuffle xor
// BLOCK: memref.store %{{.*}}, %[[WAVE_SUMS]][%[[WAVE_ID]]]
// BLOCK: gpu.barrier
// BLOCK: %[[FIRST_WAVE:.*]] = arith.cmpi eq, %[[WAVE_ID]], %[[ZERO_INDEX]]
// BLOCK: scf.if %[[FIRST_WAVE]]
// BLOCK: %[[HAS_WAVE:.*]] = arith.cmpi ult, %[[LANE_ID]], %[[WAVES_PER_BLOCK]]
// BLOCK: scf.if %[[HAS_WAVE]] -> (f32)
// BLOCK: memref.load %[[WAVE_SUMS]][%[[LANE_ID]]]
// BLOCK-COUNT-5: gpu.shuffle xor
// BLOCK: memref.store %{{.*}}, %{{.*}}[%[[ROW]]]
// BLOCK-NOT: sparsewave.spmv

// POS-LABEL: func.func @spmv(
// POS: %[[ZERO:.*]] = arith.constant 0 : index
// POS: %[[ONE:.*]] = arith.constant 1 : index
// POS: %[[BLOCK_SIZE:.*]] = arith.constant 128 : index
// POS: %[[OUTPUT_SIZE:.*]] = memref.dim %{{.*}}, %[[ZERO]]
// POS-NEXT: %[[NNZ:.*]] = memref.dim %{{.*}}, %[[ZERO]]
// POS: gpu.launch
// POS: memref.store
// POS: gpu.terminator
// POS: %[[REQUIRED_BLOCKS:.*]] = arith.ceildivui %[[NNZ]], %[[BLOCK_SIZE]]
// POS: %[[GRID_SIZE:.*]] = arith.maxui %[[REQUIRED_BLOCKS]], %[[ONE]]
// POS: %[[WORKER_COUNT:.*]] = arith.muli %[[GRID_SIZE]], %[[BLOCK_SIZE]]
// POS: gpu.launch
// POS: %[[WORKER_BASE:.*]] = arith.muli %{{.*}}, %{{.*}}
// POS-NEXT: %[[WORKER:.*]] = arith.addi %[[WORKER_BASE]], %{{.*}}
// POS: %[[ACTIVE:.*]] = arith.cmpi ult, %[[WORKER]], %[[WORKER_COUNT]]
// POS: scf.if %[[ACTIVE]]
// POS: %[[BEGIN:.*]], %[[END:.*]] = sparsewave.position_space %[[ZERO]] to %[[NNZ]] partition %[[WORKER]] of %[[WORKER_COUNT]]
// POS: scf.for %[[POSITION_VALUE:.*]] = %[[BEGIN]] to %[[END]] step %[[ONE]]
// POS: %[[ROW:.*]], %[[COLUMN:.*]] = sparsewave.csr_coordinates %{{.*}}, %{{.*}} at %[[POSITION_VALUE]]
// POS: %[[SPARSE_VALUE:.*]] = memref.load %{{.*}}[%[[POSITION_VALUE]]]
// POS: %[[VECTOR_VALUE:.*]] = memref.load %{{.*}}[%[[COLUMN]]]
// POS: %[[PRODUCT:.*]] = arith.mulf %[[SPARSE_VALUE]], %[[VECTOR_VALUE]]
// POS: memref.atomic_rmw addf %[[PRODUCT]], %{{.*}}[%[[ROW]]
// POS-NOT: sparsewave.spmv

// POS-LOWER-LABEL: func.func @spmv(
// POS-LOWER-NOT: sparsewave.position_space
// POS-LOWER-NOT: sparsewave.csr_coordinates
// POS-LOWER: scf.while
// POS-LOWER: memref.atomic_rmw addf
// POS-LOWER-NOT: sparsewave.spmv

// SEGMENT-LABEL: func.func @spmv(
// SEGMENT: %[[WAVE_SIZE:.*]] = arith.constant 32 : index
// SEGMENT: %[[ZERO:.*]] = arith.constant 0 : index
// SEGMENT: %[[ONE:.*]] = arith.constant 1 : index
// SEGMENT: %[[BLOCK_SIZE:.*]] = arith.constant 128 : index
// SEGMENT: %[[WAVES_PER_BLOCK:.*]] = arith.constant 4 : index
// SEGMENT: %[[OUTPUT_SIZE:.*]] = memref.dim %{{.*}}, %[[ZERO]]
// SEGMENT-NEXT: %[[NNZ:.*]] = memref.dim %{{.*}}, %[[ZERO]]
// SEGMENT: gpu.launch
// SEGMENT: gpu.terminator
// SEGMENT: %[[REQUIRED_BLOCKS:.*]] = arith.ceildivui %[[NNZ]], %[[BLOCK_SIZE]]
// SEGMENT: %[[GRID_SIZE:.*]] = arith.maxui %[[REQUIRED_BLOCKS]], %[[ONE]]
// SEGMENT: %[[WAVE_COUNT:.*]] = arith.muli %[[GRID_SIZE]], %[[WAVES_PER_BLOCK]]
// SEGMENT: gpu.launch
// SEGMENT: %[[WAVE_IN_BLOCK:.*]] = arith.divui %{{.*}}, %[[WAVE_SIZE]]
// SEGMENT: %[[LANE:.*]] = arith.remui %{{.*}}, %[[WAVE_SIZE]]
// SEGMENT: %[[WAVE_BASE:.*]] = arith.muli %{{.*}}, %[[WAVES_PER_BLOCK]]
// SEGMENT: %[[POSITION_WORKER:.*]] = arith.addi %[[WAVE_BASE]], %[[WAVE_IN_BLOCK]]
// SEGMENT: %[[WORKER_ACTIVE:.*]] = arith.cmpi ult, %[[POSITION_WORKER]], %[[WAVE_COUNT]]
// SEGMENT: scf.if %[[WORKER_ACTIVE]]
// SEGMENT: %[[BEGIN:.*]], %[[END:.*]] = sparsewave.position_space %[[ZERO]] to %[[NNZ]] partition %[[POSITION_WORKER]] of %[[WAVE_COUNT]]
// SEGMENT: %[[POSITION:.*]] = arith.addi %[[BEGIN]], %[[LANE]]
// SEGMENT: %[[ACTIVE:.*]] = arith.cmpi ult, %[[POSITION]], %[[END]]
// SEGMENT: scf.if %[[ACTIVE]] -> (index, f32)
// SEGMENT: sparsewave.csr_coordinates
// SEGMENT-COUNT-10: gpu.shuffle up
// SEGMENT-COUNT-2: gpu.shuffle down
// SEGMENT: memref.atomic_rmw addf
// SEGMENT-NOT: sparsewave.spmv

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
