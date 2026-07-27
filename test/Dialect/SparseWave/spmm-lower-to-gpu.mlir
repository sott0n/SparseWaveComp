// RUN: sparsewave-opt %s \
// RUN:   --convert-sparsewave-to-gpu='block-size=128 spmm-mapping=thread-per-output spmm-block-size=64' \
// RUN:   | FileCheck %s
// RUN: sparsewave-opt %s \
// RUN:   --convert-sparsewave-to-gpu='spmm-mapping=wave-per-row-tile spmm-block-size=64 wave-size=32 spmm-tile-size=4' \
// RUN:   | FileCheck %s --check-prefix=WAVE-TILE
// RUN: sparsewave-opt %s \
// RUN:   --convert-sparsewave-to-gpu='spmm-mapping=wave-per-row-tile spmm-block-size=64 wave-size=32 spmm-tile-size=4' \
// RUN:   | FileCheck %s --check-prefix=FULL-TILE
// RUN: sparsewave-opt %s \
// RUN:   --convert-sparsewave-to-gpu='spmm-mapping=wave-per-row-tile spmm-block-size=64 wave-size=32 spmm-tile-size=4' \
// RUN:   | FileCheck %s --check-prefix=TAIL-TILE

// CHECK-LABEL: func.func @spmm(
// CHECK: %[[BLOCK_SIZE:.*]] = arith.constant 64 : index
// CHECK: %[[ROWS:.*]] = memref.dim %{{.*}}, %{{.*}} : memref<?x?xf32>
// CHECK: %[[COLUMNS:.*]] = memref.dim %{{.*}}, %{{.*}} : memref<?x?xf32>
// CHECK: %[[ELEMENTS:.*]] = arith.muli %[[ROWS]], %[[COLUMNS]] : index
// CHECK: gpu.launch blocks
// CHECK-SAME: = %[[BLOCK_SIZE]],
// CHECK: %[[ROW:.*]] = arith.divui %{{.*}}, %[[COLUMNS]] : index
// CHECK: %[[COLUMN:.*]] = arith.remui %{{.*}}, %[[COLUMNS]] : index
// CHECK: %[[START_VALUE:.*]] = memref.load %{{.*}}[%[[ROW]]]
// CHECK: scf.for
// CHECK: %[[RHS_VALUE:.*]] = memref.load %{{.*}}[%{{.*}}, %[[COLUMN]]]
// CHECK: arith.mulf
// CHECK: memref.store %{{.*}}, %{{.*}}[%[[ROW]], %[[COLUMN]]]
// CHECK-NOT: sparsewave.spmm
// WAVE-TILE-LABEL: func.func @spmm(
// WAVE-TILE: %[[BLOCK_SIZE:.*]] = arith.constant 64 : index
// WAVE-TILE: %[[WAVE_SIZE:.*]] = arith.constant 32 : index
// WAVE-TILE: %[[TILE_SIZE:.*]] = arith.constant 4 : index
// WAVE-TILE: %[[ROWS:.*]] = memref.dim %{{.*}}, %{{.*}} : memref<?x?xf32>
// WAVE-TILE: %[[COLUMNS:.*]] = memref.dim %{{.*}}, %{{.*}} : memref<?x?xf32>
// WAVE-TILE: %[[TILES:.*]] = arith.ceildivui %[[COLUMNS]], %[[TILE_SIZE]]
// WAVE-TILE: gpu.launch blocks
// WAVE-TILE-SAME: threads(
// WAVE-TILE-SAME: = %[[BLOCK_SIZE]],
// WAVE-TILE: %[[LANE:.*]] = arith.remui %{{.*}}, %[[WAVE_SIZE]]
// WAVE-TILE: %[[FIRST_POSITION:.*]] = arith.addi %{{.*}}, %[[LANE]]
// WAVE-TILE: scf.for %{{.*}} = %[[FIRST_POSITION]] to %{{.*}} step %[[WAVE_SIZE]]
// WAVE-TILE: %[[SPARSE_VALUE:.*]] = memref.load %{{.*}}[%{{.*}}]
// WAVE-TILE-COUNT-4: memref.load %{{.*}}[%{{.*}}, %{{.*}}]
// WAVE-TILE-COUNT-4: gpu.shuffle xor
// WAVE-TILE-COUNT-4: memref.store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}]
// WAVE-TILE-NOT: sparsewave.spmm
// FULL-TILE-LABEL: func.func @spmm(
// FULL-TILE: %[[TILE_SIZE:.*]] = arith.constant 4 : index
// FULL-TILE: %[[FIRST_OUTPUT_COLUMN:.*]] = arith.muli %{{.*}}, %[[TILE_SIZE]]
// FULL-TILE: %[[TILE_END:.*]] = arith.addi %[[FIRST_OUTPUT_COLUMN]], %[[TILE_SIZE]]
// FULL-TILE: %[[IS_FULL_TILE:.*]] = arith.cmpi ule, %[[TILE_END]], %{{.*}}
// FULL-TILE: scf.if %[[IS_FULL_TILE]]
// FULL-TILE: scf.for
// FULL-TILE-NOT: scf.if
// FULL-TILE: scf.yield
// FULL-TILE: gpu.shuffle xor
// FULL-TILE: } else {
// TAIL-TILE-LABEL: func.func @spmm(
// TAIL-TILE: } else {
// TAIL-TILE-COUNT-4: arith.cmpi ult
// TAIL-TILE: scf.for
// TAIL-TILE: scf.if
// TAIL-TILE: } else {
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
