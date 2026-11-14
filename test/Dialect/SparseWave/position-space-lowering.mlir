// RUN: sparsewave-opt %s --lower-sparsewave-position-space | FileCheck %s
// RUN: sparsewave-opt %s --lower-sparsewave-position-space --canonicalize | FileCheck %s --check-prefix=FOLD

// CHECK-LABEL: func.func @split(
// CHECK-SAME: %[[POSITION:.*]]: index)
func.func @split(%position: index) -> (index, index) {
  // CHECK: %[[FACTOR:.*]] = arith.constant 8 : index
  // CHECK: %[[OUTER:.*]] = arith.divui %[[POSITION]], %[[FACTOR]] : index
  // CHECK: %[[INNER:.*]] = arith.remui %[[POSITION]], %[[FACTOR]] : index
  // CHECK-NOT: sparsewave.position_split
  // CHECK: return %[[OUTER]], %[[INNER]] : index, index
  %outer, %inner = sparsewave.position_split %position by 8 : index
  return %outer, %inner : index, index
}

// CHECK-LABEL: func.func @position_for(
// CHECK-SAME: %[[LOWER:[^,]+]]: index, %[[UPPER:[^,]+]]: index,
// CHECK-SAME: %[[WORKER:[^,]+]]: index, %[[OUTPUT:[^)]+]]: memref<?xindex>)
func.func @position_for(%lower: index, %upper: index, %worker: index,
                        %output: memref<?xindex>) {
  // CHECK: %[[ZERO:.*]] = arith.constant 0 : index
  // CHECK: %[[ONE:.*]] = arith.constant 1 : index
  // CHECK: %[[FACTOR:.*]] = arith.constant 8 : index
  // CHECK: %[[SPAN:.*]] = arith.subi %[[UPPER]], %[[LOWER]] : index
  // CHECK: %[[FULL:.*]] = arith.divui %[[SPAN]], %[[FACTOR]] : index
  // CHECK: %[[REMAINDER:.*]] = arith.remui %[[SPAN]], %[[FACTOR]] : index
  // CHECK: %[[HAS_REMAINDER:.*]] = arith.cmpi ne, %[[REMAINDER]], %[[ZERO]] : index
  // CHECK: %[[EXTRA:.*]] = arith.select %[[HAS_REMAINDER]], %[[ONE]], %[[ZERO]] : index
  // CHECK: %[[COUNT:.*]] = arith.addi %[[FULL]], %[[EXTRA]] : index
  // CHECK: %[[ACTIVE:.*]] = arith.cmpi ult, %[[WORKER]], %[[COUNT]] : index
  // CHECK: %[[SAFE_WORKER:.*]] = arith.select %[[ACTIVE]], %[[WORKER]], %[[ZERO]] : index
  // CHECK: %[[OFFSET:.*]] = arith.muli %[[SAFE_WORKER]], %[[FACTOR]] : index
  // CHECK: %[[BEGIN:.*]] = arith.addi %[[LOWER]], %[[OFFSET]] : index
  // CHECK: %[[REMAINING:.*]] = arith.subi %[[UPPER]], %[[BEGIN]] : index
  // CHECK: %[[BOUNDED_SIZE:.*]] = arith.minui %[[REMAINING]], %[[FACTOR]] : index
  // CHECK: %[[SIZE:.*]] = arith.select %[[ACTIVE]], %[[BOUNDED_SIZE]], %[[ZERO]] : index
  // CHECK: %[[END:.*]] = arith.addi %[[BEGIN]], %[[SIZE]] : index
  // CHECK: scf.for %[[POSITION:.*]] = %[[BEGIN]] to %[[END]] step %[[ONE]] {
  // CHECK: %[[INNER:.*]] = arith.subi %[[POSITION]], %[[BEGIN]] : index
  // CHECK: memref.store %[[INNER]], %[[OUTPUT]][%[[POSITION]]] : memref<?xindex>
  // CHECK: }
  // CHECK-NOT: sparsewave.position_for
  sparsewave.position_for %worker in %lower to %upper by 8 : index {
  ^bb0(%position: index, %inner: index):
    memref.store %inner, %output[%position] : memref<?xindex>
    sparsewave.yield
  }
  return
}

// CHECK-LABEL: func.func @position_for_with_state(
// CHECK-SAME: %[[LOWER:[^,]+]]: index, %[[UPPER:[^,]+]]: index,
// CHECK-SAME: %[[WORKER:[^,]+]]: index, %[[INITIAL:[^)]+]]: index)
// CHECK: %[[RESULT:.*]] = scf.for %[[POSITION:.*]] = %{{[^ ]+}} to %{{[^ ]+}} step %{{[^ ]+}} iter_args(%[[STATE:.*]] = %[[INITIAL]]) -> (index) {
// CHECK: %[[NEXT:.*]] = arith.addi %[[STATE]], %[[POSITION]] : index
// CHECK: scf.yield %[[NEXT]] : index
// CHECK: return %[[RESULT]] : index
// CHECK-NOT: sparsewave.position_for
func.func @position_for_with_state(%lower: index, %upper: index,
                                   %worker: index,
                                   %initial: index) -> index {
  %result = sparsewave.position_for
      %worker in %lower to %upper by 8 : index
      iter_args(%initial : index) -> (index) {
  ^bb0(%position: index, %inner: index, %state: index):
    %next = arith.addi %state, %position : index
    sparsewave.yield %next : index
  }
  return %result : index
}

// The order permutation controls loop nesting, while body arguments remain in
// logical axis order.
// CHECK-LABEL: func.func @position_reorder(
// CHECK-SAME: %[[LOWER0:[^,]+]]: index, %[[LOWER1:[^,]+]]: index,
// CHECK-SAME: %[[UPPER0:[^,]+]]: index, %[[UPPER1:[^,]+]]: index,
// CHECK-SAME: %[[OUTPUT:[^)]+]]: memref<?x?xindex>)
func.func @position_reorder(%lower0: index, %lower1: index, %upper0: index,
                            %upper1: index, %output: memref<?x?xindex>) {
  // CHECK: %[[ONE:.*]] = arith.constant 1 : index
  // CHECK: scf.for %[[AXIS1:.*]] = %[[LOWER1]] to %[[UPPER1]] step %[[ONE]] {
  // CHECK:   scf.for %[[AXIS0:.*]] = %[[LOWER0]] to %[[UPPER0]] step %[[ONE]] {
  // CHECK:     memref.store %[[AXIS0]], %[[OUTPUT]][%[[AXIS0]], %[[AXIS1]]] : memref<?x?xindex>
  // CHECK:   }
  // CHECK: }
  // CHECK-NOT: sparsewave.position_reorder
  sparsewave.position_reorder lower (%lower0, %lower1)
      upper (%upper0, %upper1) order = [1, 0] {
  ^bb0(%axis0: index, %axis1: index):
    memref.store %axis0, %output[%axis0, %axis1] : memref<?x?xindex>
    sparsewave.yield
  }
  return
}

// Recover logical coordinates in reverse linearization order without forming
// the product of the axis extents.
// CHECK-LABEL: func.func @position_collapse(
// CHECK-SAME: %[[WORKER:[^,]+]]: index, %[[LOWER0:[^,]+]]: index,
// CHECK-SAME: %[[LOWER1:[^,]+]]: index, %[[UPPER0:[^,]+]]: index,
// CHECK-SAME: %[[UPPER1:[^,]+]]: index,
// CHECK-SAME: %[[OUTPUT:[^)]+]]: memref<?x?xindex>)
func.func @position_collapse(%worker: index, %lower0: index, %lower1: index,
                             %upper0: index, %upper1: index,
                             %output: memref<?x?xindex>) {
  // CHECK: %[[ZERO:.*]] = arith.constant 0 : index
  // CHECK: %[[ONE:.*]] = arith.constant 1 : index
  // CHECK: %[[EXTENT1:.*]] = arith.subi %[[UPPER1]], %[[LOWER1]] : index
  // CHECK: %[[HAS1:.*]] = arith.cmpi ne, %[[EXTENT1]], %[[ZERO]] : index
  // CHECK: %[[SAFE1:.*]] = arith.select %[[HAS1]], %[[EXTENT1]], %[[ONE]] : index
  // CHECK: %[[OFFSET1:.*]] = arith.remui %[[WORKER]], %[[SAFE1]] : index
  // CHECK: %[[REMAINING1:.*]] = arith.divui %[[WORKER]], %[[SAFE1]] : index
  // CHECK: %[[AXIS1:.*]] = arith.addi %[[LOWER1]], %[[OFFSET1]] : index
  // CHECK: %[[EXTENT0:.*]] = arith.subi %[[UPPER0]], %[[LOWER0]] : index
  // CHECK: %[[HAS0:.*]] = arith.cmpi ne, %[[EXTENT0]], %[[ZERO]] : index
  // CHECK: %[[NONEMPTY0:.*]] = arith.andi %[[HAS1]], %[[HAS0]] : i1
  // CHECK: %[[SAFE0:.*]] = arith.select %[[HAS0]], %[[EXTENT0]], %[[ONE]] : index
  // CHECK: %[[OFFSET0:.*]] = arith.remui %[[REMAINING1]], %[[SAFE0]] : index
  // CHECK: %[[REMAINING0:.*]] = arith.divui %[[REMAINING1]], %[[SAFE0]] : index
  // CHECK: %[[AXIS0:.*]] = arith.addi %[[LOWER0]], %[[OFFSET0]] : index
  // CHECK: %[[IN_RANGE:.*]] = arith.cmpi eq, %[[REMAINING0]], %[[ZERO]] : index
  // CHECK: %[[ACTIVE:.*]] = arith.andi %[[NONEMPTY0]], %[[IN_RANGE]] : i1
  // CHECK-NOT: arith.muli
  // CHECK: scf.if %[[ACTIVE]] {
  // CHECK:   memref.store %[[AXIS0]], %[[OUTPUT]][%[[AXIS0]], %[[AXIS1]]] : memref<?x?xindex>
  // CHECK: }
  // CHECK-NOT: sparsewave.position_collapse
  sparsewave.position_collapse %worker in lower (%lower0, %lower1)
      upper (%upper0, %upper1) order = [0, 1] {
  ^bb0(%axis0: index, %axis1: index):
    memref.store %axis0, %output[%axis0, %axis1] : memref<?x?xindex>
    sparsewave.yield
  }
  return
}

// Return region values for an active worker and the fallback values otherwise.
// CHECK-LABEL: func.func @position_collapse_results(
// CHECK: %[[IN_RANGE:.*]] = arith.cmpi eq
// CHECK: %[[ACTIVE:.*]] = arith.andi %{{.*}}, %[[IN_RANGE]] : i1
// CHECK: %[[RESULTS:.*]]:2 = scf.if %[[ACTIVE]] -> (index, index) {
// CHECK:   %[[SUM:.*]] = arith.addi
// CHECK:   scf.yield %{{.*}}, %[[SUM]] : index, index
// CHECK: } else {
// CHECK:   scf.yield %{{.*}}, %{{.*}} : index, index
// CHECK: }
// CHECK: return %[[RESULTS]]#0, %[[RESULTS]]#1 : index, index
func.func @position_collapse_results(
    %worker: index, %lower0: index, %lower1: index,
    %upper0: index, %upper1: index, %fallback0: index,
    %fallback1: index) -> (index, index) {
  %result0, %result1 = sparsewave.position_collapse %worker
      in lower (%lower0, %lower1) upper (%upper0, %upper1) order = [0, 1]
      fallback (%fallback0, %fallback1 : index, index) -> (index, index) {
  ^bb0(%axis0: index, %axis1: index):
    %sum = arith.addi %axis0, %axis1 : index
    sparsewave.yield %axis0, %sum : index, index
  }
  return %result0, %result1 : index, index
}

// CHECK-LABEL: func.func @partition(
// CHECK-SAME: %[[LOWER:[^,]+]]: index, %[[UPPER:[^,]+]]: index,
// CHECK-SAME: %[[WORKER_ID:[^,]+]]: index, %[[WORKER_COUNT:[^)]+]]: index)
func.func @partition(%lower: index, %upper: index, %workerId: index,
                     %workerCount: index) -> (index, index) {
  // CHECK: %[[ZERO:.*]] = arith.constant 0 : index
  // CHECK: %[[ONE:.*]] = arith.constant 1 : index
  // CHECK: %[[SPAN:.*]] = arith.subi %[[UPPER]], %[[LOWER]] : index
  // CHECK: %[[BASE:.*]] = arith.divui %[[SPAN]], %[[WORKER_COUNT]] : index
  // CHECK: %[[REMAINDER:.*]] = arith.remui %[[SPAN]], %[[WORKER_COUNT]] : index
  // CHECK: %[[HAS_EXTRA:.*]] = arith.cmpi ult, %[[WORKER_ID]], %[[REMAINDER]] : index
  // CHECK: %[[BEFORE:.*]] = arith.select %[[HAS_EXTRA]], %[[WORKER_ID]], %[[REMAINDER]] : index
  // CHECK: %[[BASE_OFFSET:.*]] = arith.muli %[[WORKER_ID]], %[[BASE]] : index
  // CHECK: %[[PARTITION_OFFSET:.*]] = arith.addi %[[BASE_OFFSET]], %[[BEFORE]] : index
  // CHECK: %[[BEGIN:.*]] = arith.addi %[[LOWER]], %[[PARTITION_OFFSET]] : index
  // CHECK: %[[EXTRA:.*]] = arith.select %[[HAS_EXTRA]], %[[ONE]], %[[ZERO]] : index
  // CHECK: %[[SIZE:.*]] = arith.addi %[[BASE]], %[[EXTRA]] : index
  // CHECK: %[[END:.*]] = arith.addi %[[BEGIN]], %[[SIZE]] : index
  // CHECK-NOT: sparsewave.position_space
  // CHECK: return %[[BEGIN]], %[[END]] : index, index
  %begin, %end = sparsewave.position_space %lower to %upper
      partition %workerId of %workerCount : index
  return %begin, %end : index, index
}

// CHECK-LABEL: func.func @coordinates_i32(
// CHECK-SAME: %[[ROW_OFFSETS:[^,]+]]: memref<?xi32>,
// CHECK-SAME: %[[COLUMN_INDICES:[^,]+]]: memref<?xi32>,
// CHECK-SAME: %[[POSITION:[^)]+]]: index)
func.func @coordinates_i32(%rowOffsets: memref<?xi32>,
                           %columnIndices: memref<?xi32>, %position: index)
    -> (index, index) {
  // CHECK: %[[ZERO:.*]] = arith.constant 0 : index
  // CHECK: %[[ONE:.*]] = arith.constant 1 : index
  // CHECK: %[[TWO:.*]] = arith.constant 2 : index
  // CHECK: %[[OFFSETS_SIZE:.*]] = memref.dim %[[ROW_OFFSETS]], %[[ZERO]] : memref<?xi32>
  // CHECK: %[[SEARCH:.*]]:2 = scf.while (%[[LOWER:.*]] = %[[ONE]], %[[UPPER:.*]] = %[[OFFSETS_SIZE]])
  // CHECK: %[[CONTINUE:.*]] = arith.cmpi ult, %[[LOWER]], %[[UPPER]] : index
  // CHECK: scf.condition(%[[CONTINUE]]) %[[LOWER]], %[[UPPER]] : index, index
  // CHECK: do {
  // CHECK: ^bb0(%[[BODY_LOWER:.*]]: index, %[[BODY_UPPER:.*]]: index):
  // CHECK: %[[DISTANCE:.*]] = arith.subi %[[BODY_UPPER]], %[[BODY_LOWER]] : index
  // CHECK: %[[HALF:.*]] = arith.divui %[[DISTANCE]], %[[TWO]] : index
  // CHECK: %[[MIDPOINT:.*]] = arith.addi %[[BODY_LOWER]], %[[HALF]] : index
  // CHECK: %[[OFFSET_I32:.*]] = memref.load %[[ROW_OFFSETS]][%[[MIDPOINT]]] : memref<?xi32>
  // CHECK: %[[OFFSET:.*]] = arith.index_cast %[[OFFSET_I32]] : i32 to index
  // CHECK: %[[AT_OR_BEFORE:.*]] = arith.cmpi ule, %[[OFFSET]], %[[POSITION]] : index
  // CHECK: %[[NEXT_POSITION:.*]] = arith.addi %[[MIDPOINT]], %[[ONE]] : index
  // CHECK: %[[NEXT_LOWER:.*]] = arith.select %[[AT_OR_BEFORE]], %[[NEXT_POSITION]], %[[BODY_LOWER]] : index
  // CHECK: %[[NEXT_UPPER:.*]] = arith.select %[[AT_OR_BEFORE]], %[[BODY_UPPER]], %[[MIDPOINT]] : index
  // CHECK: scf.yield %[[NEXT_LOWER]], %[[NEXT_UPPER]] : index, index
  // CHECK: %[[ROW:.*]] = arith.subi %[[SEARCH]]#0, %[[ONE]] : index
  // CHECK: %[[COLUMN_I32:.*]] = memref.load %[[COLUMN_INDICES]][%[[POSITION]]] : memref<?xi32>
  // CHECK: %[[COLUMN:.*]] = arith.index_cast %[[COLUMN_I32]] : i32 to index
  // CHECK-NOT: sparsewave.csr_coordinates
  // CHECK: return %[[ROW]], %[[COLUMN]] : index, index
  %row, %column = sparsewave.csr_coordinates
      %rowOffsets, %columnIndices at %position : memref<?xi32>, memref<?xi32>
  return %row, %column : index, index
}

// CHECK-LABEL: func.func @coordinates_index(
// CHECK-NOT: arith.index_cast
func.func @coordinates_index(%rowOffsets: memref<?xindex>,
                             %columnIndices: memref<?xindex>, %position: index)
    -> (index, index) {
  %row, %column = sparsewave.csr_coordinates
      %rowOffsets, %columnIndices at %position
      : memref<?xindex>, memref<?xindex>
  // CHECK: return
  return %row, %column : index, index
}

// CHECK-LABEL: func.func @row_at_position(
// CHECK: scf.while
// CHECK-NOT: sparsewave.compressed_segment_at_position
// CHECK: return %{{.*}} : index
func.func @row_at_position(%rowOffsets: memref<?xi32>, %position: index)
    -> index {
  %row = sparsewave.compressed_segment_at_position %rowOffsets at %position
      : memref<?xi32>
  return %row : index
}

// CHECK-LABEL: func.func @classify_segment(
// CHECK-SAME: %[[OFFSETS:[^,]+]]: memref<?xi32>, %[[SEGMENT:[^)]+]]: index)
func.func @classify_segment(%offsets: memref<?xi32>, %segment: index)
    -> index {
  // CHECK-DAG: %[[CLASS2:.*]] = arith.constant 2 : index
  // CHECK-DAG: %[[BOUND1:.*]] = arith.constant 32 : index
  // CHECK-DAG: %[[BOUND0:.*]] = arith.constant 4 : index
  // CHECK-DAG: %[[CLASS0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[CLASS1:.*]] = arith.constant 1 : index
  // CHECK: %[[NEXT:.*]] = arith.addi %[[SEGMENT]], %[[CLASS1]] : index
  // CHECK: %[[BEGIN_I32:.*]] = memref.load %[[OFFSETS]][%[[SEGMENT]]] : memref<?xi32>
  // CHECK: %[[END_I32:.*]] = memref.load %[[OFFSETS]][%[[NEXT]]] : memref<?xi32>
  // CHECK: %[[BEGIN:.*]] = arith.index_cast %[[BEGIN_I32]] : i32 to index
  // CHECK: %[[END:.*]] = arith.index_cast %[[END_I32]] : i32 to index
  // CHECK: %[[LENGTH:.*]] = arith.subi %[[END]], %[[BEGIN]] : index
  // CHECK: %[[EXCEEDS0:.*]] = arith.cmpi ugt, %[[LENGTH]], %[[BOUND0]] : index
  // CHECK: %[[SELECT1:.*]] = arith.select %[[EXCEEDS0]], %[[CLASS1]], %[[CLASS0]] : index
  // CHECK: %[[EXCEEDS1:.*]] = arith.cmpi ugt, %[[LENGTH]], %[[BOUND1]] : index
  // CHECK: %[[SELECT2:.*]] = arith.select %[[EXCEEDS1]], %[[CLASS2]], %[[SELECT1]] : index
  // CHECK-NOT: sparsewave.compressed_segment_bounds
  // CHECK-NOT: sparsewave.segment_classify
  // CHECK: return %[[SELECT2]] : index
  %begin, %end = sparsewave.compressed_segment_bounds
      %offsets at %segment : memref<?xi32>
  %length = arith.subi %end, %begin : index
  %class = sparsewave.segment_classify %length upper_bounds = [4, 32] : index
  return %class : index
}

// FOLD-LABEL: func.func @classify_constant_lengths(
// FOLD: %[[SHORT:.*]] = arith.constant 0 : index
// FOLD: %[[MEDIUM:.*]] = arith.constant 1 : index
// FOLD: %[[LONG:.*]] = arith.constant 2 : index
// FOLD: return %[[SHORT]], %[[MEDIUM]], %[[LONG]] : index, index, index
func.func @classify_constant_lengths() -> (index, index, index) {
  %length4 = arith.constant 4 : index
  %length5 = arith.constant 5 : index
  %length33 = arith.constant 33 : index
  %short = sparsewave.segment_classify %length4 upper_bounds = [4, 32]
      : index
  %medium = sparsewave.segment_classify %length5 upper_bounds = [4, 32]
      : index
  %long = sparsewave.segment_classify %length33 upper_bounds = [4, 32]
      : index
  return %short, %medium, %long : index, index, index
}

// FOLD-LABEL: func.func @partition_remainder(
// FOLD: %[[BEGIN:.*]] = arith.constant 13 : index
// FOLD: %[[END:.*]] = arith.constant 15 : index
// FOLD: return %[[BEGIN]], %[[END]] : index, index
func.func @partition_remainder() -> (index, index) {
  %lower = arith.constant 10 : index
  %upper = arith.constant 17 : index
  %workerId = arith.constant 1 : index
  %workerCount = arith.constant 3 : index
  %begin, %end = sparsewave.position_space %lower to %upper
      partition %workerId of %workerCount : index
  return %begin, %end : index, index
}

// FOLD-LABEL: func.func @split_constant()
// FOLD: %[[OUTER:.*]] = arith.constant 4 : index
// FOLD: %[[INNER:.*]] = arith.constant 5 : index
// FOLD: return %[[OUTER]], %[[INNER]] : index, index
func.func @split_constant() -> (index, index) {
  %position = arith.constant 37 : index
  %outer, %inner = sparsewave.position_split %position by 8 : index
  return %outer, %inner : index, index
}

// A nonzero lower bound is included in the chunk start, and the final chunk
// stops at upper instead of executing all factor iterations.
// FOLD-LABEL: func.func @position_for_tail(
// FOLD: %[[ONE:.*]] = arith.constant 1 : index
// FOLD: %[[BEGIN:.*]] = arith.constant 13 : index
// FOLD: %[[END:.*]] = arith.constant 15 : index
// FOLD: scf.for %[[POSITION:.*]] = %[[BEGIN]] to %[[END]] step %[[ONE]] {
// FOLD: %[[INNER:.*]] = arith.subi %[[POSITION]], %[[BEGIN]] : index
// FOLD: memref.store %[[INNER]], %{{.*}}[%[[POSITION]]] : memref<?xindex>
func.func @position_for_tail(%output: memref<?xindex>) {
  %lower = arith.constant 5 : index
  %upper = arith.constant 15 : index
  %worker = arith.constant 2 : index
  sparsewave.position_for %worker in %lower to %upper by 4 : index {
  ^bb0(%position: index, %inner: index):
    memref.store %inner, %output[%position] : memref<?xindex>
    sparsewave.yield
  }
  return
}

// A worker beyond ceil((upper - lower) / factor) executes no body operations.
// FOLD-LABEL: func.func @position_for_extra_worker(
// FOLD-NOT: scf.for
// FOLD-NOT: memref.store
// FOLD: return
func.func @position_for_extra_worker(%output: memref<?xindex>) {
  %lower = arith.constant 0 : index
  %upper = arith.constant 10 : index
  %worker = arith.constant 3 : index
  sparsewave.position_for %worker in %lower to %upper by 4 : index {
  ^bb0(%position: index, %inner: index):
    memref.store %inner, %output[%position] : memref<?xindex>
    sparsewave.yield
  }
  return
}

// A collapsed worker recovers coordinates relative to nonzero lower bounds.
// For extents (3, 4), worker 5 maps to offsets (1, 1) and coordinates (3, 11).
// FOLD-LABEL: func.func @position_collapse_constant(
// FOLD: %[[AXIS1:.*]] = arith.constant 11 : index
// FOLD: %[[AXIS0:.*]] = arith.constant 3 : index
// FOLD: memref.store %[[AXIS0]], %{{.*}}[%[[AXIS0]], %[[AXIS1]]] : memref<?x?xindex>
func.func @position_collapse_constant(%output: memref<?x?xindex>) {
  %worker = arith.constant 5 : index
  %lower0 = arith.constant 2 : index
  %lower1 = arith.constant 10 : index
  %upper0 = arith.constant 5 : index
  %upper1 = arith.constant 14 : index
  sparsewave.position_collapse %worker in lower (%lower0, %lower1)
      upper (%upper0, %upper1) order = [0, 1] {
  ^bb0(%axis0: index, %axis1: index):
    memref.store %axis0, %output[%axis0, %axis1] : memref<?x?xindex>
    sparsewave.yield
  }
  return
}

// Changing the linearization order changes coordinate recovery. For the same
// extents and worker, order [1, 0] maps worker 5 to offsets (2, 1).
// FOLD-LABEL: func.func @position_collapse_reordered(
// FOLD: %[[AXIS0:.*]] = arith.constant 4 : index
// FOLD: %[[AXIS1:.*]] = arith.constant 11 : index
// FOLD: memref.store %[[AXIS0]], %{{.*}}[%[[AXIS0]], %[[AXIS1]]] : memref<?x?xindex>
func.func @position_collapse_reordered(%output: memref<?x?xindex>) {
  %worker = arith.constant 5 : index
  %lower0 = arith.constant 2 : index
  %lower1 = arith.constant 10 : index
  %upper0 = arith.constant 5 : index
  %upper1 = arith.constant 14 : index
  sparsewave.position_collapse %worker in lower (%lower0, %lower1)
      upper (%upper0, %upper1) order = [1, 0] {
  ^bb0(%axis0: index, %axis1: index):
    memref.store %axis0, %output[%axis0, %axis1] : memref<?x?xindex>
    sparsewave.yield
  }
  return
}

// A worker at the collapsed extent product is outside the iteration space.
// FOLD-LABEL: func.func @position_collapse_extra_worker(
// FOLD-NOT: memref.store
// FOLD: return
func.func @position_collapse_extra_worker(%output: memref<?x?xindex>) {
  %worker = arith.constant 12 : index
  %lower0 = arith.constant 0 : index
  %lower1 = arith.constant 0 : index
  %upper0 = arith.constant 3 : index
  %upper1 = arith.constant 4 : index
  sparsewave.position_collapse %worker in lower (%lower0, %lower1)
      upper (%upper0, %upper1) order = [0, 1] {
  ^bb0(%axis0: index, %axis1: index):
    memref.store %axis0, %output[%axis0, %axis1] : memref<?x?xindex>
    sparsewave.yield
  }
  return
}

// An empty logical axis makes the complete collapsed space empty and uses a
// safe divisor instead of issuing division by zero.
// FOLD-LABEL: func.func @position_collapse_empty_axis(
// FOLD-NOT: memref.store
// FOLD: return
func.func @position_collapse_empty_axis(%output: memref<?x?xindex>) {
  %worker = arith.constant 0 : index
  %lower0 = arith.constant 2 : index
  %lower1 = arith.constant 0 : index
  %upper0 = arith.constant 2 : index
  %upper1 = arith.constant 4 : index
  sparsewave.position_collapse %worker in lower (%lower0, %lower1)
      upper (%upper0, %upper1) order = [0, 1] {
  ^bb0(%axis0: index, %axis1: index):
    memref.store %axis0, %output[%axis0, %axis1] : memref<?x?xindex>
    sparsewave.yield
  }
  return
}

// When there are more workers than positions, trailing workers receive an
// empty partition at the upper bound.
// FOLD-LABEL: func.func @partition_empty_worker(
// FOLD: %[[BOUND:.*]] = arith.constant 3 : index
// FOLD: return %[[BOUND]], %[[BOUND]] : index, index
func.func @partition_empty_worker() -> (index, index) {
  %lower = arith.constant 0 : index
  %upper = arith.constant 3 : index
  %workerId = arith.constant 4 : index
  %workerCount = arith.constant 5 : index
  %begin, %end = sparsewave.position_space %lower to %upper
      partition %workerId of %workerCount : index
  return %begin, %end : index, index
}
