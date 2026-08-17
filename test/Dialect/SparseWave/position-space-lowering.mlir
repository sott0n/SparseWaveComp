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
      partition %workerId of %workerCount mapping = "wave" : index
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
      partition %workerId of %workerCount mapping = "thread" : index
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
      partition %workerId of %workerCount mapping = "block" : index
  return %begin, %end : index, index
}
