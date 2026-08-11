// RUN: sparsewave-opt %s --lower-sparsewave-position-space | FileCheck %s
// RUN: sparsewave-opt %s --lower-sparsewave-position-space --canonicalize | FileCheck %s --check-prefix=FOLD

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
