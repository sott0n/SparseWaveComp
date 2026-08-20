// RUN: sparsewave-opt %s | FileCheck %s

// CHECK-LABEL: func.func @position_space(
func.func @position_space(
    %lower: index, %upper: index, %workerId: index, %workerCount: index,
    %rowOffsets: memref<?xi32>, %columnIndices: memref<?xi32>) {
  // CHECK: %[[BEGIN:.*]], %[[END:.*]] = sparsewave.position_space
  // CHECK-SAME: partition %{{.*}} of %{{.*}} : index
  %begin, %end = sparsewave.position_space %lower to %upper
      partition %workerId of %workerCount : index

  // CHECK: sparsewave.csr_row_at_position
  %recoveredRow = sparsewave.csr_row_at_position %rowOffsets at %begin
      : memref<?xi32>

  // CHECK: sparsewave.csr_coordinates
  %row, %column = sparsewave.csr_coordinates
      %rowOffsets, %columnIndices at %begin : memref<?xi32>, memref<?xi32>
  return
}

// CHECK-LABEL: func.func @position_split(
func.func @position_split(%position: index) -> (index, index) {
  // CHECK: %[[OUTER:.*]], %[[INNER:.*]] = sparsewave.position_split
  // CHECK-SAME: %[[POSITION:.*]] by 32 : index
  %outer, %inner = sparsewave.position_split %position by 32 : index
  return %outer, %inner : index, index
}

// CHECK-LABEL: func.func @position_for(
func.func @position_for(%lower: index, %upper: index, %workerId: index,
                        %output: memref<?xindex>) {
  // CHECK: sparsewave.position_for %[[WORKER:.*]] in %[[LOWER:.*]] to %[[UPPER:.*]] by 8 : index {
  // CHECK: ^bb0(%[[POSITION:.*]]: index, %[[INNER:.*]]: index):
  sparsewave.position_for %workerId in %lower to %upper by 8 : index {
  ^bb0(%position: index, %inner: index):
    memref.store %inner, %output[%position] : memref<?xindex>
    sparsewave.yield
  }
  return
}

// CHECK-LABEL: func.func @position_reorder(
func.func @position_reorder(%lower0: index, %lower1: index, %upper0: index,
                            %upper1: index, %output: memref<?x?xindex>) {
  // CHECK: sparsewave.position_reorder lower(%[[LOWER0:.*]], %[[LOWER1:.*]]) upper(%[[UPPER0:.*]], %[[UPPER1:.*]]) order = [1, 0] {
  // CHECK: ^bb0(%[[AXIS0:.*]]: index, %[[AXIS1:.*]]: index):
  sparsewave.position_reorder lower (%lower0, %lower1)
      upper (%upper0, %upper1) order = [1, 0] {
  ^bb0(%axis0: index, %axis1: index):
    memref.store %axis0, %output[%axis0, %axis1] : memref<?x?xindex>
    sparsewave.yield
  }
  return
}

// CHECK-LABEL: func.func @position_collapse(
func.func @position_collapse(%workerId: index, %lower0: index, %lower1: index,
                             %upper0: index, %upper1: index,
                             %output: memref<?x?xindex>) {
  // CHECK: sparsewave.position_collapse %[[WORKER:.*]] in lower(%[[LOWER0:.*]], %[[LOWER1:.*]]) upper(%[[UPPER0:.*]], %[[UPPER1:.*]]) order = [0, 1] {
  // CHECK: ^bb0(%[[AXIS0:.*]]: index, %[[AXIS1:.*]]: index):
  sparsewave.position_collapse %workerId in lower (%lower0, %lower1)
      upper (%upper0, %upper1) order = [0, 1] {
  ^bb0(%axis0: index, %axis1: index):
    memref.store %axis0, %output[%axis0, %axis1] : memref<?x?xindex>
    sparsewave.yield
  }
  return
}

// CHECK-LABEL: func.func @position_parallel(
func.func @position_parallel(%workerCount: index,
                             %output: memref<?x?xindex>) {
  // CHECK: sparsewave.position_parallel %[[COUNT:.*]] mapping = "wave" block_size = 128 {
  // CHECK: ^bb0(%[[WORKER:.*]]: index, %[[LANE:.*]]: index, %[[WAVE_SIZE:.*]]: index):
  sparsewave.position_parallel %workerCount mapping = "wave" block_size = 128 {
  ^bb0(%worker: index, %lane: index, %waveSize: index):
    memref.store %lane, %output[%worker, %waveSize] : memref<?x?xindex>
    sparsewave.yield
  }
  return
}

// CHECK-LABEL: func.func @static_valid_partition(
func.func @static_valid_partition() {
  %lower = arith.constant 0 : index
  %upper = arith.constant 17 : index
  %workerId = arith.constant 3 : index
  %workerCount = arith.constant 8 : index
  // CHECK: sparsewave.position_space
  // CHECK-SAME: partition %{{.*}} of %{{.*}} : index
  %begin, %end = sparsewave.position_space %lower to %upper
      partition %workerId of %workerCount : index
  return
}
