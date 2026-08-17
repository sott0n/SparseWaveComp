// RUN: sparsewave-opt %s -split-input-file -verify-diagnostics

func.func @invalid_mapping(
    %lower: index, %upper: index, %workerId: index, %workerCount: index) {
  // expected-error @+1 {{mapping must be 'thread', 'wave', or 'block', but got 'lane'}}
  %begin, %end = sparsewave.position_space %lower to %upper
      partition %workerId of %workerCount mapping = "lane" : index
  return
}

// -----

func.func @nonpositive_split_factor(%position: index) {
  // expected-error @+1 {{factor must be positive, but got 0}}
  %outer, %inner = sparsewave.position_split %position by 0 : index
  return
}

// -----

func.func @negative_split_position() {
  %position = arith.constant -1 : index
  // expected-error @+1 {{position must be nonnegative, but got -1}}
  %outer, %inner = sparsewave.position_split %position by 8 : index
  return
}

// -----

func.func @nonpositive_position_for_factor(
    %lower: index, %upper: index, %workerId: index) {
  // expected-error @+1 {{factor must be positive, but got 0}}
  sparsewave.position_for %workerId in %lower to %upper by 0 : index {
  ^bb0(%position: index, %inner: index):
    sparsewave.yield
  }
  return
}

// -----

func.func @position_for_reversed_range() {
  %lower = arith.constant 9 : index
  %upper = arith.constant 4 : index
  %workerId = arith.constant 0 : index
  // expected-error @+1 {{lower bound must not exceed upper bound, but got 9 and 4}}
  sparsewave.position_for %workerId in %lower to %upper by 8 : index {
  ^bb0(%position: index, %inner: index):
    sparsewave.yield
  }
  return
}

// -----

func.func @position_for_negative_worker() {
  %lower = arith.constant 0 : index
  %upper = arith.constant 4 : index
  %workerId = arith.constant -1 : index
  // expected-error @+1 {{worker ID must be nonnegative, but got -1}}
  sparsewave.position_for %workerId in %lower to %upper by 8 : index {
  ^bb0(%position: index, %inner: index):
    sparsewave.yield
  }
  return
}

// -----

func.func @position_for_body_arguments(
    %lower: index, %upper: index, %workerId: index) {
  // expected-error @+1 {{body must have position and inner arguments, but got 1}}
  sparsewave.position_for %workerId in %lower to %upper by 8 : index {
  ^bb0(%position: index):
    sparsewave.yield
  }
  return
}

// -----

func.func @position_for_must_not_yield_values(
    %lower: index, %upper: index, %workerId: index) {
  // expected-error @+1 {{body must yield no values}}
  sparsewave.position_for %workerId in %lower to %upper by 8 : index {
  ^bb0(%position: index, %inner: index):
    sparsewave.yield %position : index
  }
  return
}

// -----

func.func @position_reorder_requires_two_axes(%lower: index, %upper: index) {
  // expected-error @+1 {{requires at least two axes, but got 1}}
  sparsewave.position_reorder lower (%lower) upper (%upper) order = [0] {
  ^bb0(%axis0: index):
    sparsewave.yield
  }
  return
}

// -----

func.func @position_reorder_bound_count(
    %lower0: index, %lower1: index, %upper0: index) {
  // expected-error @+1 {{lower and upper bounds must have the same number of axes, but got 2 and 1}}
  sparsewave.position_reorder lower (%lower0, %lower1) upper (%upper0)
      order = [0, 1] {
  ^bb0(%axis0: index, %axis1: index):
    sparsewave.yield
  }
  return
}

// -----

func.func @position_reorder_order_size(
    %lower0: index, %lower1: index, %upper0: index, %upper1: index) {
  // expected-error @+1 {{order must contain one entry per axis, but got 1 entries for 2 axes}}
  sparsewave.position_reorder lower (%lower0, %lower1)
      upper (%upper0, %upper1) order = [1] {
  ^bb0(%axis0: index, %axis1: index):
    sparsewave.yield
  }
  return
}

// -----

func.func @position_reorder_duplicate_axis(
    %lower0: index, %lower1: index, %upper0: index, %upper1: index) {
  // expected-error @+1 {{order must be a permutation, but axis 0 appears more than once}}
  sparsewave.position_reorder lower (%lower0, %lower1)
      upper (%upper0, %upper1) order = [0, 0] {
  ^bb0(%axis0: index, %axis1: index):
    sparsewave.yield
  }
  return
}

// -----

func.func @position_reorder_axis_out_of_range(
    %lower0: index, %lower1: index, %upper0: index, %upper1: index) {
  // expected-error @+1 {{order axis must be in [0, 2), but got 2}}
  sparsewave.position_reorder lower (%lower0, %lower1)
      upper (%upper0, %upper1) order = [0, 2] {
  ^bb0(%axis0: index, %axis1: index):
    sparsewave.yield
  }
  return
}

// -----

func.func @position_reorder_body_arguments(
    %lower0: index, %lower1: index, %upper0: index, %upper1: index) {
  // expected-error @+1 {{body must have one argument per logical axis, but got 1 for 2 axes}}
  sparsewave.position_reorder lower (%lower0, %lower1)
      upper (%upper0, %upper1) order = [0, 1] {
  ^bb0(%axis0: index):
    sparsewave.yield
  }
  return
}

// -----

func.func @reversed_range() {
  %lower = arith.constant 9 : index
  %upper = arith.constant 4 : index
  %workerId = arith.constant 0 : index
  %workerCount = arith.constant 1 : index
  // expected-error @+1 {{lower bound must not exceed upper bound, but got 9 and 4}}
  %begin, %end = sparsewave.position_space %lower to %upper
      partition %workerId of %workerCount mapping = "thread" : index
  return
}

// -----

func.func @negative_position_bound() {
  %lower = arith.constant -1 : index
  %upper = arith.constant 4 : index
  %workerId = arith.constant 0 : index
  %workerCount = arith.constant 1 : index
  // expected-error @+1 {{lower bound must be nonnegative, but got -1}}
  %begin, %end = sparsewave.position_space %lower to %upper
      partition %workerId of %workerCount mapping = "thread" : index
  return
}

// -----

func.func @nonpositive_worker_count() {
  %lower = arith.constant 0 : index
  %upper = arith.constant 4 : index
  %workerId = arith.constant 0 : index
  %workerCount = arith.constant 0 : index
  // expected-error @+1 {{worker count must be positive, but got 0}}
  %begin, %end = sparsewave.position_space %lower to %upper
      partition %workerId of %workerCount mapping = "wave" : index
  return
}

// -----

func.func @negative_worker_id() {
  %lower = arith.constant 0 : index
  %upper = arith.constant 4 : index
  %workerId = arith.constant -1 : index
  %workerCount = arith.constant 2 : index
  // expected-error @+1 {{worker ID must be nonnegative, but got -1}}
  %begin, %end = sparsewave.position_space %lower to %upper
      partition %workerId of %workerCount mapping = "block" : index
  return
}

// -----

func.func @worker_id_out_of_range() {
  %lower = arith.constant 0 : index
  %upper = arith.constant 4 : index
  %workerId = arith.constant 2 : index
  %workerCount = arith.constant 2 : index
  // expected-error @+1 {{worker ID must be smaller than worker count, but got 2 and 2}}
  %begin, %end = sparsewave.position_space %lower to %upper
      partition %workerId of %workerCount mapping = "thread" : index
  return
}

// -----

func.func @coordinate_buffers_must_be_rank_one(
    %rowOffsets: memref<2x3xi32>, %columnIndices: memref<?xi32>,
    %position: index) {
  // expected-error @+1 {{row offsets must be a rank-1 memref}}
  %row, %column = sparsewave.csr_coordinates
      %rowOffsets, %columnIndices at %position
      : memref<2x3xi32>, memref<?xi32>
  return
}

// -----

func.func @coordinate_index_types_must_match(
    %rowOffsets: memref<?xi32>, %columnIndices: memref<?xi64>,
    %position: index) {
  // expected-error @+1 {{row offsets and column indices must have the same element type}}
  %row, %column = sparsewave.csr_coordinates
      %rowOffsets, %columnIndices at %position
      : memref<?xi32>, memref<?xi64>
  return
}

// -----

func.func @coordinate_offsets_must_not_be_empty(
    %rowOffsets: memref<0xindex>, %columnIndices: memref<?xindex>,
    %position: index) {
  // expected-error @+1 {{row offsets must contain at least two elements, but got 0}}
  %row, %column = sparsewave.csr_coordinates
      %rowOffsets, %columnIndices at %position
      : memref<0xindex>, memref<?xindex>
  return
}

// -----

func.func @coordinate_position_must_be_in_bounds(
    %rowOffsets: memref<4xindex>, %columnIndices: memref<7xindex>) {
  %position = arith.constant 7 : index
  // expected-error @+1 {{position must be smaller than the column-indices size, but got 7 and 7}}
  %row, %column = sparsewave.csr_coordinates
      %rowOffsets, %columnIndices at %position
      : memref<4xindex>, memref<7xindex>
  return
}
