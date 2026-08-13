// RUN: sparsewave-opt %s -split-input-file -verify-diagnostics

func.func @dense_operands_must_be_rank_two(
    %rowOffsets: memref<5xi32>, %columnIndices: memref<8xi32>,
    %values: memref<8xf32>, %lhs: memref<12xf32>,
    %rhs: memref<3x4xf32>, %outputValues: memref<8xf32>) {
  // expected-error @+1 {{left-hand side must be a rank-2 memref}}
  sparsewave.sddmm %rowOffsets, %columnIndices, %values, %lhs, %rhs,
      %outputValues {
    ^bb0(%sample: f32, %dot: f32):
      sparsewave.yield %dot : f32
  }
      : memref<5xi32>, memref<8xi32>, memref<8xf32>,
        memref<12xf32>, memref<3x4xf32>, memref<8xf32>
  return
}

// -----

func.func @row_count_must_match_lhs(
    %rowOffsets: memref<6xi32>, %columnIndices: memref<8xi32>,
    %values: memref<8xf32>, %lhs: memref<4x3xf32>,
    %rhs: memref<3x4xf32>, %outputValues: memref<8xf32>) {
  // expected-error @+1 {{row offsets size must equal left-hand-side rows plus one, but got 6 and 4}}
  sparsewave.sddmm %rowOffsets, %columnIndices, %values, %lhs, %rhs,
      %outputValues {
    ^bb0(%sample: f32, %dot: f32):
      sparsewave.yield %dot : f32
  }
      : memref<6xi32>, memref<8xi32>, memref<8xf32>,
        memref<4x3xf32>, memref<3x4xf32>, memref<8xf32>
  return
}

// -----

func.func @reduction_dimensions_must_match(
    %rowOffsets: memref<5xi32>, %columnIndices: memref<8xi32>,
    %values: memref<8xf32>, %lhs: memref<4x3xf32>,
    %rhs: memref<2x4xf32>, %outputValues: memref<8xf32>) {
  // expected-error @+1 {{left-hand-side columns and right-hand-side rows must match}}
  sparsewave.sddmm %rowOffsets, %columnIndices, %values, %lhs, %rhs,
      %outputValues {
    ^bb0(%sample: f32, %dot: f32):
      sparsewave.yield %dot : f32
  }
      : memref<5xi32>, memref<8xi32>, memref<8xf32>,
        memref<4x3xf32>, memref<2x4xf32>, memref<8xf32>
  return
}

// -----

func.func @output_nonzero_count_must_match(
    %rowOffsets: memref<5xi32>, %columnIndices: memref<8xi32>,
    %values: memref<8xf32>, %lhs: memref<4x3xf32>,
    %rhs: memref<3x4xf32>, %outputValues: memref<7xf32>) {
  // expected-error @+1 {{values and output values must have the same size, but got 8 and 7}}
  sparsewave.sddmm %rowOffsets, %columnIndices, %values, %lhs, %rhs,
      %outputValues {
    ^bb0(%sample: f32, %dot: f32):
      sparsewave.yield %dot : f32
  }
      : memref<5xi32>, memref<8xi32>, memref<8xf32>,
        memref<4x3xf32>, memref<3x4xf32>, memref<7xf32>
  return
}

// -----

func.func @body_arguments_must_match_values(
    %rowOffsets: memref<5xi32>, %columnIndices: memref<8xi32>,
    %values: memref<8xf32>, %lhs: memref<4x3xf32>,
    %rhs: memref<3x4xf32>, %outputValues: memref<8xf32>) {
  // expected-error @+1 {{body arguments must have the values element type 'f32'}}
  sparsewave.sddmm %rowOffsets, %columnIndices, %values, %lhs, %rhs,
      %outputValues {
    ^bb0(%sample: f64, %dot: f32):
      sparsewave.yield %dot : f32
  }
      : memref<5xi32>, memref<8xi32>, memref<8xf32>,
        memref<4x3xf32>, memref<3x4xf32>, memref<8xf32>
  return
}

// -----

func.func @body_yield_must_match_values(
    %rowOffsets: memref<5xi32>, %columnIndices: memref<8xi32>,
    %values: memref<8xf32>, %lhs: memref<4x3xf32>,
    %rhs: memref<3x4xf32>, %outputValues: memref<8xf32>) {
  // expected-error @+1 {{body must yield one value with the values element type 'f32'}}
  sparsewave.sddmm %rowOffsets, %columnIndices, %values, %lhs, %rhs,
      %outputValues {
    ^bb0(%sample: f32, %dot: f32):
      %wrong = arith.constant 0.0 : f64
      sparsewave.yield %wrong : f64
  }
      : memref<5xi32>, memref<8xi32>, memref<8xf32>,
        memref<4x3xf32>, memref<3x4xf32>, memref<8xf32>
  return
}
