// RUN: sparsewave-opt %s --split-input-file --verify-diagnostics

func.func @invalid_row_count(
    %rowOffsets: memref<5xi32>, %columnIndices: memref<8xi32>,
    %values: memref<8xf32>, %rowValues: memref<3xf32>,
    %outputValues: memref<8xf32>) {
  // expected-error @+1 {{row offsets size must equal row values size plus one, but got 5 and 3}}
  sparsewave.csr_rowwise_map %rowOffsets, %columnIndices, %values, %rowValues,
      %outputValues {
    ^bb0(%value: f32, %rowValue: f32):
      sparsewave.yield %value : f32
  } : memref<5xi32>, memref<8xi32>, memref<8xf32>, memref<3xf32>,
      memref<8xf32>
  return
}

// -----

func.func @invalid_output_size(
    %rowOffsets: memref<5xi32>, %columnIndices: memref<8xi32>,
    %values: memref<8xf32>, %rowValues: memref<4xf32>,
    %outputValues: memref<7xf32>) {
  // expected-error @+1 {{values and output values must have the same size, but got 8 and 7}}
  sparsewave.csr_rowwise_map %rowOffsets, %columnIndices, %values, %rowValues,
      %outputValues {
    ^bb0(%value: f32, %rowValue: f32):
      sparsewave.yield %value : f32
  } : memref<5xi32>, memref<8xi32>, memref<8xf32>, memref<4xf32>,
      memref<7xf32>
  return
}

// -----

func.func @invalid_value_types(
    %rowOffsets: memref<5xi32>, %columnIndices: memref<8xi32>,
    %values: memref<8xf32>, %rowValues: memref<4xf64>,
    %outputValues: memref<8xf32>) {
  // expected-error @+1 {{values, row values, and output values must have the same element type}}
  sparsewave.csr_rowwise_map %rowOffsets, %columnIndices, %values, %rowValues,
      %outputValues {
    ^bb0(%value: f32, %rowValue: f32):
      sparsewave.yield %value : f32
  } : memref<5xi32>, memref<8xi32>, memref<8xf32>, memref<4xf64>,
      memref<8xf32>
  return
}

// -----

func.func @invalid_body_arguments(
    %rowOffsets: memref<5xi32>, %columnIndices: memref<8xi32>,
    %values: memref<8xf32>, %rowValues: memref<4xf32>,
    %outputValues: memref<8xf32>) {
  // expected-error @+1 {{body must have two arguments, but got 1}}
  sparsewave.csr_rowwise_map %rowOffsets, %columnIndices, %values, %rowValues,
      %outputValues {
    ^bb0(%value: f32):
      sparsewave.yield %value : f32
  } : memref<5xi32>, memref<8xi32>, memref<8xf32>, memref<4xf32>,
      memref<8xf32>
  return
}

// -----

func.func @invalid_yield_type(
    %rowOffsets: memref<5xi32>, %columnIndices: memref<8xi32>,
    %values: memref<8xf32>, %rowValues: memref<4xf32>,
    %outputValues: memref<8xf32>) {
  // expected-error @+1 {{body must yield one value with the values element type 'f32'}}
  sparsewave.csr_rowwise_map %rowOffsets, %columnIndices, %values, %rowValues,
      %outputValues {
    ^bb0(%value: f32, %rowValue: f32):
      %constant = arith.constant 1 : i32
      sparsewave.yield %constant : i32
  } : memref<5xi32>, memref<8xi32>, memref<8xf32>, memref<4xf32>,
      memref<8xf32>
  return
}
