// RUN: sparsewave-opt %s -split-input-file -verify-diagnostics

func.func @row_offsets_must_be_rank_one(
    %rowOffsets: memref<2x3xi32>, %columnIndices: memref<8xi32>,
    %values: memref<8xf32>, %vector: memref<4xf32>,
    %output: memref<4xf32>) {
  // expected-error @+1 {{row offsets must be a rank-1 memref}}
  sparsewave.spmv %rowOffsets, %columnIndices, %values, %vector, %output
      : memref<2x3xi32>, memref<8xi32>, memref<8xf32>,
        memref<4xf32>, memref<4xf32>
  return
}

// -----

func.func @indices_must_match(
    %rowOffsets: memref<5xi32>, %columnIndices: memref<8xi64>,
    %values: memref<8xf32>, %vector: memref<4xf32>,
    %output: memref<4xf32>) {
  // expected-error @+1 {{row offsets and column indices must have the same element type}}
  sparsewave.spmv %rowOffsets, %columnIndices, %values, %vector, %output
      : memref<5xi32>, memref<8xi64>, memref<8xf32>,
        memref<4xf32>, memref<4xf32>
  return
}

// -----

func.func @values_must_be_float(
    %rowOffsets: memref<5xi32>, %columnIndices: memref<8xi32>,
    %values: memref<8xi32>, %vector: memref<4xi32>,
    %output: memref<4xi32>) {
  // expected-error @+1 {{values must have floating-point elements}}
  sparsewave.spmv %rowOffsets, %columnIndices, %values, %vector, %output
      : memref<5xi32>, memref<8xi32>, memref<8xi32>,
        memref<4xi32>, memref<4xi32>
  return
}

// -----

func.func @row_count_must_match_output(
    %rowOffsets: memref<6xi32>, %columnIndices: memref<8xi32>,
    %values: memref<8xf32>, %vector: memref<4xf32>,
    %output: memref<4xf32>) {
  // expected-error @+1 {{row offsets size must equal output size plus one, but got 6 and 4}}
  sparsewave.spmv %rowOffsets, %columnIndices, %values, %vector, %output
      : memref<6xi32>, memref<8xi32>, memref<8xf32>,
        memref<4xf32>, memref<4xf32>
  return
}

// -----

func.func @nonzero_counts_must_match(
    %rowOffsets: memref<5xindex>, %columnIndices: memref<7xindex>,
    %values: memref<8xf64>, %vector: memref<4xf64>,
    %output: memref<4xf64>) {
  // expected-error @+1 {{column indices and values must have the same size, but got 7 and 8}}
  sparsewave.spmv %rowOffsets, %columnIndices, %values, %vector, %output
      : memref<5xindex>, memref<7xindex>, memref<8xf64>,
        memref<4xf64>, memref<4xf64>
  return
}

// -----

func.func @coo_indices_must_match(
    %rowIndices: memref<8xi32>, %columnIndices: memref<8xi64>,
    %values: memref<8xf32>, %vector: memref<4xf32>,
    %output: memref<4xf32>) {
  // expected-error @+1 {{row and column indices must have the same element type}}
  sparsewave.coo_spmv %rowIndices, %columnIndices, %values, %vector, %output
      : memref<8xi32>, memref<8xi64>, memref<8xf32>,
        memref<4xf32>, memref<4xf32>
  return
}

// -----

func.func @coo_nonzero_counts_must_match(
    %rowIndices: memref<7xindex>, %columnIndices: memref<8xindex>,
    %values: memref<8xf64>, %vector: memref<4xf64>,
    %output: memref<4xf64>) {
  // expected-error @+1 {{row indices, column indices, and values must have the same size}}
  sparsewave.coo_spmv %rowIndices, %columnIndices, %values, %vector, %output
      : memref<7xindex>, memref<8xindex>, memref<8xf64>,
        memref<4xf64>, memref<4xf64>
  return
}

// -----

func.func @spmm_rhs_must_be_rank_two(
    %rowOffsets: memref<5xi32>, %columnIndices: memref<8xi32>,
    %values: memref<8xf32>, %rhs: memref<4xf32>,
    %output: memref<4x3xf32>) {
  // expected-error @+1 {{right-hand side must be a rank-2 memref}}
  sparsewave.spmm %rowOffsets, %columnIndices, %values, %rhs, %output
      : memref<5xi32>, memref<8xi32>, memref<8xf32>,
        memref<4xf32>, memref<4x3xf32>
  return
}

// -----

func.func @spmm_columns_must_match(
    %rowOffsets: memref<5xi32>, %columnIndices: memref<8xi32>,
    %values: memref<8xf32>, %rhs: memref<4x2xf32>,
    %output: memref<4x3xf32>) {
  // expected-error @+1 {{right-hand side and output must have the same number of columns}}
  sparsewave.spmm %rowOffsets, %columnIndices, %values, %rhs, %output
      : memref<5xi32>, memref<8xi32>, memref<8xf32>,
        memref<4x2xf32>, memref<4x3xf32>
  return
}
