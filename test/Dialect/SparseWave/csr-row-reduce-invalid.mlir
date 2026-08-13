// RUN: sparsewave-opt %s -split-input-file -verify-diagnostics

func.func @kind_must_be_supported(
    %rowOffsets: memref<5xi32>, %columnIndices: memref<8xi32>,
    %values: memref<8xf32>, %output: memref<4xf32>) {
  // expected-error @+1 {{kind must be 'sum' or 'max', but got 'product'}}
  sparsewave.csr_row_reduce %rowOffsets, %columnIndices, %values, %output
      kind = "product"
      : memref<5xi32>, memref<8xi32>, memref<8xf32>, memref<4xf32>
  return
}

// -----

func.func @output_must_be_rank_one(
    %rowOffsets: memref<5xi32>, %columnIndices: memref<8xi32>,
    %values: memref<8xf32>, %output: memref<2x2xf32>) {
  // expected-error @+1 {{output must be a rank-1 memref}}
  sparsewave.csr_row_reduce %rowOffsets, %columnIndices, %values, %output
      kind = "sum"
      : memref<5xi32>, memref<8xi32>, memref<8xf32>, memref<2x2xf32>
  return
}

// -----

func.func @row_count_must_match_output(
    %rowOffsets: memref<6xi32>, %columnIndices: memref<8xi32>,
    %values: memref<8xf32>, %output: memref<4xf32>) {
  // expected-error @+1 {{row offsets size must equal output size plus one, but got 6 and 4}}
  sparsewave.csr_row_reduce %rowOffsets, %columnIndices, %values, %output
      kind = "max"
      : memref<6xi32>, memref<8xi32>, memref<8xf32>, memref<4xf32>
  return
}

// -----

func.func @value_types_must_match(
    %rowOffsets: memref<5xi32>, %columnIndices: memref<8xi32>,
    %values: memref<8xf32>, %output: memref<4xf64>) {
  // expected-error @+1 {{values and output must have the same element type}}
  sparsewave.csr_row_reduce %rowOffsets, %columnIndices, %values, %output
      kind = "sum"
      : memref<5xi32>, memref<8xi32>, memref<8xf32>, memref<4xf64>
  return
}

// -----

func.func @csr_components_must_match(
    %rowOffsets: memref<5xi32>, %columnIndices: memref<7xi32>,
    %values: memref<8xf32>, %output: memref<4xf32>) {
  // expected-error @+1 {{column indices and values must have the same size, but got 7 and 8}}
  sparsewave.csr_row_reduce %rowOffsets, %columnIndices, %values, %output
      kind = "max"
      : memref<5xi32>, memref<7xi32>, memref<8xf32>, memref<4xf32>
  return
}
