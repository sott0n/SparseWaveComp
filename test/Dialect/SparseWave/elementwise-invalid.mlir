// RUN: sparsewave-opt %s -split-input-file -verify-diagnostics

func.func @unsupported_kind(
    %lhsOffsets: memref<5xi32>, %lhsColumns: memref<8xi32>,
    %lhsValues: memref<8xf32>, %rhsOffsets: memref<5xi32>,
    %rhsColumns: memref<7xi32>, %rhsValues: memref<7xf32>,
    %outputOffsets: memref<5xi32>, %outputColumns: memref<15xi32>,
    %outputValues: memref<15xf32>, %outputNnz: memref<1xi32>) {
  // expected-error @+1 {{kind must be 'add' or 'multiply', but got 'subtract'}}
  sparsewave.csr_elementwise
      %lhsOffsets, %lhsColumns, %lhsValues,
      %rhsOffsets, %rhsColumns, %rhsValues,
      %outputOffsets, %outputColumns, %outputValues, %outputNnz
      kind = "subtract"
      : memref<5xi32>, memref<8xi32>, memref<8xf32>,
        memref<5xi32>, memref<7xi32>, memref<7xf32>,
        memref<5xi32>, memref<15xi32>, memref<15xf32>, memref<1xi32>
  return
}

// -----

func.func @row_counts_must_match(
    %lhsOffsets: memref<5xi32>, %lhsColumns: memref<8xi32>,
    %lhsValues: memref<8xf32>, %rhsOffsets: memref<6xi32>,
    %rhsColumns: memref<7xi32>, %rhsValues: memref<7xf32>,
    %outputOffsets: memref<5xi32>, %outputColumns: memref<15xi32>,
    %outputValues: memref<15xf32>, %outputNnz: memref<1xi32>) {
  // expected-error @+1 {{left, right, and output row offsets must have the same size}}
  sparsewave.csr_elementwise
      %lhsOffsets, %lhsColumns, %lhsValues,
      %rhsOffsets, %rhsColumns, %rhsValues,
      %outputOffsets, %outputColumns, %outputValues, %outputNnz kind = "add"
      : memref<5xi32>, memref<8xi32>, memref<8xf32>,
        memref<6xi32>, memref<7xi32>, memref<7xf32>,
        memref<5xi32>, memref<15xi32>, memref<15xf32>, memref<1xi32>
  return
}

// -----

func.func @index_types_must_match(
    %lhsOffsets: memref<5xi32>, %lhsColumns: memref<8xi32>,
    %lhsValues: memref<8xf32>, %rhsOffsets: memref<5xi64>,
    %rhsColumns: memref<7xi64>, %rhsValues: memref<7xf32>,
    %outputOffsets: memref<5xi32>, %outputColumns: memref<15xi32>,
    %outputValues: memref<15xf32>, %outputNnz: memref<1xi32>) {
  // expected-error @+1 {{all row offsets, column indices, and output NNZ must have the same element type}}
  sparsewave.csr_elementwise
      %lhsOffsets, %lhsColumns, %lhsValues,
      %rhsOffsets, %rhsColumns, %rhsValues,
      %outputOffsets, %outputColumns, %outputValues, %outputNnz
      kind = "multiply"
      : memref<5xi32>, memref<8xi32>, memref<8xf32>,
        memref<5xi64>, memref<7xi64>, memref<7xf32>,
        memref<5xi32>, memref<15xi32>, memref<15xf32>, memref<1xi32>
  return
}

// -----

func.func @value_types_must_match(
    %lhsOffsets: memref<5xi32>, %lhsColumns: memref<8xi32>,
    %lhsValues: memref<8xf32>, %rhsOffsets: memref<5xi32>,
    %rhsColumns: memref<7xi32>, %rhsValues: memref<7xf64>,
    %outputOffsets: memref<5xi32>, %outputColumns: memref<15xi32>,
    %outputValues: memref<15xf32>, %outputNnz: memref<1xi32>) {
  // expected-error @+1 {{left, right, and output values must have the same element type}}
  sparsewave.csr_elementwise
      %lhsOffsets, %lhsColumns, %lhsValues,
      %rhsOffsets, %rhsColumns, %rhsValues,
      %outputOffsets, %outputColumns, %outputValues, %outputNnz kind = "add"
      : memref<5xi32>, memref<8xi32>, memref<8xf32>,
        memref<5xi32>, memref<7xi32>, memref<7xf64>,
        memref<5xi32>, memref<15xi32>, memref<15xf32>, memref<1xi32>
  return
}

// -----

func.func @output_capacity_must_match(
    %lhsOffsets: memref<5xi32>, %lhsColumns: memref<8xi32>,
    %lhsValues: memref<8xf32>, %rhsOffsets: memref<5xi32>,
    %rhsColumns: memref<7xi32>, %rhsValues: memref<7xf32>,
    %outputOffsets: memref<5xi32>, %outputColumns: memref<15xi32>,
    %outputValues: memref<14xf32>, %outputNnz: memref<1xi32>) {
  // expected-error @+1 {{output column indices and values must have the same capacity, but got 15 and 14}}
  sparsewave.csr_elementwise
      %lhsOffsets, %lhsColumns, %lhsValues,
      %rhsOffsets, %rhsColumns, %rhsValues,
      %outputOffsets, %outputColumns, %outputValues, %outputNnz kind = "add"
      : memref<5xi32>, memref<8xi32>, memref<8xf32>,
        memref<5xi32>, memref<7xi32>, memref<7xf32>,
        memref<5xi32>, memref<15xi32>, memref<14xf32>, memref<1xi32>
  return
}

// -----

func.func @output_nnz_must_be_scalar_buffer(
    %lhsOffsets: memref<5xi32>, %lhsColumns: memref<8xi32>,
    %lhsValues: memref<8xf32>, %rhsOffsets: memref<5xi32>,
    %rhsColumns: memref<7xi32>, %rhsValues: memref<7xf32>,
    %outputOffsets: memref<5xi32>, %outputColumns: memref<15xi32>,
    %outputValues: memref<15xf32>, %outputNnz: memref<2xi32>) {
  // expected-error @+1 {{output NNZ must contain one element, but got 2}}
  sparsewave.csr_elementwise
      %lhsOffsets, %lhsColumns, %lhsValues,
      %rhsOffsets, %rhsColumns, %rhsValues,
      %outputOffsets, %outputColumns, %outputValues, %outputNnz
      kind = "multiply"
      : memref<5xi32>, memref<8xi32>, memref<8xf32>,
        memref<5xi32>, memref<7xi32>, memref<7xf32>,
        memref<5xi32>, memref<15xi32>, memref<15xf32>, memref<2xi32>
  return
}

// -----

func.func @add_output_capacity_must_cover_union(
    %lhsOffsets: memref<5xi32>, %lhsColumns: memref<8xi32>,
    %lhsValues: memref<8xf32>, %rhsOffsets: memref<5xi32>,
    %rhsColumns: memref<7xi32>, %rhsValues: memref<7xf32>,
    %outputOffsets: memref<5xi32>, %outputColumns: memref<14xi32>,
    %outputValues: memref<14xf32>, %outputNnz: memref<1xi32>) {
  // expected-error @+1 {{output capacity for 'add' must be at least 15, but got 14}}
  sparsewave.csr_elementwise
      %lhsOffsets, %lhsColumns, %lhsValues,
      %rhsOffsets, %rhsColumns, %rhsValues,
      %outputOffsets, %outputColumns, %outputValues, %outputNnz kind = "add"
      : memref<5xi32>, memref<8xi32>, memref<8xf32>,
        memref<5xi32>, memref<7xi32>, memref<7xf32>,
        memref<5xi32>, memref<14xi32>, memref<14xf32>, memref<1xi32>
  return
}

// -----

func.func @multiply_output_capacity_must_cover_intersection(
    %lhsOffsets: memref<5xi32>, %lhsColumns: memref<8xi32>,
    %lhsValues: memref<8xf32>, %rhsOffsets: memref<5xi32>,
    %rhsColumns: memref<7xi32>, %rhsValues: memref<7xf32>,
    %outputOffsets: memref<5xi32>, %outputColumns: memref<6xi32>,
    %outputValues: memref<6xf32>, %outputNnz: memref<1xi32>) {
  // expected-error @+1 {{output capacity for 'multiply' must be at least 7, but got 6}}
  sparsewave.csr_elementwise
      %lhsOffsets, %lhsColumns, %lhsValues,
      %rhsOffsets, %rhsColumns, %rhsValues,
      %outputOffsets, %outputColumns, %outputValues, %outputNnz
      kind = "multiply"
      : memref<5xi32>, memref<8xi32>, memref<8xf32>,
        memref<5xi32>, memref<7xi32>, memref<7xf32>,
        memref<5xi32>, memref<6xi32>, memref<6xf32>, memref<1xi32>
  return
}
