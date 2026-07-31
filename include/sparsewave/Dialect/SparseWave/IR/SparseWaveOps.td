#ifndef SPARSEWAVE_DIALECT_SPARSEWAVE_IR_SPARSEWAVEOPS_TD
#define SPARSEWAVE_DIALECT_SPARSEWAVE_IR_SPARSEWAVEOPS_TD

include "sparsewave/Dialect/SparseWave/IR/SparseWaveDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def SparseWave_SpMVOp : SparseWave_Op<"spmv"> {
  let summary = "Multiply a CSR matrix by a dense vector";
  let description = [{
    `sparsewave.spmv` computes `output = matrix * vector`, where `matrix` is
    represented in compressed sparse row (CSR) form by row offsets, column
    indices, and nonzero values.

    All operands are one-dimensional memrefs. Row offsets and column indices
    use the same integer or index element type. Values, vector, and output use
    the same floating-point element type.

    Example:

    ```mlir
    sparsewave.spmv %row_offsets, %column_indices, %values, %vector, %output
        : memref<5xi32>, memref<8xi32>, memref<8xf32>,
          memref<4xf32>, memref<4xf32>
    ```
  }];

  let arguments = (ins Arg<AnyMemRef, "CSR row offsets", [MemRead]>:$rowOffsets,
                       Arg<AnyMemRef, "CSR column indices",
                           [MemRead]>:$columnIndices,
                       Arg<AnyMemRef, "CSR nonzero values", [MemRead]>:$values,
                       Arg<AnyMemRef, "dense input vector", [MemRead]>:$vector,
                       Arg<AnyMemRef, "dense output vector",
                           [MemWrite]>:$output);

  let assemblyFormat = [{
    $rowOffsets `,` $columnIndices `,` $values `,` $vector `,` $output
    attr-dict `:` type($rowOffsets) `,` type($columnIndices) `,`
    type($values) `,` type($vector) `,` type($output)
  }];

  let hasVerifier = 1;
}

def SparseWave_COOSpMVOp : SparseWave_Op<"coo_spmv"> {
  let summary = "Multiply a COO matrix by a dense vector";
  let description = [{
    `sparsewave.coo_spmv` computes `output = matrix * vector`, where `matrix`
    is represented in coordinate (COO) form by parallel row-index,
    column-index, and nonzero-value arrays.

    The output must be zero-initialized before accumulating products. The GPU
    lowering performs this initialization and atomically accumulates one
    product per nonzero entry.

    Example:

    ```mlir
    sparsewave.coo_spmv %row_indices, %column_indices, %values, %vector, %output
        : memref<8xi32>, memref<8xi32>, memref<8xf32>,
          memref<4xf32>, memref<4xf32>
    ```
  }];

  let arguments = (ins Arg<AnyMemRef, "COO row indices", [MemRead]>:$rowIndices,
                       Arg<AnyMemRef, "COO column indices",
                           [MemRead]>:$columnIndices,
                       Arg<AnyMemRef, "COO nonzero values", [MemRead]>:$values,
                       Arg<AnyMemRef, "dense input vector", [MemRead]>:$vector,
                       Arg<AnyMemRef, "dense output vector",
                           [MemRead, MemWrite]>:$output);

  let assemblyFormat = [{
    $rowIndices `,` $columnIndices `,` $values `,` $vector `,` $output
    attr-dict `:` type($rowIndices) `,` type($columnIndices) `,`
    type($values) `,` type($vector) `,` type($output)
  }];

  let hasVerifier = 1;
}

def SparseWave_SpMMOp : SparseWave_Op<"spmm"> {
  let summary = "Multiply a CSR matrix by a dense matrix";
  let description = [{
    `sparsewave.spmm` computes `output = sparse_matrix * dense_matrix`.
    The sparse left-hand side is represented in compressed sparse row (CSR)
    form. Each row of `output` has the same number of columns as
    `dense_matrix`.

    Example:

    ```mlir
    sparsewave.spmm %row_offsets, %column_indices, %values, %rhs, %output
        : memref<5xi32>, memref<8xi32>, memref<8xf32>,
          memref<4x3xf32>, memref<4x3xf32>
    ```
  }];

  let arguments = (ins Arg<AnyMemRef, "CSR row offsets", [MemRead]>:$rowOffsets,
                       Arg<AnyMemRef, "CSR column indices",
                           [MemRead]>:$columnIndices,
                       Arg<AnyMemRef, "CSR nonzero values", [MemRead]>:$values,
                       Arg<AnyMemRef, "dense right-hand-side matrix",
                           [MemRead]>:$rhs,
                       Arg<AnyMemRef, "dense output matrix",
                           [MemWrite]>:$output);

  let assemblyFormat = [{
    $rowOffsets `,` $columnIndices `,` $values `,` $rhs `,` $output
    attr-dict `:` type($rowOffsets) `,` type($columnIndices) `,`
    type($values) `,` type($rhs) `,` type($output)
  }];

  let hasVerifier = 1;
}

def SparseWave_SDDMMOp : SparseWave_Op<"sddmm"> {
  let summary = "Sample a dense matrix product using a CSR matrix";
  let description = [{
    `sparsewave.sddmm` computes a value-weighted sampled dense-dense matrix
    multiplication. For every CSR entry at `(row, column)`, it writes

    ```
    output_values[position] =
        values[position] * dot(lhs[row, :], rhs[:, column])
    ```

    Row offsets and column indices are preserved, so the output values have
    the same sparse structure as the input CSR matrix.

    Example:

    ```mlir
    sparsewave.sddmm %row_offsets, %column_indices, %values,
        %lhs, %rhs, %output_values
        : memref<5xi32>, memref<8xi32>, memref<8xf32>,
          memref<4x3xf32>, memref<3x4xf32>, memref<8xf32>
    ```
  }];

  let arguments = (ins Arg<AnyMemRef, "CSR row offsets", [MemRead]>:$rowOffsets,
                       Arg<AnyMemRef, "CSR column indices",
                           [MemRead]>:$columnIndices,
                       Arg<AnyMemRef, "CSR nonzero values", [MemRead]>:$values,
                       Arg<AnyMemRef, "dense left-hand-side matrix",
                           [MemRead]>:$lhs,
                       Arg<AnyMemRef, "dense right-hand-side matrix",
                           [MemRead]>:$rhs,
                       Arg<AnyMemRef, "CSR output values",
                           [MemWrite]>:$outputValues);

  let assemblyFormat = [{
    $rowOffsets `,` $columnIndices `,` $values `,` $lhs `,` $rhs `,`
    $outputValues attr-dict `:` type($rowOffsets) `,`
    type($columnIndices) `,` type($values) `,` type($lhs) `,` type($rhs) `,`
    type($outputValues)
  }];

  let hasVerifier = 1;
}

#endif // SPARSEWAVE_DIALECT_SPARSEWAVE_IR_SPARSEWAVEOPS_TD
