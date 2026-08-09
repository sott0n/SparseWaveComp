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

def SparseWave_BSRSpMMOp : SparseWave_Op<"bsr_spmm"> {
  let summary = "Multiply a BSR matrix by a dense matrix";
  let description = [{
    `sparsewave.bsr_spmm` computes `output = sparse_matrix * dense_matrix`.
    The sparse left-hand side is represented in block sparse row (BSR) form
    using square blocks.

    `blockRowOffsets` and `blockColumnIndices` address blocks rather than
    scalar entries. `blockValues` is a flat row-major buffer whose logical
    shape is `[nnzb][block_size][block_size]`, where `nnzb` is the number of
    stored blocks. Matrix row and column counts must be divisible by
    `block_size` in this initial BSR contract.

    Example:

    ```mlir
    sparsewave.bsr_spmm %block_row_offsets, %block_column_indices,
        %block_values, %rhs, %output block_size = 2
        : memref<3xi32>, memref<3xi32>, memref<12xf32>,
          memref<4x3xf32>, memref<4x3xf32>
    ```
  }];

  let arguments = (ins Arg<AnyMemRef, "BSR block-row offsets",
                           [MemRead]>:$blockRowOffsets,
                       Arg<AnyMemRef, "BSR block-column indices",
                           [MemRead]>:$blockColumnIndices,
                       Arg<AnyMemRef, "packed BSR block values",
                           [MemRead]>:$blockValues,
                       Arg<AnyMemRef, "dense right-hand-side matrix",
                           [MemRead]>:$rhs,
                       Arg<AnyMemRef, "dense output matrix",
                           [MemWrite]>:$output,
                       I64Attr:$blockSize);

  let assemblyFormat = [{
    $blockRowOffsets `,` $blockColumnIndices `,` $blockValues `,` $rhs `,`
    $output `block_size` `=` $blockSize attr-dict `:`
    type($blockRowOffsets) `,` type($blockColumnIndices) `,`
    type($blockValues) `,` type($rhs) `,` type($output)
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

def SparseWave_CSRElementwiseOp : SparseWave_Op<"csr_elementwise"> {
  let summary = "Apply an elementwise operation to two CSR matrices";
  let description = [{
    `sparsewave.csr_elementwise` coiterates two CSR matrices with sorted,
    unique column indices in each row and assembles a CSR result.

    `kind = "add"` traverses the union of the input coordinates. A coordinate
    present in both inputs receives the sum of both values, while a coordinate
    present in only one input preserves that value. `kind = "multiply"`
    traverses the intersection and emits the product at coordinates present in
    both inputs. Structural entries are retained even when the computed value
    is zero.

    The caller preallocates the three output buffers. For addition, the column
    and value capacity must be at least the sum of both input NNZ counts. For
    multiplication, it must be at least the smaller input NNZ count. The
    operation writes the actual result NNZ to the one-element `outputNnz`
    buffer, and the final row offset equals this value. Output buffers must not
    alias either input.

    Example:

    ```mlir
    sparsewave.csr_elementwise
        %lhs_offsets, %lhs_columns, %lhs_values,
        %rhs_offsets, %rhs_columns, %rhs_values,
        %out_offsets, %out_columns, %out_values, %out_nnz kind = "add"
        : memref<5xi32>, memref<8xi32>, memref<8xf32>,
          memref<5xi32>, memref<7xi32>, memref<7xf32>,
          memref<5xi32>, memref<15xi32>, memref<15xf32>, memref<1xi32>
    ```
  }];

  let arguments = (ins
      Arg<AnyMemRef, "left CSR row offsets", [MemRead]>:$lhsRowOffsets,
      Arg<AnyMemRef, "left CSR column indices", [MemRead]>:$lhsColumnIndices,
      Arg<AnyMemRef, "left CSR nonzero values", [MemRead]>:$lhsValues,
      Arg<AnyMemRef, "right CSR row offsets", [MemRead]>:$rhsRowOffsets,
      Arg<AnyMemRef, "right CSR column indices", [MemRead]>:$rhsColumnIndices,
      Arg<AnyMemRef, "right CSR nonzero values", [MemRead]>:$rhsValues,
      Arg<AnyMemRef, "output CSR row offsets", [MemWrite]>:$outputRowOffsets,
      Arg<AnyMemRef, "output CSR column indices",
          [MemWrite]>:$outputColumnIndices,
      Arg<AnyMemRef, "output CSR values", [MemWrite]>:$outputValues,
      Arg<AnyMemRef, "actual output NNZ", [MemWrite]>:$outputNnz,
      StrAttr:$kind);

  let assemblyFormat = [{
    $lhsRowOffsets `,` $lhsColumnIndices `,` $lhsValues `,`
    $rhsRowOffsets `,` $rhsColumnIndices `,` $rhsValues `,`
    $outputRowOffsets `,` $outputColumnIndices `,` $outputValues `,` $outputNnz
    `kind` `=` $kind attr-dict `:`
    type($lhsRowOffsets) `,` type($lhsColumnIndices) `,` type($lhsValues) `,`
    type($rhsRowOffsets) `,` type($rhsColumnIndices) `,` type($rhsValues) `,`
    type($outputRowOffsets) `,` type($outputColumnIndices) `,`
    type($outputValues) `,` type($outputNnz)
  }];

  let hasVerifier = 1;
}

#endif // SPARSEWAVE_DIALECT_SPARSEWAVE_IR_SPARSEWAVEOPS_TD
