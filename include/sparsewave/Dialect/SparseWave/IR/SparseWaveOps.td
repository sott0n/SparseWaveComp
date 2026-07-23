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

#endif // SPARSEWAVE_DIALECT_SPARSEWAVE_IR_SPARSEWAVEOPS_TD
