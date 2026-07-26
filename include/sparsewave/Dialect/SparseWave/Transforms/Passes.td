#ifndef SPARSEWAVE_DIALECT_SPARSEWAVE_TRANSFORMS_PASSES_TD
#define SPARSEWAVE_DIALECT_SPARSEWAVE_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

def ConvertLinalgSpMVToSparseWave
    : Pass<"convert-linalg-spmv-to-sparsewave", "::mlir::ModuleOp"> {
  let summary = "Recognize sparse Linalg SpMV operations";
  let description = [{
    Rewrites a canonical Linalg matrix-vector contraction with a CSR-encoded
    SparseTensor matrix and a zero-filled dense output to sparsewave.spmv.
    SparseTensor storage extraction operations expose the CSR positions,
    coordinates, and values buffers without defining a separate SparseWave
    storage representation.
  }];
  let dependentDialects = [
    "bufferization::BufferizationDialect",
    "linalg::LinalgDialect",
    "sparse_tensor::SparseTensorDialect",
    "sparsewave::SparseWaveDialect"
  ];
}

def ConvertLinalgSpMMToSparseWave
    : Pass<"convert-linalg-spmm-to-sparsewave", "::mlir::ModuleOp"> {
  let summary = "Recognize sparse Linalg SpMM operations";
  let description = [{
    Rewrites a canonical Linalg matrix-matrix contraction with a CSR-encoded
    SparseTensor left-hand side and a zero-filled dense output to
    sparsewave.spmm.
  }];
  let dependentDialects = [
    "bufferization::BufferizationDialect",
    "linalg::LinalgDialect",
    "sparse_tensor::SparseTensorDialect",
    "sparsewave::SparseWaveDialect"
  ];
}

def ConvertSparseWaveToGPU
    : Pass<"convert-sparsewave-to-gpu", "::mlir::ModuleOp"> {
  let summary = "Map SparseWave operations to GPU launches";
  let description = [{
    Lowers sparsewave.spmv and sparsewave.spmm to one-dimensional GPU
    launches. The SpMV work mapping can assign one GPU thread, one GPU wave,
    or one GPU block to each CSR matrix row. SpMM assigns one thread to each
    dense output element.
  }];
  let options = [
    Option<"mapping", "mapping", "std::string",
           /*default=*/"\"thread-per-row\"",
           "SpMV work mapping strategy">,
    Option<"blockSize", "block-size", "int64_t",
           /*default=*/"256",
           "Number of GPU threads in each one-dimensional block">,
    Option<"waveSize", "wave-size", "int64_t",
           /*default=*/"32",
           "Number of GPU lanes in a wave">
  ];
  let dependentDialects = [
    "arith::ArithDialect",
    "gpu::GPUDialect",
    "memref::MemRefDialect",
    "scf::SCFDialect"
  ];
}

#endif // SPARSEWAVE_DIALECT_SPARSEWAVE_TRANSFORMS_PASSES_TD
