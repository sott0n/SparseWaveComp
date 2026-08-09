#ifndef SPARSEWAVE_DIALECT_SPARSEWAVE_TRANSFORMS_PASSES_TD
#define SPARSEWAVE_DIALECT_SPARSEWAVE_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

def ConvertLinalgSpMVToSparseWave
    : Pass<"convert-linalg-spmv-to-sparsewave", "::mlir::ModuleOp"> {
  let summary = "Recognize sparse Linalg SpMV operations";
  let description = [{
    Rewrites a canonical Linalg matrix-vector contraction with a CSR- or
    COO-encoded SparseTensor matrix and a zero-filled dense output to the
    corresponding SparseWave bridge operation. SparseTensor storage extraction
    operations expose the format buffers without defining a separate
    SparseWave storage representation.
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
    Rewrites a canonical Linalg matrix-matrix contraction with a CSR- or
    canonical square BSR-encoded SparseTensor left-hand side and a zero-filled
    dense output to sparsewave.spmm or sparsewave.bsr_spmm, respectively.
  }];
  let dependentDialects = [
    "bufferization::BufferizationDialect",
    "linalg::LinalgDialect",
    "sparse_tensor::SparseTensorDialect",
    "sparsewave::SparseWaveDialect"
  ];
}

def ConvertLinalgSDDMMToSparseWave
    : Pass<"convert-linalg-sddmm-to-sparsewave", "::mlir::ModuleOp"> {
  let summary = "Recognize sparse Linalg SDDMM operations";
  let description = [{
    Rewrites the canonical Linalg SDDMM form with dense left- and
    right-hand-side inputs and a CSR SparseTensor sample to sparsewave.sddmm.
    The sample also backs a zero-filled destination, which proves that the
    result preserves its CSR structure while replacing its values with the
    value-weighted sampled dense product.
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
    Lowers sparsewave.spmv, sparsewave.coo_spmv, sparsewave.spmm,
    sparsewave.bsr_spmm, sparsewave.sddmm, and
    sparsewave.csr_elementwise to one-dimensional GPU launches.
    CSR SpMV can assign one GPU thread, one GPU wave, or one GPU block to each
    matrix row. COO SpMV assigns one thread to each nonzero and atomically
    accumulates the output. CSR SpMM can assign one GPU thread to each dense
    output element or one GPU wave to a CSR row and a tile of dense output
    columns. BSR SpMM initially assigns one GPU thread to each dense output
    element. SDDMM assigns one thread to each CSR row and computes the sampled
    dot products sequentially. CSR elementwise operations use parallel
    row-wise symbolic and numeric coiteration around a sequential prefix scan.
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
           "Number of GPU lanes in a wave">,
    Option<"spmmMapping", "spmm-mapping", "std::string",
           /*default=*/"\"thread-per-output\"",
           "SpMM work mapping strategy">,
    Option<"spmmBlockSize", "spmm-block-size", "int64_t",
           /*default=*/"256",
           "Number of GPU threads in each SpMM block">,
    Option<"spmmTileSize", "spmm-tile-size", "int64_t",
           /*default=*/"4",
           "Number of dense output columns in each SpMM wave tile">,
    Option<"sddmmBlockSize", "sddmm-block-size", "int64_t",
           /*default=*/"256",
           "Number of GPU threads in each SDDMM block">,
    Option<"elementwiseBlockSize", "elementwise-block-size", "int64_t",
           /*default=*/"256",
           "Number of GPU threads in each sparse elementwise block">
  ];
  let dependentDialects = [
    "arith::ArithDialect",
    "gpu::GPUDialect",
    "memref::MemRefDialect",
    "scf::SCFDialect"
  ];
}

#endif // SPARSEWAVE_DIALECT_SPARSEWAVE_TRANSFORMS_PASSES_TD
