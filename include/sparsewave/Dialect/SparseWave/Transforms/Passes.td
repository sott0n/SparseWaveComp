#ifndef SPARSEWAVE_DIALECT_SPARSEWAVE_TRANSFORMS_PASSES_TD
#define SPARSEWAVE_DIALECT_SPARSEWAVE_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

def ConvertTorchToSparseWave
    : Pass<"convert-torch-to-sparsewave", "::mlir::ModuleOp"> {
  let summary = "Lower imported Torch sparse operations to SparseWave";
  let description = [{
    Converts a shape-specialized torch.aten._sparse_mm imported by torch-mlir
    into sparsewave.spmm. The Torch tensor function boundary is expanded to
    the SparseWave buffer ABI: CSR row offsets, column indices, values, dense
    right-hand side, and caller-provided output.

    The pass consumes generic Torch dialect assembly so SparseWave does not
    need to build or link a second copy of torch-mlir. The initial contract is
    rank-two, static-shape f32 tensors and i32 CSR indices.
  }];
  let dependentDialects = [
    "func::FuncDialect",
    "sparsewave::SparseWaveDialect"
  ];
}

def PrepareGPUBarePtrABI
    : Pass<"prepare-gpu-bare-ptr-abi", "::mlir::ModuleOp"> {
  let summary = "Prepare GPU launch captures for a bare-pointer kernel ABI";
  let description = [{
    Rewrites dynamic memrefs captured by gpu.launch to static views when their
    layout has no dynamic offset or strides and every in-kernel use implements
    the indexed-memory-access interface. These conditions prove that kernel
    address calculation does not consume runtime shape metadata. Unsupported
    descriptor-dependent captures are rejected instead of being specialized.
  }];
  let dependentDialects = [
    "gpu::GPUDialect",
    "memref::MemRefDialect"
  ];
}

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

def LowerSparseWavePositionSpace
    : Pass<"lower-sparsewave-position-space", "::mlir::ModuleOp"> {
  let summary = "Lower sparse position-space primitives to structured control flow";
  let description = [{
    Lowers sparsewave.position_split to quotient and remainder arithmetic,
    sparsewave.position_for to a bounded sequential loop,
    sparsewave.position_reorder to an SCF loop nest in the requested order,
    sparsewave.position_collapse to coordinate recovery and a guarded region,
    and sparsewave.position_space to integer arithmetic implementing a balanced
    contiguous partition. Lowers sparsewave.csr_coordinates to an upper-bound
    binary search over CSR row offsets and a column-index load. The resulting
    IR is target-independent and uses only Arith, MemRef, and SCF operations.
  }];
  let dependentDialects = [
    "arith::ArithDialect",
    "memref::MemRefDialect",
    "scf::SCFDialect"
  ];
}

def DecomposePositionSpMV
    : Pass<"decompose-position-spmv", "::mlir::ModuleOp"> {
  let summary = "Decompose CSR SpMV into a keyed position reduction";
  let description = [{
    Rewrites sparsewave.spmv into sparsewave.position_reduce. The resulting
    operator-independent reduction region recovers each stored position's CSR
    coordinates and yields the output row together with the matrix-vector
    product. No parallel mapping or GPU launch strategy is selected.
  }];
  let dependentDialects = [
    "arith::ArithDialect",
    "memref::MemRefDialect",
    "sparsewave::SparseWaveDialect"
  ];
}

def DecomposePositionSpMM
    : Pass<"decompose-position-spmm", "::mlir::ModuleOp"> {
  let summary = "Decompose CSR SpMM into a keyed position reduction";
  let description = [{
    Rewrites sparsewave.spmm into a rank-2 sparsewave.position_reduce whose
    logical axes are sparse position and dense output column in canonical
    order. The reduction region remains independent from collapse arithmetic
    and yields the flattened output coordinate together with the sparse-dense
    product. Axis reordering, parallel mapping, and GPU launch strategy are
    selected by later scheduling passes.
  }];
  let dependentDialects = [
    "arith::ArithDialect",
    "memref::MemRefDialect",
    "sparsewave::SparseWaveDialect"
  ];
}

def ReorderSparseWavePosition
    : Pass<"reorder-sparsewave-position", "::mlir::ModuleOp"> {
  let summary = "Reorder logical axes of keyed position reductions";
  let description = [{
    Resolves an operator-independent list of logical axis names to a
    permutation on every sparsewave.position_reduce. Bounds and region
    arguments remain in logical axis order; only the order used to collapse
    the iteration domain changes. This separates operator decomposition from
    iteration-space scheduling.
  }];
  let options = [
    ListOption<"order", "order", "std::string",
               "Logical axis names from outermost to innermost">
  ];
}

def ScheduleSparseWavePosition
    : Pass<"schedule-sparsewave-position", "::mlir::ModuleOp"> {
  let summary = "Schedule keyed sparse position reductions";
  let description = [{
    Rewrites sparsewave.position_reduce into explicit position scheduling
    operations. The pass is independent of the operator and storage format
    that produced each keyed contribution. It collapses an arbitrary-rank
    logical position domain according to its order permutation and recovers
    logical coordinates inside each worker. Thread mapping splits the
    collapsed range into configurable sequential chunks. It can either
    atomically accumulate every iteration or combine adjacent equal-key
    contributions locally before atomic accumulation. For a rank-1 CSR
    position domain, the segmented form also carries the recovered row across
    the chunk. Wave mapping uses balanced collapsed-space partitions and first
    combines adjacent equal keys with a segmented reduction. When a
    cooperative axis is named, each wave instead owns one group of that axis
    and its lanes select individual axis coordinates. Read-only scalar values
    that do not depend on the cooperative axis are loaded by lane zero and
    broadcast across the wave.
  }];
  let options = [
    Option<"mapping", "mapping", "std::string",
           /*default=*/"\"thread\"",
           "Position worker mapping strategy">,
    Option<"blockSize", "block-size", "int64_t",
           /*default=*/"256",
           "Number of target threads in each scheduled block">,
    Option<"waveSize", "wave-size", "int64_t",
           /*default=*/"32",
           "Number of target lanes in a scheduled wave">,
    Option<"cooperativeAxis", "cooperative-axis", "std::string",
           /*default=*/"\"\"",
           "Logical axis distributed across cooperating wave lanes">,
    Option<"threadChunkSize", "thread-chunk-size", "int64_t",
           /*default=*/"1",
           "Number of consecutive positions processed by each thread worker">,
    Option<"threadReduction", "thread-reduction", "std::string",
           /*default=*/"\"atomic\"",
           "Thread chunk reduction strategy: atomic or segmented">
  ];
  let dependentDialects = [
    "arith::ArithDialect",
    "gpu::GPUDialect",
    "memref::MemRefDialect",
    "scf::SCFDialect",
    "sparsewave::SparseWaveDialect"
  ];
}

def ConvertSparseWaveToGPU
    : Pass<"convert-sparsewave-to-gpu", "::mlir::ModuleOp"> {
  let summary = "Map SparseWave operations to GPU launches";
  let description = [{
    Lowers sparsewave.spmv, sparsewave.coo_spmv, sparsewave.spmm,
    sparsewave.bsr_spmm, sparsewave.sddmm, sparsewave.csr_rowwise_map, and
    sparsewave.csr_elementwise to one-dimensional GPU launches. Also lowers
    sparsewave.position_parallel by mapping its logical workers and
    participants to threads, waves, or blocks.
    CSR SpMV can assign one GPU thread, one GPU wave, or one GPU block to each
    matrix row. Position-space reductions are first made explicit by
    schedule-sparsewave-position, then this pass maps their logical parallel
    workers to the GPU. COO SpMV
    assigns one thread to each nonzero and atomically accumulates the output.
    CSR SpMM can assign one GPU thread to each dense
    output element or one GPU wave to a CSR row and a tile of dense output
    columns. BSR SpMM initially assigns one GPU thread to each dense output
    element. SDDMM and CSR row reduction assign one thread to each CSR row.
    SDDMM computes sampled dot products sequentially. CSR elementwise operations use parallel
    row-wise symbolic and numeric coiteration around a sequential prefix scan.
  }];
  let options = [
    Option<"mapping", "mapping", "std::string",
           /*default=*/"\"thread-per-row\"",
           "Row-space SpMV work mapping strategy">,
    Option<"blockSize", "block-size", "int64_t",
           /*default=*/"256",
           "Number of GPU threads in each one-dimensional block">,
    Option<"waveSize", "wave-size", "int64_t",
           /*default=*/"32",
           "Number of GPU lanes in a wave">,
    Option<"positionBlockSize", "position-block-size", "int64_t",
           /*default=*/"256",
           "Number of GPU threads in a position-parallel block">,
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
    Option<"rowReductionBlockSize", "row-reduction-block-size", "int64_t",
           /*default=*/"256",
           "Number of GPU threads in each CSR row-reduction block">,
    Option<"rowwiseMapBlockSize", "rowwise-map-block-size", "int64_t",
           /*default=*/"256",
           "Number of GPU threads in each CSR row-wise map block">,
    Option<"elementwiseBlockSize", "elementwise-block-size", "int64_t",
           /*default=*/"256",
           "Number of GPU threads in each sparse elementwise block">
  ];
  let dependentDialects = [
    "arith::ArithDialect",
    "gpu::GPUDialect",
    "memref::MemRefDialect",
    "scf::SCFDialect",
    "sparsewave::SparseWaveDialect"
  ];
}

#endif // SPARSEWAVE_DIALECT_SPARSEWAVE_TRANSFORMS_PASSES_TD
