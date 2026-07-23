#ifndef SPARSEWAVE_DIALECT_SPARSEWAVE_TRANSFORMS_PASSES_TD
#define SPARSEWAVE_DIALECT_SPARSEWAVE_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

def ConvertSparseWaveToGPU
    : Pass<"convert-sparsewave-to-gpu", "::mlir::ModuleOp"> {
  let summary = "Map SparseWave operations to GPU launches";
  let description = [{
    Lowers sparsewave.spmv to a one-dimensional GPU launch. Each GPU thread
    computes one CSR matrix row and writes one output element.
  }];
  let dependentDialects = [
    "arith::ArithDialect",
    "gpu::GPUDialect",
    "memref::MemRefDialect",
    "scf::SCFDialect"
  ];
}

#endif // SPARSEWAVE_DIALECT_SPARSEWAVE_TRANSFORMS_PASSES_TD
