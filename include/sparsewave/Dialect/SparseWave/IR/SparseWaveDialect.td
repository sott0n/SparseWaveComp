#ifndef SPARSEWAVE_DIALECT_SPARSEWAVE_IR_SPARSEWAVEDIALECT_TD
#define SPARSEWAVE_DIALECT_SPARSEWAVE_IR_SPARSEWAVEDIALECT_TD

include "mlir/IR/OpBase.td"

def SparseWave_Dialect : Dialect {
  let name = "sparsewave";
  let summary = "Sparse computations intended for wavefront execution";
  let description = [{
    The SparseWave dialect represents sparse computations before they are
    mapped to GPU threads and AMD wavefronts.
  }];
  let cppNamespace = "::mlir::sparsewave";
}

class SparseWave_Op<string mnemonic, list<Trait> traits = []>
    : Op<SparseWave_Dialect, mnemonic, traits>;

#endif // SPARSEWAVE_DIALECT_SPARSEWAVE_IR_SPARSEWAVEDIALECT_TD
