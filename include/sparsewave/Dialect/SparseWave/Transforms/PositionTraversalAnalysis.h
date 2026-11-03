#ifndef SPARSEWAVE_DIALECT_SPARSEWAVE_TRANSFORMS_POSITIONTRAVERSALANALYSIS_H
#define SPARSEWAVE_DIALECT_SPARSEWAVE_TRANSFORMS_POSITIONTRAVERSALANALYSIS_H

#include "sparsewave/Dialect/SparseWave/IR/SparseWaveOps.h"

#include "llvm/ADT/StringRef.h"

#include <optional>

namespace mlir::sparsewave {

enum class PositionTraversalKind {
  Nondecreasing,
  MayReset,
  Unknown,
};

llvm::StringRef stringifyPositionTraversalKind(PositionTraversalKind kind);

/// Describes how logical coordinates change while a position reduction's
/// collapsed iteration domain is traversed in its requested order.
class PositionTraversalAnalysis {
public:
  explicit PositionTraversalAnalysis(PositionReduceOp reduction)
      : reduction(reduction.getOperation()) {}

  PositionTraversalKind getAxisTraversal(unsigned axis) const;
  PositionTraversalKind getValueTraversal(Value value) const;
  std::optional<unsigned> getAxisForValue(Value value) const;

private:
  Operation *reduction;
};

} // namespace mlir::sparsewave

#endif // SPARSEWAVE_DIALECT_SPARSEWAVE_TRANSFORMS_POSITIONTRAVERSALANALYSIS_H
