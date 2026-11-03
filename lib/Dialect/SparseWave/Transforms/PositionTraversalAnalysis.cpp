#include "sparsewave/Dialect/SparseWave/Transforms/PositionTraversalAnalysis.h"

#include "sparsewave/Dialect/SparseWave/Transforms/Passes.h"

#include "mlir/IR/BuiltinOps.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"

namespace mlir::sparsewave {
#define GEN_PASS_DEF_REPORTSPARSEWAVEPOSITIONTRAVERSAL
#include "sparsewave/Dialect/SparseWave/Transforms/Passes.h.inc"

llvm::StringRef stringifyPositionTraversalKind(PositionTraversalKind kind) {
  switch (kind) {
  case PositionTraversalKind::Nondecreasing:
    return "nondecreasing";
  case PositionTraversalKind::MayReset:
    return "may-reset";
  case PositionTraversalKind::Unknown:
    return "unknown";
  }
  llvm_unreachable("unknown position traversal kind");
}

std::optional<unsigned>
PositionTraversalAnalysis::getAxisForValue(Value value) const {
  PositionReduceOp reductionOp = cast<PositionReduceOp>(reduction);
  auto argument = dyn_cast<BlockArgument>(value);
  if (!argument || argument.getOwner() != &reductionOp.getBody().front())
    return std::nullopt;
  return argument.getArgNumber();
}

PositionTraversalKind
PositionTraversalAnalysis::getAxisTraversal(unsigned axis) const {
  PositionReduceOp reductionOp = cast<PositionReduceOp>(reduction);
  if (axis >= reductionOp.getLower().size() || reductionOp.getOrder().empty())
    return PositionTraversalKind::Unknown;
  if (reductionOp.getOrder().front() == static_cast<int64_t>(axis))
    return PositionTraversalKind::Nondecreasing;
  return PositionTraversalKind::MayReset;
}

PositionTraversalKind
PositionTraversalAnalysis::getValueTraversal(Value value) const {
  std::optional<unsigned> axis = getAxisForValue(value);
  if (!axis)
    return PositionTraversalKind::Unknown;
  return getAxisTraversal(*axis);
}

namespace {

class ReportSparseWavePositionTraversal
    : public impl::ReportSparseWavePositionTraversalBase<
          ReportSparseWavePositionTraversal> {
public:
  using impl::ReportSparseWavePositionTraversalBase<
      ReportSparseWavePositionTraversal>::ReportSparseWavePositionTraversalBase;

  void runOnOperation() override {
    getOperation().walk([](PositionReduceOp reduction) {
      PositionTraversalAnalysis analysis(reduction);
      for (auto [axis, name] : llvm::enumerate(reduction.getAxes())) {
        PositionTraversalKind traversal = analysis.getAxisTraversal(axis);
        reduction.emitRemark()
            << "axis '" << cast<StringAttr>(name).getValue()
            << "' traversal is " << stringifyPositionTraversalKind(traversal);
      }
    });
    markAllAnalysesPreserved();
  }
};

} // namespace
} // namespace mlir::sparsewave
