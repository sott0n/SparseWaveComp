#include "sparsewave/Dialect/SparseWave/Transforms/Passes.h"

#include "sparsewave/Dialect/SparseWave/IR/SparseWaveOps.h"

#include "mlir/IR/BuiltinOps.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"

#include <iterator>

namespace mlir::sparsewave {
#define GEN_PASS_DEF_REORDERSPARSEWAVEPOSITION
#include "sparsewave/Dialect/SparseWave/Transforms/Passes.h.inc"

namespace {

class ReorderSparseWavePosition
    : public impl::ReorderSparseWavePositionBase<ReorderSparseWavePosition> {
public:
  using impl::ReorderSparseWavePositionBase<
      ReorderSparseWavePosition>::ReorderSparseWavePositionBase;

  void runOnOperation() override {
    if (order.empty()) {
      getOperation().emitError(
          "position reorder requires a non-empty axis permutation");
      signalPassFailure();
      return;
    }

    llvm::SmallSet<StringRef, 4> uniqueAxes;
    for (const std::string &axis : order) {
      if (!uniqueAxes.insert(axis).second) {
        getOperation().emitError()
            << "position reorder must not repeat axis '" << axis << "'";
        signalPassFailure();
        return;
      }
    }

    WalkResult result = getOperation().walk([&](PositionReduceOp op) {
      int64_t rank = op.getLower().size();
      if (static_cast<int64_t>(order.size()) != rank) {
        op.emitError() << "position reorder rank mismatch: got " << order.size()
                       << " axes for a rank-" << rank << " reduction";
        return WalkResult::interrupt();
      }

      SmallVector<int64_t> permutation;
      permutation.reserve(rank);
      for (const std::string &requestedAxis : order) {
        auto axis = llvm::find_if(op.getAxes(), [&](Attribute attribute) {
          return cast<StringAttr>(attribute).getValue() == requestedAxis;
        });
        if (axis == op.getAxes().end()) {
          op.emitError() << "position reorder references unknown axis '"
                         << requestedAxis << "'";
          return WalkResult::interrupt();
        }
        permutation.push_back(std::distance(op.getAxes().begin(), axis));
      }

      op->setAttr("order", DenseI64ArrayAttr::get(&getContext(), permutation));
      return WalkResult::advance();
    });
    if (result.wasInterrupted())
      signalPassFailure();
  }
};

} // namespace
} // namespace mlir::sparsewave
