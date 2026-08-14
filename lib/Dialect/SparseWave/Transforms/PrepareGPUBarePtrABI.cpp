#include "sparsewave/Dialect/SparseWave/Transforms/Passes.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/RegionUtils.h"

#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <limits>

namespace mlir::sparsewave {
#define GEN_PASS_DEF_PREPAREGPUBAREPTRABI
#include "sparsewave/Dialect/SparseWave/Transforms/Passes.h.inc"

namespace {

using CaptureUses = llvm::MapVector<Value, SmallVector<OpOperand *>>;

FailureOr<MemRefType> getBarePtrViewType(Value capture) {
  auto type = dyn_cast<MemRefType>(capture.getType());
  if (!type || type.hasStaticShape())
    return failure();
  if (!type.getLayout().isIdentity())
    return failure();

  SmallVector<int64_t> strides;
  int64_t offset;
  if (failed(type.getStridesAndOffset(strides, offset)) ||
      ShapedType::isDynamic(offset) ||
      llvm::any_of(strides, [](int64_t stride) {
        return ShapedType::isDynamic(stride);
      }))
    return failure();

  constexpr int64_t maxStaticExtent = std::numeric_limits<int64_t>::max();
  SmallVector<int64_t> shape(type.getShape());
  llvm::replace(shape, ShapedType::kDynamic, maxStaticExtent);
  return MemRefType::get(shape, type.getElementType(), type.getLayout(),
                         type.getMemorySpace());
}

LogicalResult prepareLaunch(gpu::LaunchOp launch) {
  CaptureUses captures;
  visitUsedValuesDefinedAbove(launch.getBody(), launch.getBody(),
                              [&](OpOperand *operand) {
                                if (isa<MemRefType>(operand->get().getType()))
                                  captures[operand->get()].push_back(operand);
                              });

  OpBuilder builder(launch);
  for (auto &[capture, uses] : captures) {
    auto type = cast<MemRefType>(capture.getType());
    if (type.hasStaticShape())
      continue;

    FailureOr<MemRefType> viewType = getBarePtrViewType(capture);
    if (failed(viewType))
      return launch.emitError()
             << "bare-pointer kernel capture requires a dynamic memref with "
                "identity layout and static offset/strides, got "
             << type;
    for (OpOperand *use : uses) {
      if (!isa<memref::IndexedAccessOpInterface>(use->getOwner()))
        return use->getOwner()->emitError()
               << "dynamic memref capture requires runtime descriptor "
                  "metadata and cannot use the bare-pointer kernel ABI";
    }

    SmallVector<int64_t> strides;
    int64_t offset;
    if (failed(viewType->getStridesAndOffset(strides, offset)))
      return launch.emitError("failed to compute bare-pointer view layout");
    Value view = memref::ReinterpretCastOp::create(
        builder, launch.getLoc(), *viewType, capture, offset,
        viewType->getShape(), strides);
    for (OpOperand *use : uses)
      use->set(view);
  }
  return success();
}

class PrepareGPUBarePtrABI
    : public impl::PrepareGPUBarePtrABIBase<PrepareGPUBarePtrABI> {
public:
  void runOnOperation() override {
    WalkResult result = getOperation().walk([](gpu::LaunchOp launch) {
      return failed(prepareLaunch(launch)) ? WalkResult::interrupt()
                                           : WalkResult::advance();
    });
    if (result.wasInterrupted())
      signalPassFailure();
  }
};

} // namespace
} // namespace mlir::sparsewave
