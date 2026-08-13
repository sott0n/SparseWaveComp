#include "sparsewave/Dialect/SparseWave/Transforms/Passes.h"

#include "sparsewave/Dialect/SparseWave/IR/SparseWaveOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/StringExtras.h"

namespace mlir::sparsewave {
#define GEN_PASS_DEF_CONVERTTORCHTOSPARSEWAVE
#include "sparsewave/Dialect/SparseWave/Transforms/Passes.h.inc"

namespace {

struct TorchValueTensorType {
  SmallVector<int64_t> shape;
  bool hasEncoding = false;
  bool hasCanonicalI32CSR = false;
};

FailureOr<TorchValueTensorType> parseTorchValueTensorType(Type type) {
  auto opaque = dyn_cast<OpaqueType>(type);
  if (!opaque || opaque.getDialectNamespace().getValue() != "torch")
    return failure();

  StringRef data = opaque.getTypeData().trim();
  if (!data.consume_front("vtensor<[") || !data.consume_back(">"))
    return failure();

  size_t shapeEnd = data.find(']');
  if (shapeEnd == StringRef::npos)
    return failure();

  TorchValueTensorType result;
  SmallVector<StringRef> dimensions;
  data.take_front(shapeEnd).split(dimensions, ',', /*MaxSplit=*/-1,
                                  /*KeepEmpty=*/false);
  if (dimensions.empty())
    return failure();
  for (StringRef dimension : dimensions) {
    int64_t size;
    if (dimension.trim().getAsInteger(10, size) || size < 0)
      return failure();
    result.shape.push_back(size);
  }

  data = data.drop_front(shapeEnd + 1).trim();
  if (!data.consume_front(","))
    return failure();
  data = data.trim();
  if (!data.consume_front("f32"))
    return failure();
  data = data.trim();
  if (data.empty())
    return result;
  if (!data.consume_front(",") || data.trim().empty())
    return failure();
  result.hasEncoding = true;
  std::string compactEncoding;
  for (char character : data.trim()) {
    if (!llvm::isSpace(character))
      compactEncoding.push_back(character);
  }
  result.hasCanonicalI32CSR =
      compactEncoding ==
      "#sparse_tensor.encoding<{map=(d0,d1)->(d0:dense,d1:compressed),"
      "posWidth=32,crdWidth=32}>";
  return result;
}

LogicalResult validateTorchSpMMFunction(func::FuncOp function,
                                        Operation *&torchOperator,
                                        TorchValueTensorType &lhs,
                                        TorchValueTensorType &rhs,
                                        TorchValueTensorType &output) {
  if (function.isExternal() || !function.getBody().hasOneBlock())
    return function.emitError(
        "expected a defined single-block function for Torch CSR SpMM");

  Block &body = function.getBody().front();
  if (body.getNumArguments() != 2 || function.getNumResults() != 1)
    return function.emitError(
        "expected Torch CSR SpMM signature with two inputs and one result");

  SmallVector<Operation *> operations;
  for (Operation &operation : body)
    operations.push_back(&operation);
  if (operations.size() != 2 ||
      operations[0]->getName().getStringRef() != "torch.operator" ||
      !isa<func::ReturnOp>(operations[1]))
    return function.emitError(
        "expected torch.operator followed by func.return");

  torchOperator = operations[0];
  auto operatorName = torchOperator->getAttrOfType<StringAttr>("name");
  if (!operatorName || operatorName.getValue() != "torch.aten._sparse_mm")
    return torchOperator->emitError(
        "expected torch.aten._sparse_mm for the initial Torch bridge");
  if (torchOperator->getNumOperands() != 2 ||
      torchOperator->getNumResults() != 1 ||
      torchOperator->getOperand(0) != body.getArgument(0) ||
      torchOperator->getOperand(1) != body.getArgument(1))
    return torchOperator->emitError(
        "expected torch.aten._sparse_mm to consume both function arguments");

  auto returnOp = cast<func::ReturnOp>(operations[1]);
  if (returnOp.getNumOperands() != 1 ||
      returnOp.getOperand(0) != torchOperator->getResult(0))
    return returnOp.emitError("expected the sparse matrix product result");

  FailureOr<TorchValueTensorType> parsedLhs =
      parseTorchValueTensorType(body.getArgument(0).getType());
  FailureOr<TorchValueTensorType> parsedRhs =
      parseTorchValueTensorType(body.getArgument(1).getType());
  FailureOr<TorchValueTensorType> parsedOutput =
      parseTorchValueTensorType(function.getResultTypes().front());
  if (failed(parsedLhs) || failed(parsedRhs) || failed(parsedOutput))
    return function.emitError(
        "expected static-shape Torch f32 value tensor types");

  lhs = *parsedLhs;
  rhs = *parsedRhs;
  output = *parsedOutput;
  if (!lhs.hasCanonicalI32CSR || rhs.hasEncoding || output.hasEncoding)
    return function.emitError(
        "expected an i32 CSR left-hand side and dense RHS/result tensors");
  if (lhs.shape.size() != 2 || rhs.shape.size() != 2 ||
      output.shape.size() != 2)
    return function.emitError("expected rank-two Torch tensors");
  if (lhs.shape[1] != rhs.shape[0] || lhs.shape[0] != output.shape[0] ||
      rhs.shape[1] != output.shape[1])
    return function.emitError("incompatible Torch CSR SpMM shapes");

  return success();
}

LogicalResult lowerTorchSpMMFunction(func::FuncOp function) {
  Operation *torchOperator = nullptr;
  TorchValueTensorType lhs;
  TorchValueTensorType rhs;
  TorchValueTensorType output;
  if (failed(
          validateTorchSpMMFunction(function, torchOperator, lhs, rhs, output)))
    return failure();

  MLIRContext *context = function.getContext();
  Type indexType = IntegerType::get(context, 32);
  Type elementType = Float32Type::get(context);
  int64_t dynamic = ShapedType::kDynamic;
  SmallVector<Type> argumentTypes{
      MemRefType::get({lhs.shape[0] + 1}, indexType),
      MemRefType::get({dynamic}, indexType),
      MemRefType::get({dynamic}, elementType),
      MemRefType::get(rhs.shape, elementType),
      MemRefType::get(output.shape, elementType),
  };

  Location operatorLocation = torchOperator->getLoc();
  function.getBody().getBlocks().clear();
  function.setType(FunctionType::get(context, argumentTypes, {}));
  function->removeAttr("arg_attrs");
  function->removeAttr("res_attrs");

  Block *entry = function.addEntryBlock();
  OpBuilder builder = OpBuilder::atBlockEnd(entry);
  SpMMOp::create(builder, operatorLocation, entry->getArgument(0),
                 entry->getArgument(1), entry->getArgument(2),
                 entry->getArgument(3), entry->getArgument(4));
  func::ReturnOp::create(builder, function.getLoc());
  return success();
}

class ConvertTorchToSparseWave
    : public impl::ConvertTorchToSparseWaveBase<ConvertTorchToSparseWave> {
public:
  void runOnOperation() override {
    SmallVector<func::FuncOp> functions;
    getOperation().walk([&](func::FuncOp function) {
      bool containsTorchOperation = false;
      function.walk([&](Operation *operation) {
        containsTorchOperation |=
            operation->getName().getDialectNamespace() == "torch";
      });
      if (containsTorchOperation)
        functions.push_back(function);
    });

    for (func::FuncOp function : functions) {
      if (failed(lowerTorchSpMMFunction(function))) {
        signalPassFailure();
        return;
      }
    }
  }
};

} // namespace
} // namespace mlir::sparsewave
