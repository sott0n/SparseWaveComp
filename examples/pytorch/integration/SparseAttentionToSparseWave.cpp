#include "SparseAttentionToSparseWave.h"

#include "sparsewave/Dialect/SparseWave/IR/SparseWaveOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringExtras.h"

#include <cmath>

namespace mlir::sparsewave::example {
namespace {

struct TorchTensorType {
  SmallVector<int64_t> shape;
  bool hasEncoding = false;
  bool hasCanonicalI32CSR = false;
};

struct SparseAttentionMatch {
  Operation *sampledAddMM = nullptr;
  TorchTensorType mask;
  TorchTensorType query;
  TorchTensorType key;
  TorchTensorType value;
  TorchTensorType output;
  double beta = 0.0;
  double alpha = 0.0;
};

Attribute getProperty(Operation *operation, StringRef name) {
  auto properties =
      dyn_cast_or_null<DictionaryAttr>(operation->getPropertiesAsAttribute());
  return properties ? properties.get(name) : Attribute();
}

bool hasOperatorName(Operation *operation, StringRef name) {
  if (operation->getName().getStringRef() != "torch.operator")
    return false;
  auto operatorName =
      dyn_cast_or_null<StringAttr>(getProperty(operation, "name"));
  return operatorName && operatorName.getValue() == name;
}

FailureOr<int64_t> getIntegerConstant(Operation *operation) {
  if (operation->getName().getStringRef() != "torch.constant.int")
    return failure();
  auto value = dyn_cast_or_null<IntegerAttr>(getProperty(operation, "value"));
  if (!value)
    return failure();
  return value.getInt();
}

FailureOr<double> getFloatConstant(Operation *operation) {
  if (operation->getName().getStringRef() != "torch.constant.float")
    return failure();
  auto value = dyn_cast_or_null<FloatAttr>(getProperty(operation, "value"));
  if (!value)
    return failure();
  return value.getValueAsDouble();
}

FailureOr<TorchTensorType> parseTensorType(Type type) {
  auto opaque = dyn_cast<OpaqueType>(type);
  if (!opaque || opaque.getDialectNamespace().getValue() != "torch")
    return failure();

  StringRef data = opaque.getTypeData().trim();
  if (!data.consume_front("vtensor<[") || !data.consume_back(">"))
    return failure();
  size_t shapeEnd = data.find(']');
  if (shapeEnd == StringRef::npos)
    return failure();

  TorchTensorType result;
  SmallVector<StringRef> dimensions;
  data.take_front(shapeEnd).split(dimensions, ',', /*MaxSplit=*/-1,
                                  /*KeepEmpty=*/false);
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
  for (char character : data.trim())
    if (!llvm::isSpace(character))
      compactEncoding.push_back(character);
  result.hasCanonicalI32CSR =
      compactEncoding ==
      "#sparse_tensor.encoding<{map=(d0,d1)->(d0:dense,d1:compressed),"
      "posWidth=32,crdWidth=32}>";
  return result;
}

LogicalResult validate(func::FuncOp function, SparseAttentionMatch &match) {
  if (function.isExternal() || !function.getBody().hasOneBlock())
    return function.emitError(
        "expected a defined single-block Torch SparseAttention function");
  Block &body = function.getBody().front();
  if (body.getNumArguments() != 4 || function.getNumResults() != 1)
    return function.emitError(
        "expected four SparseAttention inputs and one result");

  SmallVector<Operation *> operations;
  for (Operation &operation : body)
    operations.push_back(&operation);
  if (operations.size() != 15)
    return function.emitError(
        "expected the PyTorch 2.13 SparseAttention example graph");

  FailureOr<int64_t> transposeDim0 = getIntegerConstant(operations[0]);
  FailureOr<int64_t> transposeDim1 = getIntegerConstant(operations[1]);
  Operation *transpose = operations[2];
  FailureOr<double> beta = getFloatConstant(operations[3]);
  FailureOr<double> alpha = getFloatConstant(operations[4]);
  Operation *sampledAddMM = operations[5];
  Operation *toSparse = operations[9];
  FailureOr<int64_t> softmaxDim = getIntegerConstant(operations[10]);
  Operation *softmax = operations[12];
  Operation *sparseMM = operations[13];
  auto returnOp = dyn_cast<func::ReturnOp>(operations[14]);

  if (failed(transposeDim0) || failed(transposeDim1) || *transposeDim0 != 0 ||
      *transposeDim1 != 1 ||
      transpose->getName().getStringRef() != "torch.aten.transpose.int" ||
      transpose->getNumOperands() != 3 || transpose->getNumResults() != 1 ||
      transpose->getOperand(0) != body.getArgument(2) ||
      transpose->getOperand(1) != operations[0]->getResult(0) ||
      transpose->getOperand(2) != operations[1]->getResult(0))
    return transpose->emitError("expected key.transpose(0, 1)");
  if (failed(beta) || failed(alpha) || !std::isfinite(*beta) ||
      !std::isfinite(*alpha))
    return sampledAddMM->emitError("expected finite constant beta and alpha");
  if (!hasOperatorName(sampledAddMM, "torch.aten.sparse_sampled_addmm") ||
      sampledAddMM->getNumOperands() != 5 ||
      sampledAddMM->getOperand(0) != body.getArgument(0) ||
      sampledAddMM->getOperand(1) != body.getArgument(1) ||
      sampledAddMM->getOperand(2) != transpose->getResult(0) ||
      sampledAddMM->getOperand(3) != operations[3]->getResult(0) ||
      sampledAddMM->getOperand(4) != operations[4]->getResult(0))
    return sampledAddMM->emitError("expected sampled_addmm(mask, query, key)");

  for (unsigned index : {6u, 7u, 8u, 11u})
    if (operations[index]->getName().getStringRef() != "torch.constant.none")
      return operations[index]->emitError("expected a Torch none constant");
  if (!hasOperatorName(toSparse, "torch.aten.to_sparse") ||
      toSparse->getNumOperands() != 4 ||
      toSparse->getOperand(0) != sampledAddMM->getResult(0))
    return toSparse->emitError("expected CSR-to-COO conversion after SDDMM");
  if (failed(softmaxDim) || *softmaxDim != 1)
    return operations[10]->emitError(
        "expected SparseAttention softmax along dimension 1");
  if (!hasOperatorName(softmax, "torch.aten._sparse_softmax.int") ||
      softmax->getNumOperands() != 3 ||
      softmax->getOperand(0) != toSparse->getResult(0))
    return softmax->emitError("expected sparse softmax after COO conversion");
  if (!hasOperatorName(sparseMM, "torch.aten._sparse_mm") ||
      sparseMM->getNumOperands() != 2 ||
      sparseMM->getOperand(0) != softmax->getResult(0) ||
      sparseMM->getOperand(1) != body.getArgument(3))
    return sparseMM->emitError("expected sparse_mm(probabilities, value)");
  if (!returnOp || returnOp.getNumOperands() != 1 ||
      returnOp.getOperand(0) != sparseMM->getResult(0))
    return operations[14]->emitError("expected the SparseAttention result");

  FailureOr<TorchTensorType> mask =
      parseTensorType(body.getArgument(0).getType());
  FailureOr<TorchTensorType> query =
      parseTensorType(body.getArgument(1).getType());
  FailureOr<TorchTensorType> key =
      parseTensorType(body.getArgument(2).getType());
  FailureOr<TorchTensorType> value =
      parseTensorType(body.getArgument(3).getType());
  FailureOr<TorchTensorType> output =
      parseTensorType(function.getResultTypes().front());
  if (failed(mask) || failed(query) || failed(key) || failed(value) ||
      failed(output))
    return function.emitError("expected static-shape f32 Torch tensors");
  if (!mask->hasCanonicalI32CSR || query->hasEncoding || key->hasEncoding ||
      value->hasEncoding || output->hasEncoding)
    return function.emitError("expected an i32 CSR mask and dense operands");
  if (mask->shape.size() != 2 || query->shape.size() != 2 ||
      key->shape.size() != 2 || value->shape.size() != 2 ||
      output->shape.size() != 2)
    return function.emitError("expected rank-two tensors");
  if (mask->shape[0] != query->shape[0] || mask->shape[1] != key->shape[0] ||
      mask->shape[1] != value->shape[0] || query->shape[1] != key->shape[1] ||
      output->shape[0] != mask->shape[0] || output->shape[1] != value->shape[1])
    return function.emitError("incompatible SparseAttention shapes");

  match = {sampledAddMM, *mask, *query, *key, *value, *output, *beta, *alpha};
  return success();
}

void addYieldingRegion(Operation *operation, Type elementType, Location loc,
                       function_ref<Value(OpBuilder &, Value, Value)> build) {
  Block *body = new Block();
  operation->getRegion(0).push_back(body);
  Value lhs = body->addArgument(elementType, loc);
  Value rhs = body->addArgument(elementType, loc);
  OpBuilder builder = OpBuilder::atBlockEnd(body);
  YieldOp::create(builder, loc, build(builder, lhs, rhs));
}

template <typename OpTy> OpTy setKernelName(OpTy operation, StringRef name) {
  operation->setAttr("sparsewave.kernel_name",
                     StringAttr::get(operation.getContext(), name));
  return operation;
}

LogicalResult lower(func::FuncOp function) {
  SparseAttentionMatch match;
  if (failed(validate(function, match)))
    return failure();

  MLIRContext *context = function.getContext();
  Type indexType = IntegerType::get(context, 32);
  Type elementType = Float32Type::get(context);
  int64_t dynamic = ShapedType::kDynamic;
  int64_t rows = match.mask.shape[0];
  SmallVector<Type> arguments{
      MemRefType::get({rows + 1}, indexType),
      MemRefType::get({dynamic}, indexType),
      MemRefType::get({dynamic}, elementType),
      MemRefType::get(match.query.shape, elementType),
      MemRefType::get({match.key.shape[1], match.key.shape[0]}, elementType),
      MemRefType::get(match.value.shape, elementType),
      MemRefType::get({dynamic}, elementType),
      MemRefType::get({rows}, elementType),
      MemRefType::get({rows}, elementType),
      MemRefType::get(match.output.shape, elementType)};

  Location loc = match.sampledAddMM->getLoc();
  function.getBody().getBlocks().clear();
  function.setType(FunctionType::get(context, arguments, {}));
  function->removeAttr("arg_attrs");
  function->removeAttr("res_attrs");
  Block *entry = function.addEntryBlock();
  OpBuilder builder = OpBuilder::atBlockEnd(entry);
  Value rowOffsets = entry->getArgument(0);
  Value columns = entry->getArgument(1);
  Value maskValues = entry->getArgument(2);
  Value query = entry->getArgument(3);
  Value transposedKey = entry->getArgument(4);
  Value value = entry->getArgument(5);
  Value scores = entry->getArgument(6);
  Value rowMaximum = entry->getArgument(7);
  Value rowSum = entry->getArgument(8);
  Value output = entry->getArgument(9);

  auto sddmm =
      setKernelName(SDDMMOp::create(builder, loc, rowOffsets, columns,
                                    maskValues, query, transposedKey, scores),
                    "sparse_attention_scores");
  addYieldingRegion(
      sddmm, elementType, loc, [&](OpBuilder &nested, Value sample, Value dot) {
        Value beta = arith::ConstantOp::create(
            nested, loc, nested.getF32FloatAttr(match.beta));
        Value alpha = arith::ConstantOp::create(
            nested, loc, nested.getF32FloatAttr(match.alpha));
        Value scaledSample = arith::MulFOp::create(nested, loc, sample, beta);
        Value scaledDot = arith::MulFOp::create(nested, loc, dot, alpha);
        return arith::AddFOp::create(nested, loc, scaledSample, scaledDot);
      });

  setKernelName(CSRRowReduceOp::create(builder, loc, rowOffsets, columns,
                                       scores, rowMaximum, "max"),
                "sparse_attention_row_max");
  auto exponentiate =
      setKernelName(CSRRowwiseMapOp::create(builder, loc, rowOffsets, columns,
                                            scores, rowMaximum, scores),
                    "sparse_attention_exp");
  addYieldingRegion(exponentiate, elementType, loc,
                    [&](OpBuilder &nested, Value input, Value maximum) {
                      Value shifted =
                          arith::SubFOp::create(nested, loc, input, maximum);
                      return math::ExpOp::create(nested, loc, shifted);
                    });
  setKernelName(CSRRowReduceOp::create(builder, loc, rowOffsets, columns,
                                       scores, rowSum, "sum"),
                "sparse_attention_row_sum");
  auto normalize =
      setKernelName(CSRRowwiseMapOp::create(builder, loc, rowOffsets, columns,
                                            scores, rowSum, scores),
                    "sparse_attention_normalize");
  addYieldingRegion(normalize, elementType, loc,
                    [&](OpBuilder &nested, Value numerator, Value denominator) {
                      return arith::DivFOp::create(nested, loc, numerator,
                                                   denominator);
                    });
  setKernelName(
      SpMMOp::create(builder, loc, rowOffsets, columns, scores, value, output),
      "sparse_attention_output");
  func::ReturnOp::create(builder, function.getLoc());
  return success();
}

class ConvertTorchSparseAttentionToSparseWave
    : public PassWrapper<ConvertTorchSparseAttentionToSparseWave,
                         OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      ConvertTorchSparseAttentionToSparseWave)

  StringRef getArgument() const final {
    return "convert-torch-sparse-attention-to-sparsewave";
  }
  StringRef getDescription() const final {
    return "Lower the PyTorch SparseAttention example to SparseWave";
  }
  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<arith::ArithDialect, func::FuncDialect, math::MathDialect,
                    SparseWaveDialect>();
  }
  void runOnOperation() final {
    SmallVector<func::FuncOp> functions;
    getOperation().walk([&](func::FuncOp function) {
      if (function.getNumArguments() == 4)
        functions.push_back(function);
    });
    if (functions.size() != 1 || failed(lower(functions.front())))
      signalPassFailure();
  }
};

} // namespace

void registerSparseAttentionToSparseWavePass() {
  PassRegistration<ConvertTorchSparseAttentionToSparseWave>();
}

} // namespace mlir::sparsewave::example
