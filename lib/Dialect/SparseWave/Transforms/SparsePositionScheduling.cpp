#include "sparsewave/Dialect/SparseWave/Transforms/Passes.h"

#include "sparsewave/Dialect/SparseWave/IR/SparseWaveOps.h"

#include "SparseGPUUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseSet.h"

namespace mlir::sparsewave {
#define GEN_PASS_DEF_SCHEDULESPARSEWAVEPOSITION
#include "sparsewave/Dialect/SparseWave/Transforms/Passes.h.inc"

namespace {

struct KeyedContribution {
  Value key;
  Value value;
};

struct CompressedSegmentState {
  Value segment;
  Value nextBoundary;
};

Value buildCollapsedIterationCount(OpBuilder &builder, Location loc,
                                   PositionReduceOp reduction) {
  Value count;
  for (auto [lower, upper] :
       llvm::zip(reduction.getLower(), reduction.getUpper())) {
    Value extent;
    if (matchPattern(lower, m_Zero()))
      extent = upper;
    else
      extent = arith::SubIOp::create(builder, loc, upper, lower);
    if (count)
      count = arith::MulIOp::create(builder, loc, count, extent);
    else
      count = extent;
  }
  return count;
}

SmallVector<Value> buildLogicalCoordinates(OpBuilder &builder, Location loc,
                                           PositionReduceOp reduction,
                                           Value linearIteration) {
  if (reduction.getLower().size() == 1) {
    Value lower = reduction.getLower().front();
    if (matchPattern(lower, m_Zero()))
      return {linearIteration};
    return {arith::AddIOp::create(builder, loc, lower, linearIteration)};
  }

  SmallVector<Value> coordinates(reduction.getLower().size());
  Value remaining = linearIteration;
  ArrayRef<int64_t> order = reduction.getOrder();
  for (size_t orderIndex = order.size(); orderIndex > 0; --orderIndex) {
    int64_t axis = order[orderIndex - 1];
    Value offset = remaining;
    if (orderIndex > 1) {
      Value extent = arith::SubIOp::create(
          builder, loc, reduction.getUpper()[axis], reduction.getLower()[axis]);
      offset = arith::RemUIOp::create(builder, loc, remaining, extent);
      remaining = arith::DivUIOp::create(builder, loc, remaining, extent);
    }
    Value lower = reduction.getLower()[axis];
    if (matchPattern(lower, m_Zero()))
      coordinates[axis] = offset;
    else
      coordinates[axis] = arith::AddIOp::create(builder, loc, lower, offset);
  }
  return coordinates;
}

std::optional<unsigned> findAxis(PositionReduceOp reduction,
                                 StringRef axisName) {
  for (auto [index, attribute] : llvm::enumerate(reduction.getAxes()))
    if (cast<StringAttr>(attribute).getValue() == axisName)
      return index;
  return std::nullopt;
}

SmallVector<Value> buildCooperativeCoordinates(OpBuilder &builder, Location loc,
                                               PositionReduceOp reduction,
                                               Value linearIteration,
                                               unsigned cooperativeAxis,
                                               Value axisGroup, Value lane,
                                               Value participantCount) {
  unsigned rank = reduction.getLower().size();
  SmallVector<Value> coordinates(rank);
  Value remaining = linearIteration;
  ArrayRef<int64_t> order = reduction.getOrder();
  // Decode the non-cooperative position domain independently from the axis
  // group assigned to wave lanes. This lets one wave traverse consecutive
  // points on that domain without changing its cooperative coordinates.
  for (size_t orderIndex = order.size(); orderIndex > 0; --orderIndex) {
    unsigned axis = order[orderIndex - 1];
    if (axis == cooperativeAxis)
      continue;
    Value extent = arith::SubIOp::create(
        builder, loc, reduction.getUpper()[axis], reduction.getLower()[axis]);
    Value offset = arith::RemUIOp::create(builder, loc, remaining, extent);
    remaining = arith::DivUIOp::create(builder, loc, remaining, extent);
    Value lower = reduction.getLower()[axis];
    coordinates[axis] =
        matchPattern(lower, m_Zero())
            ? offset
            : arith::AddIOp::create(builder, loc, lower, offset);
  }

  Value groupOffset =
      arith::MulIOp::create(builder, loc, axisGroup, participantCount);
  Value laneOffset = arith::AddIOp::create(builder, loc, groupOffset, lane);
  Value lower = reduction.getLower()[cooperativeAxis];
  coordinates[cooperativeAxis] =
      matchPattern(lower, m_Zero())
          ? laneOffset
          : arith::AddIOp::create(builder, loc, lower, laneOffset);
  return coordinates;
}

Value buildNonCooperativeIterationCount(OpBuilder &builder, Location loc,
                                        PositionReduceOp reduction,
                                        unsigned cooperativeAxis) {
  Value count = arith::ConstantIndexOp::create(builder, loc, 1);
  for (unsigned axis = 0; axis < reduction.getLower().size(); ++axis) {
    if (axis == cooperativeAxis)
      continue;
    Value extent = arith::SubIOp::create(
        builder, loc, reduction.getUpper()[axis], reduction.getLower()[axis]);
    count = arith::MulIOp::create(builder, loc, count, extent);
  }
  return count;
}

PositionParallelOp buildOutputInitialization(PatternRewriter &rewriter,
                                             Location loc, Value output,
                                             Value outputSize, Value zero,
                                             int64_t blockSize) {
  auto parallel =
      PositionParallelOp::create(rewriter, loc, outputSize, "thread",
                                 rewriter.getI64IntegerAttr(blockSize));
  Block *body =
      rewriter.createBlock(&parallel.getBody(), parallel.getBody().end(),
                           {rewriter.getIndexType(), rewriter.getIndexType(),
                            rewriter.getIndexType()},
                           {loc, loc, loc});
  rewriter.setInsertionPointToStart(body);
  memref::StoreOp::create(rewriter, loc, zero, output, body->getArgument(0));
  YieldOp::create(rewriter, loc);
  return parallel;
}

KeyedContribution cloneContributionBody(OpBuilder &builder,
                                        PositionReduceOp reduction,
                                        ValueRange coordinates) {
  Block &source = reduction.getBody().front();
  IRMapping mapping;
  for (auto [argument, coordinate] :
       llvm::zip(source.getArguments(), coordinates))
    mapping.map(argument, coordinate);
  for (Operation &operation : source.without_terminator())
    builder.clone(operation, mapping);
  auto yield = cast<YieldOp>(source.getTerminator());
  return {mapping.lookup(yield.getResults()[0]),
          mapping.lookup(yield.getResults()[1])};
}

Value buildZero(OpBuilder &builder, Location loc, Type type) {
  if (type.isIndex())
    return arith::ConstantIndexOp::create(builder, loc, 0);
  return arith::ConstantOp::create(builder, loc, builder.getZeroAttr(type));
}

Value buildLaneZeroBroadcast(OpBuilder &builder, Location loc, Value value,
                             int64_t waveSize) {
  if (value.getType().isIndex()) {
    Value valueI64 =
        arith::IndexCastOp::create(builder, loc, builder.getI64Type(), value);
    Value broadcastI64 = gpu::ShuffleOp::create(builder, loc, valueI64, 0,
                                                waveSize, gpu::ShuffleMode::IDX)
                             .getShuffleResult();
    return arith::IndexCastOp::create(builder, loc, builder.getIndexType(),
                                      broadcastI64);
  }
  return gpu::ShuffleOp::create(builder, loc, value, 0, waveSize,
                                gpu::ShuffleMode::IDX)
      .getShuffleResult();
}

bool isReadOnlyBroadcastCandidate(Operation &operation) {
  if (operation.getNumRegions() != 0 || operation.getNumResults() == 0)
    return false;
  if (!llvm::all_of(operation.getResultTypes(), [](Type type) {
        return type.isIndex() || isa<IntegerType, FloatType>(type);
      }))
    return false;
  std::optional<SmallVector<MemoryEffects::EffectInstance>> effects =
      getEffectsRecursively(&operation);
  return effects && !effects->empty() &&
         llvm::all_of(*effects, [](MemoryEffects::EffectInstance &effect) {
           return isa<MemoryEffects::Read>(effect.getEffect());
         });
}

bool isMappedOrCaptured(Value value, Block &source, IRMapping &mapping) {
  if (mapping.contains(value))
    return true;
  auto argument = dyn_cast<BlockArgument>(value);
  if (argument)
    return argument.getOwner() != &source;
  Operation *definition = value.getDefiningOp();
  return !definition || definition->getBlock() != &source;
}

struct CooperativeBodyMapping {
  IRMapping values;
  llvm::DenseSet<Operation *> hoistedOperations;
};

CooperativeBodyMapping
buildCooperativeBodyMapping(OpBuilder &builder, Location loc,
                            PositionReduceOp reduction, ValueRange coordinates,
                            unsigned cooperativeAxis, Value lane,
                            int64_t waveSize) {
  Block &source = reduction.getBody().front();
  CooperativeBodyMapping result;
  for (auto [argument, coordinate] :
       llvm::zip(source.getArguments(), coordinates))
    result.values.map(argument, coordinate);

  llvm::DenseSet<Value> laneDependentValues;
  laneDependentValues.insert(source.getArgument(cooperativeAxis));
  Value zeroIndex = arith::ConstantIndexOp::create(builder, loc, 0);
  Value laneIsZero = arith::CmpIOp::create(
      builder, loc, arith::CmpIPredicate::eq, lane, zeroIndex);

  // Track dependence on the selected axis through the contribution body.
  // Uniform pure operations are hoisted. Uniform reads execute only in lane
  // zero and are broadcast before the active-lane guard so the shuffle still
  // has a complete hardware wave, including a partial final axis group.
  for (Operation &operation : source.without_terminator()) {
    bool laneDependent =
        llvm::any_of(operation.getOperands(), [&](Value value) {
          return laneDependentValues.contains(value);
        });
    if (laneDependent) {
      laneDependentValues.insert(operation.getResults().begin(),
                                 operation.getResults().end());
      continue;
    }

    bool operandsAvailable =
        llvm::all_of(operation.getOperands(), [&](Value value) {
          return isMappedOrCaptured(value, source, result.values);
        });
    if (!operandsAvailable)
      continue;

    if (isMemoryEffectFree(&operation)) {
      Operation *clone = builder.clone(operation, result.values);
      for (auto [original, replacement] :
           llvm::zip(operation.getResults(), clone->getResults()))
        result.values.map(original, replacement);
      result.hoistedOperations.insert(&operation);
      continue;
    }
    if (!isReadOnlyBroadcastCandidate(operation))
      continue;

    auto laneZero = scf::IfOp::create(
        builder, loc, laneIsZero,
        [&](OpBuilder &thenBuilder, Location thenLoc) {
          Operation *clone = thenBuilder.clone(operation, result.values);
          scf::YieldOp::create(thenBuilder, thenLoc, clone->getResults());
        },
        [&](OpBuilder &elseBuilder, Location elseLoc) {
          SmallVector<Value> zeros;
          for (Type type : operation.getResultTypes())
            zeros.push_back(buildZero(elseBuilder, elseLoc, type));
          scf::YieldOp::create(elseBuilder, elseLoc, zeros);
        });
    for (auto [original, laneZeroValue] :
         llvm::zip(operation.getResults(), laneZero.getResults()))
      result.values.map(original, buildLaneZeroBroadcast(
                                      builder, loc, laneZeroValue, waveSize));
    result.hoistedOperations.insert(&operation);
  }
  return result;
}

KeyedContribution
cloneRemainingContributionBody(OpBuilder &builder, PositionReduceOp reduction,
                               CooperativeBodyMapping &mapping) {
  Block &source = reduction.getBody().front();
  for (Operation &operation : source.without_terminator())
    if (!mapping.hoistedOperations.contains(&operation))
      builder.clone(operation, mapping.values);
  auto yield = cast<YieldOp>(source.getTerminator());
  return {mapping.values.lookup(yield.getResults()[0]),
          mapping.values.lookup(yield.getResults()[1])};
}

KeyedContribution
buildActiveCooperativeContribution(OpBuilder &builder, Location loc,
                                   PositionReduceOp reduction,
                                   CooperativeBodyMapping &mapping,
                                   Value active, Value zeroIndex, Value zero) {
  auto entry = scf::IfOp::create(
      builder, loc, TypeRange{builder.getIndexType(), zero.getType()}, active,
      /*withElseRegion=*/true);
  builder.setInsertionPointToStart(&entry.getThenRegion().front());
  KeyedContribution contribution =
      cloneRemainingContributionBody(builder, reduction, mapping);
  scf::YieldOp::create(builder, loc,
                       ValueRange{contribution.key, contribution.value});
  builder.setInsertionPointToStart(&entry.getElseRegion().front());
  scf::YieldOp::create(builder, loc, ValueRange{zeroIndex, zero});
  builder.setInsertionPointAfter(entry);
  return {entry.getResult(0), entry.getResult(1)};
}

std::optional<CompressedSegmentAtPositionOp>
matchCompressedSegmentRecovery(PositionReduceOp reduction) {
  Block &body = reduction.getBody().front();
  if (body.getNumArguments() != 1)
    return std::nullopt;
  auto yield = cast<YieldOp>(body.getTerminator());
  for (Operation &operation : body.without_terminator()) {
    auto recovery = dyn_cast<CompressedSegmentAtPositionOp>(operation);
    if (recovery && recovery.getPosition() == body.getArgument(0) &&
        yield.getResults()[0] == recovery.getSegment())
      return recovery;
  }
  return std::nullopt;
}

KeyedContribution cloneContributionBodyWithCompressedSegment(
    OpBuilder &builder, PositionReduceOp reduction, ValueRange coordinates,
    CompressedSegmentAtPositionOp recovery, Value segment) {
  Block &source = reduction.getBody().front();
  IRMapping mapping;
  for (auto [argument, coordinate] :
       llvm::zip(source.getArguments(), coordinates))
    mapping.map(argument, coordinate);
  mapping.map(recovery.getSegment(), segment);
  for (Operation &operation : source.without_terminator()) {
    if (&operation == recovery.getOperation())
      continue;
    builder.clone(operation, mapping);
  }
  auto yield = cast<YieldOp>(source.getTerminator());
  return {mapping.lookup(yield.getResults()[0]),
          mapping.lookup(yield.getResults()[1])};
}

Value loadNextCompressedSegmentBoundary(OpBuilder &builder, Location loc,
                                        Value offsets, Value segment) {
  Value one = arith::ConstantIndexOp::create(builder, loc, 1);
  Value nextSegment = arith::AddIOp::create(builder, loc, segment, one);
  Value boundaryValue =
      memref::LoadOp::create(builder, loc, offsets, nextSegment);
  return castToIndex(builder, loc, boundaryValue);
}

CompressedSegmentState
buildNextCompressedSegmentState(OpBuilder &builder, Location loc, Value offsets,
                                Value segment, Value nextBoundary,
                                Value position) {
  Type indexType = builder.getIndexType();
  Value one = arith::ConstantIndexOp::create(builder, loc, 1);
  auto advance =
      scf::WhileOp::create(builder, loc, TypeRange{indexType, indexType},
                           ValueRange{segment, nextBoundary});
  SmallVector<Location> argumentLocations(2, loc);
  Block *before = builder.createBlock(
      &advance.getBefore(), {}, advance.getResultTypes(), argumentLocations);
  builder.setInsertionPointToStart(before);
  Value currentSegment = before->getArgument(0);
  Value currentBoundary = before->getArgument(1);
  Value crossesBoundary = arith::CmpIOp::create(
      builder, loc, arith::CmpIPredicate::ule, currentBoundary, position);
  scf::ConditionOp::create(builder, loc, crossesBoundary,
                           ValueRange{currentSegment, currentBoundary});

  Block *after = builder.createBlock(
      &advance.getAfter(), {}, advance.getResultTypes(), argumentLocations);
  builder.setInsertionPointToStart(after);
  currentSegment = after->getArgument(0);
  Value advancedSegment =
      arith::AddIOp::create(builder, loc, currentSegment, one);
  Value advancedBoundary =
      loadNextCompressedSegmentBoundary(builder, loc, offsets, advancedSegment);
  scf::YieldOp::create(builder, loc,
                       ValueRange{advancedSegment, advancedBoundary});

  builder.setInsertionPointAfter(advance);
  return {advance.getResult(0), advance.getResult(1)};
}

class ThreadPositionReducePattern : public OpRewritePattern<PositionReduceOp> {
public:
  ThreadPositionReducePattern(MLIRContext *context, int64_t blockSize,
                              int64_t chunkSize)
      : OpRewritePattern<PositionReduceOp>(context), blockSize(blockSize),
        chunkSize(chunkSize) {}

  LogicalResult matchAndRewrite(PositionReduceOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value zeroIndex = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value chunkSizeValue =
        arith::ConstantIndexOp::create(rewriter, loc, chunkSize);
    Value outputSize =
        memref::DimOp::create(rewriter, loc, op.getOutput(), zeroIndex);
    Type valueType =
        cast<MemRefType>(op.getOutput().getType()).getElementType();
    Value zero = arith::ConstantOp::create(rewriter, loc,
                                           rewriter.getZeroAttr(valueType));
    PositionParallelOp initialization = buildOutputInitialization(
        rewriter, loc, op.getOutput(), outputSize, zero, blockSize);

    rewriter.setInsertionPointAfter(initialization);
    Value iterationCount = buildCollapsedIterationCount(rewriter, loc, op);
    Value workerCount = arith::CeilDivUIOp::create(
        rewriter, loc, iterationCount, chunkSizeValue);
    auto parallel =
        PositionParallelOp::create(rewriter, loc, workerCount, "thread",
                                   rewriter.getI64IntegerAttr(blockSize));
    Block *body =
        rewriter.createBlock(&parallel.getBody(), parallel.getBody().end(),
                             {rewriter.getIndexType(), rewriter.getIndexType(),
                              rewriter.getIndexType()},
                             {loc, loc, loc});
    rewriter.setInsertionPointToStart(body);
    Value worker = body->getArgument(0);
    auto chunk =
        PositionForOp::create(rewriter, loc, zeroIndex, iterationCount, worker,
                              rewriter.getI64IntegerAttr(chunkSize));
    Block *chunkBody = rewriter.createBlock(
        &chunk.getBody(), chunk.getBody().end(),
        {rewriter.getIndexType(), rewriter.getIndexType()}, {loc, loc});
    rewriter.setInsertionPointToStart(chunkBody);
    SmallVector<Value> coordinates =
        buildLogicalCoordinates(rewriter, loc, op, chunkBody->getArgument(0));
    KeyedContribution contribution =
        cloneContributionBody(rewriter, op, coordinates);
    memref::AtomicRMWOp::create(rewriter, loc, arith::AtomicRMWKind::addf,
                                contribution.value, op.getOutput(),
                                ValueRange{contribution.key});
    YieldOp::create(rewriter, loc);
    rewriter.setInsertionPointAfter(chunk);
    YieldOp::create(rewriter, loc);
    rewriter.eraseOp(op);
    return success();
  }

private:
  int64_t blockSize;
  int64_t chunkSize;
};

class ThreadSegmentedPositionReducePattern
    : public OpRewritePattern<PositionReduceOp> {
public:
  ThreadSegmentedPositionReducePattern(MLIRContext *context, int64_t blockSize,
                                       int64_t chunkSize)
      : OpRewritePattern<PositionReduceOp>(context), blockSize(blockSize),
        chunkSize(chunkSize) {}

  LogicalResult matchAndRewrite(PositionReduceOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value zeroIndex = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value oneIndex = arith::ConstantIndexOp::create(rewriter, loc, 1);
    Value chunkSizeValue =
        arith::ConstantIndexOp::create(rewriter, loc, chunkSize);
    Value outputSize =
        memref::DimOp::create(rewriter, loc, op.getOutput(), zeroIndex);
    Type valueType =
        cast<MemRefType>(op.getOutput().getType()).getElementType();
    Value zero = arith::ConstantOp::create(rewriter, loc,
                                           rewriter.getZeroAttr(valueType));
    PositionParallelOp initialization = buildOutputInitialization(
        rewriter, loc, op.getOutput(), outputSize, zero, blockSize);

    rewriter.setInsertionPointAfter(initialization);
    Value iterationCount = buildCollapsedIterationCount(rewriter, loc, op);
    Value workerCount = arith::CeilDivUIOp::create(
        rewriter, loc, iterationCount, chunkSizeValue);
    auto parallel =
        PositionParallelOp::create(rewriter, loc, workerCount, "thread",
                                   rewriter.getI64IntegerAttr(blockSize));
    Block *body =
        rewriter.createBlock(&parallel.getBody(), parallel.getBody().end(),
                             {rewriter.getIndexType(), rewriter.getIndexType(),
                              rewriter.getIndexType()},
                             {loc, loc, loc});
    rewriter.setInsertionPointToStart(body);
    Value worker = body->getArgument(0);
    Value chunkOffset =
        arith::MulIOp::create(rewriter, loc, worker, chunkSizeValue);
    Value begin = chunkOffset;
    Value remaining =
        arith::SubIOp::create(rewriter, loc, iterationCount, begin);
    Value boundedSize =
        arith::MinUIOp::create(rewriter, loc, remaining, chunkSizeValue);
    Value end = arith::AddIOp::create(rewriter, loc, begin, boundedSize);

    std::optional<CompressedSegmentAtPositionOp> segmentRecovery =
        matchCompressedSegmentRecovery(op);
    SmallVector<Value> firstCoordinates =
        buildLogicalCoordinates(rewriter, loc, op, begin);
    KeyedContribution first;
    if (segmentRecovery) {
      Value firstSegment = CompressedSegmentAtPositionOp::create(
          rewriter, loc, rewriter.getIndexType(), segmentRecovery->getOffsets(),
          firstCoordinates[0]);
      first = cloneContributionBodyWithCompressedSegment(
          rewriter, op, firstCoordinates, *segmentRecovery, firstSegment);
    } else {
      first = cloneContributionBody(rewriter, op, firstCoordinates);
    }

    Value nextPosition = arith::AddIOp::create(rewriter, loc, begin, oneIndex);
    SmallVector<Value> reductionInitializers{first.key, first.value};
    if (segmentRecovery) {
      Value firstBoundary = loadNextCompressedSegmentBoundary(
          rewriter, loc, segmentRecovery->getOffsets(), first.key);
      reductionInitializers.push_back(firstBoundary);
    }
    auto reduction = scf::ForOp::create(
        rewriter, loc, nextPosition, end, oneIndex, reductionInitializers,
        [&](OpBuilder &builder, Location bodyLoc, Value linearIteration,
            ValueRange iterArgs) {
          Value currentKey = iterArgs[0];
          Value currentValue = iterArgs[1];
          Value nextBoundary;
          SmallVector<Value> coordinates =
              buildLogicalCoordinates(builder, bodyLoc, op, linearIteration);
          KeyedContribution contribution;
          if (segmentRecovery) {
            CompressedSegmentState segmentState =
                buildNextCompressedSegmentState(
                    builder, bodyLoc, segmentRecovery->getOffsets(), currentKey,
                    iterArgs[2], coordinates[0]);
            contribution = cloneContributionBodyWithCompressedSegment(
                builder, op, coordinates, *segmentRecovery,
                segmentState.segment);
            nextBoundary = segmentState.nextBoundary;
          } else {
            contribution = cloneContributionBody(builder, op, coordinates);
          }
          Value sameKey =
              arith::CmpIOp::create(builder, bodyLoc, arith::CmpIPredicate::eq,
                                    currentKey, contribution.key);
          Value combined = arith::AddFOp::create(builder, bodyLoc, currentValue,
                                                 contribution.value);
          scf::IfOp::create(
              builder, bodyLoc, sameKey,
              [&](OpBuilder &thenBuilder, Location thenLoc) {
                scf::YieldOp::create(thenBuilder, thenLoc);
              },
              [&](OpBuilder &elseBuilder, Location elseLoc) {
                memref::AtomicRMWOp::create(
                    elseBuilder, elseLoc, arith::AtomicRMWKind::addf,
                    currentValue, op.getOutput(), ValueRange{currentKey});
                scf::YieldOp::create(elseBuilder, elseLoc);
              });
          Value nextValue = arith::SelectOp::create(
              builder, bodyLoc, sameKey, combined, contribution.value);
          SmallVector<Value> nextState{contribution.key, nextValue};
          if (segmentRecovery)
            nextState.push_back(nextBoundary);
          scf::YieldOp::create(builder, bodyLoc, nextState);
        });
    memref::AtomicRMWOp::create(rewriter, loc, arith::AtomicRMWKind::addf,
                                reduction.getResult(1), op.getOutput(),
                                ValueRange{reduction.getResult(0)});
    YieldOp::create(rewriter, loc);
    rewriter.eraseOp(op);
    return success();
  }

private:
  int64_t blockSize;
  int64_t chunkSize;
};

class WavePositionReducePattern : public OpRewritePattern<PositionReduceOp> {
public:
  WavePositionReducePattern(MLIRContext *context, int64_t blockSize,
                            int64_t waveSize)
      : OpRewritePattern<PositionReduceOp>(context), blockSize(blockSize),
        waveSize(waveSize) {}

  LogicalResult matchAndRewrite(PositionReduceOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value zeroIndex = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value oneIndex = arith::ConstantIndexOp::create(rewriter, loc, 1);
    Value blockSizeValue =
        arith::ConstantIndexOp::create(rewriter, loc, blockSize);
    Value wavesPerBlockValue =
        arith::ConstantIndexOp::create(rewriter, loc, blockSize / waveSize);
    Value outputSize =
        memref::DimOp::create(rewriter, loc, op.getOutput(), zeroIndex);
    Type valueType =
        cast<MemRefType>(op.getOutput().getType()).getElementType();
    Value zero = arith::ConstantOp::create(rewriter, loc,
                                           rewriter.getZeroAttr(valueType));
    PositionParallelOp initialization = buildOutputInitialization(
        rewriter, loc, op.getOutput(), outputSize, zero, blockSize);

    rewriter.setInsertionPointAfter(initialization);
    Value iterationCount = buildCollapsedIterationCount(rewriter, loc, op);
    Value requiredBlocks = arith::CeilDivUIOp::create(
        rewriter, loc, iterationCount, blockSizeValue);
    Value gridSize =
        arith::MaxUIOp::create(rewriter, loc, requiredBlocks, oneIndex);
    Value waveCount =
        arith::MulIOp::create(rewriter, loc, gridSize, wavesPerBlockValue);
    auto parallel =
        PositionParallelOp::create(rewriter, loc, waveCount, "wave",
                                   rewriter.getI64IntegerAttr(blockSize));
    Block *body =
        rewriter.createBlock(&parallel.getBody(), parallel.getBody().end(),
                             {rewriter.getIndexType(), rewriter.getIndexType(),
                              rewriter.getIndexType()},
                             {loc, loc, loc});
    rewriter.setInsertionPointToStart(body);
    Value wave = body->getArgument(0);
    Value lane = body->getArgument(1);
    auto partition = PositionSpaceOp::create(
        rewriter, loc, rewriter.getIndexType(), rewriter.getIndexType(),
        zeroIndex, iterationCount, wave, waveCount);
    Value linearIteration =
        arith::AddIOp::create(rewriter, loc, partition.getBegin(), lane);
    Value active =
        arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::ult,
                              linearIteration, partition.getEnd());
    auto entry = scf::IfOp::create(
        rewriter, loc, active,
        [&](OpBuilder &builder, Location bodyLoc) {
          SmallVector<Value> coordinates =
              buildLogicalCoordinates(builder, bodyLoc, op, linearIteration);
          KeyedContribution contribution =
              cloneContributionBody(builder, op, coordinates);
          scf::YieldOp::create(
              builder, bodyLoc,
              ValueRange{contribution.key, contribution.value});
        },
        [&](OpBuilder &builder, Location bodyLoc) {
          scf::YieldOp::create(builder, bodyLoc, ValueRange{zeroIndex, zero});
        });
    Value key = entry.getResult(0);
    Value keyI64 =
        arith::IndexCastOp::create(rewriter, loc, rewriter.getI64Type(), key);
    WaveSegmentedReduction reduction = buildWavePrefixSegmentedReduction(
        rewriter, loc, keyI64, entry.getResult(1), active, waveSize);
    scf::IfOp::create(rewriter, loc, reduction.segmentEnd,
                      [&](OpBuilder &builder, Location bodyLoc) {
                        memref::AtomicRMWOp::create(
                            builder, bodyLoc, arith::AtomicRMWKind::addf,
                            reduction.inclusiveValue, op.getOutput(),
                            ValueRange{key});
                        scf::YieldOp::create(builder, bodyLoc);
                      });
    YieldOp::create(rewriter, loc);
    rewriter.eraseOp(op);
    return success();
  }

private:
  int64_t blockSize;
  int64_t waveSize;
};

class CooperativeWavePositionReducePattern
    : public OpRewritePattern<PositionReduceOp> {
public:
  CooperativeWavePositionReducePattern(MLIRContext *context, int64_t blockSize,
                                       int64_t waveSize,
                                       StringRef cooperativeAxis,
                                       int64_t chunkSize)
      : OpRewritePattern<PositionReduceOp>(context), blockSize(blockSize),
        waveSize(waveSize), cooperativeAxis(cooperativeAxis),
        chunkSize(chunkSize) {}

  LogicalResult matchAndRewrite(PositionReduceOp op,
                                PatternRewriter &rewriter) const override {
    std::optional<unsigned> axis = findAxis(op, cooperativeAxis);
    if (!axis)
      return rewriter.notifyMatchFailure(op, "cooperative axis was not found");

    Location loc = op.getLoc();
    Value zeroIndex = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value outputSize =
        memref::DimOp::create(rewriter, loc, op.getOutput(), zeroIndex);
    Type valueType =
        cast<MemRefType>(op.getOutput().getType()).getElementType();
    Value zero = arith::ConstantOp::create(rewriter, loc,
                                           rewriter.getZeroAttr(valueType));
    PositionParallelOp initialization = buildOutputInitialization(
        rewriter, loc, op.getOutput(), outputSize, zero, blockSize);

    rewriter.setInsertionPointAfter(initialization);
    Value waveSizeValue =
        arith::ConstantIndexOp::create(rewriter, loc, waveSize);
    Value axisExtent = arith::SubIOp::create(
        rewriter, loc, op.getUpper()[*axis], op.getLower()[*axis]);
    Value axisGroupCount =
        arith::CeilDivUIOp::create(rewriter, loc, axisExtent, waveSizeValue);
    Value iterationCount =
        buildNonCooperativeIterationCount(rewriter, loc, op, *axis);
    Value workerCount;
    Value chunkSizeValue;
    if (chunkSize == 1) {
      workerCount =
          arith::MulIOp::create(rewriter, loc, iterationCount, axisGroupCount);
    } else {
      chunkSizeValue = arith::ConstantIndexOp::create(rewriter, loc, chunkSize);
      Value chunkCount = arith::CeilDivUIOp::create(
          rewriter, loc, iterationCount, chunkSizeValue);
      workerCount =
          arith::MulIOp::create(rewriter, loc, chunkCount, axisGroupCount);
    }

    auto parallel =
        PositionParallelOp::create(rewriter, loc, workerCount, "wave",
                                   rewriter.getI64IntegerAttr(blockSize));
    Block *body =
        rewriter.createBlock(&parallel.getBody(), parallel.getBody().end(),
                             {rewriter.getIndexType(), rewriter.getIndexType(),
                              rewriter.getIndexType()},
                             {loc, loc, loc});
    rewriter.setInsertionPointToStart(body);
    Value worker = body->getArgument(0);
    Value lane = body->getArgument(1);
    Value participantCount = body->getArgument(2);
    Value axisGroup =
        arith::RemUIOp::create(rewriter, loc, worker, axisGroupCount);
    Value chunk = arith::DivUIOp::create(rewriter, loc, worker, axisGroupCount);
    if (chunkSize == 1) {
      SmallVector<Value> coordinates = buildCooperativeCoordinates(
          rewriter, loc, op, chunk, *axis, axisGroup, lane, participantCount);
      Value active =
          arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::ult,
                                coordinates[*axis], op.getUpper()[*axis]);
      CooperativeBodyMapping mapping = buildCooperativeBodyMapping(
          rewriter, loc, op, coordinates, *axis, lane, waveSize);
      scf::IfOp::create(
          rewriter, loc, active, [&](OpBuilder &builder, Location bodyLoc) {
            KeyedContribution contribution =
                cloneRemainingContributionBody(builder, op, mapping);
            memref::AtomicRMWOp::create(builder, bodyLoc,
                                        arith::AtomicRMWKind::addf,
                                        contribution.value, op.getOutput(),
                                        ValueRange{contribution.key});
            scf::YieldOp::create(builder, bodyLoc);
          });
      YieldOp::create(rewriter, loc);
      rewriter.eraseOp(op);
      return success();
    }

    Value oneIndex = arith::ConstantIndexOp::create(rewriter, loc, 1);
    Value begin = arith::MulIOp::create(rewriter, loc, chunk, chunkSizeValue);
    Value remaining =
        arith::SubIOp::create(rewriter, loc, iterationCount, begin);
    Value boundedSize =
        arith::MinUIOp::create(rewriter, loc, remaining, chunkSizeValue);
    Value end = arith::AddIOp::create(rewriter, loc, begin, boundedSize);
    SmallVector<Value> coordinates = buildCooperativeCoordinates(
        rewriter, loc, op, begin, *axis, axisGroup, lane, participantCount);
    Value active =
        arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::ult,
                              coordinates[*axis], op.getUpper()[*axis]);
    CooperativeBodyMapping mapping = buildCooperativeBodyMapping(
        rewriter, loc, op, coordinates, *axis, lane, waveSize);
    KeyedContribution first = buildActiveCooperativeContribution(
        rewriter, loc, op, mapping, active, zeroIndex, zero);

    Value nextIteration = arith::AddIOp::create(rewriter, loc, begin, oneIndex);
    auto reduction = scf::ForOp::create(
        rewriter, loc, nextIteration, end, oneIndex,
        ValueRange{first.key, first.value},
        [&](OpBuilder &builder, Location bodyLoc, Value linearIteration,
            ValueRange iterArgs) {
          SmallVector<Value> nextCoordinates = buildCooperativeCoordinates(
              builder, bodyLoc, op, linearIteration, *axis, axisGroup, lane,
              participantCount);
          CooperativeBodyMapping nextMapping = buildCooperativeBodyMapping(
              builder, bodyLoc, op, nextCoordinates, *axis, lane, waveSize);
          KeyedContribution next = buildActiveCooperativeContribution(
              builder, bodyLoc, op, nextMapping, active, zeroIndex, zero);
          Value sameKey =
              arith::CmpIOp::create(builder, bodyLoc, arith::CmpIPredicate::eq,
                                    iterArgs[0], next.key);
          Value keyChanged = arith::XOrIOp::create(
              builder, bodyLoc, sameKey,
              arith::ConstantIntOp::create(builder, bodyLoc, 1, 1));
          Value flush =
              arith::AndIOp::create(builder, bodyLoc, active, keyChanged);
          scf::IfOp::create(builder, bodyLoc, flush,
                            [&](OpBuilder &flushBuilder, Location flushLoc) {
                              memref::AtomicRMWOp::create(
                                  flushBuilder, flushLoc,
                                  arith::AtomicRMWKind::addf, iterArgs[1],
                                  op.getOutput(), ValueRange{iterArgs[0]});
                              scf::YieldOp::create(flushBuilder, flushLoc);
                            });
          Value combined =
              arith::AddFOp::create(builder, bodyLoc, iterArgs[1], next.value);
          Value nextKey = arith::SelectOp::create(builder, bodyLoc, sameKey,
                                                  iterArgs[0], next.key);
          Value nextValue = arith::SelectOp::create(builder, bodyLoc, sameKey,
                                                    combined, next.value);
          scf::YieldOp::create(builder, bodyLoc,
                               ValueRange{nextKey, nextValue});
        });
    scf::IfOp::create(
        rewriter, loc, active, [&](OpBuilder &builder, Location bodyLoc) {
          memref::AtomicRMWOp::create(builder, bodyLoc,
                                      arith::AtomicRMWKind::addf,
                                      reduction.getResult(1), op.getOutput(),
                                      ValueRange{reduction.getResult(0)});
          scf::YieldOp::create(builder, bodyLoc);
        });
    YieldOp::create(rewriter, loc);
    rewriter.eraseOp(op);
    return success();
  }

private:
  int64_t blockSize;
  int64_t waveSize;
  std::string cooperativeAxis;
  int64_t chunkSize;
};

class ScheduleSparseWavePosition
    : public impl::ScheduleSparseWavePositionBase<ScheduleSparseWavePosition> {
public:
  using impl::ScheduleSparseWavePositionBase<
      ScheduleSparseWavePosition>::ScheduleSparseWavePositionBase;

  void runOnOperation() override {
    if (mapping != "thread" && mapping != "wave") {
      getOperation().emitError() << "unsupported position mapping '" << mapping
                                 << "'; expected 'thread' or 'wave'";
      signalPassFailure();
      return;
    }
    if (blockSize < 1 || blockSize > 1024) {
      getOperation().emitError()
          << "position block size must be between 1 and 1024, but got "
          << blockSize.getValue();
      signalPassFailure();
      return;
    }
    if (mapping == "thread" && !cooperativeAxis.empty()) {
      getOperation().emitError()
          << "cooperative position axis applies only to wave mapping";
      signalPassFailure();
      return;
    }
    if (mapping == "wave" && cooperativeAxis.empty() && waveSize != 32) {
      getOperation().emitError()
          << "wave position mapping currently requires Wave32, but got "
          << waveSize.getValue();
      signalPassFailure();
      return;
    }
    if (mapping == "wave" && !cooperativeAxis.empty() && waveSize != 32 &&
        waveSize != 64) {
      getOperation().emitError()
          << "cooperative wave position mapping requires a wave size of 32 "
             "or 64, but got "
          << waveSize.getValue();
      signalPassFailure();
      return;
    }
    if (mapping == "wave" && blockSize % waveSize != 0) {
      getOperation().emitError()
          << "wave position mapping requires the block size to be a multiple "
             "of "
          << waveSize.getValue() << ", but got " << blockSize.getValue();
      signalPassFailure();
      return;
    }
    if (threadChunkSize < 1) {
      getOperation().emitError()
          << "thread position chunk size must be positive, but got "
          << threadChunkSize.getValue();
      signalPassFailure();
      return;
    }
    if (cooperativeChunkSize < 1) {
      getOperation().emitError()
          << "cooperative position chunk size must be positive, but got "
          << cooperativeChunkSize.getValue();
      signalPassFailure();
      return;
    }
    if (threadReduction != "atomic" && threadReduction != "segmented") {
      getOperation().emitError()
          << "unsupported thread position reduction '" << threadReduction
          << "'; expected 'atomic' or 'segmented'";
      signalPassFailure();
      return;
    }
    if (mapping == "wave" && threadChunkSize != 1) {
      getOperation().emitError()
          << "thread position chunk size applies only to thread mapping";
      signalPassFailure();
      return;
    }
    if ((mapping != "wave" || cooperativeAxis.empty()) &&
        cooperativeChunkSize != 1) {
      getOperation().emitError()
          << "cooperative position chunk size applies only to cooperative "
             "wave mapping";
      signalPassFailure();
      return;
    }
    if (mapping == "wave" && threadReduction != "atomic") {
      getOperation().emitError()
          << "thread position reduction applies only to thread mapping";
      signalPassFailure();
      return;
    }
    if (!cooperativeAxis.empty()) {
      WalkResult result = getOperation().walk([&](PositionReduceOp op) {
        if (findAxis(op, cooperativeAxis))
          return WalkResult::advance();
        op.emitError() << "does not define cooperative axis '"
                       << cooperativeAxis << "'";
        return WalkResult::interrupt();
      });
      if (result.wasInterrupted()) {
        signalPassFailure();
        return;
      }
    }

    RewritePatternSet patterns(&getContext());
    if (mapping == "thread") {
      if (threadReduction == "segmented")
        patterns.add<ThreadSegmentedPositionReducePattern>(
            &getContext(), blockSize, threadChunkSize);
      else
        patterns.add<ThreadPositionReducePattern>(&getContext(), blockSize,
                                                  threadChunkSize);
    } else if (cooperativeAxis.empty()) {
      patterns.add<WavePositionReducePattern>(&getContext(), blockSize,
                                              waveSize);
    } else {
      patterns.add<CooperativeWavePositionReducePattern>(
          &getContext(), blockSize, waveSize, cooperativeAxis,
          cooperativeChunkSize);
    }
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
} // namespace mlir::sparsewave
