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
#include "llvm/Support/MathExtras.h"

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

struct CompressedSegmentRecovery {
  CompressedSegmentAtPositionOp operation;
  unsigned positionAxis;
  bool mayReset;
};

/// Materializes the operator semantics independently from the selected
/// position schedule.
class PositionReductionSemantics {
public:
  static PositionReductionSemantics get(PositionReduceOp reduction) {
    assert(reduction.getKind() == "sum" &&
           "expected a verified position reduction kind");
    return PositionReductionSemantics();
  }

  Value buildIdentity(OpBuilder &builder, Location loc, Type valueType) const {
    return arith::ConstantOp::create(builder, loc,
                                     builder.getZeroAttr(valueType));
  }

  Value combine(OpBuilder &builder, Location loc, Value lhs, Value rhs) const {
    return arith::AddFOp::create(builder, loc, lhs, rhs);
  }

  void atomicAccumulate(OpBuilder &builder, Location loc, Value value,
                        Value output, Value key) const {
    memref::AtomicRMWOp::create(builder, loc, arith::AtomicRMWKind::addf, value,
                                output, ValueRange{key});
  }

private:
  PositionReductionSemantics() = default;
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

template <typename BodyBuilder>
SmallVector<Value>
buildCollapsedLogicalPoint(OpBuilder &builder, Location loc,
                           Value linearIteration, ValueRange lower,
                           ValueRange upper, DenseI64ArrayAttr order,
                           ValueRange fallbacks, BodyBuilder buildBody) {
  if (lower.size() == 1) {
    Value lowerBound = lower.front();
    Value coordinate =
        matchPattern(lowerBound, m_Zero())
            ? linearIteration
            : arith::AddIOp::create(builder, loc, lowerBound, linearIteration);
    return buildBody(builder, loc, ValueRange{coordinate});
  }

  SmallVector<Type> resultTypes;
  llvm::transform(fallbacks, std::back_inserter(resultTypes),
                  [](Value value) { return value.getType(); });
  auto collapse =
      PositionCollapseOp::create(builder, loc, resultTypes, linearIteration,
                                 lower, upper, fallbacks, order);
  SmallVector<Type> coordinateTypes(lower.size(), builder.getIndexType());
  SmallVector<Location> coordinateLocations(coordinateTypes.size(), loc);
  Block *body =
      builder.createBlock(&collapse.getBody(), collapse.getBody().end(),
                          coordinateTypes, coordinateLocations);
  builder.setInsertionPointToStart(body);
  SmallVector<Value> results = buildBody(builder, loc, body->getArguments());
  YieldOp::create(builder, loc, results);
  builder.setInsertionPointAfter(collapse);
  SmallVector<Value> collapsedResults;
  llvm::append_range(collapsedResults, collapse.getResults());
  return collapsedResults;
}

template <typename BodyBuilder>
SmallVector<Value>
buildCollapsedLogicalPoint(OpBuilder &builder, Location loc,
                           PositionReduceOp reduction, Value linearIteration,
                           ValueRange fallbacks, BodyBuilder buildBody) {
  return buildCollapsedLogicalPoint(
      builder, loc, linearIteration, reduction.getLower(), reduction.getUpper(),
      reduction.getOrderAttr(), fallbacks, buildBody);
}

std::optional<unsigned> findAxis(PositionReduceOp reduction,
                                 StringRef axisName) {
  for (auto [index, attribute] : llvm::enumerate(reduction.getAxes()))
    if (cast<StringAttr>(attribute).getValue() == axisName)
      return index;
  return std::nullopt;
}

Value buildCooperativeAxisCoordinate(OpBuilder &builder, Location loc,
                                     PositionReduceOp reduction,
                                     unsigned cooperativeAxis, Value axisGroup,
                                     Value lane, Value participantCount) {
  Value groupOffset =
      arith::MulIOp::create(builder, loc, axisGroup, participantCount);
  Value laneOffset = arith::AddIOp::create(builder, loc, groupOffset, lane);
  Value cooperativeLower = reduction.getLower()[cooperativeAxis];
  return matchPattern(cooperativeLower, m_Zero())
             ? laneOffset
             : arith::AddIOp::create(builder, loc, cooperativeLower,
                                     laneOffset);
}

template <typename BodyBuilder>
SmallVector<Value> buildCooperativeLogicalPoint(
    OpBuilder &builder, Location loc, PositionReduceOp reduction,
    Value linearIteration, unsigned cooperativeAxis,
    Value cooperativeCoordinate, ValueRange fallbacks, BodyBuilder buildBody) {
  Value zero = arith::ConstantIndexOp::create(builder, loc, 0);
  Value one = arith::ConstantIndexOp::create(builder, loc, 1);
  SmallVector<Value> lower(reduction.getLower());
  SmallVector<Value> upper(reduction.getUpper());
  lower[cooperativeAxis] = zero;
  upper[cooperativeAxis] = one;
  return buildCollapsedLogicalPoint(
      builder, loc, linearIteration, lower, upper, reduction.getOrderAttr(),
      fallbacks,
      [&](OpBuilder &nestedBuilder, Location nestedLoc,
          ValueRange collapsedCoordinates) {
        SmallVector<Value> coordinates(collapsedCoordinates);
        coordinates[cooperativeAxis] = cooperativeCoordinate;
        return buildBody(nestedBuilder, nestedLoc, coordinates);
      });
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

Value buildWaveLaneBroadcast(OpBuilder &builder, Location loc, Value value,
                             Value sourceLane, int64_t waveSize) {
  Value sourceLaneI32 = arith::IndexCastOp::create(
      builder, loc, builder.getI32Type(), sourceLane);
  Value waveSizeI32 = arith::ConstantIntOp::create(builder, loc, waveSize, 32);
  if (value.getType().isIndex()) {
    Value valueI64 =
        arith::IndexCastOp::create(builder, loc, builder.getI64Type(), value);
    Value broadcastI64 =
        gpu::ShuffleOp::create(builder, loc, valueI64, sourceLaneI32,
                               waveSizeI32, gpu::ShuffleMode::IDX)
            .getShuffleResult();
    return arith::IndexCastOp::create(builder, loc, builder.getIndexType(),
                                      broadcastI64);
  }
  return gpu::ShuffleOp::create(builder, loc, value, sourceLaneI32, waveSizeI32,
                                gpu::ShuffleMode::IDX)
      .getShuffleResult();
}

Value buildLaneZeroBroadcast(OpBuilder &builder, Location loc, Value value,
                             int64_t waveSize) {
  Value zero = arith::ConstantIndexOp::create(builder, loc, 0);
  return buildWaveLaneBroadcast(builder, loc, value, zero, waveSize);
}

struct WavePositionSegment {
  Value wave;
  Value lane;
  Value key;
  Value value;
  Value segmentOrdinal;
  Value lastActiveLane;
};

template <typename ContributionBuilder, typename SegmentBuilder>
PositionParallelOp buildWavePositionReductionStage(
    PatternRewriter &rewriter, Location loc, Value iterationCount,
    Value waveCount, Value zeroIndex, Value zero,
    PositionReductionSemantics semantics, int64_t blockSize, int64_t waveSize,
    bool recoverBoundaryMetadata, ContributionBuilder buildContribution,
    SegmentBuilder buildSegment) {
  auto parallel = PositionParallelOp::create(
      rewriter, loc, waveCount, "wave", rewriter.getI64IntegerAttr(blockSize));
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
  Value active = arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::ult,
                                       linearIteration, partition.getEnd());
  Value waveActive =
      arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::ult,
                            partition.getBegin(), partition.getEnd());
  KeyedContribution contribution =
      buildContribution(rewriter, loc, linearIteration, active, waveActive,
                        partition.getBegin(), lane);
  Value key = contribution.key;
  Value keyI64 =
      arith::IndexCastOp::create(rewriter, loc, rewriter.getI64Type(), key);
  WaveSegmentedReduction reduction = buildWavePrefixSegmentedReduction(
      rewriter, loc, keyI64, contribution.value, active, waveSize,
      [&](OpBuilder &builder, Location bodyLoc, Value lhs, Value rhs) {
        return semantics.combine(builder, bodyLoc, lhs, rhs);
      },
      recoverBoundaryMetadata);

  Value lastActiveLane = lane;
  if (recoverBoundaryMetadata) {
    Value activeLaneCount = arith::SubIOp::create(
        rewriter, loc, partition.getEnd(), partition.getBegin());
    Value one = arith::ConstantIndexOp::create(rewriter, loc, 1);
    Value safeLaneCount =
        arith::MaxUIOp::create(rewriter, loc, activeLaneCount, one);
    lastActiveLane = arith::SubIOp::create(rewriter, loc, safeLaneCount, one);
  }

  scf::IfOp::create(rewriter, loc, reduction.segmentEnd,
                    [&](OpBuilder &builder, Location bodyLoc) {
                      buildSegment(builder, bodyLoc,
                                   WavePositionSegment{wave, lane, key,
                                                       reduction.inclusiveValue,
                                                       reduction.segmentOrdinal,
                                                       lastActiveLane});
                      scf::YieldOp::create(builder, bodyLoc);
                    });
  YieldOp::create(rewriter, loc);
  rewriter.setInsertionPointAfter(parallel);
  return parallel;
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

std::optional<CompressedSegmentRecovery>
matchCompressedSegmentRecovery(PositionReduceOp reduction) {
  Block &body = reduction.getBody().front();
  for (Operation &operation : body.without_terminator()) {
    auto recovery = dyn_cast<CompressedSegmentAtPositionOp>(operation);
    if (!recovery)
      continue;
    auto position = dyn_cast<BlockArgument>(recovery.getPosition());
    if (!position || position.getOwner() != &body)
      continue;
    unsigned positionAxis = position.getArgNumber();
    // Only the outermost collapsed axis is globally nondecreasing. Other axes
    // restart whenever a preceding axis advances.
    bool mayReset =
        reduction.getOrder().front() != static_cast<int64_t>(positionAxis);
    return CompressedSegmentRecovery{recovery, positionAxis, mayReset};
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

CompressedSegmentState buildReusableCompressedSegmentState(
    OpBuilder &builder, Location loc, Value offsets, Value segment,
    Value nextBoundary, Value previousPosition, Value position) {
  Value resets = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::ult,
                                       position, previousPosition);
  // Reuse the current segment while the compressed position advances. If a
  // reordered multidimensional traversal restarts that axis, recover its
  // segment with a new search before resuming incremental boundary updates.
  auto state = scf::IfOp::create(
      builder, loc, TypeRange{builder.getIndexType(), builder.getIndexType()},
      resets,
      /*withElseRegion=*/true);
  builder.setInsertionPointToStart(&state.getThenRegion().front());
  Value resetSegment = CompressedSegmentAtPositionOp::create(
      builder, loc, builder.getIndexType(), offsets, position);
  Value resetBoundary =
      loadNextCompressedSegmentBoundary(builder, loc, offsets, resetSegment);
  scf::YieldOp::create(builder, loc, ValueRange{resetSegment, resetBoundary});
  builder.setInsertionPointToStart(&state.getElseRegion().front());
  CompressedSegmentState advanced = buildNextCompressedSegmentState(
      builder, loc, offsets, segment, nextBoundary, position);
  scf::YieldOp::create(builder, loc,
                       ValueRange{advanced.segment, advanced.nextBoundary});
  builder.setInsertionPointAfter(state);
  return {state.getResult(0), state.getResult(1)};
}

Value buildWaveIndexMax(OpBuilder &builder, Location loc, Value value,
                        int64_t waveSize) {
  Value waveSizeI32 = arith::ConstantIntOp::create(builder, loc, waveSize, 32);
  Value maximum =
      arith::IndexCastOp::create(builder, loc, builder.getI64Type(), value);
  for (int64_t offset = 1; offset < waveSize; offset *= 2) {
    Value offsetI32 = arith::ConstantIntOp::create(builder, loc, offset, 32);
    Value shuffled = gpu::ShuffleOp::create(builder, loc, maximum, offsetI32,
                                            waveSizeI32, gpu::ShuffleMode::XOR)
                         .getShuffleResult();
    maximum = arith::MaxUIOp::create(builder, loc, maximum, shuffled);
  }
  return arith::IndexCastOp::create(builder, loc, builder.getIndexType(),
                                    maximum);
}

Value buildPositiveCeilDivByPowerOfTwo(OpBuilder &builder, Location loc,
                                       Value value, int64_t divisor) {
  assert(llvm::isPowerOf2_64(divisor) && "expected a power-of-two divisor");
  Value one = arith::ConstantIndexOp::create(builder, loc, 1);
  Value shift = arith::ConstantIndexOp::create(
      builder, loc, llvm::Log2_64(static_cast<uint64_t>(divisor)));
  Value decremented = arith::SubIOp::create(builder, loc, value, one);
  Value quotient = arith::ShRUIOp::create(builder, loc, decremented, shift);
  return arith::AddIOp::create(builder, loc, quotient, one);
}

Value buildWaveCompressedSegmentSearch(OpBuilder &builder, Location loc,
                                       Value offsets, Value position,
                                       Value lane, int64_t waveSize) {
  Type indexType = builder.getIndexType();
  Value zero = arith::ConstantIndexOp::create(builder, loc, 0);
  Value one = arith::ConstantIndexOp::create(builder, loc, 1);
  Value waveWidth = arith::ConstantIndexOp::create(builder, loc, waveSize);
  Value boundaryCount = memref::DimOp::create(builder, loc, offsets, zero);

  // Each round samples one boundary per lane and retains the interval after
  // the last sample at or before the target position. The interval shrinks by
  // up to the wave width rather than by two as in a scalar binary search.
  auto search =
      scf::WhileOp::create(builder, loc, TypeRange{indexType, indexType},
                           ValueRange{zero, boundaryCount});
  SmallVector<Location> argumentLocations(2, loc);
  Block *before = builder.createBlock(
      &search.getBefore(), {}, search.getResultTypes(), argumentLocations);
  builder.setInsertionPointToStart(before);
  Value lower = before->getArgument(0);
  Value upper = before->getArgument(1);
  Value distance = arith::SubIOp::create(builder, loc, upper, lower);
  Value needsAnotherRound = arith::CmpIOp::create(
      builder, loc, arith::CmpIPredicate::ugt, distance, waveWidth);
  scf::ConditionOp::create(builder, loc, needsAnotherRound,
                           ValueRange{lower, upper});

  Block *after = builder.createBlock(
      &search.getAfter(), {}, search.getResultTypes(), argumentLocations);
  builder.setInsertionPointToStart(after);
  lower = after->getArgument(0);
  upper = after->getArgument(1);
  distance = arith::SubIOp::create(builder, loc, upper, lower);
  Value stride =
      buildPositiveCeilDivByPowerOfTwo(builder, loc, distance, waveSize);
  Value laneOffset = arith::MulIOp::create(builder, loc, lane, stride);
  Value candidate = arith::AddIOp::create(builder, loc, lower, laneOffset);
  Value lastBoundary = arith::SubIOp::create(builder, loc, upper, one);
  candidate = arith::MinUIOp::create(builder, loc, candidate, lastBoundary);
  Value boundaryValue =
      memref::LoadOp::create(builder, loc, offsets, candidate);
  Value boundary = castToIndex(builder, loc, boundaryValue);
  Value qualifies = arith::CmpIOp::create(
      builder, loc, arith::CmpIPredicate::ule, boundary, position);
  Value localLower =
      arith::SelectOp::create(builder, loc, qualifies, candidate, lower);
  Value nextLower = buildWaveIndexMax(builder, loc, localLower, waveSize);
  Value nextCandidate = arith::AddIOp::create(builder, loc, nextLower, stride);
  Value nextUpper = arith::MinUIOp::create(builder, loc, nextCandidate, upper);
  scf::YieldOp::create(builder, loc, ValueRange{nextLower, nextUpper});

  builder.setInsertionPointAfter(search);
  lower = search.getResult(0);
  upper = search.getResult(1);
  Value finalCandidate = arith::AddIOp::create(builder, loc, lower, lane);
  Value candidateIsValid = arith::CmpIOp::create(
      builder, loc, arith::CmpIPredicate::ult, finalCandidate, upper);
  Value finalBoundary = arith::SubIOp::create(builder, loc, upper, one);
  Value safeCandidate =
      arith::MinUIOp::create(builder, loc, finalCandidate, finalBoundary);
  Value finalBoundaryValue =
      memref::LoadOp::create(builder, loc, offsets, safeCandidate);
  Value finalBoundaryPosition = castToIndex(builder, loc, finalBoundaryValue);
  Value boundaryQualifies = arith::CmpIOp::create(
      builder, loc, arith::CmpIPredicate::ule, finalBoundaryPosition, position);
  Value finalQualifies =
      arith::AndIOp::create(builder, loc, candidateIsValid, boundaryQualifies);
  Value localSegment = arith::SelectOp::create(builder, loc, finalQualifies,
                                               finalCandidate, lower);
  return buildWaveIndexMax(builder, loc, localSegment, waveSize);
}

CompressedSegmentState
buildWaveCompressedSegmentState(OpBuilder &builder, Location loc, Value offsets,
                                Value position, Value lane, int64_t waveSize) {
  Value firstPosition =
      buildLaneZeroBroadcast(builder, loc, position, waveSize);
  Value segment = buildWaveCompressedSegmentSearch(
      builder, loc, offsets, firstPosition, lane, waveSize);
  Value nextBoundary =
      loadNextCompressedSegmentBoundary(builder, loc, offsets, segment);
  return buildNextCompressedSegmentState(builder, loc, offsets, segment,
                                         nextBoundary, position);
}

template <typename SegmentBuilder>
PositionParallelOp buildChunkedWavePositionReductionStage(
    PatternRewriter &rewriter, Location loc, PositionReduceOp reduction,
    std::optional<CompressedSegmentRecovery> segmentRecovery,
    Value iterationCount, Value waveCount, Value zeroIndex, Value zero,
    PositionReductionSemantics semantics, int64_t blockSize, int64_t waveSize,
    int64_t chunkSize, bool recoverBoundaryMetadata,
    SegmentBuilder buildSegment) {
  // Assign each physical wave a balanced contiguous partition containing at
  // most `chunkSize` lane-width tiles. The sequential tile loop carries one
  // compressed segment state per lane, so only the first tile performs the
  // cooperative search. Each tile remains an independent reduction partition
  // to preserve the ordering expected by the boundary workspace.
  bool canReuseSegment = segmentRecovery && !segmentRecovery->mayReset;
  auto parallel = PositionParallelOp::create(
      rewriter, loc, waveCount, "wave", rewriter.getI64IntegerAttr(blockSize));
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
  Value one = arith::ConstantIndexOp::create(rewriter, loc, 1);
  Value waveWidth = arith::ConstantIndexOp::create(rewriter, loc, waveSize);
  Value chunkSizeValue =
      arith::ConstantIndexOp::create(rewriter, loc, chunkSize);
  Value tileCapacity =
      arith::MulIOp::create(rewriter, loc, waveCount, chunkSizeValue);
  auto tiles = PositionForOp::create(
      rewriter, loc,
      TypeRange{rewriter.getIndexType(), rewriter.getIndexType()}, zeroIndex,
      tileCapacity, wave, ValueRange{zeroIndex, zeroIndex},
      rewriter.getI64IntegerAttr(chunkSize));
  Block *tileBody =
      rewriter.createBlock(&tiles.getBody(), tiles.getBody().end(),
                           {rewriter.getIndexType(), rewriter.getIndexType(),
                            rewriter.getIndexType(), rewriter.getIndexType()},
                           {loc, loc, loc, loc});
  rewriter.setInsertionPointToStart(tileBody);
  Value logicalTile = tileBody->getArgument(0);
  Value tile = tileBody->getArgument(1);
  Value currentSegment = tileBody->getArgument(2);
  Value currentBoundary = tileBody->getArgument(3);
  Value tileOffset = arith::MulIOp::create(rewriter, loc, tile, waveWidth);
  Value tileBegin =
      arith::AddIOp::create(rewriter, loc, partition.getBegin(), tileOffset);
  Value unboundedTileEnd =
      arith::AddIOp::create(rewriter, loc, tileBegin, waveWidth);
  Value tileEnd = arith::MinUIOp::create(rewriter, loc, unboundedTileEnd,
                                         partition.getEnd());
  Value waveActive = arith::CmpIOp::create(
      rewriter, loc, arith::CmpIPredicate::ult, tileBegin, tileEnd);
  Value linearIteration = arith::AddIOp::create(rewriter, loc, tileBegin, lane);
  Value active = arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::ult,
                                       linearIteration, tileEnd);
  auto waveEntry = scf::IfOp::create(
      rewriter, loc,
      TypeRange{rewriter.getIndexType(), zero.getType(),
                rewriter.getIndexType(), rewriter.getIndexType()},
      waveActive,
      /*withElseRegion=*/true);
  rewriter.setInsertionPointToStart(&waveEntry.getThenRegion().front());
  Value safeIteration = arith::SelectOp::create(rewriter, loc, active,
                                                linearIteration, tileBegin);
  SmallVector<Value> results = buildCollapsedLogicalPoint(
      rewriter, loc, reduction, safeIteration,
      ValueRange{zeroIndex, zero, currentSegment, currentBoundary},
      [&](OpBuilder &builder, Location bodyLoc, ValueRange coordinates) {
        Value nextSegment = currentSegment;
        Value nextBoundary = currentBoundary;
        if (canReuseSegment) {
          Value firstTile = arith::CmpIOp::create(
              builder, bodyLoc, arith::CmpIPredicate::eq, tile, zeroIndex);
          auto state = scf::IfOp::create(
              builder, bodyLoc,
              TypeRange{builder.getIndexType(), builder.getIndexType()},
              firstTile,
              /*withElseRegion=*/true);
          builder.setInsertionPointToStart(&state.getThenRegion().front());
          CompressedSegmentState first = buildWaveCompressedSegmentState(
              builder, bodyLoc, segmentRecovery->operation.getOffsets(),
              coordinates[segmentRecovery->positionAxis], lane, waveSize);
          scf::YieldOp::create(builder, bodyLoc,
                               ValueRange{first.segment, first.nextBoundary});
          builder.setInsertionPointToStart(&state.getElseRegion().front());
          CompressedSegmentState advanced = buildNextCompressedSegmentState(
              builder, bodyLoc, segmentRecovery->operation.getOffsets(),
              currentSegment, currentBoundary,
              coordinates[segmentRecovery->positionAxis]);
          scf::YieldOp::create(
              builder, bodyLoc,
              ValueRange{advanced.segment, advanced.nextBoundary});
          builder.setInsertionPointAfter(state);
          nextSegment = state.getResult(0);
          nextBoundary = state.getResult(1);
        }

        auto activeEntry = scf::IfOp::create(
            builder, bodyLoc, TypeRange{builder.getIndexType(), zero.getType()},
            active,
            /*withElseRegion=*/true);
        builder.setInsertionPointToStart(&activeEntry.getThenRegion().front());
        KeyedContribution contribution;
        if (canReuseSegment) {
          contribution = cloneContributionBodyWithCompressedSegment(
              builder, reduction, coordinates, segmentRecovery->operation,
              nextSegment);
        } else {
          contribution = cloneContributionBody(builder, reduction, coordinates);
        }
        scf::YieldOp::create(builder, bodyLoc,
                             ValueRange{contribution.key, contribution.value});
        builder.setInsertionPointToStart(&activeEntry.getElseRegion().front());
        scf::YieldOp::create(builder, bodyLoc, ValueRange{zeroIndex, zero});
        builder.setInsertionPointAfter(activeEntry);
        return SmallVector<Value>{activeEntry.getResult(0),
                                  activeEntry.getResult(1), nextSegment,
                                  nextBoundary};
      });
  scf::YieldOp::create(rewriter, loc, results);
  rewriter.setInsertionPointToStart(&waveEntry.getElseRegion().front());
  scf::YieldOp::create(
      rewriter, loc,
      ValueRange{zeroIndex, zero, currentSegment, currentBoundary});
  rewriter.setInsertionPointAfter(waveEntry);

  Value key = waveEntry.getResult(0);
  Value keyI64 =
      arith::IndexCastOp::create(rewriter, loc, rewriter.getI64Type(), key);
  WaveSegmentedReduction waveReduction = buildWavePrefixSegmentedReduction(
      rewriter, loc, keyI64, waveEntry.getResult(1), active, waveSize,
      [&](OpBuilder &builder, Location bodyLoc, Value lhs, Value rhs) {
        return semantics.combine(builder, bodyLoc, lhs, rhs);
      },
      recoverBoundaryMetadata);

  Value lastActiveLane = lane;
  if (recoverBoundaryMetadata) {
    Value activeLaneCount =
        arith::SubIOp::create(rewriter, loc, tileEnd, tileBegin);
    Value safeLaneCount =
        arith::MaxUIOp::create(rewriter, loc, activeLaneCount, one);
    lastActiveLane = arith::SubIOp::create(rewriter, loc, safeLaneCount, one);
  }
  scf::IfOp::create(
      rewriter, loc, waveReduction.segmentEnd,
      [&](OpBuilder &builder, Location bodyLoc) {
        buildSegment(builder, bodyLoc,
                     WavePositionSegment{
                         logicalTile, lane, key, waveReduction.inclusiveValue,
                         waveReduction.segmentOrdinal, lastActiveLane});
        scf::YieldOp::create(builder, bodyLoc);
      });
  YieldOp::create(rewriter, loc,
                  ValueRange{waveEntry.getResult(2), waveEntry.getResult(3)});
  rewriter.setInsertionPointAfter(tiles);
  YieldOp::create(rewriter, loc);
  rewriter.setInsertionPointAfter(parallel);
  return parallel;
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
    PositionReductionSemantics semantics = PositionReductionSemantics::get(op);
    Value zeroIndex = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value chunkSizeValue =
        arith::ConstantIndexOp::create(rewriter, loc, chunkSize);
    Value outputSize =
        memref::DimOp::create(rewriter, loc, op.getOutput(), zeroIndex);
    Type valueType =
        cast<MemRefType>(op.getOutput().getType()).getElementType();
    Value zero = semantics.buildIdentity(rewriter, loc, valueType);
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
    auto chunk = PositionForOp::create(rewriter, loc, TypeRange{}, zeroIndex,
                                       iterationCount, worker, ValueRange{},
                                       rewriter.getI64IntegerAttr(chunkSize));
    Block *chunkBody = rewriter.createBlock(
        &chunk.getBody(), chunk.getBody().end(),
        {rewriter.getIndexType(), rewriter.getIndexType()}, {loc, loc});
    rewriter.setInsertionPointToStart(chunkBody);
    buildCollapsedLogicalPoint(
        rewriter, loc, op, chunkBody->getArgument(0), ValueRange{},
        [&](OpBuilder &builder, Location bodyLoc, ValueRange coordinates) {
          KeyedContribution contribution =
              cloneContributionBody(builder, op, coordinates);
          semantics.atomicAccumulate(builder, bodyLoc, contribution.value,
                                     op.getOutput(), contribution.key);
          return SmallVector<Value>{};
        });
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
    PositionReductionSemantics semantics = PositionReductionSemantics::get(op);
    Value zeroIndex = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value oneIndex = arith::ConstantIndexOp::create(rewriter, loc, 1);
    Value chunkSizeValue =
        arith::ConstantIndexOp::create(rewriter, loc, chunkSize);
    Value outputSize =
        memref::DimOp::create(rewriter, loc, op.getOutput(), zeroIndex);
    Type valueType =
        cast<MemRefType>(op.getOutput().getType()).getElementType();
    Value zero = semantics.buildIdentity(rewriter, loc, valueType);
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

    std::optional<CompressedSegmentRecovery> segmentRecovery =
        matchCompressedSegmentRecovery(op);
    SmallVector<Value> firstFallbacks{zeroIndex, zero};
    if (segmentRecovery) {
      firstFallbacks.push_back(zeroIndex);
      if (segmentRecovery->mayReset)
        firstFallbacks.push_back(zeroIndex);
    }
    SmallVector<Value> firstResults = buildCollapsedLogicalPoint(
        rewriter, loc, op, begin, firstFallbacks,
        [&](OpBuilder &builder, Location bodyLoc, ValueRange coordinates) {
          KeyedContribution first;
          if (segmentRecovery) {
            Value firstSegment = CompressedSegmentAtPositionOp::create(
                builder, bodyLoc, builder.getIndexType(),
                segmentRecovery->operation.getOffsets(),
                coordinates[segmentRecovery->positionAxis]);
            first = cloneContributionBodyWithCompressedSegment(
                builder, op, coordinates, segmentRecovery->operation,
                firstSegment);
            SmallVector<Value> results{first.key, first.value, firstSegment};
            if (segmentRecovery->mayReset)
              results.push_back(coordinates[segmentRecovery->positionAxis]);
            return results;
          } else {
            first = cloneContributionBody(builder, op, coordinates);
          }
          return SmallVector<Value>{first.key, first.value};
        });
    KeyedContribution first{firstResults[0], firstResults[1]};

    Value nextPosition = arith::AddIOp::create(rewriter, loc, begin, oneIndex);
    SmallVector<Value> reductionInitializers{first.key, first.value};
    if (segmentRecovery) {
      Value firstSegment = firstResults[2];
      Value firstBoundary = loadNextCompressedSegmentBoundary(
          rewriter, loc, segmentRecovery->operation.getOffsets(), firstSegment);
      reductionInitializers.push_back(firstSegment);
      reductionInitializers.push_back(firstBoundary);
      if (segmentRecovery->mayReset)
        reductionInitializers.push_back(firstResults[3]);
    }
    SmallVector<Type> stateTypes;
    llvm::transform(reductionInitializers, std::back_inserter(stateTypes),
                    [](Value value) { return value.getType(); });
    auto reduction = PositionForOp::create(
        rewriter, loc, stateTypes, nextPosition, end, zeroIndex,
        reductionInitializers, rewriter.getI64IntegerAttr(chunkSize));
    SmallVector<Type> bodyTypes{rewriter.getIndexType(),
                                rewriter.getIndexType()};
    llvm::append_range(bodyTypes, stateTypes);
    SmallVector<Location> bodyLocations(bodyTypes.size(), loc);
    Block *reductionBody =
        rewriter.createBlock(&reduction.getBody(), reduction.getBody().end(),
                             bodyTypes, bodyLocations);
    rewriter.setInsertionPointToStart(reductionBody);
    Value linearIteration = reductionBody->getArgument(0);
    ValueRange iterArgs = reductionBody->getArguments().drop_front(2);
    Value currentKey = iterArgs[0];
    Value currentValue = iterArgs[1];
    SmallVector<Value> fallbacks{zeroIndex, zero};
    if (segmentRecovery) {
      fallbacks.push_back(iterArgs[2]);
      fallbacks.push_back(iterArgs[3]);
      if (segmentRecovery->mayReset)
        fallbacks.push_back(iterArgs[4]);
    }
    SmallVector<Value> contributionResults = buildCollapsedLogicalPoint(
        rewriter, loc, op, linearIteration, fallbacks,
        [&](OpBuilder &builder, Location bodyLoc, ValueRange coordinates) {
          KeyedContribution contribution;
          SmallVector<Value> results;
          if (segmentRecovery) {
            Value position = coordinates[segmentRecovery->positionAxis];
            CompressedSegmentState segmentState;
            if (segmentRecovery->mayReset) {
              segmentState = buildReusableCompressedSegmentState(
                  builder, bodyLoc, segmentRecovery->operation.getOffsets(),
                  iterArgs[2], iterArgs[3], iterArgs[4], position);
            } else {
              segmentState = buildNextCompressedSegmentState(
                  builder, bodyLoc, segmentRecovery->operation.getOffsets(),
                  iterArgs[2], iterArgs[3], position);
            }
            contribution = cloneContributionBodyWithCompressedSegment(
                builder, op, coordinates, segmentRecovery->operation,
                segmentState.segment);
            results = {contribution.key, contribution.value,
                       segmentState.segment, segmentState.nextBoundary};
            if (segmentRecovery->mayReset)
              results.push_back(position);
          } else {
            contribution = cloneContributionBody(builder, op, coordinates);
            results = {contribution.key, contribution.value};
          }
          return results;
        });
    KeyedContribution contribution{contributionResults[0],
                                   contributionResults[1]};
    Value sameKey = arith::CmpIOp::create(
        rewriter, loc, arith::CmpIPredicate::eq, currentKey, contribution.key);
    Value combined =
        semantics.combine(rewriter, loc, currentValue, contribution.value);
    scf::IfOp::create(
        rewriter, loc, sameKey,
        [&](OpBuilder &thenBuilder, Location thenLoc) {
          scf::YieldOp::create(thenBuilder, thenLoc);
        },
        [&](OpBuilder &elseBuilder, Location elseLoc) {
          semantics.atomicAccumulate(elseBuilder, elseLoc, currentValue,
                                     op.getOutput(), currentKey);
          scf::YieldOp::create(elseBuilder, elseLoc);
        });
    Value nextValue = arith::SelectOp::create(rewriter, loc, sameKey, combined,
                                              contribution.value);
    SmallVector<Value> nextState{contribution.key, nextValue};
    if (segmentRecovery) {
      nextState.push_back(contributionResults[2]);
      nextState.push_back(contributionResults[3]);
      if (segmentRecovery->mayReset)
        nextState.push_back(contributionResults[4]);
    }
    YieldOp::create(rewriter, loc, nextState);
    rewriter.setInsertionPointAfter(reduction);
    semantics.atomicAccumulate(rewriter, loc, reduction.getResult(1),
                               op.getOutput(), reduction.getResult(0));
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
                            int64_t waveSize, int64_t chunkSize,
                            bool useWorkspace)
      : OpRewritePattern<PositionReduceOp>(context), blockSize(blockSize),
        waveSize(waveSize), chunkSize(chunkSize), useWorkspace(useWorkspace) {}

  LogicalResult matchAndRewrite(PositionReduceOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    PositionReductionSemantics semantics = PositionReductionSemantics::get(op);
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
    Value zero = semantics.buildIdentity(rewriter, loc, valueType);
    PositionParallelOp initialization = buildOutputInitialization(
        rewriter, loc, op.getOutput(), outputSize, zero, blockSize);

    rewriter.setInsertionPointAfter(initialization);
    Value iterationCount = buildCollapsedIterationCount(rewriter, loc, op);
    auto buildWaveCount = [&](Value count) {
      Value requiredBlocks =
          arith::CeilDivUIOp::create(rewriter, loc, count, blockSizeValue);
      Value gridSize =
          arith::MaxUIOp::create(rewriter, loc, requiredBlocks, oneIndex);
      return arith::MulIOp::create(rewriter, loc, gridSize, wavesPerBlockValue);
    };
    Value scheduledCount = iterationCount;
    if (chunkSize > 1) {
      Value chunkSizeValue =
          arith::ConstantIndexOp::create(rewriter, loc, chunkSize);
      scheduledCount = arith::CeilDivUIOp::create(rewriter, loc, iterationCount,
                                                  chunkSizeValue);
    }
    Value waveCount = buildWaveCount(scheduledCount);
    std::optional<CompressedSegmentRecovery> segmentRecovery =
        matchCompressedSegmentRecovery(op);
    bool canReuseSegmentAcrossWave =
        segmentRecovery && !segmentRecovery->mayReset;
    auto buildContribution = [&](OpBuilder &builder, Location bodyLoc,
                                 Value linearIteration, Value active,
                                 Value waveActive, Value waveBegin,
                                 Value lane) {
      auto waveEntry = scf::IfOp::create(
          builder, bodyLoc, TypeRange{builder.getIndexType(), zero.getType()},
          waveActive,
          /*withElseRegion=*/true);
      builder.setInsertionPointToStart(&waveEntry.getThenRegion().front());
      // Inactive lanes still participate in wave shuffles. Give them the
      // first valid logical point so coordinate recovery remains in range.
      Value safeIteration = arith::SelectOp::create(builder, bodyLoc, active,
                                                    linearIteration, waveBegin);
      SmallVector<Value> results = buildCollapsedLogicalPoint(
          builder, bodyLoc, op, safeIteration, ValueRange{zeroIndex, zero},
          [&](OpBuilder &nestedBuilder, Location nestedLoc,
              ValueRange coordinates) {
            Value recoveredSegment;
            if (canReuseSegmentAcrossWave) {
              Value position = coordinates[segmentRecovery->positionAxis];
              recoveredSegment = buildWaveCompressedSegmentState(
                                     nestedBuilder, nestedLoc,
                                     segmentRecovery->operation.getOffsets(),
                                     position, lane, waveSize)
                                     .segment;
            }
            auto activeEntry = scf::IfOp::create(
                nestedBuilder, nestedLoc,
                TypeRange{nestedBuilder.getIndexType(), zero.getType()}, active,
                /*withElseRegion=*/true);
            nestedBuilder.setInsertionPointToStart(
                &activeEntry.getThenRegion().front());
            KeyedContribution contribution;
            if (canReuseSegmentAcrossWave) {
              contribution = cloneContributionBodyWithCompressedSegment(
                  nestedBuilder, op, coordinates, segmentRecovery->operation,
                  recoveredSegment);
            } else {
              contribution =
                  cloneContributionBody(nestedBuilder, op, coordinates);
            }
            scf::YieldOp::create(
                nestedBuilder, nestedLoc,
                ValueRange{contribution.key, contribution.value});
            nestedBuilder.setInsertionPointToStart(
                &activeEntry.getElseRegion().front());
            scf::YieldOp::create(nestedBuilder, nestedLoc,
                                 ValueRange{zeroIndex, zero});
            nestedBuilder.setInsertionPointAfter(activeEntry);
            return SmallVector<Value>{activeEntry.getResult(0),
                                      activeEntry.getResult(1)};
          });
      scf::YieldOp::create(builder, bodyLoc, results);
      builder.setInsertionPointToStart(&waveEntry.getElseRegion().front());
      scf::YieldOp::create(builder, bodyLoc, ValueRange{zeroIndex, zero});
      builder.setInsertionPointAfter(waveEntry);
      return KeyedContribution{waveEntry.getResult(0), waveEntry.getResult(1)};
    };
    auto buildPrimaryStage = [&](bool recoverBoundaryMetadata,
                                 auto buildSegment) {
      if (chunkSize == 1) {
        return buildWavePositionReductionStage(
            rewriter, loc, iterationCount, waveCount, zeroIndex, zero,
            semantics, blockSize, waveSize, recoverBoundaryMetadata,
            buildContribution, buildSegment);
      }
      return buildChunkedWavePositionReductionStage(
          rewriter, loc, op, segmentRecovery, iterationCount, waveCount,
          zeroIndex, zero, semantics, blockSize, waveSize, chunkSize,
          recoverBoundaryMetadata, buildSegment);
    };

    if (!useWorkspace) {
      buildPrimaryStage(
          /*recoverBoundaryMetadata=*/false,
          [&](OpBuilder &builder, Location bodyLoc,
              WavePositionSegment segment) {
            semantics.atomicAccumulate(builder, bodyLoc, segment.value,
                                       op.getOutput(), segment.key);
          });
      rewriter.eraseOp(op);
      return success();
    }

    // Every wave reserves two ordered slots for its first and last segment.
    // Interior segments are complete within that wave and can be accumulated
    // immediately. A second wave stage merges the boundary partials that may
    // continue across neighboring wave partitions.
    Value two = arith::ConstantIndexOp::create(rewriter, loc, 2);
    Value workspacePartitionCount = waveCount;
    if (chunkSize > 1) {
      Value chunkSizeValue =
          arith::ConstantIndexOp::create(rewriter, loc, chunkSize);
      workspacePartitionCount =
          arith::MulIOp::create(rewriter, loc, waveCount, chunkSizeValue);
    }
    Value workspaceSize =
        arith::MulIOp::create(rewriter, loc, workspacePartitionCount, two);
    auto keyWorkspaceType =
        MemRefType::get({ShapedType::kDynamic}, rewriter.getIndexType());
    auto valueWorkspaceType =
        MemRefType::get({ShapedType::kDynamic}, valueType);
    Value keyWorkspace =
        gpu::AllocOp::create(rewriter, loc, keyWorkspaceType, Type(),
                             ValueRange(), ValueRange{workspaceSize},
                             ValueRange())
            .getMemref();
    Value valueWorkspace =
        gpu::AllocOp::create(rewriter, loc, valueWorkspaceType, Type(),
                             ValueRange(), ValueRange{workspaceSize},
                             ValueRange())
            .getMemref();

    auto workspaceInitialization =
        PositionParallelOp::create(rewriter, loc, workspaceSize, "thread",
                                   rewriter.getI64IntegerAttr(blockSize));
    Block *initializationBody =
        rewriter.createBlock(&workspaceInitialization.getBody(),
                             workspaceInitialization.getBody().end(),
                             {rewriter.getIndexType(), rewriter.getIndexType(),
                              rewriter.getIndexType()},
                             {loc, loc, loc});
    rewriter.setInsertionPointToStart(initializationBody);
    Value workspaceIndex = initializationBody->getArgument(0);
    memref::StoreOp::create(rewriter, loc, zeroIndex, keyWorkspace,
                            workspaceIndex);
    memref::StoreOp::create(rewriter, loc, zero, valueWorkspace,
                            workspaceIndex);
    YieldOp::create(rewriter, loc);
    rewriter.setInsertionPointAfter(workspaceInitialization);

    buildPrimaryStage(
        /*recoverBoundaryMetadata=*/true,
        [&](OpBuilder &builder, Location bodyLoc, WavePositionSegment segment) {
          Value oneI32 = arith::ConstantIntOp::create(builder, bodyLoc, 1, 32);
          Value isFirst =
              arith::CmpIOp::create(builder, bodyLoc, arith::CmpIPredicate::eq,
                                    segment.segmentOrdinal, oneI32);
          // The last active lane always terminates the wave's final segment,
          // so no cross-lane segment-count broadcast is needed here.
          Value isLast =
              arith::CmpIOp::create(builder, bodyLoc, arith::CmpIPredicate::eq,
                                    segment.lane, segment.lastActiveLane);
          Value isBoundary =
              arith::OrIOp::create(builder, bodyLoc, isFirst, isLast);
          scf::IfOp::create(
              builder, bodyLoc, isBoundary,
              [&](OpBuilder &boundaryBuilder, Location boundaryLoc) {
                Value base = arith::MulIOp::create(boundaryBuilder, boundaryLoc,
                                                   segment.wave, two);
                Value lastSlot = arith::AddIOp::create(
                    boundaryBuilder, boundaryLoc, base, oneIndex);
                Value slot = arith::SelectOp::create(
                    boundaryBuilder, boundaryLoc, isFirst, base, lastSlot);
                memref::StoreOp::create(boundaryBuilder, boundaryLoc,
                                        segment.key, keyWorkspace, slot);
                memref::StoreOp::create(boundaryBuilder, boundaryLoc,
                                        segment.value, valueWorkspace, slot);
                Value isOnlySegment = arith::AndIOp::create(
                    boundaryBuilder, boundaryLoc, isFirst, isLast);
                scf::IfOp::create(
                    boundaryBuilder, boundaryLoc, isOnlySegment,
                    [&](OpBuilder &singleBuilder, Location singleLoc) {
                      memref::StoreOp::create(singleBuilder, singleLoc,
                                              segment.key, keyWorkspace,
                                              lastSlot);
                      memref::StoreOp::create(singleBuilder, singleLoc, zero,
                                              valueWorkspace, lastSlot);
                      scf::YieldOp::create(singleBuilder, singleLoc);
                    });
                scf::YieldOp::create(boundaryBuilder, boundaryLoc);
              },
              [&](OpBuilder &interiorBuilder, Location interiorLoc) {
                semantics.atomicAccumulate(interiorBuilder, interiorLoc,
                                           segment.value, op.getOutput(),
                                           segment.key);
                scf::YieldOp::create(interiorBuilder, interiorLoc);
              });
        });

    Value workspaceWaveCount = buildWaveCount(workspaceSize);
    PositionParallelOp merge = buildWavePositionReductionStage(
        rewriter, loc, workspaceSize, workspaceWaveCount, zeroIndex, zero,
        semantics, blockSize, waveSize, /*recoverBoundaryMetadata=*/false,
        [&](OpBuilder &builder, Location bodyLoc, Value linearIteration,
            Value active, Value, Value, Value) {
          auto entry = scf::IfOp::create(
              builder, bodyLoc,
              TypeRange{builder.getIndexType(), zero.getType()}, active,
              /*withElseRegion=*/true);
          builder.setInsertionPointToStart(&entry.getThenRegion().front());
          Value key = memref::LoadOp::create(builder, bodyLoc, keyWorkspace,
                                             linearIteration);
          Value value = memref::LoadOp::create(builder, bodyLoc, valueWorkspace,
                                               linearIteration);
          scf::YieldOp::create(builder, bodyLoc, ValueRange{key, value});
          builder.setInsertionPointToStart(&entry.getElseRegion().front());
          scf::YieldOp::create(builder, bodyLoc, ValueRange{zeroIndex, zero});
          builder.setInsertionPointAfter(entry);
          return KeyedContribution{entry.getResult(0), entry.getResult(1)};
        },
        [&](OpBuilder &builder, Location bodyLoc, WavePositionSegment segment) {
          semantics.atomicAccumulate(builder, bodyLoc, segment.value,
                                     op.getOutput(), segment.key);
        });
    rewriter.setInsertionPointAfter(merge);
    gpu::DeallocOp::create(rewriter, loc, Type(), ValueRange(), keyWorkspace);
    gpu::DeallocOp::create(rewriter, loc, Type(), ValueRange(), valueWorkspace);
    rewriter.eraseOp(op);
    return success();
  }

private:
  int64_t blockSize;
  int64_t waveSize;
  int64_t chunkSize;
  bool useWorkspace;
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
    PositionReductionSemantics semantics = PositionReductionSemantics::get(op);
    Value zeroIndex = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value outputSize =
        memref::DimOp::create(rewriter, loc, op.getOutput(), zeroIndex);
    Type valueType =
        cast<MemRefType>(op.getOutput().getType()).getElementType();
    Value zero = semantics.buildIdentity(rewriter, loc, valueType);
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
      Value cooperativeCoordinate = buildCooperativeAxisCoordinate(
          rewriter, loc, op, *axis, axisGroup, lane, participantCount);
      buildCooperativeLogicalPoint(
          rewriter, loc, op, chunk, *axis, cooperativeCoordinate, ValueRange{},
          [&](OpBuilder &builder, Location bodyLoc, ValueRange coordinates) {
            Value active = arith::CmpIOp::create(
                builder, bodyLoc, arith::CmpIPredicate::ult, coordinates[*axis],
                op.getUpper()[*axis]);
            CooperativeBodyMapping mapping = buildCooperativeBodyMapping(
                builder, bodyLoc, op, coordinates, *axis, lane, waveSize);
            scf::IfOp::create(
                builder, bodyLoc, active,
                [&](OpBuilder &nestedBuilder, Location nestedLoc) {
                  KeyedContribution contribution =
                      cloneRemainingContributionBody(nestedBuilder, op,
                                                     mapping);
                  semantics.atomicAccumulate(nestedBuilder, nestedLoc,
                                             contribution.value, op.getOutput(),
                                             contribution.key);
                  scf::YieldOp::create(nestedBuilder, nestedLoc);
                });
            return SmallVector<Value>{};
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
    Value cooperativeCoordinate = buildCooperativeAxisCoordinate(
        rewriter, loc, op, *axis, axisGroup, lane, participantCount);
    Value active =
        arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::ult,
                              cooperativeCoordinate, op.getUpper()[*axis]);
    SmallVector<Value> firstResults = buildCooperativeLogicalPoint(
        rewriter, loc, op, begin, *axis, cooperativeCoordinate,
        ValueRange{zeroIndex, zero},
        [&](OpBuilder &builder, Location bodyLoc, ValueRange coordinates) {
          CooperativeBodyMapping mapping = buildCooperativeBodyMapping(
              builder, bodyLoc, op, coordinates, *axis, lane, waveSize);
          KeyedContribution first = buildActiveCooperativeContribution(
              builder, bodyLoc, op, mapping, active, zeroIndex, zero);
          return SmallVector<Value>{first.key, first.value};
        });
    KeyedContribution first{firstResults[0], firstResults[1]};

    Value nextIteration = arith::AddIOp::create(rewriter, loc, begin, oneIndex);
    SmallVector<Type> stateTypes{first.key.getType(), first.value.getType()};
    auto reduction =
        PositionForOp::create(rewriter, loc, stateTypes, nextIteration, end,
                              zeroIndex, ValueRange{first.key, first.value},
                              rewriter.getI64IntegerAttr(chunkSize));
    Block *reductionBody =
        rewriter.createBlock(&reduction.getBody(), reduction.getBody().end(),
                             {rewriter.getIndexType(), rewriter.getIndexType(),
                              stateTypes[0], stateTypes[1]},
                             {loc, loc, loc, loc});
    rewriter.setInsertionPointToStart(reductionBody);
    Value linearIteration = reductionBody->getArgument(0);
    ValueRange iterArgs = reductionBody->getArguments().drop_front(2);
    SmallVector<Value> nextResults = buildCooperativeLogicalPoint(
        rewriter, loc, op, linearIteration, *axis, cooperativeCoordinate,
        ValueRange{zeroIndex, zero},
        [&](OpBuilder &builder, Location bodyLoc, ValueRange coordinates) {
          CooperativeBodyMapping nextMapping = buildCooperativeBodyMapping(
              builder, bodyLoc, op, coordinates, *axis, lane, waveSize);
          KeyedContribution next = buildActiveCooperativeContribution(
              builder, bodyLoc, op, nextMapping, active, zeroIndex, zero);
          return SmallVector<Value>{next.key, next.value};
        });
    KeyedContribution next{nextResults[0], nextResults[1]};
    Value sameKey = arith::CmpIOp::create(
        rewriter, loc, arith::CmpIPredicate::eq, iterArgs[0], next.key);
    Value keyChanged = arith::XOrIOp::create(
        rewriter, loc, sameKey,
        arith::ConstantIntOp::create(rewriter, loc, 1, 1));
    Value flush = arith::AndIOp::create(rewriter, loc, active, keyChanged);
    scf::IfOp::create(
        rewriter, loc, flush, [&](OpBuilder &builder, Location bodyLoc) {
          semantics.atomicAccumulate(builder, bodyLoc, iterArgs[1],
                                     op.getOutput(), iterArgs[0]);
          scf::YieldOp::create(builder, bodyLoc);
        });
    Value combined = semantics.combine(rewriter, loc, iterArgs[1], next.value);
    Value nextKey =
        arith::SelectOp::create(rewriter, loc, sameKey, iterArgs[0], next.key);
    Value nextValue =
        arith::SelectOp::create(rewriter, loc, sameKey, combined, next.value);
    YieldOp::create(rewriter, loc, ValueRange{nextKey, nextValue});
    rewriter.setInsertionPointAfter(reduction);
    scf::IfOp::create(
        rewriter, loc, active, [&](OpBuilder &builder, Location bodyLoc) {
          semantics.atomicAccumulate(builder, bodyLoc, reduction.getResult(1),
                                     op.getOutput(), reduction.getResult(0));
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
    if (waveChunkSize < 1) {
      getOperation().emitError()
          << "wave position chunk size must be positive, but got "
          << waveChunkSize.getValue();
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
    if ((mapping != "wave" || !cooperativeAxis.empty()) && waveChunkSize != 1) {
      getOperation().emitError()
          << "wave position chunk size applies only to non-cooperative wave "
             "mapping";
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
    if (waveReduction != "atomic" && waveReduction != "workspace") {
      getOperation().emitError()
          << "unsupported wave position reduction '" << waveReduction
          << "'; expected 'atomic' or 'workspace'";
      signalPassFailure();
      return;
    }
    if ((mapping != "wave" || !cooperativeAxis.empty()) &&
        waveReduction != "atomic") {
      getOperation().emitError()
          << "wave position reduction applies only to non-cooperative wave "
             "mapping";
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
                                              waveSize, waveChunkSize,
                                              waveReduction == "workspace");
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
