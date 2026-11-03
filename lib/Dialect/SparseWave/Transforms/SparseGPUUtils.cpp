#include "SparseGPUUtils.h"

#include "sparsewave/Dialect/SparseWave/IR/SparseWaveOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

namespace mlir::sparsewave {

namespace {

Value buildWaveInclusiveScan(OpBuilder &builder, Location loc, Value lane,
                             Value value, int64_t waveSize) {
  for (int32_t offset = 1; offset < waveSize; offset <<= 1) {
    auto predecessor = gpu::ShuffleOp::create(builder, loc, value, offset,
                                              waveSize, gpu::ShuffleMode::UP);
    Value total = arith::AddIOp::create(builder, loc, value,
                                        predecessor.getShuffleResult());
    Value offsetValue = arith::ConstantIndexOp::create(builder, loc, offset);
    Value hasPredecessor = arith::CmpIOp::create(
        builder, loc, arith::CmpIPredicate::uge, lane, offsetValue);
    value = arith::SelectOp::create(builder, loc, hasPredecessor, total, value);
  }
  return value;
}

Value buildWorkgroupInclusiveScan(OpBuilder &builder, Location loc,
                                  gpu::LaunchOp launch, Value participant,
                                  Value value, Value zero, int64_t blockSize,
                                  int64_t waveSize) {
  int64_t wavesPerBlock = (blockSize + waveSize - 1) / waveSize;
  auto workgroupAddressSpace = gpu::AddressSpaceAttr::get(
      builder.getContext(), gpu::AddressSpace::Workgroup);
  auto waveTotalsType = MemRefType::get({wavesPerBlock}, value.getType(),
                                        MemRefLayoutAttrInterface{},
                                        Attribute(workgroupAddressSpace));
  Value waveTotals = launch.addWorkgroupAttribution(waveTotalsType, loc);

  Value waveSizeValue = arith::ConstantIndexOp::create(builder, loc, waveSize);
  Value wave = arith::DivUIOp::create(builder, loc, participant, waveSizeValue);
  Value lane = arith::RemUIOp::create(builder, loc, participant, waveSizeValue);
  Value wavePrefix =
      buildWaveInclusiveScan(builder, loc, lane, value, waveSize);

  // The final active lane records the total for its wave. This also handles a
  // block whose final wave contains fewer than waveSize participants.
  Value one = arith::ConstantIndexOp::create(builder, loc, 1);
  Value waveBase = arith::MulIOp::create(builder, loc, wave, waveSizeValue);
  Value fullWaveEnd =
      arith::AddIOp::create(builder, loc, waveBase, waveSizeValue);
  Value blockSizeValue =
      arith::ConstantIndexOp::create(builder, loc, blockSize);
  Value waveEnd =
      arith::MinUIOp::create(builder, loc, fullWaveEnd, blockSizeValue);
  Value finalParticipant = arith::SubIOp::create(builder, loc, waveEnd, one);
  Value isFinalLane = arith::CmpIOp::create(
      builder, loc, arith::CmpIPredicate::eq, participant, finalParticipant);
  scf::IfOp::create(builder, loc, isFinalLane,
                    [&](OpBuilder &storeBuilder, Location storeLoc) {
                      memref::StoreOp::create(storeBuilder, storeLoc,
                                              wavePrefix, waveTotals, wave);
                      scf::YieldOp::create(storeBuilder, storeLoc);
                    });
  gpu::BarrierOp::create(builder, loc);

  // The first wave scans one total per wave and writes the inclusive wave
  // prefixes back to LDS. AMD blocks contain at most one wave of wave totals.
  Value zeroIndex = arith::ConstantIndexOp::create(builder, loc, 0);
  Value isFirstWave = arith::CmpIOp::create(
      builder, loc, arith::CmpIPredicate::eq, wave, zeroIndex);
  scf::IfOp::create(
      builder, loc, isFirstWave, [&](OpBuilder &scanBuilder, Location scanLoc) {
        Value waveCount =
            arith::ConstantIndexOp::create(scanBuilder, scanLoc, wavesPerBlock);
        Value hasWave = arith::CmpIOp::create(
            scanBuilder, scanLoc, arith::CmpIPredicate::ult, lane, waveCount);
        auto waveTotal =
            scf::IfOp::create(scanBuilder, scanLoc, TypeRange{value.getType()},
                              hasWave, /*withElseRegion=*/true);
        scanBuilder.setInsertionPointToStart(
            &waveTotal.getThenRegion().front());
        Value loaded =
            memref::LoadOp::create(scanBuilder, scanLoc, waveTotals, lane);
        scf::YieldOp::create(scanBuilder, scanLoc, loaded);
        scanBuilder.setInsertionPointToStart(
            &waveTotal.getElseRegion().front());
        scf::YieldOp::create(scanBuilder, scanLoc, zero);
        scanBuilder.setInsertionPointAfter(waveTotal);

        Value total = buildWaveInclusiveScan(scanBuilder, scanLoc, lane,
                                             waveTotal.getResult(0), waveSize);
        scf::IfOp::create(scanBuilder, scanLoc, hasWave,
                          [&](OpBuilder &storeBuilder, Location storeLoc) {
                            memref::StoreOp::create(storeBuilder, storeLoc,
                                                    total, waveTotals, lane);
                            scf::YieldOp::create(storeBuilder, storeLoc);
                          });
        scf::YieldOp::create(scanBuilder, scanLoc);
      });
  gpu::BarrierOp::create(builder, loc);

  Value hasPriorWave = arith::CmpIOp::create(
      builder, loc, arith::CmpIPredicate::ugt, wave, zeroIndex);
  auto priorWaveTotal =
      scf::IfOp::create(builder, loc, TypeRange{value.getType()}, hasPriorWave,
                        /*withElseRegion=*/true);
  builder.setInsertionPointToStart(&priorWaveTotal.getThenRegion().front());
  Value priorWave = arith::SubIOp::create(builder, loc, wave, one);
  Value priorTotal =
      memref::LoadOp::create(builder, loc, waveTotals, priorWave);
  scf::YieldOp::create(builder, loc, priorTotal);
  builder.setInsertionPointToStart(&priorWaveTotal.getElseRegion().front());
  scf::YieldOp::create(builder, loc, zero);
  builder.setInsertionPointAfter(priorWaveTotal);
  return arith::AddIOp::create(builder, loc, wavePrefix,
                               priorWaveTotal.getResult(0));
}

} // namespace

LinearThreadWorkDistribution
buildLinearThreadWorkDistribution(PatternRewriter &rewriter, Location loc,
                                  Value workUnitCount, Value oneIndex,
                                  Value blockSize) {
  Value requiredBlocks =
      arith::CeilDivUIOp::create(rewriter, loc, workUnitCount, blockSize);
  Value gridSize =
      arith::MaxUIOp::create(rewriter, loc, requiredBlocks, oneIndex);

  gpu::LaunchOp launch =
      gpu::LaunchOp::create(rewriter, loc, gridSize, oneIndex, oneIndex,
                            blockSize, oneIndex, oneIndex);
  rewriter.setInsertionPointToStart(&launch.getBody().front());

  Value workUnitBase = arith::MulIOp::create(
      rewriter, loc, launch.getBlockIds().x, launch.getBlockSize().x);
  Value workUnit = arith::AddIOp::create(rewriter, loc, workUnitBase,
                                         launch.getThreadIds().x);
  Value workUnitIsActive = arith::CmpIOp::create(
      rewriter, loc, arith::CmpIPredicate::ult, workUnit, workUnitCount);
  return {launch, workUnit, workUnitIsActive};
}

BlockWorkDistribution buildBlockWorkDistribution(PatternRewriter &rewriter,
                                                 Location loc,
                                                 Value workUnitCount,
                                                 Value oneIndex,
                                                 Value blockSize) {
  Value gridSize =
      arith::MaxUIOp::create(rewriter, loc, workUnitCount, oneIndex);
  gpu::LaunchOp launch =
      gpu::LaunchOp::create(rewriter, loc, gridSize, oneIndex, oneIndex,
                            blockSize, oneIndex, oneIndex);
  rewriter.setInsertionPointToStart(&launch.getBody().front());

  Value workUnit = launch.getBlockIds().x;
  Value participant = launch.getThreadIds().x;
  Value participantCount = launch.getBlockSize().x;
  Value workUnitIsActive = arith::CmpIOp::create(
      rewriter, loc, arith::CmpIPredicate::ult, workUnit, workUnitCount);
  return {launch, participant, participantCount, workUnit, workUnitIsActive};
}

gpu::LaunchOp buildThreadPerDenseOutputElement(
    PatternRewriter &rewriter, Location loc, Value output, Value zeroIndex,
    Value oneIndex, Value blockSize,
    ThreadDenseOutputElementBodyBuilder buildBody) {
  Value rowCount = memref::DimOp::create(rewriter, loc, output, zeroIndex);
  Value columnCount = memref::DimOp::create(rewriter, loc, output, oneIndex);
  Value outputElementCount =
      arith::MulIOp::create(rewriter, loc, rowCount, columnCount);
  LinearThreadWorkDistribution distribution = buildLinearThreadWorkDistribution(
      rewriter, loc, outputElementCount, oneIndex, blockSize);

  scf::IfOp::create(rewriter, loc, distribution.workUnitIsActive,
                    [&](OpBuilder &builder, Location bodyLoc) {
                      Value row = arith::DivUIOp::create(
                          builder, bodyLoc, distribution.workUnit, columnCount);
                      Value column = arith::RemUIOp::create(
                          builder, bodyLoc, distribution.workUnit, columnCount);
                      Value result =
                          buildBody(builder, bodyLoc,
                                    ThreadDenseOutputElement{row, column});
                      memref::StoreOp::create(builder, bodyLoc, result, output,
                                              ValueRange{row, column});
                      scf::YieldOp::create(builder, bodyLoc);
                    },
                    {});

  rewriter.setInsertionPointToEnd(&distribution.launch.getBody().front());
  gpu::TerminatorOp::create(rewriter, loc);
  return distribution.launch;
}

gpu::LaunchOp buildThreadPerCompressedSegment(
    PatternRewriter &rewriter, Location loc, Value segmentCount, Value offsets,
    Value oneIndex, Value blockSize,
    ThreadPerCompressedSegmentBodyBuilder buildBody) {
  LinearThreadWorkDistribution distribution = buildLinearThreadWorkDistribution(
      rewriter, loc, segmentCount, oneIndex, blockSize);

  scf::IfOp::create(
      rewriter, loc, distribution.workUnitIsActive,
      [&](OpBuilder &builder, Location bodyLoc) {
        CompressedSegmentBounds bounds = buildCompressedSegmentBounds(
            builder, bodyLoc, offsets, distribution.workUnit, oneIndex);
        buildBody(builder, bodyLoc, distribution.workUnit, bounds);
        scf::YieldOp::create(builder, bodyLoc);
      },
      {});

  rewriter.setInsertionPointToEnd(&distribution.launch.getBody().front());
  gpu::TerminatorOp::create(rewriter, loc);
  return distribution.launch;
}

gpu::LaunchOp buildThreadPerCompressedSegmentPair(
    PatternRewriter &rewriter, Location loc, Value segmentCount,
    Value lhsOffsets, Value rhsOffsets, Value oneIndex, Value blockSize,
    ThreadPerCompressedSegmentPairBodyBuilder buildBody) {
  LinearThreadWorkDistribution distribution = buildLinearThreadWorkDistribution(
      rewriter, loc, segmentCount, oneIndex, blockSize);

  scf::IfOp::create(
      rewriter, loc, distribution.workUnitIsActive,
      [&](OpBuilder &builder, Location bodyLoc) {
        CompressedSegmentBounds lhsBounds = buildCompressedSegmentBounds(
            builder, bodyLoc, lhsOffsets, distribution.workUnit, oneIndex);
        CompressedSegmentBounds rhsBounds = buildCompressedSegmentBounds(
            builder, bodyLoc, rhsOffsets, distribution.workUnit, oneIndex);
        buildBody(builder, bodyLoc,
                  ThreadCompressedSegmentPair{distribution.workUnit, lhsBounds,
                                              rhsBounds});
        scf::YieldOp::create(builder, bodyLoc);
      },
      {});

  rewriter.setInsertionPointToEnd(&distribution.launch.getBody().front());
  gpu::TerminatorOp::create(rewriter, loc);
  return distribution.launch;
}

CompressedOutputAssembly buildThreadPerCompressedOutputAssembly(
    PatternRewriter &rewriter, Location loc, Value segmentCount,
    Value outputOffsets, Value outputElementCount, Value outputCapacity,
    Value zeroIndex, Value oneIndex, Value blockSize, int64_t staticBlockSize,
    int64_t waveSize, CompressedOutputCountBuilder buildCount,
    CompressedOutputValuesBuilder buildValues) {
  Type offsetType = cast<MemRefType>(outputOffsets.getType()).getElementType();

  // Symbolic phase: count the output elements owned by each segment.
  LinearThreadWorkDistribution symbolicDistribution =
      buildLinearThreadWorkDistribution(rewriter, loc, segmentCount, oneIndex,
                                        blockSize);
  scf::IfOp::create(
      rewriter, loc, symbolicDistribution.workUnitIsActive,
      [&](OpBuilder &builder, Location bodyLoc) {
        Value count =
            buildCount(builder, bodyLoc, symbolicDistribution.workUnit);
        Value nextSegment = arith::AddIOp::create(
            builder, bodyLoc, symbolicDistribution.workUnit, oneIndex);
        Value storedCount =
            castIndexToType(builder, bodyLoc, count, offsetType);
        memref::StoreOp::create(builder, bodyLoc, storedCount, outputOffsets,
                                nextSegment);
        scf::YieldOp::create(builder, bodyLoc);
      },
      {});
  rewriter.setInsertionPointToEnd(
      &symbolicDistribution.launch.getBody().front());
  gpu::TerminatorOp::create(rewriter, loc);
  gpu::LaunchOp symbolic = symbolicDistribution.launch;

  // Prefix phase 1: scan each block of segment counts independently. The last
  // offset in each block also serves as its block total, avoiding a temporary
  // device allocation for the next phase.
  rewriter.setInsertionPointAfter(symbolic);
  Value prefixBlockCount =
      arith::CeilDivUIOp::create(rewriter, loc, segmentCount, blockSize);
  Value prefixGridSize =
      arith::MaxUIOp::create(rewriter, loc, prefixBlockCount, oneIndex);
  gpu::LaunchOp localPrefix =
      gpu::LaunchOp::create(rewriter, loc, prefixGridSize, oneIndex, oneIndex,
                            blockSize, oneIndex, oneIndex);
  rewriter.setInsertionPointToStart(&localPrefix.getBody().front());
  Value zeroOffset = arith::ConstantOp::create(
      rewriter, loc, rewriter.getZeroAttr(offsetType));
  Value segmentBase = arith::MulIOp::create(
      rewriter, loc, localPrefix.getBlockIds().x, blockSize);
  Value segment = arith::AddIOp::create(rewriter, loc, segmentBase,
                                        localPrefix.getThreadIds().x);
  Value segmentIsActive = arith::CmpIOp::create(
      rewriter, loc, arith::CmpIPredicate::ult, segment, segmentCount);
  auto count =
      scf::IfOp::create(rewriter, loc, TypeRange{offsetType}, segmentIsActive,
                        /*withElseRegion=*/true);
  rewriter.setInsertionPointToStart(&count.getThenRegion().front());
  Value nextSegment = arith::AddIOp::create(rewriter, loc, segment, oneIndex);
  Value storedCount =
      memref::LoadOp::create(rewriter, loc, outputOffsets, nextSegment);
  scf::YieldOp::create(rewriter, loc, storedCount);
  rewriter.setInsertionPointToStart(&count.getElseRegion().front());
  scf::YieldOp::create(rewriter, loc, zeroOffset);
  rewriter.setInsertionPointAfter(count);
  Value localTotal = buildWorkgroupInclusiveScan(
      rewriter, loc, localPrefix, localPrefix.getThreadIds().x,
      count.getResult(0), zeroOffset, staticBlockSize, waveSize);
  scf::IfOp::create(rewriter, loc, segmentIsActive,
                    [&](OpBuilder &builder, Location bodyLoc) {
                      Value outputIndex = arith::AddIOp::create(
                          builder, bodyLoc, segment, oneIndex);
                      memref::StoreOp::create(builder, bodyLoc, localTotal,
                                              outputOffsets, outputIndex);
                      scf::YieldOp::create(builder, bodyLoc);
                    });
  Value blockIsFirst =
      arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::eq,
                            localPrefix.getBlockIds().x, zeroIndex);
  Value threadIsFirst =
      arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::eq,
                            localPrefix.getThreadIds().x, zeroIndex);
  Value initializesFirstOffset =
      arith::AndIOp::create(rewriter, loc, blockIsFirst, threadIsFirst);
  scf::IfOp::create(rewriter, loc, initializesFirstOffset,
                    [&](OpBuilder &builder, Location bodyLoc) {
                      memref::StoreOp::create(builder, bodyLoc, zeroOffset,
                                              outputOffsets, zeroIndex);
                      scf::YieldOp::create(builder, bodyLoc);
                    });
  gpu::TerminatorOp::create(rewriter, loc);

  // Prefix phase 2: one workgroup scans the block totals. Each participant
  // first scans a contiguous chunk, then the workgroup scan carries the sum
  // between chunks. This also handles more blocks than workgroup threads.
  rewriter.setInsertionPointAfter(localPrefix);
  gpu::LaunchOp blockPrefix =
      gpu::LaunchOp::create(rewriter, loc, oneIndex, oneIndex, oneIndex,
                            blockSize, oneIndex, oneIndex);
  rewriter.setInsertionPointToStart(&blockPrefix.getBody().front());
  Value blocksPerParticipant =
      arith::CeilDivUIOp::create(rewriter, loc, prefixBlockCount, blockSize);
  Value firstBlock = arith::MulIOp::create(
      rewriter, loc, blockPrefix.getThreadIds().x, blocksPerParticipant);
  Value blockLimit =
      arith::AddIOp::create(rewriter, loc, firstBlock, blocksPerParticipant);
  Value lastBlock =
      arith::MinUIOp::create(rewriter, loc, blockLimit, prefixBlockCount);
  auto localBlockScan = scf::ForOp::create(
      rewriter, loc, firstBlock, lastBlock, oneIndex, ValueRange{zeroOffset},
      [&](OpBuilder &builder, Location bodyLoc, Value block,
          ValueRange iterArgs) {
        Value nextBlock =
            arith::AddIOp::create(builder, bodyLoc, block, oneIndex);
        Value fullBlockEnd =
            arith::MulIOp::create(builder, bodyLoc, nextBlock, blockSize);
        Value blockEnd = arith::MinUIOp::create(builder, bodyLoc, fullBlockEnd,
                                                segmentCount);
        Value blockTotal =
            memref::LoadOp::create(builder, bodyLoc, outputOffsets, blockEnd);
        Value total = arith::AddIOp::create(builder, bodyLoc, iterArgs.front(),
                                            blockTotal);
        memref::StoreOp::create(builder, bodyLoc, total, outputOffsets,
                                blockEnd);
        scf::YieldOp::create(builder, bodyLoc, total);
      });
  Value inclusiveChunkTotal = buildWorkgroupInclusiveScan(
      rewriter, loc, blockPrefix, blockPrefix.getThreadIds().x,
      localBlockScan.getResult(0), zeroOffset, staticBlockSize, waveSize);
  Value priorChunkTotal = arith::SubIOp::create(
      rewriter, loc, inclusiveChunkTotal, localBlockScan.getResult(0));
  scf::ForOp::create(
      rewriter, loc, firstBlock, lastBlock, oneIndex, ValueRange{},
      [&](OpBuilder &builder, Location bodyLoc, Value block, ValueRange) {
        Value nextBlock =
            arith::AddIOp::create(builder, bodyLoc, block, oneIndex);
        Value fullBlockEnd =
            arith::MulIOp::create(builder, bodyLoc, nextBlock, blockSize);
        Value blockEnd = arith::MinUIOp::create(builder, bodyLoc, fullBlockEnd,
                                                segmentCount);
        Value localEnd =
            memref::LoadOp::create(builder, bodyLoc, outputOffsets, blockEnd);
        Value globalEnd =
            arith::AddIOp::create(builder, bodyLoc, localEnd, priorChunkTotal);
        memref::StoreOp::create(builder, bodyLoc, globalEnd, outputOffsets,
                                blockEnd);
        scf::YieldOp::create(builder, bodyLoc);
      });
  gpu::BarrierOp::create(rewriter, loc);
  Value blockPrefixLeader =
      arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::eq,
                            blockPrefix.getThreadIds().x, zeroIndex);
  scf::IfOp::create(rewriter, loc, blockPrefixLeader,
                    [&](OpBuilder &builder, Location bodyLoc) {
                      Value total = memref::LoadOp::create(
                          builder, bodyLoc, outputOffsets, segmentCount);
                      memref::StoreOp::create(builder, bodyLoc, total,
                                              outputElementCount, zeroIndex);
                      scf::YieldOp::create(builder, bodyLoc);
                    });
  gpu::TerminatorOp::create(rewriter, loc);

  // Prefix phase 3: add each preceding block total to the locally scanned
  // offsets. Block endpoints already contain their global totals from phase 2.
  rewriter.setInsertionPointAfter(blockPrefix);
  LinearThreadWorkDistribution addDistribution =
      buildLinearThreadWorkDistribution(rewriter, loc, segmentCount, oneIndex,
                                        blockSize);
  Value outputIndex =
      arith::AddIOp::create(rewriter, loc, addDistribution.workUnit, oneIndex);
  Value block = arith::DivUIOp::create(rewriter, loc, addDistribution.workUnit,
                                       blockSize);
  Value nextBlock = arith::AddIOp::create(rewriter, loc, block, oneIndex);
  Value fullBlockEnd =
      arith::MulIOp::create(rewriter, loc, nextBlock, blockSize);
  Value blockEnd =
      arith::MinUIOp::create(rewriter, loc, fullBlockEnd, segmentCount);
  Value hasPriorBlock = arith::CmpIOp::create(
      rewriter, loc, arith::CmpIPredicate::ugt, block, zeroIndex);
  Value isNotBlockEnd = arith::CmpIOp::create(
      rewriter, loc, arith::CmpIPredicate::ult, outputIndex, blockEnd);
  Value needsBlockOffset = arith::AndIOp::create(
      rewriter, loc, addDistribution.workUnitIsActive,
      arith::AndIOp::create(rewriter, loc, hasPriorBlock, isNotBlockEnd));
  scf::IfOp::create(
      rewriter, loc, needsBlockOffset,
      [&](OpBuilder &builder, Location bodyLoc) {
        Value priorBlockEnd =
            arith::MulIOp::create(builder, bodyLoc, block, blockSize);
        Value blockOffset = memref::LoadOp::create(
            builder, bodyLoc, outputOffsets, priorBlockEnd);
        Value localOffset = memref::LoadOp::create(builder, bodyLoc,
                                                   outputOffsets, outputIndex);
        Value globalOffset =
            arith::AddIOp::create(builder, bodyLoc, localOffset, blockOffset);
        memref::StoreOp::create(builder, bodyLoc, globalOffset, outputOffsets,
                                outputIndex);
        scf::YieldOp::create(builder, bodyLoc);
      });
  rewriter.setInsertionPointToEnd(&addDistribution.launch.getBody().front());
  gpu::TerminatorOp::create(rewriter, loc);

  // Numeric phase: replay the caller's traversal within the assigned output
  // range for each segment.
  rewriter.setInsertionPointAfter(addDistribution.launch);
  LinearThreadWorkDistribution numericDistribution =
      buildLinearThreadWorkDistribution(rewriter, loc, segmentCount, oneIndex,
                                        blockSize);
  scf::IfOp::create(
      rewriter, loc, numericDistribution.workUnitIsActive,
      [&](OpBuilder &builder, Location bodyLoc) {
        Value outputStartValue = memref::LoadOp::create(
            builder, bodyLoc, outputOffsets, numericDistribution.workUnit);
        Value outputStart = castToIndex(builder, bodyLoc, outputStartValue);
        buildValues(builder, bodyLoc,
                    ThreadCompressedOutputSegment{numericDistribution.workUnit,
                                                  outputStart, outputCapacity});
        scf::YieldOp::create(builder, bodyLoc);
      },
      {});
  rewriter.setInsertionPointToEnd(
      &numericDistribution.launch.getBody().front());
  gpu::TerminatorOp::create(rewriter, loc);

  return {symbolic, localPrefix, blockPrefix, addDistribution.launch,
          numericDistribution.launch};
}

gpu::LaunchOp buildWavePerCompressedSegment(
    PatternRewriter &rewriter, Location loc, Value segmentCount, Value offsets,
    Value oneIndex, Value blockSize, Value waveSize, Value wavesPerBlock,
    WavePerCompressedSegmentBodyBuilder buildBody) {
  return buildWavePerCompressedWorkUnit(
      rewriter, loc, segmentCount, offsets, oneIndex, blockSize, waveSize,
      wavesPerBlock,
      [](OpBuilder &, Location, Value workUnit) { return workUnit; },
      [&](OpBuilder &builder, Location bodyLoc, WaveCompressedWorkUnit work) {
        buildBody(builder, bodyLoc,
                  WaveCompressedSegment{work.segment, work.lane, work.bounds,
                                        work.positions});
      });
}

gpu::LaunchOp buildWavePerCompressedWorkUnit(
    PatternRewriter &rewriter, Location loc, Value workUnitCount, Value offsets,
    Value oneIndex, Value blockSize, Value waveSize, Value wavesPerBlock,
    CompressedSegmentMappingBuilder mapSegment,
    WavePerCompressedWorkUnitBodyBuilder buildBody) {
  WaveWorkDistribution distribution =
      buildWaveWorkDistribution(rewriter, loc, workUnitCount, oneIndex,
                                blockSize, waveSize, wavesPerBlock);

  scf::IfOp::create(
      rewriter, loc, distribution.workUnitIsActive,
      [&](OpBuilder &builder, Location bodyLoc) {
        Value segment = mapSegment(builder, bodyLoc, distribution.workUnit);
        CompressedSegmentBounds bounds = buildCompressedSegmentBounds(
            builder, bodyLoc, offsets, segment, oneIndex);
        StridedPositionRange positions = buildStridedPositionRange(
            builder, bodyLoc, bounds, distribution.lane,
            distribution.positionStride);
        buildBody(builder, bodyLoc,
                  WaveCompressedWorkUnit{distribution.workUnit, segment,
                                         distribution.lane, bounds, positions});
        scf::YieldOp::create(builder, bodyLoc);
      },
      {});

  rewriter.setInsertionPointToEnd(&distribution.launch.getBody().front());
  gpu::TerminatorOp::create(rewriter, loc);
  return distribution.launch;
}

WaveWorkDistribution
buildWaveWorkDistribution(PatternRewriter &rewriter, Location loc,
                          Value workUnitCount, Value oneIndex, Value blockSize,
                          Value waveSize, Value wavesPerBlock) {
  Value requiredBlocks =
      arith::CeilDivUIOp::create(rewriter, loc, workUnitCount, wavesPerBlock);
  Value gridSize =
      arith::MaxUIOp::create(rewriter, loc, requiredBlocks, oneIndex);

  gpu::LaunchOp launch =
      gpu::LaunchOp::create(rewriter, loc, gridSize, oneIndex, oneIndex,
                            blockSize, oneIndex, oneIndex);
  rewriter.setInsertionPointToStart(&launch.getBody().front());

  Value waveInBlock =
      arith::DivUIOp::create(rewriter, loc, launch.getThreadIds().x, waveSize);
  Value lane =
      arith::RemUIOp::create(rewriter, loc, launch.getThreadIds().x, waveSize);
  Value workUnitBase = arith::MulIOp::create(
      rewriter, loc, launch.getBlockIds().x, wavesPerBlock);
  Value workUnit =
      arith::AddIOp::create(rewriter, loc, workUnitBase, waveInBlock);
  Value workUnitIsActive = arith::CmpIOp::create(
      rewriter, loc, arith::CmpIPredicate::ult, workUnit, workUnitCount);
  return {launch, waveInBlock, lane, workUnit, workUnitIsActive, waveSize};
}

CompressedSegmentBounds
buildCompressedSegmentBounds(OpBuilder &builder, Location loc, Value offsets,
                             Value segment, Value oneIndex) {
  Value nextSegment = arith::AddIOp::create(builder, loc, segment, oneIndex);
  Value startValue = memref::LoadOp::create(builder, loc, offsets, segment);
  Value endValue = memref::LoadOp::create(builder, loc, offsets, nextSegment);
  return {castToIndex(builder, loc, startValue),
          castToIndex(builder, loc, endValue)};
}

StridedPositionRange buildStridedPositionRange(OpBuilder &builder, Location loc,
                                               CompressedSegmentBounds bounds,
                                               Value participantOffset,
                                               Value stride) {
  Value first =
      arith::AddIOp::create(builder, loc, bounds.start, participantOffset);
  return {first, bounds.end, stride};
}

SmallVector<Value> buildCompressedPositionTraversal(
    OpBuilder &builder, Location loc, Value coordinates, Value values,
    StridedPositionRange positions, ValueRange initialValues,
    CompressedPositionBodyBuilder buildBody) {
  auto loop = scf::ForOp::create(
      builder, loc, positions.first, positions.end, positions.stride,
      initialValues,
      [&](OpBuilder &loopBuilder, Location loopLoc, Value position,
          ValueRange iterArgs) {
        Value coordinateValue =
            memref::LoadOp::create(loopBuilder, loopLoc, coordinates, position);
        Value coordinate = castToIndex(loopBuilder, loopLoc, coordinateValue);
        Value sparseValue =
            memref::LoadOp::create(loopBuilder, loopLoc, values, position);
        SmallVector<Value> nextValues = buildBody(
            loopBuilder, loopLoc,
            CompressedPosition{position, coordinate, sparseValue}, iterArgs);
        scf::YieldOp::create(loopBuilder, loopLoc, nextValues);
      });
  return SmallVector<Value>(loop.getResults());
}

SmallVector<Value> buildCompressedCoiteration(
    OpBuilder &builder, Location loc, Value lhsCoordinates,
    CompressedSegmentBounds lhsBounds, Value rhsCoordinates,
    CompressedSegmentBounds rhsBounds, CompressedCoiterationKind kind,
    Value oneIndex, ValueRange initialValues,
    CompressedCoiterationBodyBuilder buildBody) {
  SmallVector<Value> initialState{lhsBounds.start, rhsBounds.start};
  llvm::append_range(initialState, initialValues);
  TypeRange stateTypes = ValueRange(initialState).getTypes();
  auto loop = scf::WhileOp::create(builder, loc, stateTypes, initialState);

  OpBuilder::InsertionGuard guard(builder);
  SmallVector<Location> argumentLocations(stateTypes.size(), loc);
  Block *before =
      builder.createBlock(&loop.getBefore(), {}, stateTypes, argumentLocations);
  builder.setInsertionPointToStart(before);
  Value lhsPosition = before->getArgument(0);
  Value rhsPosition = before->getArgument(1);
  Value lhsActive = arith::CmpIOp::create(
      builder, loc, arith::CmpIPredicate::ult, lhsPosition, lhsBounds.end);
  Value rhsActive = arith::CmpIOp::create(
      builder, loc, arith::CmpIPredicate::ult, rhsPosition, rhsBounds.end);
  Value condition;
  if (kind == CompressedCoiterationKind::Union)
    condition = arith::OrIOp::create(builder, loc, lhsActive, rhsActive);
  else
    condition = arith::AndIOp::create(builder, loc, lhsActive, rhsActive);
  scf::ConditionOp::create(builder, loc, condition, before->getArguments());

  Block *after =
      builder.createBlock(&loop.getAfter(), {}, stateTypes, argumentLocations);
  builder.setInsertionPointToStart(after);
  lhsPosition = after->getArgument(0);
  rhsPosition = after->getArgument(1);
  ValueRange iterArgs = after->getArguments().drop_front(2);
  lhsActive = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::ult,
                                    lhsPosition, lhsBounds.end);
  rhsActive = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::ult,
                                    rhsPosition, rhsBounds.end);

  // A union iteration may have exhausted one input. Select the active input as
  // the fallback so neither branch performs an out-of-bounds coordinate load.
  auto lhsCoordinateSelection = scf::IfOp::create(
      builder, loc, lhsActive,
      [&](OpBuilder &thenBuilder, Location thenLoc) {
        Value coordinateValue = memref::LoadOp::create(
            thenBuilder, thenLoc, lhsCoordinates, lhsPosition);
        Value coordinate = castToIndex(thenBuilder, thenLoc, coordinateValue);
        scf::YieldOp::create(thenBuilder, thenLoc, coordinate);
      },
      [&](OpBuilder &elseBuilder, Location elseLoc) {
        Value coordinateValue = memref::LoadOp::create(
            elseBuilder, elseLoc, rhsCoordinates, rhsPosition);
        Value coordinate = castToIndex(elseBuilder, elseLoc, coordinateValue);
        scf::YieldOp::create(elseBuilder, elseLoc, coordinate);
      });
  auto rhsCoordinateSelection = scf::IfOp::create(
      builder, loc, rhsActive,
      [&](OpBuilder &thenBuilder, Location thenLoc) {
        Value coordinateValue = memref::LoadOp::create(
            thenBuilder, thenLoc, rhsCoordinates, rhsPosition);
        Value coordinate = castToIndex(thenBuilder, thenLoc, coordinateValue);
        scf::YieldOp::create(thenBuilder, thenLoc, coordinate);
      },
      [&](OpBuilder &elseBuilder, Location elseLoc) {
        Value coordinateValue = memref::LoadOp::create(
            elseBuilder, elseLoc, lhsCoordinates, lhsPosition);
        Value coordinate = castToIndex(elseBuilder, elseLoc, coordinateValue);
        scf::YieldOp::create(elseBuilder, elseLoc, coordinate);
      });
  Value lhsCoordinate = lhsCoordinateSelection.getResult(0);
  Value rhsCoordinate = rhsCoordinateSelection.getResult(0);

  Value trueValue = arith::ConstantIntOp::create(builder, loc, 1, 1);
  Value lhsOnly = arith::XOrIOp::create(builder, loc, rhsActive, trueValue);
  Value rhsOnly = arith::XOrIOp::create(builder, loc, lhsActive, trueValue);
  Value lhsPrecedes = arith::CmpIOp::create(
      builder, loc, arith::CmpIPredicate::ule, lhsCoordinate, rhsCoordinate);
  Value rhsPrecedes = arith::CmpIOp::create(
      builder, loc, arith::CmpIPredicate::ule, rhsCoordinate, lhsCoordinate);
  Value takeLhs = arith::AndIOp::create(
      builder, loc, lhsActive,
      arith::OrIOp::create(builder, loc, lhsOnly, lhsPrecedes));
  Value takeRhs = arith::AndIOp::create(
      builder, loc, rhsActive,
      arith::OrIOp::create(builder, loc, rhsOnly, rhsPrecedes));
  Value coordinate = arith::SelectOp::create(builder, loc, takeLhs,
                                             lhsCoordinate, rhsCoordinate);

  Value nextLhsPosition = arith::SelectOp::create(
      builder, loc, takeLhs,
      arith::AddIOp::create(builder, loc, lhsPosition, oneIndex), lhsPosition);
  Value nextRhsPosition = arith::SelectOp::create(
      builder, loc, takeRhs,
      arith::AddIOp::create(builder, loc, rhsPosition, oneIndex), rhsPosition);
  Value emit = kind == CompressedCoiterationKind::Union
                   ? trueValue
                   : arith::AndIOp::create(builder, loc, takeLhs, takeRhs);

  auto update = scf::IfOp::create(
      builder, loc, emit,
      [&](OpBuilder &thenBuilder, Location thenLoc) {
        SmallVector<Value> nextValues =
            buildBody(thenBuilder, thenLoc,
                      CompressedCoiterationEntry{coordinate, lhsPosition,
                                                 rhsPosition, takeLhs, takeRhs},
                      iterArgs);
        scf::YieldOp::create(thenBuilder, thenLoc, nextValues);
      },
      [&](OpBuilder &elseBuilder, Location elseLoc) {
        scf::YieldOp::create(elseBuilder, elseLoc, iterArgs);
      });
  SmallVector<Value> nextState{nextLhsPosition, nextRhsPosition};
  llvm::append_range(nextState, update.getResults());
  scf::YieldOp::create(builder, loc, nextState);

  return SmallVector<Value>(loop.getResults().drop_front(2));
}

BoundedTile buildBoundedTile(OpBuilder &builder, Location loc,
                             Value firstCoordinate, Value coordinateCount,
                             int64_t tileSize) {
  SmallVector<Value> coordinates;
  coordinates.reserve(tileSize);
  for (int64_t offset = 0; offset < tileSize; ++offset) {
    Value offsetValue = arith::ConstantIndexOp::create(builder, loc, offset);
    coordinates.push_back(
        arith::AddIOp::create(builder, loc, firstCoordinate, offsetValue));
  }

  Value tileSizeValue = arith::ConstantIndexOp::create(builder, loc, tileSize);
  Value tileEnd =
      arith::AddIOp::create(builder, loc, firstCoordinate, tileSizeValue);
  Value isFull = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::ule,
                                       tileEnd, coordinateCount);
  return {std::move(coordinates), coordinateCount, isFull};
}

SmallVector<Value>
buildFullOrPartialTileValues(OpBuilder &builder, Location loc,
                             const BoundedTile &tile, TypeRange resultTypes,
                             BoundedTileValuesBuilder buildValues) {
  auto selection = scf::IfOp::create(builder, loc, resultTypes, tile.isFull,
                                     /*withElseRegion=*/true);
  builder.setInsertionPointToStart(&selection.getThenRegion().front());
  SmallVector<Value> fullValues = buildValues(
      builder, loc, tile.coordinates, /*coordinateIsValid=*/ValueRange{});
  scf::YieldOp::create(builder, loc, fullValues);

  builder.setInsertionPointToStart(&selection.getElseRegion().front());
  SmallVector<Value> coordinateIsValid;
  coordinateIsValid.reserve(tile.coordinates.size());
  for (Value coordinate : tile.coordinates) {
    coordinateIsValid.push_back(
        arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::ult,
                              coordinate, tile.coordinateCount));
  }
  SmallVector<Value> partialValues =
      buildValues(builder, loc, tile.coordinates, coordinateIsValid);
  scf::YieldOp::create(builder, loc, partialValues);

  builder.setInsertionPointAfter(selection);
  return SmallVector<Value>(selection.getResults());
}

void buildValidTileResults(OpBuilder &builder, Location loc,
                           const BoundedTile &tile, ValueRange results,
                           BoundedTileResultBuilder buildResult) {
  scf::IfOp::create(
      builder, loc, tile.isFull,
      [&](OpBuilder &fullBuilder, Location fullLoc) {
        for (auto [coordinate, result] :
             llvm::zip_equal(tile.coordinates, results))
          buildResult(fullBuilder, fullLoc, coordinate, result);
        scf::YieldOp::create(fullBuilder, fullLoc);
      },
      [&](OpBuilder &partialBuilder, Location partialLoc) {
        for (size_t index = 0; index < tile.coordinates.size(); ++index) {
          Value coordinate = tile.coordinates[index];
          Value result = results[index];
          Value isValid = arith::CmpIOp::create(
              partialBuilder, partialLoc, arith::CmpIPredicate::ult, coordinate,
              tile.coordinateCount);
          scf::IfOp::create(partialBuilder, partialLoc, isValid,
                            [&](OpBuilder &resultBuilder, Location resultLoc) {
                              buildResult(resultBuilder, resultLoc, coordinate,
                                          result);
                              scf::YieldOp::create(resultBuilder, resultLoc);
                            });
        }
        scf::YieldOp::create(partialBuilder, partialLoc);
      });
}

Value buildWaveReduction(OpBuilder &builder, Location loc, Value value,
                         int64_t waveSize, ReductionCombinerBuilder combine) {
  for (int32_t offset = 1; offset < waveSize; offset <<= 1) {
    Value shuffled = gpu::ShuffleOp::create(builder, loc, value, offset,
                                            waveSize, gpu::ShuffleMode::XOR)
                         .getShuffleResult();
    value = combine(builder, loc, value, shuffled);
  }
  return value;
}

Value buildWaveReduction(OpBuilder &builder, Location loc, Value value,
                         int64_t waveSize) {
  return buildWaveReduction(
      builder, loc, value, waveSize,
      [](OpBuilder &sumBuilder, Location sumLoc, Value lhs, Value rhs) {
        return arith::AddFOp::create(sumBuilder, sumLoc, lhs, rhs);
      });
}

void buildBlockReduction(OpBuilder &builder, Location loc, gpu::LaunchOp launch,
                         Value partialValue, Value identity, Value waveInBlock,
                         Value lane, Value zeroIndex, int64_t wavesPerBlock,
                         int64_t waveSize, ReductionCombinerBuilder combine,
                         BlockReductionResultBuilder buildResult) {
  auto workgroupAddressSpace = gpu::AddressSpaceAttr::get(
      builder.getContext(), gpu::AddressSpace::Workgroup);
  auto waveValuesType = MemRefType::get({wavesPerBlock}, partialValue.getType(),
                                        MemRefLayoutAttrInterface{},
                                        Attribute(workgroupAddressSpace));
  Value waveValues = launch.addWorkgroupAttribution(waveValuesType, loc);

  Value waveValue =
      buildWaveReduction(builder, loc, partialValue, waveSize, combine);
  Value laneIsZero = arith::CmpIOp::create(
      builder, loc, arith::CmpIPredicate::eq, lane, zeroIndex);
  scf::IfOp::create(builder, loc, laneIsZero,
                    [&](OpBuilder &storeBuilder, Location storeLoc) {
                      memref::StoreOp::create(storeBuilder, storeLoc, waveValue,
                                              waveValues, waveInBlock);
                      scf::YieldOp::create(storeBuilder, storeLoc);
                    });

  gpu::BarrierOp::create(builder, loc);

  Value waveIsZero = arith::CmpIOp::create(
      builder, loc, arith::CmpIPredicate::eq, waveInBlock, zeroIndex);
  scf::IfOp::create(
      builder, loc, waveIsZero, [&](OpBuilder &waveBuilder, Location waveLoc) {
        Value waveCount =
            arith::ConstantIndexOp::create(waveBuilder, waveLoc, wavesPerBlock);
        Value laneHasWave = arith::CmpIOp::create(
            waveBuilder, waveLoc, arith::CmpIPredicate::ult, lane, waveCount);
        auto initialValue = scf::IfOp::create(
            waveBuilder, waveLoc, TypeRange{partialValue.getType()},
            laneHasWave, /*withElseRegion=*/true);
        waveBuilder.setInsertionPointToStart(
            &initialValue.getThenRegion().front());
        Value value =
            memref::LoadOp::create(waveBuilder, waveLoc, waveValues, lane);
        scf::YieldOp::create(waveBuilder, waveLoc, value);
        waveBuilder.setInsertionPointToStart(
            &initialValue.getElseRegion().front());
        scf::YieldOp::create(waveBuilder, waveLoc, identity);
        waveBuilder.setInsertionPointAfter(initialValue);

        Value blockValue = buildWaveReduction(
            waveBuilder, waveLoc, initialValue.getResult(0), waveSize, combine);
        Value blockLeader = arith::CmpIOp::create(
            waveBuilder, waveLoc, arith::CmpIPredicate::eq, lane, zeroIndex);
        scf::IfOp::create(waveBuilder, waveLoc, blockLeader,
                          [&](OpBuilder &resultBuilder, Location resultLoc) {
                            buildResult(resultBuilder, resultLoc, blockValue);
                            scf::YieldOp::create(resultBuilder, resultLoc);
                          });
        scf::YieldOp::create(waveBuilder, waveLoc);
      });
}

WaveSegmentedReduction buildWavePrefixSegmentedReduction(
    OpBuilder &builder, Location loc, Value key, Value value, Value active,
    Value lane, int64_t waveSize, ReductionCombinerBuilder combine,
    bool computeSegmentOrdinal) {
  Value activeI32 =
      arith::ExtUIOp::create(builder, loc, builder.getI32Type(), active);
  Value zeroIndex = arith::ConstantIndexOp::create(builder, loc, 0);
  Value lastLane = arith::ConstantIndexOp::create(builder, loc, waveSize - 1);
  Value zeroI32 = arith::ConstantIntOp::create(builder, loc, 0, 32);
  Value trueValue = arith::ConstantIntOp::create(builder, loc, 1, 1);

  auto previousKey = gpu::ShuffleOp::create(builder, loc, key, 1, waveSize,
                                            gpu::ShuffleMode::UP);
  auto previousActive = gpu::ShuffleOp::create(builder, loc, activeI32, 1,
                                               waveSize, gpu::ShuffleMode::UP);
  Value previousLaneIsActive =
      arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::ne,
                            previousActive.getShuffleResult(), zeroI32);
  Value hasPreviousLane = arith::CmpIOp::create(
      builder, loc, arith::CmpIPredicate::ne, lane, zeroIndex);
  Value continuesPreviousSegment = arith::AndIOp::create(
      builder, loc, hasPreviousLane,
      arith::AndIOp::create(
          builder, loc, previousLaneIsActive,
          arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::eq, key,
                                previousKey.getShuffleResult())));
  Value startsSegment = arith::AndIOp::create(
      builder, loc, active,
      arith::XOrIOp::create(builder, loc, continuesPreviousSegment, trueValue));
  Value startsSegmentI32 =
      arith::ExtUIOp::create(builder, loc, builder.getI32Type(), startsSegment);
  Value segmentOrdinal;
  if (computeSegmentOrdinal) {
    // Count segment starts instead of comparing against the first key. Equal
    // keys may legally occur in multiple non-adjacent segments.
    segmentOrdinal = startsSegmentI32;
    for (int32_t offset = 1; offset < waveSize; offset <<= 1) {
      auto shuffledOrdinal = gpu::ShuffleOp::create(
          builder, loc, segmentOrdinal, offset, waveSize, gpu::ShuffleMode::UP);
      Value accumulated = arith::AddIOp::create(
          builder, loc, segmentOrdinal, shuffledOrdinal.getShuffleResult());
      Value shouldAccumulate = arith::AndIOp::create(
          builder, loc, active, shuffledOrdinal.getValid());
      segmentOrdinal = arith::SelectOp::create(builder, loc, shouldAccumulate,
                                               accumulated, segmentOrdinal);
    }
  }

  // Carry segment-start markers with the partial sums so equal keys separated
  // by another segment are never combined at larger shuffle offsets.
  for (int32_t offset = 1; offset < waveSize; offset <<= 1) {
    auto shuffledValue = gpu::ShuffleOp::create(builder, loc, value, offset,
                                                waveSize, gpu::ShuffleMode::UP);
    auto shuffledSegmentStart = gpu::ShuffleOp::create(
        builder, loc, startsSegmentI32, offset, waveSize, gpu::ShuffleMode::UP);
    Value hasSegmentStart = arith::CmpIOp::create(
        builder, loc, arith::CmpIPredicate::ne, startsSegmentI32, zeroI32);
    Value shouldCombine = arith::AndIOp::create(
        builder, loc, active,
        arith::AndIOp::create(
            builder, loc, shuffledValue.getValid(),
            arith::XOrIOp::create(builder, loc, hasSegmentStart, trueValue)));
    Value accumulated =
        combine(builder, loc, value, shuffledValue.getShuffleResult());
    value = arith::SelectOp::create(builder, loc, shouldCombine, accumulated,
                                    value);

    Value shuffledHasSegmentStart = arith::AndIOp::create(
        builder, loc, shuffledSegmentStart.getValid(),
        arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::ne,
                              shuffledSegmentStart.getShuffleResult(),
                              zeroI32));
    Value propagatedSegmentStart = arith::OrIOp::create(
        builder, loc, hasSegmentStart, shuffledHasSegmentStart);
    startsSegmentI32 = arith::ExtUIOp::create(
        builder, loc, builder.getI32Type(), propagatedSegmentStart);
  }

  auto nextKey = gpu::ShuffleOp::create(builder, loc, key, 1, waveSize,
                                        gpu::ShuffleMode::DOWN);
  auto nextActive = gpu::ShuffleOp::create(builder, loc, activeI32, 1, waveSize,
                                           gpu::ShuffleMode::DOWN);
  Value nextLaneIsActive =
      arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::ne,
                            nextActive.getShuffleResult(), zeroI32);
  Value nextIsSameSegment = arith::AndIOp::create(
      builder, loc,
      arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::ne, lane,
                            lastLane),
      arith::AndIOp::create(builder, loc, nextLaneIsActive,
                            arith::CmpIOp::create(builder, loc,
                                                  arith::CmpIPredicate::eq, key,
                                                  nextKey.getShuffleResult())));
  Value nextIsDifferent =
      arith::XOrIOp::create(builder, loc, nextIsSameSegment, trueValue);
  Value segmentEnd =
      arith::AndIOp::create(builder, loc, active, nextIsDifferent);
  return {value, segmentEnd, segmentOrdinal};
}

SmallVector<Value> buildWaveReductions(OpBuilder &builder, Location loc,
                                       ValueRange values, int64_t waveSize,
                                       ReductionCombinerBuilder combine) {
  SmallVector<Value> reducedValues;
  reducedValues.reserve(values.size());
  for (Value value : values)
    reducedValues.push_back(
        buildWaveReduction(builder, loc, value, waveSize, combine));
  return reducedValues;
}

gpu::LaunchOp buildWavePerCompressedTile(
    PatternRewriter &rewriter, Location loc, Value segmentCount, Value offsets,
    Value coordinateCount, Value oneIndex, Value blockSize, Value waveSizeValue,
    Value wavesPerBlock, Value tileSizeValue, int64_t waveSize,
    int64_t tileSize, TypeRange resultTypes, ReductionCombinerBuilder combine,
    WaveCompressedTileValuesBuilder buildLaneValues,
    WaveCompressedTileResultBuilder buildResult) {
  Value zeroIndex = arith::ConstantIndexOp::create(rewriter, loc, 0);
  Value tilesPerSegment =
      arith::CeilDivUIOp::create(rewriter, loc, coordinateCount, tileSizeValue);
  Value workUnitCount =
      arith::MulIOp::create(rewriter, loc, segmentCount, tilesPerSegment);

  return buildWavePerCompressedWorkUnit(
      rewriter, loc, workUnitCount, offsets, oneIndex, blockSize, waveSizeValue,
      wavesPerBlock,
      [&](OpBuilder &builder, Location bodyLoc, Value workUnit) {
        return arith::DivUIOp::create(builder, bodyLoc, workUnit,
                                      tilesPerSegment);
      },
      [&](OpBuilder &builder, Location bodyLoc, WaveCompressedWorkUnit work) {
        Value tile = arith::RemUIOp::create(builder, bodyLoc, work.workUnit,
                                            tilesPerSegment);
        Value firstCoordinate =
            arith::MulIOp::create(builder, bodyLoc, tile, tileSizeValue);
        BoundedTile boundedTile = buildBoundedTile(
            builder, bodyLoc, firstCoordinate, coordinateCount, tileSize);

        SmallVector<Value> laneValues = buildFullOrPartialTileValues(
            builder, bodyLoc, boundedTile, resultTypes,
            [&](OpBuilder &tileBuilder, Location tileLoc,
                ValueRange coordinates,
                ValueRange activeCoordinates) -> SmallVector<Value> {
              return buildLaneValues(
                  tileBuilder, tileLoc,
                  WaveCompressedTile{work, coordinates, activeCoordinates});
            });
        SmallVector<Value> reducedValues = buildWaveReductions(
            builder, bodyLoc, laneValues, waveSize, combine);

        Value laneIsZero = arith::CmpIOp::create(
            builder, bodyLoc, arith::CmpIPredicate::eq, work.lane, zeroIndex);
        scf::IfOp::create(
            builder, bodyLoc, laneIsZero,
            [&](OpBuilder &laneBuilder, Location laneLoc) {
              buildValidTileResults(
                  laneBuilder, laneLoc, boundedTile, reducedValues,
                  [&](OpBuilder &resultBuilder, Location resultLoc,
                      Value coordinate, Value result) {
                    buildResult(resultBuilder, resultLoc,
                                WaveCompressedTileResult{work.segment,
                                                         coordinate, result});
                  });
              scf::YieldOp::create(laneBuilder, laneLoc);
            });
      });
}

} // namespace mlir::sparsewave
