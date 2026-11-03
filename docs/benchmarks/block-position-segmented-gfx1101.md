# Hierarchical block-local segmented position reduction on gfx1101

This report evaluates SparseWave's hierarchical block-local segmented
reduction for the generic position scheduler. A block owns one contiguous
partition of the collapsed position domain. The atomic baseline updates the
output once per position. The segmented strategy scans adjacent equal keys
with wave shuffles, emits wave-interior segments directly, and stages only the
first and last partial segments of each wave in LDS for block-level merging.

The experiment asks when the reduction in dynamic atomics repays the wave scan
and boundary-merge cost. It covers both SpMV and SpMM because the scheduler
operates on generic keyed contributions rather than named operators.

## Environment

| Component | Value |
| --- | --- |
| Date | 2026-11-03 |
| GPU | AMD Radeon RX 7800 XT |
| Target chip | gfx1101 |
| Wavefront size | 32 |
| SparseWave base commit | `55cc13558a3a929356ad391f5e9a975575bb61a3` |
| LLVM commit | `64b593c2371b7f7225b0ec190a37cd8b672e4c5d` |
| ROCm version | 6.4.4-129 |
| rocSPARSE version | 300400, Git revision `8fbfc797` |
| GPU performance level | Automatic |
| Block size | 128 |

The measurements include the lane-boundary correction documented below and
committed with this report. SparseWave times are the sum of GPU kernel
timestamps reported by `rocprofv3`; initialization is included for position
mappings. rocSPARSE uses HIP events around the complete compute call. GPU
clocks were not locked, so
the results support comparisons within these runs rather than cross-run
absolute-latency claims. All 90 recorded configurations passed CPU-reference
validation. The combined rocprof/rocSPARSE sweep stalled after the SparseWave
cases, so the six rocSPARSE cases were invoked directly; their reported values
still use the same runner's HIP-event timing path.

## SpMV

The SpMV sweep used 8,192-by-8,192 generated matrices, 10 warmup dispatches,
and 50 measured dispatches. `uniform` gives every row the requested length.
`skewed` gives seven rows one entry and the eighth row `8N - 7` entries while
preserving mean row length `N`.

`Best fixed` is the minimum of thread-, wave-, and block-per-row. Speedup is
the block atomic median divided by the block segmented median.

| Distribution | Mean NNZ/row | Block atomic | Block segmented | Speedup | Best fixed | rocSPARSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Uniform | 4 | 11.36 us | 14.96 us | 0.76x | 4.40 us | 12.22 us |
| Uniform | 32 | 308.65 us | 50.02 us | 6.17x | 10.96 us | 35.80 us |
| Uniform | 128 | 4,074.46 us | 124.18 us | 32.81x | 14.92 us | 25.34 us |
| Skewed | 4 | 33.10 us | 14.80 us | 2.24x | 5.96 us | 13.88 us |
| Skewed | 32 | 1,455.43 us | 40.72 us | 35.74x | 14.64 us | 27.80 us |
| Skewed | 128 | 20,522.01 us | 126.94 us | **161.67x** | 29.60 us | 33.66 us |

The benefit grows with contiguous same-row runs. In SpMV the output key is the
row, and CSR positions for one row are adjacent. Wave-local scanning therefore
turns up to 32 competing updates into one partial. Equal boundary partials are
then merged across the four waves in a 128-thread block. The skewed long-row
input makes the atomic baseline especially costly because many blocks
concurrently update the same long row.

The segmented schedule is nevertheless slower than a row-owned mapping in all
six workloads. It must recover row keys from positions, synchronize the block,
and atomically flush segments split at block boundaries. Row ownership avoids
all three costs. It also remains slower than rocSPARSE in all six workloads,
so this result validates a reusable position-space primitive rather than
replacing the existing SpMV mapping policy.

Compared with the earlier full-block LDS scan, the hierarchical implementation
is 3.24x faster on skewed 32-NNZ rows and 2.04x faster on skewed 128-NNZ rows.
It is 1.34x to 1.67x slower on the other four workloads. These are separate
runs with unlocked clocks, so they are directional rather than a controlled
kernel A/B. The result nevertheless shows the tradeoff: removing block-wide
barriers helps irregular long segments, while the extra wave and boundary
bookkeeping is not repaid by short or uniform segments.

## SpMM axis order

The SpMM block-only sweep used 4,096-by-4,096 generated matrices, 5 warmup
dispatches, and 20 measured dispatches. Each table entry is
`atomic / segmented (speedup)`. Uniform inputs cover mean row lengths 4, 32,
and 128. This rerun focuses on uniform inputs because the earlier skewed result
already established the key-locality effect and long sequential profiler
sweeps are not stable on this machine.

| Distribution | NNZ/row | RHS | Position-major | RHS-major |
| --- | ---: | ---: | ---: | ---: |
| Uniform | 4 | 4 | 16.32 / 24.08 us (0.68x) | 16.00 / 21.20 us (0.75x) |
| Uniform | 4 | 16 | 33.68 / 51.48 us (0.65x) | 44.14 / 46.54 us (0.95x) |
| Uniform | 32 | 4 | 166.46 / 240.49 us (0.69x) | 574.63 / 71.34 us (8.05x) |
| Uniform | 32 | 16 | 438.65 / 647.10 us (0.68x) | 2,328.58 / 265.49 us (8.77x) |
| Uniform | 128 | 4 | 2,225.75 / 2,693.84 us (0.83x) | 6,561.85 / 240.25 us (27.31x) |
| Uniform | 128 | 16 | 6,406.70 / 7,497.66 us (0.85x) | 24,598.33 / 934.42 us (26.32x) |

The output key for SpMM is `(row, RHS column)`. Position-major order changes
the RHS column at adjacent collapsed iterations, so equal keys rarely form a
useful segment. Segmentation is then 15% to 35% slower: wave and boundary
bookkeeping remains while few atomics disappear.

RHS-major order fixes one RHS column while adjacent iterations traverse sparse
positions. Positions from the same row now share an output key, exposing the
same contiguous segments as SpMV. This makes segmented reduction 8.05x to
27.31x faster on uniform medium and long rows. The result demonstrates that
iteration order and reduction strategy must be selected together;
segmentation alone is not a generally beneficial mapping.

## Comparison with fixed SpMM mappings

The table compares the fastest block-segmented order with thread-per-output,
wave-per-row-tile with tile 4, and rocSPARSE on the same uniform workloads.
`Best fixed` selects the faster SparseWave mapping. `vs rocSPARSE` is
rocSPARSE divided by block segmented, so values below 1.0 mean that the block
schedule is slower.

| NNZ/row | RHS | Best block segmented | Best fixed | rocSPARSE | vs rocSPARSE |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 4 | 21.20 us | 10.48 us | 22.82 us | 1.08x |
| 4 | 16 | 46.54 us | 19.84 us | 21.12 us | 0.45x |
| 32 | 4 | 71.34 us | 21.24 us | 23.04 us | 0.32x |
| 32 | 16 | 265.49 us | 46.88 us | 19.82 us | 0.07x |
| 128 | 4 | 240.25 us | 30.38 us | 40.68 us | 0.17x |
| 128 | 16 | 934.42 us | 112.48 us | 40.18 us | 0.04x |

Block segmentation removes the dominant contention inside the position
schedule but does not make it competitive with output- or row-owned SpMM.
The 4-NNZ, RHS-4 case is 1.08x faster than rocSPARSE in this run, but the
thread-per-output mapping is another 2.02x faster, so it does not justify
selecting block position scheduling for that workload.
Every block still reconstructs logical coordinates and may emit another
partial atomic for the same output. The scan also executes for all blocks even
when a simpler ownership mapping can retain the complete sum in registers.

## GPU resources and ISA

Resource counts are invariant across each measured generated shape. None of
the kernels spill registers or use scratch memory.

| Operation and order | Reduction | VGPR | SGPR | LDS | Scratch | Spills |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| SpMV | Atomic | 20 | 33 | 0 B | 0 B | 0 |
| SpMV | Segmented | 23 | 40 | 112 B | 0 B | 0 |
| SpMM position-major | Atomic | 29 | 46 | 0 B | 0 B | 0 |
| SpMM position-major | Segmented | 26 | 40 | 112 B | 0 B | 0 |
| SpMM RHS-major | Atomic | 31 | 46 | 0 B | 0 B | 0 |
| SpMM RHS-major | Segmented | 28 | 40 | 112 B | 0 B | 0 |

For 128 threads in Wave32 mode, the segmented kernel allocates two boundary
slots per wave plus one segment count per wave. Four waves therefore require
`8 * (8-byte key + 4-byte value) + 4 * 4-byte count = 112` bytes of LDS. The
five prefix steps at offsets 1 through 16 use wave shuffles. Only one barrier
is needed before thread zero merges the staged wave boundaries. This replaces
the earlier 2,048-byte full-block staging and 16 static barriers.

Static disassembly of representative uniform, 128-NNZ/row kernels gives:

| Kernel | Reduction | Instructions | Atomic sites | Barriers | LDS load/store sites |
| --- | --- | ---: | ---: | ---: | ---: |
| SpMV | Atomic | 541 | 2 | 0 | 0 |
| SpMV | Segmented | 1,105 | 3 | 1 | 8 |
| SpMM RHS-major, RHS 16 | Atomic | 920 | 2 | 0 | 0 |
| SpMM RHS-major, RHS 16 | Segmented | 1,478 | 3 | 1 | 8 |

The atomic baseline reaches its update path once per contribution. The three
segmented sites separately handle wave-interior segments, completed boundary
segments, and the final boundary accumulator. Each is reached only for the
corresponding adjacent equal-key segment. The hierarchical kernel has fewer
LDS operations and barriers but more static instructions for wave scans and
boundary classification; performance improves only when enough contributions
collapse into each dynamic atomic.

During measurement, inputs with many short segments exposed a correctness
issue in the shared wave-prefix helper: the scan relied on the `valid` result
of `gpu.shuffle` as its sole lane-boundary predicate. That assumption was not
sufficient on the tested AMD path, allowing out-of-range shuffle values into
the segment ordinal and partial sum. The implementation no longer depends on
that target behavior: it derives predecessor availability explicitly from
`lane >= offset` at every scan step. A full-block runtime regression with 32
four-entry rows covers all shuffle offsets.

## Conclusions

- Block-local segmentation is effective when the collapsed iteration order
  produces long adjacent runs of the same output key.
- SpMV naturally has this property in row-ordered position space.
- SpMM needs RHS-major order to expose it; position-major usually cannot repay
  the fixed scan cost.
- Long or skewed rows amplify the benefit by increasing both segment length
  and atomic contention.
- Fixed ownership remains preferable when the compiler can assign an entire
  output reduction to one thread, wave, or block.

These conditions provide a concrete input to a future mapping decision stage:
select block segmentation from expected key locality and contention, rather
than from the operator or storage-format name.

## Reproduction

Run the SpMV sweep with:

```sh
python3 benchmark/run_spmv_benchmark.py \
  --formats=csr \
  --mappings=thread-per-row,wave-per-row,block-per-row,wave-per-position,block-per-position \
  --rows=8192 \
  --columns=8192 \
  --nnz-per-row=4,32,128 \
  --distributions=uniform,skewed \
  --position-reductions=atomic,segmented,workspace \
  --block-sizes=128 \
  --warmup=10 \
  --iterations=50 \
  --chip=gfx1101 \
  --wave-size=32 \
  --rocsparse \
  --keep-artifacts
```

Run the uniform SpMM block comparison with:

```sh
python3 benchmark/run_spmm_benchmark.py \
  --formats=csr \
  --mappings=block-per-position \
  --rows=4096 \
  --columns=4096 \
  --nnz-per-row=4,32,128 \
  --distributions=uniform \
  --rhs-columns=4,16 \
  --position-orders=position-major,rhs-major \
  --position-reductions=atomic,segmented \
  --block-sizes=128 \
  --warmup=5 \
  --iterations=20 \
  --chip=gfx1101 \
  --wave-size=32 \
  --keep-artifacts
```

Run the matching fixed-mapping and rocSPARSE baselines with:

```sh
python3 benchmark/run_spmm_benchmark.py \
  --formats=csr \
  --mappings=thread-per-output,wave-per-row-tile \
  --rows=4096 \
  --columns=4096 \
  --nnz-per-row=4,32,128 \
  --distributions=uniform \
  --rhs-columns=4,16 \
  --tile-sizes=4 \
  --block-sizes=128 \
  --warmup=5 \
  --iterations=20 \
  --chip=gfx1101 \
  --wave-size=32 \
  --rocsparse \
  --keep-artifacts
```
