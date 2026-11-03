# Block-local segmented position reduction on gfx1101

This report evaluates SparseWave's block-local segmented reduction for the
generic position scheduler. A block owns one contiguous partition of the
collapsed position domain. The atomic baseline updates the output once per
position; the segmented strategy stages one keyed contribution per thread in
LDS, scans adjacent equal keys, and lets only each local segment end perform an
atomic update.

The experiment asks when the reduction in dynamic atomics repays the LDS scan
and synchronization cost. It covers both SpMV and SpMM because the scheduler
operates on generic keyed contributions rather than named operators.

## Environment

| Component | Value |
| --- | --- |
| Date | 2026-11-03 |
| GPU | AMD Radeon RX 7800 XT |
| Target chip | gfx1101 |
| Wavefront size | 32 |
| SparseWave commit | `88b2716814a621c805f4570f2caa2925f7cb4ae1` |
| LLVM commit | `64b593c2371b7f7225b0ec190a37cd8b672e4c5d` |
| ROCm version | 6.4.4-129 |
| rocSPARSE version | 300400, Git revision `8fbfc797` |
| GPU performance level | Automatic |
| Block size | 128 |

SparseWave times are the sum of GPU kernel timestamps reported by
`rocprofv3`; initialization is included for position mappings. rocSPARSE uses
HIP events around the complete compute call. GPU clocks were not locked, so
the results support comparisons within these runs rather than cross-run
absolute-latency claims. All 106 recorded configurations passed CPU-reference
validation.

## SpMV

The SpMV sweep used 8,192-by-8,192 generated matrices, 10 warmup dispatches,
and 50 measured dispatches. `uniform` gives every row the requested length.
`skewed` gives seven rows one entry and the eighth row `8N - 7` entries while
preserving mean row length `N`.

`Best fixed` is the minimum of thread-, wave-, and block-per-row. Speedup is
the block atomic median divided by the block segmented median.

| Distribution | Mean NNZ/row | Block atomic | Block segmented | Speedup | Best fixed | rocSPARSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Uniform | 4 | 11.02 us | 10.68 us | 1.03x | 4.56 us | 12.80 us |
| Uniform | 32 | 295.60 us | 29.96 us | 9.87x | 7.84 us | 16.08 us |
| Uniform | 128 | 3,809.37 us | 92.86 us | 41.02x | 14.60 us | 24.52 us |
| Skewed | 4 | 31.40 us | 11.04 us | 2.84x | 5.64 us | 13.44 us |
| Skewed | 32 | 1,349.85 us | 131.96 us | 10.23x | 13.80 us | 93.44 us |
| Skewed | 128 | 66,565.47 us | 258.60 us | **257.41x** | 121.98 us | 50.08 us |

The benefit grows with contiguous same-row runs. In SpMV the output key is the
row, and CSR positions for one row are adjacent. A 128-thread block therefore
turns up to 128 competing row updates into one local segment update. The
skewed long-row input makes the atomic baseline especially costly because many
blocks concurrently update the same long row.

The segmented schedule is nevertheless slower than a row-owned mapping in all
six workloads. It must recover row keys from positions, synchronize the block,
and atomically flush segments split at block boundaries. Row ownership avoids
all three costs. It also remains slower than rocSPARSE for medium and long
rows, so this result validates a reusable position-space primitive rather than
replacing the existing SpMV mapping policy.

## SpMM axis order

The SpMM block-only sweep used 4,096-by-4,096 generated matrices, 5 warmup
dispatches, and 20 measured dispatches. Each table entry is
`atomic / segmented (speedup)`. Uniform inputs cover mean row lengths 4, 32,
and 128. The skewed sweep was limited to 4 and 32 entries per row after long
sequential profiler sweeps caused repeated GPU hangs; no incomplete result is
used below.

| Distribution | NNZ/row | RHS | Position-major | RHS-major |
| --- | ---: | ---: | ---: | ---: |
| Uniform | 4 | 4 | 44.22 / 72.68 us (0.61x) | 65.92 / 59.92 us (1.10x) |
| Uniform | 4 | 16 | 155.92 / 163.40 us (0.95x) | 180.70 / 118.14 us (1.53x) |
| Uniform | 32 | 4 | 617.76 / 747.80 us (0.83x) | 1,598.63 / 142.38 us (11.23x) |
| Uniform | 32 | 16 | 1,124.72 / 1,187.78 us (0.95x) | 4,648.76 / 392.42 us (11.85x) |
| Uniform | 128 | 4 | 4,119.84 / 4,221.12 us (0.98x) | 10,778.43 / 334.38 us (32.23x) |
| Uniform | 128 | 16 | 9,817.70 / 9,067.48 us (1.08x) | 32,092.46 / 1,003.67 us (31.98x) |
| Skewed | 4 | 4 | 130.67 / 158.59 us (0.82x) | 175.23 / 41.66 us (4.21x) |
| Skewed | 4 | 16 | 243.42 / 276.86 us (0.88x) | 1,204.36 / 145.47 us (8.28x) |
| Skewed | 32 | 4 | 4,844.79 / 4,776.32 us (1.01x) | 12,085.69 / 169.37 us (71.36x) |
| Skewed | 32 | 16 | 11,964.72 / 10,494.83 us (1.14x) | 41,690.96 / 567.74 us (73.43x) |

The output key for SpMM is `(row, RHS column)`. Position-major order changes
the RHS column at adjacent collapsed iterations, so equal keys rarely form a
useful segment. Segmentation is then between 18% faster and 39% slower: the
LDS scan and barriers remain while few atomics disappear.

RHS-major order fixes one RHS column while adjacent iterations traverse sparse
positions. Positions from the same row now share an output key, exposing the
same contiguous segments as SpMV. This makes segmented reduction 11.23x to
32.23x faster on uniform medium and long rows, and up to 73.43x faster on the
measured skewed inputs. The result demonstrates that iteration order and
reduction strategy must be selected together; segmentation alone is not a
generally beneficial mapping.

## Comparison with fixed SpMM mappings

The table compares the fastest block-segmented order with thread-per-output,
wave-per-row-tile with tile 4, and rocSPARSE on the same uniform workloads.
`Best fixed` selects the faster SparseWave mapping. `vs rocSPARSE` is
rocSPARSE divided by block segmented, so values below 1.0 mean that the block
schedule is slower.

| NNZ/row | RHS | Best block segmented | Best fixed | rocSPARSE | vs rocSPARSE |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 4 | 59.92 us | 14.22 us | 28.42 us | 0.47x |
| 4 | 16 | 118.14 us | 14.42 us | 26.40 us | 0.22x |
| 32 | 4 | 142.38 us | 16.48 us | 26.42 us | 0.19x |
| 32 | 16 | 392.42 us | 30.70 us | 25.58 us | 0.07x |
| 128 | 4 | 334.38 us | 11.12 us | 39.86 us | 0.12x |
| 128 | 16 | 1,003.67 us | 28.32 us | 41.60 us | 0.04x |

Block segmentation removes the dominant contention inside the position
schedule but does not make it competitive with output- or row-owned SpMM.
Every block still reconstructs logical coordinates and may emit another
partial atomic for the same output. The scan also executes for all blocks even
when a simpler ownership mapping can retain the complete sum in registers.

## GPU resources and ISA

Resource counts are invariant across each measured generated shape. None of
the kernels spill registers or use scratch memory.

| Operation and order | Reduction | VGPR | SGPR | LDS | Scratch | Spills |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| SpMV | Atomic | 20 | 33 | 0 B | 0 B | 0 |
| SpMV | Segmented | 22 | 27 | 2,048 B | 0 B | 0 |
| SpMM position-major | Atomic | 29 | 46 | 0 B | 0 B | 0 |
| SpMM position-major | Segmented | 26 | 34 | 2,048 B | 0 B | 0 |
| SpMM RHS-major | Atomic | 31 | 46 | 0 B | 0 B | 0 |
| SpMM RHS-major | Segmented | 28 | 34 | 2,048 B | 0 B | 0 |

For 128 threads, the segmented kernel allocates an i64 key, f32 value, and
i32 segment flag per participant: `128 * (8 + 4 + 4) = 2,048` bytes of LDS.
It performs a seven-step inclusive segmented scan at offsets 1 through 64.
The initial staging and flag construction add two barriers, and each scan step
adds a read/write barrier pair, producing 16 static barriers.

Static disassembly of representative uniform, 128-NNZ/row kernels gives:

| Kernel | Reduction | Instructions | Atomic sites | Barriers | LDS load/store sites |
| --- | --- | ---: | ---: | ---: | ---: |
| SpMV | Atomic | 477 | 1 | 0 | 0 |
| SpMV | Segmented | 752 | 1 | 16 | 29 |
| SpMM RHS-major, RHS 16 | Atomic | 873 | 1 | 0 | 0 |
| SpMM RHS-major, RHS 16 | Segmented | 1,149 | 1 | 16 | 29 |

The single static atomic site in both variants has different dynamic behavior.
The atomic baseline reaches it once per contribution. The segmented kernel
reaches it only at the end of an adjacent equal-key segment. Its static kernel
is larger because it adds LDS traffic and synchronization; performance
improves only when enough contributions collapse into each dynamic atomic.

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
