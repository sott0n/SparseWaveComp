# SpMV segmented position reduction on gfx1101

This report evaluates chunk-local segmented reduction in SparseWave's CSR
`thread-per-position` SpMV schedule. The previous split-only schedule recovers
the CSR row and atomically updates the output for every stored position. The
segmented schedule instead recovers the first row once, carries it across the
chunk, accumulates consecutive products for the same row, and atomically
flushes only at a row boundary or the end of the chunk.

The experiment asks whether eliminating repeated row searches and per-position
atomics makes position splitting competitive with wave- and row-owned
schedules. The results are a snapshot for the revisions below, not a general
performance guarantee.

## Environment

| Component | Value |
| --- | --- |
| Date | 2026-08-21 |
| GPU | AMD Radeon RX 7800 XT |
| Target chip | gfx1101 |
| Wavefront size | 32 |
| SparseWave commit | `5abb51cc1f88056d994dffa6440b6ac1e64209be` |
| LLVM commit | `64b593c2371b7f7225b0ec190a37cd8b672e4c5d` |
| ROCm version | 6.4.4-129 |
| rocSPARSE version | 300400, Git revision `8fbfc797` |
| rocSPARSE algorithm | Default CSR SpMV |
| Warmup dispatches | 10 |
| Measured dispatches | 50 |

SparseWave times are kernel timestamps reported by `rocprofv3`.
`thread-per-position` and `wave-per-position` include both output
initialization and accumulation kernels. Row mappings overwrite each output
element in one kernel. The rocSPARSE runner uses HIP events around the complete
compute call. Compilation, input generation, serialization, and preprocessing
are excluded from steady-state time. All 441 measured configurations passed
CPU-reference validation.

## Workloads and method

Every matrix has 65,536 rows and columns. Each distribution preserves the
requested mean row length:

- `uniform`: every row has the mean length;
- `alternating`: rows alternate between 1 and `2 * mean - 1` entries;
- `skewed`: seven rows have 1 entry, followed by one row with
  `8 * mean - 7` entries.

Atomic and segmented reduction were measured for chunk sizes 1, 2, 4, and 8
at block sizes 64, 128, 256, and 512. Each entry below independently selects
the lowest median block size for the reduction strategy. The speedup is the
best atomic median divided by the best segmented median.

| Distribution | Mean | Best atomic | Best segmented | Segmented speedup |
| --- | ---: | ---: | ---: | ---: |
| uniform | 4 | 34.28 us (c2, b256) | 18.76 us (c4, b256) | 1.83x |
| uniform | 32 | 230.54 us (c8, b64) | 76.96 us (c8, b512) | 3.00x |
| uniform | 256 | 18,150.62 us (c8, b128) | 2,355.04 us (c8, b128) | 7.71x |
| alternating | 4 | 37.04 us (c2, b64) | 18.76 us (c8, b128) | 1.97x |
| alternating | 32 | 410.82 us (c8, b64) | 103.12 us (c8, b64) | 3.98x |
| alternating | 256 | 25,587.32 us (c8, b64) | 3,891.23 us (c8, b64) | 6.58x |
| skewed | 4 | 48.94 us (c8, b128) | 26.36 us (c8, b128) | 1.86x |
| skewed | 32 | 2,304.24 us (c8, b128) | 362.32 us (c8, b512) | 6.36x |
| skewed | 256 | 67,464.93 us (c8, b64) | 8,330.89 us (c8, b64) | 8.10x |

Segmented reduction improves the best atomic configuration in every workload.
The advantage grows with row length because a larger fraction of each chunk
stays in one row. Short rows improve by 1.83x to 1.97x, while the longest rows
improve by 6.58x to 8.10x.

## Effect of chunk size

The table below reports the segmented-to-atomic speedup at each chunk size,
again selecting the best block size independently for each strategy. Values
below 1.0x mean that segmentation is slower.

| Distribution | Mean | Chunk 1 | Chunk 2 | Chunk 4 | Chunk 8 |
| --- | ---: | ---: | ---: | ---: | ---: |
| uniform | 4 | 0.96x | 1.49x | 1.97x | 4.60x |
| uniform | 32 | 1.01x | 2.02x | 3.49x | 3.00x |
| uniform | 256 | 1.01x | 1.90x | 3.46x | 7.71x |
| alternating | 4 | 0.95x | 1.30x | 1.87x | 2.11x |
| alternating | 32 | 1.00x | 1.92x | 3.61x | 3.98x |
| alternating | 256 | 1.01x | 2.02x | 3.53x | 6.58x |
| skewed | 4 | 0.99x | 1.62x | 1.85x | 1.86x |
| skewed | 32 | 1.00x | 1.85x | 3.70x | 6.36x |
| skewed | 256 | 1.01x | 2.04x | 4.56x | 8.10x |

Chunk 1 cannot combine two products, so segmentation provides no reduction in
dynamic atomic updates and is up to 5% slower from its additional control
flow. Larger chunks expose reuse: when a chunk remains in one row, one atomic
update replaces one update per position. Chunk 8 is best for every medium- and
long-row workload. Uniform four-entry rows select chunk 4 because it exactly
matches the row length; a chunk of 8 crosses two rows and still requires two
segment flushes.

## Mapping comparison

The comparison below independently selects the fastest block size for the best
atomic and segmented chunks, `wave-per-position`, and the three row mappings.
Each timing is median / p95 in microseconds.

| Distribution | Mean | Best atomic | Best segmented | Wave position | Best row mapping | rocSPARSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| uniform | 4 | 34.28 / 34.64 | 18.76 / 19.08 | 34.32 / 34.64 | 8.12 / 8.28 (thread) | 38.70 / 39.48 |
| uniform | 32 | 230.54 / 231.56 | 76.96 / 77.68 | 204.02 / 205.00 | 38.82 / 39.00 (wave) | 35.92 / 37.60 |
| uniform | 256 | 18,150.62 / 18,481.99 | 2,355.04 / 2,379.06 | 2,023.91 / 2,032.55 | 259.28 / 266.88 (wave) | 261.16 / 266.36 |
| alternating | 4 | 37.04 / 37.36 | 18.76 / 19.12 | 33.76 / 96.24 | 9.92 / 10.08 (thread) | 16.78 / 17.16 |
| alternating | 32 | 410.82 / 413.64 | 103.12 / 103.60 | 204.20 / 204.64 | 49.36 / 49.52 (wave) | 36.24 / 39.16 |
| alternating | 256 | 25,587.32 / 25,826.10 | 3,891.23 / 3,918.11 | 2,308.96 / 2,317.22 | 379.78 / 392.52 (wave) | 533.24 / 534.24 |
| skewed | 4 | 48.94 / 155.64 | 26.36 / 26.80 | 33.60 / 96.04 | 24.62 / 75.36 (thread) | 45.88 / 47.20 |
| skewed | 32 | 2,304.24 / 2,327.22 | 362.32 / 367.60 | 226.58 / 227.60 | 60.90 / 171.08 (wave) | 138.84 / 139.20 |
| skewed | 256 | 67,464.93 / 68,060.99 | 8,330.89 / 8,594.20 | 5,677.49 / 5,703.01 | 304.08 / 305.68 (block) | 321.94 / 332.92 |

Segmented reduction makes the position split faster than
`wave-per-position` for all short-row workloads and for uniform and
alternating 32-entry rows. It does not win on skewed medium rows or any
long-row workload. A row-owned mapping remains the fastest SparseWave strategy
for every measured workload because it avoids output atomics entirely and does
not recover rows from positions.

The segmented schedule is faster than the measured rocSPARSE baseline for the
uniform and skewed four-entry workloads, but it is not the fastest SparseWave
mapping in either case. The result is therefore evidence for the compiler
transformation, not evidence that position-space SpMV should replace the
existing row-owned mappings.

## GPU resources and ISA

The compute kernels use no scratch memory or register spills. Atomic kernels
use 22 VGPRs and 31 SGPRs. Segmented kernels use 20 VGPRs and 23 to 25 SGPRs,
depending on block and chunk size.

Static disassembly of the uniform-256, chunk-8, block-128 compute kernels gives:

| Reduction | Instructions | Atomic compare-and-swap sites | Scalar branches | VGPR | SGPR |
| --- | ---: | ---: | ---: | ---: | ---: |
| atomic | 602 | 1 | 10 | 22 | 31 |
| segmented | 525 | 2 | 11 | 20 | 25 |

The number of static atomic sites does not represent the number of atomic
operations executed. The atomic schedule has one site inside the position
loop and executes it once per stored position. The segmented schedule has one
site for a row-boundary flush and one for the final chunk flush, but executes
them only once per segment. It also performs one binary CSR row search at the
start of the chunk instead of one search per position. This explains why its
static kernel is smaller and why the performance improvement grows with chunk
size despite the additional boundary branch.

The remaining gap to row-owned mappings comes from work that segmentation
cannot remove: position workers must still recover their starting row, chunks
can split one row across multiple workers, and different workers still issue
atomics to the same output row. Larger chunks and carrying the next row
boundary in SSA registers are useful follow-up experiments, but should remain
explicit scheduling choices rather than replacing the current mappings.

## Reproduction

Run the complete sweep with:

```sh
python3 benchmark/run_spmv_benchmark.py \
  --chip=gfx1101 \
  --rows=65536 \
  --columns=65536 \
  --nnz-per-row=4,32,256 \
  --distributions=uniform,alternating,skewed \
  --position-chunk-sizes=1,2,4,8 \
  --position-reductions=atomic,segmented \
  --rocsparse \
  --block-sizes=64,128,256,512 \
  --warmup=10 \
  --iterations=50 \
  --keep-artifacts
```

Inspect a generated compute kernel with:

```sh
/opt/rocm/llvm/bin/llvm-objdump \
  -d \
  --no-show-raw-insn \
  --disassemble-symbols=spmv_kernel \
  /path/to/kernel.hsaco
```
