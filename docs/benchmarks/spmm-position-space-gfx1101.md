# Position-space SpMM baseline on gfx1101

This report validates that SparseWave's position-space scheduling mechanism is
reusable beyond SpMV. It also establishes the initial performance baseline for
a flattened CSR SpMM schedule on an AMD Radeon RX 7800 XT. The result is a
research baseline, not a claim that position-space SpMM is currently faster
than the existing row-owned mappings.

## Schedule

The decomposition describes sparse-position and dense-column axes in a rank-2
keyed reduction. The generic position scheduler collapses them into one logical
worker space:

```text
linear iteration i in [0, NNZ * rhsColumns)
                 |
                 +-- position = i / rhsColumns
                 +-- rhsColumn = i % rhsColumns
                 |
                 +-- recover CSR row(position)
                 +-- column = columnIndices[position]
                 +-- product = values[position] * rhs[column, rhsColumn]
                 +-- atomic add output[row, rhsColumn]
```

The output is viewed as a flat row-major buffer, so the keyed reduction uses
`row * rhsColumns + rhsColumn`. The existing operator-independent
`sparsewave.position_reduce`, thread position scheduler, chunking mechanism,
and GPU work-distribution lowering are reused without an SpMM-specific GPU
pattern.

The mapping launches one output-initialization kernel and one accumulation
kernel per dispatch. Reported position-space times include both kernels. The
existing mappings overwrite their output in one kernel.

## Environment

| Component | Value |
| --- | --- |
| Date | 2026-08-22 |
| GPU | AMD Radeon RX 7800 XT |
| Target chip | gfx1101 |
| Wavefront size | 32 |
| SparseWave base commit | `aca8b90db646984a734037a12e955891d0028d26` plus this change |
| LLVM commit | `64b593c2371b7f7225b0ec190a37cd8b672e4c5d` |
| ROCm version | 6.4.4-129 |
| Matrix | SuiteSparse SNAP/ca-GrQc |
| Shape | 5,242 x 5,242, 28,980 NNZ |
| Warmup dispatches | 10 |
| Measured dispatches | 50 |

All 24 measured configurations passed the benchmark's CPU-reference output
validation.

## Results

The table reports the best block size for each mapping and configuration.

| RHS columns | Mapping | Tile/chunk | Best block | Median | p95 | GFLOP/s |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 8 | thread-per-output | - | 256 | 40.76 us | 43.16 us | 11.38 |
| 8 | wave-per-row-tile | tile 8 | 64 | **27.22 us** | 27.80 us | **17.03** |
| 8 | thread-per-position | chunk 1 | 256 | 165.56 us | 174.00 us | 2.80 |
| 8 | thread-per-position | chunk 4 | 256 | 395.87 us | 428.85 us | 1.17 |
| 8 | thread-per-position | chunk 8 | 64 | 494.23 us | 522.33 us | 0.94 |
| 32 | thread-per-output | - | 256 | **53.72 us** | 57.36 us | **34.53** |
| 32 | wave-per-row-tile | tile 16 | 256 | 53.80 us | 74.64 us | 34.47 |
| 32 | thread-per-position | chunk 1 | 256 | 280.60 us | 415.32 us | 6.61 |
| 32 | thread-per-position | chunk 4 | 256 | 292.98 us | 301.88 us | 6.33 |
| 32 | thread-per-position | chunk 8 | 256 | 476.95 us | 501.29 us | 3.89 |

The best position-space result is 6.08x slower than the best existing mapping
at 8 RHS columns and 5.22x slower at 32 RHS columns. Chunk 1 is the fastest
position configuration at both widths.

## Resource behavior

Every position chunk uses 27 VGPRs and 36 SGPRs, with no register spills, LDS,
or scratch memory. The unchanged resource counts show that chunking changes a
runtime loop bound rather than statically unrolling the body. The slowdown at
larger chunks therefore comes from reducing the number of parallel workers and
serializing more position-column pairs per thread, not from additional static
register pressure.

For comparison, `thread-per-output` uses 20 VGPRs and 20 SGPRs. The tiled
mapping uses 55 VGPRs and 54 SGPRs at tile 8, and 95 VGPRs and 86 SGPRs at tile
16; none of these configurations spill.

## Interpretation

The experiment confirms the architectural objective: the same keyed position
reduction and scheduling pass now lower both SpMV and SpMM. It also isolates
why the direct extension is not yet competitive:

- every sparse-dense product performs an atomic output update;
- each flattened iteration independently recovers its CSR row;
- position-major ordering interleaves output columns, so adjacent iterations
  do not share a reduction key and cannot use the existing segmented sum;
- a larger chunk reduces GPU parallelism without explicitly retaining and
  reusing the sparse value or recovered row across RHS columns.

The next SpMM position-space optimization should change the derived schedule,
not merely increase the chunk size. Candidate transformations are reordering
the logical axes to expose row segments, grouping multiple RHS columns under
one recovered sparse position, or adding a local accumulator before the atomic
boundary.

## Reproduction

```sh
python3 benchmark/run_spmm_benchmark.py \
  --chip=gfx1101 \
  --matrix=/path/to/ca-GrQc.mtx \
  --formats=csr \
  --mappings=thread-per-output,wave-per-row-tile,thread-per-position \
  --rhs-columns=8,32 \
  --block-sizes=64,256 \
  --tile-sizes=8,16 \
  --position-chunk-sizes=1,4,8 \
  --warmup=10 \
  --iterations=50
```
