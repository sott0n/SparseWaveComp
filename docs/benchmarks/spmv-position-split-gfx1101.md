# SpMV position-split ablation on gfx1101

This report isolates the split factor in SparseWave's CSR
`thread-per-position` SpMV schedule. A worker processes one stored position at
factor 1 and a consecutive chunk of 2, 4, or 8 positions at larger factors.
The experiment asks whether reducing the number of position workers is useful
before adding coordinate reuse or a different reduction strategy.

The results are a snapshot for the revisions below, not a general performance
guarantee.

## Environment

| Component | Value |
| --- | --- |
| Date | 2026-08-21 |
| GPU | AMD Radeon RX 7800 XT |
| Target chip | gfx1101 |
| Wavefront size | 32 |
| SparseWave commit | `60c87146ebf9a51f0f3124c552abbf08889569a4` |
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
are excluded from steady-state time. All 297 measured configurations passed
CPU-reference validation.

## Workloads and method

Every matrix has 65,536 rows and columns. Each distribution preserves the
requested mean row length:

- `uniform`: every row has the mean length;
- `alternating`: rows alternate between 1 and `2 * mean - 1` entries;
- `skewed`: seven rows have 1 entry, followed by one row with
  `8 * mean - 7` entries.

For every distribution and mean row length, the split factors 1, 2, 4, and 8
were measured at block sizes 64, 128, 256, and 512. Each table entry below is
the lowest median for that factor; the selected block size is in parentheses.
The final column reports the speedup of the best factor over factor 1.

| Distribution | Mean | Min / max | Chunk 1 | Chunk 2 | Chunk 4 | Chunk 8 | Best / speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| uniform | 4 | 4 / 4 | 83.80 us (256) | 34.00 us (512) | 38.38 us (512) | 42.92 us (512) | 2 / 2.46x |
| uniform | 32 | 32 / 32 | 2,655.86 us (512) | 1,375.13 us (512) | 442.23 us (512) | 236.57 us (512) | 8 / 11.23x |
| uniform | 256 | 256 / 256 | 126,885.76 us (128) | 56,127.19 us (128) | 26,371.32 us (64) | 17,986.23 us (128) | 8 / 7.05x |
| alternating | 4 | 1 / 7 | 53.96 us (512) | 38.48 us (512) | 38.18 us (256) | 102.44 us (128) | 4 / 1.41x |
| alternating | 32 | 1 / 63 | 3,710.56 us (256) | 2,375.37 us (256) | 1,230.79 us (256) | 415.01 us (256) | 8 / 8.94x |
| alternating | 256 | 1 / 511 | 212,481.90 us (128) | 96,255.21 us (128) | 45,476.53 us (64) | 25,425.30 us (64) | 8 / 8.36x |
| skewed | 4 | 1 / 25 | 209.54 us (512) | 111.28 us (512) | 135.64 us (256) | 97.34 us (256) | 8 / 2.15x |
| skewed | 32 | 1 / 249 | 12,184.81 us (256) | 6,023.69 us (256) | 3,308.47 us (256) | 2,286.61 us (256) | 8 / 5.33x |
| skewed | 256 | 1 / 2,041 | 658,855.66 us (64) | 301,441.22 us (128) | 140,969.36 us (64) | 66,957.84 us (64) | 8 / 9.84x |

Chunking helps every measured workload relative to factor 1. The best factor
is workload-dependent for short rows: uniform rows prefer 2, alternating rows
prefer 4, and skewed rows prefer 8. All medium- and long-row cases select the
largest tested factor, so this experiment does not establish their optimum.
It only establishes that factors through 8 have not yet reached it.

## Mapping comparison

The comparison below independently selects the fastest block size for the best
split, `wave-per-position`, and the three row mappings. Each timing is
median / p95 in microseconds.

| Distribution | Mean | Best split | Wave position | Best row mapping | rocSPARSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| uniform | 4 | 34.00 / 34.52 (c2, b512) | 35.08 / 35.28 (b512) | 16.84 / 17.12 (thread, b256) | 16.88 / 17.20 |
| uniform | 32 | 236.57 / 239.16 (c8, b512) | 203.58 / 204.96 (b256) | 39.44 / 39.72 (wave, b256) | 35.04 / 36.64 |
| uniform | 256 | 17,986.23 / 18,166.98 (c8, b128) | 1,995.55 / 2,001.77 (b64) | 253.93 / 256.97 (block, b64) | 262.41 / 267.76 |
| alternating | 4 | 38.18 / 38.80 (c4, b256) | 34.90 / 101.00 (b512) | 19.36 / 20.04 (thread, b256) | 36.72 / 37.56 |
| alternating | 32 | 415.01 / 418.61 (c8, b256) | 207.16 / 208.44 (b256) | 48.68 / 48.92 (wave, b256) | 36.76 / 94.04 |
| alternating | 256 | 25,425.30 / 25,649.19 (c8, b64) | 2,276.51 / 2,288.13 (b64) | 271.00 / 272.37 (block, b64) | 533.73 / 534.93 |
| skewed | 4 | 97.34 / 99.04 (c8, b256) | 35.36 / 35.84 (b512) | 26.66 / 81.04 (thread, b512) | 20.18 / 20.60 |
| skewed | 32 | 2,286.61 / 2,309.17 (c8, b256) | 231.12 / 685.93 (b128) | 81.18 / 81.40 (wave, b128) | 138.88 / 139.48 |
| skewed | 256 | 66,957.84 / 67,717.61 (c8, b64) | 5,617.38 / 5,655.97 (b64) | 302.46 / 303.64 (block, b64) | 321.54 / 331.72 |

Factor 2 is 1.03x faster than `wave-per-position` for uniform 4-entry
rows. This is the only measured region where the split schedule wins that
comparison, and it remains about 2x slower than both thread-per-row and
rocSPARSE. For longer rows, the segmented wave reduction is 1.16x to 11.92x
faster than the best tested split. A row-owned mapping is fastest among the
SparseWave strategies in every workload.

The result is therefore not evidence that split alone is a competitive SpMV
schedule. It shows that worker granularity is a meaningful transformation and
that its benefit grows when factor 1 creates heavy contention, while preserving
row ownership or combining products before accumulation remains much more
important.

## GPU resources and ISA

All split factors use 22 VGPRs and 31 SGPRs, with no LDS, scratch memory, or
register spills. Static disassembly of the block-256 compute kernels gives:

| Chunk | Instructions | Atomic compare-and-swap sites | Scalar branches |
| ---: | ---: | ---: | ---: |
| 1 | 419 | 1 | 8 |
| 2 | 603 | 1 | 10 |
| 4 | 602 | 1 | 10 |
| 8 | 602 | 1 | 10 |

The factor-1 loop is canonicalized away. Factors 2, 4, and 8 retain the same
runtime chunk loop, so changing its upper bound does not replicate the loop
body or increase register pressure. The floating-point atomic add still
lowers to one static compare-and-swap loop, and every stored position still
performs an atomic update and an independent CSR row search.

The speedup therefore comes from launching fewer workers and serializing a
larger share of nearby updates inside each worker. This reduces simultaneous
contention on the same output row, but does not reduce the total number of
atomic updates or coordinate searches. That explains both the large gain over
factor 1 on long rows and the remaining gap to wave reduction and row-owned
schedules.

The next optimization should carry the recovered CSR row across positions in
a chunk and combine consecutive products for the same row before issuing an
atomic update. The current implementation and measurements provide the
split-only ablation baseline for that change.

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
  --rocsparse \
  --block-sizes=64,128,256,512 \
  --warmup=10 \
  --iterations=50 \
  --keep-artifacts
```

The reported measurements were run one workload per invocation so that a
completed `results.csv` was retained after each long-running case. This does
not change the generated inputs, compilation options, or timing method used by
the combined command above.

Inspect a generated compute kernel with:

```sh
/opt/rocm/llvm/bin/llvm-objdump \
  -d \
  --no-show-raw-insn \
  --disassemble-symbols=spmv_kernel \
  /path/to/kernel.hsaco
```
