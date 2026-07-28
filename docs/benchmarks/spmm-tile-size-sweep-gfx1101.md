# SpMM tile-size sweep on gfx1101

This report records a SparseWave SpMM mapping experiment on an AMD Radeon
RX 7800 XT. It is a snapshot for the compiler and software revisions below,
not a general performance guarantee.

## Environment

| Component | Value |
| --- | --- |
| Date | 2026-07-28 |
| GPU | AMD Radeon RX 7800 XT |
| Target chip | gfx1101 |
| Wavefront size | 32 |
| SparseWave commit | `4aa7a658512cb880847fe1e262cc5b5245f7aa26` |
| LLVM commit | `64b593c2371b7f7225b0ec190a37cd8b672e4c5d` |
| ROCm version | 6.4.4-129 |
| Warmup dispatches | 10 |
| Measured dispatches | 50 |

The sweep compares the `thread-per-output` baseline with
`wave-per-row-tile`. RHS column counts are 8, 32, and 128; block sizes are 64
and 256; and tile sizes are 1, 2, 4, 8, and 16. Every reported configuration
passed the benchmark's output validation.

## Matrices

The input matrices come from the
[SuiteSparse Matrix Collection](https://sparse.tamu.edu/).

| Matrix | Rows | Columns | NNZ | Mean NNZ/row | Characteristic |
| --- | ---: | ---: | ---: | ---: | --- |
| [SNAP/ca-GrQc](https://sparse.tamu.edu/SNAP/ca-GrQc) | 5,242 | 5,242 | 28,980 | 5.53 | Small, irregular graph |
| [SNAP/ca-AstroPh](https://sparse.tamu.edu/SNAP/ca-AstroPh) | 18,772 | 18,772 | 396,160 | 21.10 | Longer graph rows |
| [Williams/mac_econ_fwd500](https://sparse.tamu.edu/Williams/mac_econ_fwd500) | 206,500 | 206,500 | 1,273,389 | 6.17 | Large, short-row economic model |

## Reproduction

Run the following command for each Matrix Market file:

```sh
python3 benchmark/run_spmm_benchmark.py \
  --chip=gfx1101 \
  --matrix=/path/to/matrix.mtx \
  --rhs-columns=8,32,128 \
  --block-sizes=64,256 \
  --tile-sizes=1,2,4,8,16 \
  --warmup=10 \
  --iterations=50
```

## Results

The values below are median kernel times in microseconds. Each cell is the
better result from block sizes 64 and 256 for that mapping and tile size. Bold
marks the fastest configuration in each row.

| Matrix | RHS columns | Thread baseline | Tile 1 | Tile 2 | Tile 4 | Tile 8 | Tile 16 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ca-GrQc | 8 | 40.78 | 99.32 | 61.52 | 37.42 | **27.34** | 47.90 |
| ca-GrQc | 32 | **46.16** | 372.11 | 163.71 | 127.25 | 56.20 | 72.86 |
| ca-GrQc | 128 | **36.86** | 454.02 | 269.15 | 151.96 | 96.40 | 82.18 |
| ca-AstroPh | 8 | 263.65 | 355.93 | 215.19 | 127.70 | **86.46** | 176.47 |
| ca-AstroPh | 32 | 366.19 | 1,053.64 | 592.94 | 319.01 | 259.53 | **167.69** |
| ca-AstroPh | 128 | 326.55 | 1,730.81 | 1,016.53 | 570.22 | 361.23 | **309.99** |
| mac_econ_fwd500 | 8 | **341.07** | 5,384.01 | 3,125.82 | 1,542.95 | 883.61 | 1,583.77 |
| mac_econ_fwd500 | 32 | **569.54** | 4,392.12 | 2,601.79 | 1,457.04 | 904.10 | 772.90 |
| mac_econ_fwd500 | 128 | **1,377.22** | 17,642.87 | 10,498.99 | 5,887.82 | 3,672.27 | 3,033.27 |

The fastest baseline and tiled configurations, including their block sizes,
are:

| Matrix | RHS | Best baseline | Best tiled | Fastest mapping |
| --- | ---: | --- | --- | --- |
| ca-GrQc | 8 | 40.78 us, block 256 | 27.34 us, tile 8, block 64 | Tiled, 1.49x |
| ca-GrQc | 32 | 46.16 us, block 256 | 56.20 us, tile 8, block 256 | Baseline, 1.22x |
| ca-GrQc | 128 | 36.86 us, block 256 | 82.18 us, tile 16, block 64 | Baseline, 2.23x |
| ca-AstroPh | 8 | 263.65 us, block 256 | 86.46 us, tile 8, block 64 | Tiled, 3.05x |
| ca-AstroPh | 32 | 366.19 us, block 256 | 167.69 us, tile 16, block 256 | Tiled, 2.18x |
| ca-AstroPh | 128 | 326.55 us, block 64 | 309.99 us, tile 16, block 64 | Tiled, 1.05x |
| mac_econ_fwd500 | 8 | 341.07 us, block 64 | 883.61 us, tile 8, block 256 | Baseline, 2.59x |
| mac_econ_fwd500 | 32 | 569.54 us, block 256 | 772.90 us, tile 16, block 256 | Baseline, 1.36x |
| mac_econ_fwd500 | 128 | 1,377.22 us, block 64 | 3,033.27 us, tile 16, block 64 | Baseline, 2.20x |

## Observations

- Tile 8 is the best tiled configuration for 8 RHS columns. Tile 16 is best
  for 32 and 128 RHS columns.
- Tile 16 performs extra guarded work when the RHS has only 8 columns, so it
  loses to tile 8 in that case.
- Tile 4 is not the fastest tiled configuration in any measured row. The
  current default is therefore conservative rather than performance-optimal
  for these inputs.
- `wave-per-row-tile` is most effective on `ca-AstroPh`, whose mean row length
  is substantially larger than the other two matrices.
- `thread-per-output` remains decisively better on the large, short-row
  `mac_econ_fwd500` matrix. A single mapping should not be selected solely from
  RHS width.
- Block size also affects the winner, so tile size and launch configuration
  should eventually be selected together.

## Excluded matrix

[HB/bcsstk17](https://sparse.tamu.edu/HB/bcsstk17) was evaluated but excluded
from the performance tables. Its values span approximately `1e-11` to
`3.9e9` and include cancellation. The first benchmark case reported 174
elements outside the current relative tolerance of `1e-4`. This is consistent
with floating-point accumulation-order and FMA differences between the host
reference and GPU execution, but the failed validation makes its timing
unsuitable for this comparison.
