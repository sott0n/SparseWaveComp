# Cooperative position-space SpMM on gfx1101

This report evaluates `wave-per-position-tile`, which assigns one sparse
position and one group of RHS columns to a wave. Lane-independent values such
as the sparse value and recovered row are computed by lane zero and broadcast
within the wave. Each active lane computes one RHS-column product and
atomically accumulates it into the output.

The experiment asks whether sharing position work across RHS columns can make
the position-space SpMM schedule competitive with row-owned mappings.

## Environment

| Component | Value |
| --- | --- |
| Date | 2026-08-30 |
| GPU | AMD Radeon RX 7800 XT |
| Target chip | gfx1101 |
| Wavefront size | 32 |
| SparseWave commit | `c0af0a65b792eb56a49fd5767b53d1cda442b848` |
| LLVM commit | `64b593c2371b7f7225b0ec190a37cd8b672e4c5d` |
| ROCm version | 6.4.4-129 |
| rocSPARSE version | 300400, Git revision `8fbfc797` |
| rocSPARSE algorithm | Default CSR SpMM |
| GPU performance level | Automatic |

SparseWave times are the sum of the output-initialization and compute kernels
reported by `rocprofv3`. rocSPARSE times use HIP events around the complete
operation. Each case used 10 warmup and 100 measured dispatches. GPU clocks
were not locked, so the results support large relative differences within
these runs rather than cross-run absolute-latency claims.

CSR conversion took 1.79 ms for `ca-GrQc` and 18.35 ms for `ca-AstroPh`.
rocSPARSE preprocessing took 4.03--6.01 us depending on the matrix and RHS
width. These one-time costs are excluded from the steady-state kernel times.

All configurations in the result tables passed CPU-reference validation.

## Inputs and configurations

| Matrix | Rows | Columns | NNZ | Mean NNZ/row | Maximum NNZ/row |
| --- | ---: | ---: | ---: | ---: | ---: |
| [SNAP/ca-GrQc](https://sparse.tamu.edu/SNAP/ca-GrQc) | 5,242 | 5,242 | 28,980 | 5.53 | 81 |
| [SNAP/ca-AstroPh](https://sparse.tamu.edu/SNAP/ca-AstroPh) | 18,772 | 18,772 | 396,160 | 21.10 | 504 |

The sweep used RHS widths 8, 32, and 128 and block sizes 64 and 256. It tested
tile sizes 8 and 16 for `wave-per-row-tile`, and chunk sizes 4 and 8 with
RHS-major order and segmented reduction for `thread-per-position`. Each table
entry is the lowest median from the tested configurations. Cells show
`median / p95` in microseconds.

## Results

| Matrix | RHS | Thread output | Wave row tile | Thread position | Wave position tile | rocSPARSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ca-GrQc | 8 | 42.14 / 61.16 | **27.36 / 27.92** | 97.55 / 126.43 | 397.73 / 409.05 | 31.52 / 32.00 |
| ca-GrQc | 32 | **18.92 / 19.56** | 24.20 / 65.56 | 105.27 / 106.11 | 122.45 / 123.27 | 24.08 / 24.56 |
| ca-GrQc | 128 | 35.84 / 36.44 | 82.52 / 82.79 | 496.37 / 1,700.76 | 430.15 / 1,352.99 | **31.00 / 45.24** |
| ca-AstroPh | 8 | 260.10 / 276.66 | 93.87 / 95.43 | 761.95 / 769.35 | 10,599.53 / 11,223.70 | **80.45 / 99.63** |
| ca-AstroPh | 32 | 120.01 / 122.47 | 81.08 / 81.83 | 1,348.69 / 1,355.22 | 3,916.98 / 3,956.17 | **46.86 / 54.80** |
| ca-AstroPh | 128 | 323.90 / 922.58 | 308.30 / 309.10 | 6,785.03 / 6,839.39 | 15,498.10 / 15,765.58 | **124.51 / 149.75** |

The best cooperative configurations and effective throughput were:

| Matrix | RHS | Block | Median | GProduct/s | GFLOP/s | vs best existing SparseWave | vs rocSPARSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ca-GrQc | 8 | 256 | 397.73 us | 0.58 | 1.17 | 14.54x slower | 0.08x |
| ca-GrQc | 32 | 64 | 122.45 us | 7.57 | 15.15 | 6.47x slower | 0.20x |
| ca-GrQc | 128 | 64 | 430.15 us | 8.62 | 17.25 | 12.00x slower | 0.07x |
| ca-AstroPh | 8 | 256 | 10,599.53 us | 0.30 | 0.60 | 112.91x slower | 0.01x |
| ca-AstroPh | 32 | 64 | 3,916.98 us | 3.24 | 6.47 | 48.31x slower | 0.01x |
| ca-AstroPh | 128 | 64 | 15,498.10 us | 3.27 | 6.54 | 50.27x slower | 0.01x |

The cooperative mapping does not beat the best existing mapping on either
matrix or any tested RHS width. It does improve on the chunked
`thread-per-position` schedule for `ca-GrQc` at RHS 128 (430.15 vs 496.37 us),
but remains slower than the row-owned mappings in that case.

## Why the cooperative mapping loses

The mapping shares work only across RHS columns for one sparse position:

```text
one wave
  owns: one CSR position x up to 32 RHS columns
  lane 0: recover row and load the sparse value
  all lanes: receive shared values through wave shuffle
  active lane: load dense RHS, multiply, atomic-add output
```

For RHS 32, `ca-GrQc` launches 28,980 logical waves, or 927,360 GPU threads,
for the compute kernel. By comparison:

| Mapping | Logical work for ca-GrQc, RHS 32 |
| --- | ---: |
| `thread-per-output` | 167,744 threads |
| `wave-per-row-tile`, tile 16 | 10,484 waves |
| `thread-per-position`, chunk 8 | 115,920 threads |
| `wave-per-position-tile` | 28,980 waves / 927,360 threads |

RHS 8 launches the same number of cooperative waves, but only 8 of 32 lanes
produce output. The remaining 75% of lanes are inactive. `ca-AstroPh` expands
this to 396,160 waves, or 12,677,120 threads, for RHS 8 or 32.

The generated gfx1101 ISA confirms that register pressure is not the limiting
factor. The following static counts use `ca-GrQc`, RHS 32, block 64, tile 16,
and position chunk 8. Instruction counts exclude `s_code_end` padding.

| Mapping | Instructions | VGPR | SGPR | Spills | `ds_bpermute_b32` | Atomic CAS sites | Reciprocal sites |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Thread output | 307 | 20 | 20 | 0 | 0 | 0 | 2 |
| Wave row tile | 1,640 | 95 | 86 | 0 | 80 | 0 | 3 |
| Thread position | 976 | 32 | 41 | 0 | 0 | 2 | 8 |
| Wave position tile | 741 | 27 | 29 | 0 | 4 | 1 | 7 |

The cooperative kernel has no VGPR or SGPR spills, LDS, or scratch usage. Its
ISA instead retains expensive coordinate recovery and division sequences.
Each product ends in a floating-point `memref.atomic_rmw`, which lowers on this
target to a `global_atomic_cmpswap_b32` retry loop with cache invalidations.
The static table shows one CAS site, but it executes once per active product.

Row-owned mappings amortize CSR traversal across all nonzeros in a row and
accumulate before writing the output. The cooperative position mapping
broadcasts one sparse value across RHS lanes, but it still recovers the row and
performs a global atomic update for every product. The saved sparse-value load
does not compensate for the larger launch topology, coordinate recovery, and
atomic serialization.

This is a useful negative result for the reusable cooperative-axis mechanism:
wave cooperation alone is insufficient for SpMM when it is applied after
splitting to individual sparse positions. A follow-up schedule should combine
cooperation with a position chunk or row segment so that one wave processes
multiple positions and accumulates locally before crossing the atomic
boundary.

## Cooperative position chunks

The follow-up schedule assigns a consecutive chunk of the non-cooperative
position domain to each wave. Every lane still owns one RHS column. Within the
chunk, it accumulates adjacent contributions with the same flattened output
key in a register. It performs an atomic add only when the key changes or the
chunk ends. A CSR row boundary therefore ends the current segment instead of
requiring any preprocessing or a new nonzero array.

Chunk size 1 retains the original direct lowering. The following results were
measured on 2026-09-02 with the same GPU, LLVM, ROCm, warmup, and iteration
counts as the initial experiment. Each row selects the lowest median across
block sizes 64 and 256. The speedup compares chunking against chunk size 1 in
this follow-up run; GPU clocks remained unlocked.

| Matrix | RHS | Chunk 1 median | Best chunk | Block | Median / p95 | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ca-GrQc | 8 | 362.92 us | 2 | 64 | 272.92 / 274.12 us | 1.33x |
| ca-GrQc | 32 | 264.56 us | 4 | 256 | 90.42 / 91.20 us | 2.93x |
| ca-GrQc | 128 | 430.06 us | 4 | 64 | 315.94 / 317.52 us | 1.36x |
| ca-AstroPh | 8 | 11,574.13 us | 16 | 256 | 2,346.44 / 2,372.08 us | 4.93x |
| ca-AstroPh | 32 | 3,954.97 us | 8 | 64 | 1,153.48 / 1,159.60 us | 3.43x |
| ca-AstroPh | 128 | 15,578.62 us | 8 | 64 | 4,464.29 / 4,507.24 us | 3.49x |

Chunked kernels use 34 VGPRs and 45 SGPRs, compared with 27 VGPRs and 29
SGPRs for the direct chunk-1 kernel. Neither variant spills or uses LDS or
scratch memory. The larger chunks reduce waves and atomic executions, but
eventually lose parallelism; `ca-GrQc` RHS 8 is fastest at chunk 2, while the
longer-row `ca-AstroPh` benefits through chunk 8 or 16. Chunking substantially
improves the cooperative schedule, but the prior row-owned and rocSPARSE
results remain faster on these matrices.

## Excluded matrix

[Williams/mac_econ_fwd500](https://sparse.tamu.edu/Williams/mac_econ_fwd500)
was also attempted. A multi-configuration run reported two elements outside
the CPU-reference tolerance, and a cooperative-only sweep reproduced the
failure. Its performance values are excluded until that correctness issue is
understood. A single RHS-8, block-64 cooperative case passed, so the failure is
not treated as evidence against all configurations of the mapping.

### Correctness follow-up

On 2026-08-31, RHS 32 / block 256 reproduced four mismatches in zero-based row
20,925, at columns 4, 11, 18, and 25. The RHS repeats every seven columns.
Ten subsequent diagnostic executions of that configuration produced one
failure and nine passes. The failing outputs shared these values:

| Quantity | Value |
| --- | ---: |
| Products in the row | 44 |
| Sum of absolute products | approximately 32,498.33 |
| Sequential f32 CPU reference | 7.05335712 |
| f64 CPU reference from f32 inputs | 7.05322437 |
| Observed GPU value | 7.05198336 |
| Previous tolerance | 0.00070534 |

This row has strong cancellation. Randomizing the f32 accumulation order on
the CPU over 10,000 permutations also exceeded the previous tolerance, with
results ranging from 7.04785156 to 7.05581331. The non-deterministic atomic
accumulation order therefore explains the observed discrepancy; reproducing
the discrepancy does not require a GPU indexing or synchronization error.

SpMM validation now uses an f64 reference and an input-dependent f32 rounding
bound shared with the rocSPARSE baseline:

```text
u = 2^-24
gamma = (n + 1) * u / (1 - (n + 1) * u)
tolerance = max(1e-4 * max(1, abs(reference)),
                gamma * sum(abs(a_i * b_i)))
```

The bound accounts for rounded products and any accumulation order, assuming
finite arithmetic without overflow or flush-to-zero underflow. It is
conservative, not a claim that every result within it has equal accuracy.
Non-finite results and errors exceeding the bound still fail validation.
The original performance tables above are unchanged.

With the updated check, the full `mac_econ_fwd500` sweep passed all 39
configurations: 36 SparseWave configurations and three rocSPARSE baselines,
using the same RHS widths, block sizes, tiles, chunks, warmup, and iteration
counts as the reproduction command below. Compiler scheduling and generated
GPU arithmetic were not changed by this validation fix.

## Reproduction

Run the following command for each Matrix Market file to reproduce the
cooperative chunk sweep:

```sh
python3 benchmark/run_spmm_benchmark.py \
  --chip=gfx1101 \
  --matrix=/path/to/matrix.mtx \
  --formats=csr \
  --mappings=wave-per-position-tile \
  --rhs-columns=8,32,128 \
  --block-sizes=64,256 \
  --position-chunk-sizes=1,2,4,8,16 \
  --warmup=10 \
  --iterations=100
```

To retain HSACO files for ISA inspection, add `--keep-artifacts` and
`--output-dir=/path/to/results`.
