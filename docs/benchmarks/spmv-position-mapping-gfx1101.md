# SpMV position-mapping study on gfx1101

This report evaluates SparseWave's CSR `thread-per-position` and
`wave-per-position` SpMV mappings on an AMD Radeon RX 7800 XT. It compares the
position-space approaches with the best measured row mapping and rocSPARSE.
The results are a snapshot for the revisions below, not a general performance
guarantee.

## Environment

| Component | Value |
| --- | --- |
| Date | 2026-08-12 |
| GPU | AMD Radeon RX 7800 XT |
| Target chip | gfx1101 |
| Wavefront size | 32 |
| SparseWave commit | `e8b6df0a1bf8ba06a759e592daa89fc91ad8b888` |
| LLVM commit | `64b593c2371b7f7225b0ec190a37cd8b672e4c5d` |
| ROCm version | 6.4.4-129 |
| rocSPARSE version | 300400, Git revision `8fbfc797` |
| rocSPARSE algorithm | Default CSR SpMV |
| Warmup dispatches | 10 |
| Measured dispatches | 50 |

SparseWave times are kernel timestamps reported by `rocprofv3`. Row mappings
overwrite each output element in one kernel. Position mappings use one kernel
to initialize the output and a second kernel to accumulate into it; their
reported time is the sum of both kernels for each dispatch. The rocSPARSE
runner uses HIP events around the complete compute call. Buffer sizing,
preprocessing, host conversion, and compilation are excluded from steady-state
time. All configurations passed CPU-reference validation.

## Mapping mechanisms

Both position mappings evenly partition the CSR stored-position range instead
of assigning complete rows to workers. Each active worker recovers its row and
column from the CSR storage before computing a product.

- `thread-per-position` atomically adds every product to its output row.
- `wave-per-position` assigns a contiguous position range to each wave. A
  segmented Wave32 scan combines adjacent products with the same row, and only
  segment-ending lanes atomically add partial sums. A row crossing a wave
  boundary receives one partial sum from each wave segment.

The wave mapping therefore reduces atomic contention and preserves balanced
position work, but adds shuffle operations and still performs coordinate
recovery for each active lane.

## Controlled row-length experiment

Every generated matrix has 65,536 rows and columns. `uniform` gives every row
the requested number of entries. `skewed` gives seven of every eight rows one
entry and the eighth row `8 * mean - 7` entries, preserving the same total NNZ
and mean. Each SparseWave entry is the fastest block size among 64, 128, 256,
and 512. The row baseline is the fastest of thread-, wave-, and block-per-row.
Speedup is relative to the rocSPARSE median.

| Distribution | Mean | Max | Mapping | Block | Median | p95 | vs rocSPARSE |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| uniform | 4 | 4 | thread-per-position | 512 | 105.33 us | 119.17 us | 0.34x |
| uniform | 4 | 4 | wave-per-position | 512 | 87.81 us | 88.65 us | 0.41x |
| uniform | 4 | 4 | thread-per-row | 64 | **20.08 us** | **20.60 us** | **1.77x** |
| uniform | 4 | 4 | rocSPARSE default | - | 35.56 us | 36.56 us | 1.00x |
| uniform | 32 | 32 | thread-per-position | 128 | 2,324.79 us | 2,342.47 us | 0.02x |
| uniform | 32 | 32 | wave-per-position | 128 | 203.41 us | 204.37 us | 0.18x |
| uniform | 32 | 32 | wave-per-row | 64 | 38.56 us | 38.80 us | 0.93x |
| uniform | 32 | 32 | rocSPARSE default | - | **35.98 us** | **37.00 us** | **1.00x** |
| uniform | 256 | 256 | thread-per-position | 64 | 123,129.33 us | 124,343.74 us | 0.00x |
| uniform | 256 | 256 | wave-per-position | 64 | 1,999.76 us | 2,010.40 us | 0.13x |
| uniform | 256 | 256 | block-per-row | 64 | **255.14 us** | **258.02 us** | **1.02x** |
| uniform | 256 | 256 | rocSPARSE default | - | 259.75 us | 265.01 us | 1.00x |
| skewed | 4 | 25 | thread-per-position | 512 | 209.09 us | 212.33 us | 0.23x |
| skewed | 4 | 25 | wave-per-position | 64 | 33.08 us | **33.36 us** | **1.47x** |
| skewed | 4 | 25 | thread-per-row | 128 | **24.80 us** | 77.09 us | **1.96x** |
| skewed | 4 | 25 | rocSPARSE default | - | 48.72 us | 50.00 us | 1.00x |
| skewed | 32 | 249 | thread-per-position | 64 | 8,555.90 us | 8,806.93 us | 0.02x |
| skewed | 32 | 249 | wave-per-position | 64 | 223.75 us | 672.72 us | 0.59x |
| skewed | 32 | 249 | wave-per-row | 64 | **60.38 us** | **60.52 us** | **2.19x** |
| skewed | 32 | 249 | rocSPARSE default | - | 132.29 us | 133.81 us | 1.00x |
| skewed | 256 | 2,041 | thread-per-position | 64 | 655,741.72 us | 659,400.12 us | 0.00x |
| skewed | 256 | 2,041 | wave-per-position | 64 | 5,693.76 us | 5,728.27 us | 0.06x |
| skewed | 256 | 2,041 | block-per-row | 64 | **305.52 us** | 579.75 us | **1.05x** |
| skewed | 256 | 2,041 | rocSPARSE default | - | 321.31 us | **331.73 us** | 1.00x |

`wave-per-position` is between 1.20x and 115.17x faster than
`thread-per-position` across these controlled cases. Its clearest region is
the skewed short-row workload: it is 1.47x faster than rocSPARSE and has a
33.36 us p95, while the faster-median thread-per-row configuration reaches a
77.09 us p95. For uniform rows and longer skewed rows, preserving row ownership
is substantially cheaper than position-space coordinate recovery, segmented
reduction, and atomic output updates.

## SuiteSparse matrices

The real-matrix experiment uses the following matrices from the
[SuiteSparse Matrix Collection](https://sparse.tamu.edu/).

| Matrix | Rows | Columns | NNZ | Mean NNZ/row | Maximum NNZ/row |
| --- | ---: | ---: | ---: | ---: | ---: |
| [SNAP/ca-GrQc](https://sparse.tamu.edu/SNAP/ca-GrQc) | 5,242 | 5,242 | 28,980 | 5.53 | 81 |
| [SNAP/ca-AstroPh](https://sparse.tamu.edu/SNAP/ca-AstroPh) | 18,772 | 18,772 | 396,160 | 21.10 | 504 |
| [Williams/mac_econ_fwd500](https://sparse.tamu.edu/Williams/mac_econ_fwd500) | 206,500 | 206,500 | 1,273,389 | 6.17 | 44 |

Each SparseWave entry again selects the fastest measured block size. The row
baseline is the fastest row mapping.

| Matrix | Mapping | Block | Median | p95 | vs rocSPARSE |
| --- | --- | ---: | ---: | ---: | ---: |
| ca-GrQc | thread-per-position | 512 | 85.68 us | 88.96 us | 0.29x |
| ca-GrQc | wave-per-position | 512 | 20.62 us | 26.84 us | 1.20x |
| ca-GrQc | wave-per-row | 512 | **12.26 us** | **16.60 us** | **2.02x** |
| ca-GrQc | rocSPARSE default | - | 24.72 us | 25.08 us | 1.00x |
| ca-AstroPh | thread-per-position | 512 | 1,896.19 us | 1,976.67 us | 0.03x |
| ca-AstroPh | wave-per-position | 512 | 110.80 us | 143.44 us | 0.59x |
| ca-AstroPh | wave-per-row | 512 | **39.80 us** | **40.36 us** | **1.64x** |
| ca-AstroPh | rocSPARSE default | - | 65.08 us | 68.12 us | 1.00x |
| mac_econ_fwd500 | thread-per-position | 512 | 1,083.63 us | 1,094.60 us | 0.06x |
| mac_econ_fwd500 | wave-per-position | 512 | 339.81 us | 342.25 us | 0.19x |
| mac_econ_fwd500 | thread-per-row | 512 | 120.14 us | 122.40 us | 0.53x |
| mac_econ_fwd500 | rocSPARSE default | - | **63.80 us** | **64.48 us** | **1.00x** |

The position-wave mapping improves over position-thread by 4.16x on ca-GrQc,
17.11x on ca-AstroPh, and 3.19x on mac_econ_fwd500. It exceeds rocSPARSE only
on ca-GrQc, and the row mapping remains faster on every real matrix. These
results show that reducing per-position atomics is necessary but not sufficient
to outperform a mapping that avoids coordinate recovery and atomics entirely.

## GPU resources and ISA

Resource counts are independent of input shape. None of the mappings spills or
uses scratch memory.

| Mapping | VGPR | SGPR | LDS | Scratch | VGPR spills | SGPR spills |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| thread-per-row | 14 | 16 | 0 B | 0 B | 0 | 0 |
| thread-per-position | 22 | 31 | 0 B | 0 B | 0 | 0 |
| wave-per-position | 21 | 31 | 0 B | 0 B | 0 | 0 |
| wave-per-row | 14 | 20 | 0 B | 0 B | 0 | 0 |
| block-per-row | 15 | 28 | `block size / 8` B | 0 B | 0 | 0 |

Static disassembly of the block-64 compute kernels gives:

| Mapping | Instructions | `ds_bpermute_b32` | Atomic compare-and-swap sites |
| --- | ---: | ---: | ---: |
| thread-per-position | 569 | 0 | 1 |
| wave-per-position | 698 | 23 | 1 |
| wave-per-row | 298 | 5 | 0 |

The floating-point atomic add lowers to a compare-and-swap loop, so one static
atomic site can execute repeatedly and can contend across workers. The
wave-position kernel reduces the number of lanes entering that loop, but pays
23 `ds_bpermute_b32` instructions: each of five scan stages shuffles the 64-bit
row key, partial value, and active state, and the final segment-boundary check
shuffles the key and active state once more. The row-wave kernel needs only the
five-value butterfly reduction and performs a non-atomic store.

This gives two concrete optimization targets for the position mapping:

1. Reduce coordinate-recovery work by deriving or incrementally tracking rows
   across contiguous positions.
2. Specialize the segmented reduction for prefix-active Wave32 partitions to
   remove redundant key or active-state shuffles.

Any such optimization should retain the current mapping as an ablation
baseline and remeasure atomic count, shuffle count, register usage, and tail
latency.

## Reproduction

Run the controlled experiment with:

```sh
python3 benchmark/run_spmv_benchmark.py \
  --chip=gfx1101 \
  --rows=65536 \
  --columns=65536 \
  --nnz-per-row=4,32,256 \
  --distributions=uniform,skewed \
  --rocsparse \
  --block-sizes=64,128,256,512 \
  --warmup=10 \
  --iterations=50 \
  --keep-artifacts
```

Run each SuiteSparse matrix with:

```sh
python3 benchmark/run_spmv_benchmark.py \
  --chip=gfx1101 \
  --matrix=/path/to/matrix.mtx \
  --rocsparse \
  --block-sizes=64,128,256,512 \
  --warmup=10 \
  --iterations=50 \
  --keep-artifacts
```

With `--keep-artifacts`, inspect a generated compute kernel using:

```sh
/opt/rocm/llvm/bin/llvm-objdump \
  -d \
  --no-show-raw-insn \
  --disassemble-symbols=spmv_kernel \
  /path/to/kernel.hsaco
```
