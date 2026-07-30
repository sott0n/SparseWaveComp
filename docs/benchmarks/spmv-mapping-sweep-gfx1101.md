# SpMV mapping sweep on gfx1101

This report compares SparseWave's `thread-per-row`, `wave-per-row`, and
`block-per-row` SpMV mappings with rocSPARSE on an AMD Radeon RX 7800 XT. It is
a snapshot for the compiler and software revisions below, not a general
performance guarantee.

## Environment

| Component | Value |
| --- | --- |
| Date | 2026-07-30 |
| GPU | AMD Radeon RX 7800 XT |
| Target chip | gfx1101 |
| Wavefront size | 32 |
| SparseWave commit | `1e11c4b162fa2c53d27915d093f6ef6f41048e2a` |
| LLVM commit | `64b593c2371b7f7225b0ec190a37cd8b672e4c5d` |
| ROCm version | 6.4.4-129 |
| rocSPARSE version | 300400, Git revision `8fbfc797` |
| rocSPARSE algorithm | Default CSR SpMV |
| Warmup dispatches | 10 |
| Measured dispatches | 50 |

SparseWave times are kernel timestamps reported by `rocprofv3`. The rocSPARSE
runner uses HIP events around the complete compute call because one library
operation may launch multiple kernels. rocSPARSE buffer sizing and
preprocessing are excluded from steady-state time and recorded separately.
All reported configurations passed output validation.

## Benchmark method

The benchmark accepts either generated matrices or Matrix Market coordinate
matrices with `real`, `integer`, or `pattern` entries in `general` or
`symmetric` form. Symmetric off-diagonal entries are expanded. Matrix Market
data is converted to a compact temporary CSR binary, avoiding MLIR source size
and compilation time proportional to NNZ.

The runner uses the stored sparse values and an all-ones input vector and
checks every output row against a host reference. Each run prints timing and
GPU-resource tables and writes `results.csv` and `metadata.json` under
`build/benchmark/results`. VGPRs, SGPRs, spills, fixed LDS, per-work-item
scratch, wave size, and maximum workgroup size are read from the generated
HSACO metadata. Generated MLIR, extracted HSACOs, CSR binaries, and profiler
traces are retained only with `--keep-artifacts`.

Synthetic inputs support `uniform`, `alternating`, and `skewed` row-length
distributions. The cooperative mappings currently require Wave32.

## SuiteSparse matrices

The real inputs are the same
[SuiteSparse Matrix Collection](https://sparse.tamu.edu/) matrices used by the
SpMM tile-size study.

| Matrix | Rows | Columns | NNZ | Mean NNZ/row | Maximum NNZ/row |
| --- | ---: | ---: | ---: | ---: | ---: |
| [SNAP/ca-GrQc](https://sparse.tamu.edu/SNAP/ca-GrQc) | 5,242 | 5,242 | 28,980 | 5.53 | 81 |
| [SNAP/ca-AstroPh](https://sparse.tamu.edu/SNAP/ca-AstroPh) | 18,772 | 18,772 | 396,160 | 21.10 | 504 |
| [Williams/mac_econ_fwd500](https://sparse.tamu.edu/Williams/mac_econ_fwd500) | 206,500 | 206,500 | 1,273,389 | 6.17 | 44 |

The table reports the best block size for each SparseWave mapping. Speedup is
relative to the rocSPARSE median for the same matrix.

| Matrix | Implementation | Mapping | Block | Median | p95 | GNNZ/s | vs rocSPARSE |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| ca-GrQc | rocSPARSE | default | - | 38.32 us | 38.88 us | 0.76 | 1.00x |
| ca-GrQc | SparseWave | thread-per-row | 64 | 33.42 us | 34.64 us | 0.87 | 1.15x |
| ca-GrQc | SparseWave | wave-per-row | 128 | **16.12 us** | **16.56 us** | **1.80** | **2.38x** |
| ca-GrQc | SparseWave | block-per-row | 64 | 26.22 us | 26.64 us | 1.11 | 1.46x |
| ca-AstroPh | rocSPARSE | default | - | 68.40 us | 85.52 us | 5.79 | 1.00x |
| ca-AstroPh | SparseWave | thread-per-row | 128 | 219.32 us | 225.60 us | 1.81 | 0.31x |
| ca-AstroPh | SparseWave | wave-per-row | 128 | **49.16 us** | **50.20 us** | **8.06** | **1.39x** |
| ca-AstroPh | SparseWave | block-per-row | 64 | 76.74 us | 77.24 us | 5.16 | 0.89x |
| mac_econ_fwd500 | rocSPARSE | default | - | **64.94 us** | **98.56 us** | **19.61** | **1.00x** |
| mac_econ_fwd500 | SparseWave | thread-per-row | 256 | 132.86 us | 135.04 us | 9.58 | 0.49x |
| mac_econ_fwd500 | SparseWave | wave-per-row | 256 | 334.76 us | 335.76 us | 3.80 | 0.19x |
| mac_econ_fwd500 | SparseWave | block-per-row | 64 | 1,017.33 us | 1,024.85 us | 1.25 | 0.06x |

rocSPARSE preprocessing took 194.28 us for ca-GrQc, 281.14 us for
ca-AstroPh, and 1,202.86 us for mac_econ_fwd500. These one-time costs are not
included in the steady-state table.

## Controlled row-length experiment

A synthetic experiment isolates average row length and row-length skew. Every
matrix has 65,536 rows and columns. `uniform` gives every row the requested
NNZ count. `skewed` gives seven of every eight rows one entry and the eighth
row `8 * NNZ/row - 7` entries, preserving the same total NNZ and average.

Only the fastest SparseWave mapping and block size are shown below. The final
column uses the rocSPARSE median as the baseline.

| Distribution | Mean NNZ/row | Maximum NNZ/row | rocSPARSE median | rocSPARSE p95 | Best SparseWave configuration | SparseWave median | SparseWave p95 | vs rocSPARSE |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| uniform | 4 | 4 | 38.72 us | 39.48 us | thread-per-row, block 64 | **20.52 us** | **21.16 us** | **1.89x** |
| uniform | 32 | 32 | 89.96 us | 97.64 us | wave-per-row, block 512 | **74.40 us** | **75.04 us** | **1.21x** |
| uniform | 256 | 256 | 261.70 us | 276.08 us | block-per-row, block 64 | **256.18 us** | 497.97 us | 1.02x |
| skewed | 4 | 25 | **23.44 us** | **23.84 us** | thread-per-row, block 128 | 24.18 us | 75.88 us | 0.97x |
| skewed | 32 | 249 | 132.28 us | 134.84 us | wave-per-row, block 64 | **60.52 us** | **60.80 us** | **2.19x** |
| skewed | 256 | 2,041 | 322.94 us | 324.60 us | block-per-row, block 64 | **304.04 us** | **305.72 us** | **1.06x** |

The uniform 256 result is effectively parity at the median, not a robust win:
its block-per-row p95 is 497.97 us versus 276.08 us for rocSPARSE.

## GPU resources and ISA

The resource counts were identical across the measured matrix shapes. LDS
usage for `block-per-row` is four bytes per wave in the block.

| Mapping | VGPR | SGPR | LDS | Scratch | VGPR spills | SGPR spills |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| thread-per-row | 14 | 16 | 0 B | 0 B | 0 | 0 |
| wave-per-row | 14 | 20 | 0 B | 0 B | 0 | 0 |
| block-per-row | 15 | 28 | `block size / 8` B | 0 B | 0 | 0 |

Disassembly of the block-64 kernels explains the reduction cost:

| Mapping | Reduction structure | Characteristic gfx1101 instructions |
| --- | --- | --- |
| thread-per-row | Sequential accumulation by one lane | No shuffle, LDS, or barrier |
| wave-per-row | Five-stage Wave32 butterfly reduction | 5 `ds_bpermute_b32` |
| block-per-row | Per-wave reduction, LDS exchange, then reduction of wave partials | 10 `ds_bpermute_b32`, 1 `ds_store_b32`, 1 `s_barrier`, 1 `ds_load_b32` |

No mapping spills registers, so the measured differences are primarily caused
by available row-level parallelism, lane utilization, and reduction or
synchronization overhead rather than register pressure.

## Interpretation

- `thread-per-row` is effective when many independent short rows already
  provide enough parallel work. This is visible in the uniform four-entry case.
- `wave-per-row` creates one wave of cooperative work per row. It improves
  latency hiding when the matrix has too few rows for thread-per-row and uses
  its lanes efficiently as rows grow. This explains the wins on ca-GrQc,
  ca-AstroPh, and the skewed 32-entry workload.
- The 206,500 short rows in mac_econ_fwd500 already expose substantial
  row-level parallelism. Assigning a whole wave or block to each row wastes
  lanes, while rocSPARSE remains 2.05x faster than SparseWave's best mapping.
- `block-per-row` pays a cross-wave LDS exchange and a block barrier. Block 64
  is consistently its best measured configuration; larger blocks add wave
  partials and synchronization work that short or medium rows cannot use.
- Very long skewed rows provide an application region for `block-per-row`, but
  the 1.06x result at mean 256 NNZ/row is modest. Its main value is currently
  exposing a distinct long-row strategy for future scheduling work.
- Average NNZ/row alone cannot select a mapping. Total row count and row-length
  distribution are also required: ca-GrQc and mac_econ_fwd500 have similar
  means but opposite mapping outcomes.

These results support an inspectable mapping decision based on at least row
count, mean and maximum row length, and row-length skew. They also provide
concrete baseline regions for the planned reusable scheduling interface and
TACO-style load-balancing experiments.

## Reproduction

Run the real-matrix sweep once for each Matrix Market file:

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
  --iterations=50
```

With `--keep-artifacts`, inspect a generated kernel using:

```sh
/opt/rocm/llvm/bin/llvm-objdump \
  -d \
  --no-show-raw-insn \
  --disassemble-symbols=spmv_kernel \
  /path/to/kernel.hsaco
```
