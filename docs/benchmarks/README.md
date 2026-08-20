# SparseWave benchmarks

SparseWave's benchmark suite validates generated sparse kernels on AMD GPUs,
measures steady-state execution, and reports resources from the generated
HSACO. Each runner can evaluate the same sparse workload across its supported
storage formats and GPU mappings while keeping input generation, correctness
validation, and measurement consistent. Equivalent rocSPARSE operations can
be included as CSR baselines.

## Supported formats

| Runner | Format | Format selection and parameters | GPU mappings |
| --- | --- | --- | --- |
| SpMV | CSR | `--formats=csr`; `--position-chunk-sizes` selects consecutive positions processed by each `thread-per-position` worker | `thread-per-row`, `thread-per-position`, `wave-per-position`, `wave-per-row`, `block-per-row` |
| SpMV | COO | `--formats=coo` | `thread-per-nonzero` |
| SpMM | CSR | `--formats=csr`; `--tile-sizes` selects output-column tile widths | `thread-per-output`, `wave-per-row-tile` |
| SpMM | BSR | `--formats=bsr`; `--bsr-block-sizes` selects square storage block sizes | `thread-per-output` |

Multiple formats can be evaluated in one invocation, for example
`--formats=csr,coo` for SpMV or `--formats=csr,bsr` for SpMM. The common
`--block-sizes` option controls the GPU workgroup size; it is distinct from
the BSR storage block size. `--wave-size` selects the AMDGPU wavefront size,
subject to the mappings currently supported by each runner. CSR is used for
rocSPARSE comparisons when `--rocsparse` is present.

Matrix Market entries are sorted deterministically without coalescing duplicate
coordinates. Temporary CSR and COO binaries therefore preserve the same
mathematical input, including repeated coordinates. `results.csv` records the
format and its host conversion/serialization time separately from GPU kernel
time. COO timing includes both the output-initialization and atomic-compute
kernels by summing their GPU durations for each SpMV dispatch.

BSR conversion groups entries into row-major dense blocks and coalesces
duplicate coordinates by addition. Matrix dimensions that are not divisible
by the selected block size are extended with zero rows and columns. The BSR
binary also retains the original Matrix Market coordinates, so CPU correctness
validation is independent from the converted blocks. Reports include the
number and density of nonzero blocks, the fraction of unused scalar slots
inside stored blocks, and the ratio of stored scalar slots to occupied input
coordinates.

For example, a small CSR/BSR comparison can be run with:

```sh
python3 benchmark/run_spmm_benchmark.py \
  --matrix test/Benchmark/Inputs/tiny.mtx \
  --formats=csr,bsr \
  --bsr-block-sizes=2,4,8 \
  --rhs-columns=4 \
  --block-size=64 \
  --tile-size=4
```

A TACO-style position split ablation can compare multiple thread chunk factors
against the row, wave-position, and rocSPARSE baselines in one run:

```sh
python3 benchmark/run_spmv_benchmark.py \
  --rows=65536 \
  --columns=65536 \
  --nnz-per-row=4,32,256 \
  --distributions=uniform,skewed \
  --position-chunk-sizes=1,2,4,8 \
  --rocsparse
```

The `chunk` result column is the split factor. A value of one is the original
one-position-per-thread schedule; larger values trade parallel worker count for
sequential reuse within each thread. Reorder and collapse are not included in
this one-dimensional SpMV ablation because they do not change its iteration
order or shape.

Each recorded result contains its workload definition, measurement method,
environment, reproduction commands, performance data, and interpretation:

- [gfx1101 SpMV mapping sweep](spmv-mapping-sweep-gfx1101.md) compares
  thread-, wave-, and block-per-row mappings across SuiteSparse and controlled
  synthetic inputs.
- [gfx1101 SpMV position-mapping study](spmv-position-mapping-gfx1101.md)
  evaluates thread- and wave-per-position mappings, including segmented
  reduction costs and comparisons with row mappings and rocSPARSE.
- [gfx1101 SpMM tile-size sweep](spmm-tile-size-sweep-gfx1101.md) compares
  thread-per-output and wave-per-row-tile mappings across RHS widths and tile
  sizes.
