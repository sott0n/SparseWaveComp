# SparseWave benchmarks

SparseWave's benchmark suite validates generated sparse kernels on AMD GPUs,
measures steady-state execution, can compare equivalent rocSPARSE operations,
and reports resources from the generated HSACO. The SpMV runner can execute
the same Matrix Market or synthetic workload as CSR and COO with
`--formats=csr,coo`; CSR remains the rocSPARSE comparison format.

Matrix Market entries are sorted deterministically without coalescing duplicate
coordinates. Temporary CSR and COO binaries therefore preserve the same
mathematical input, including repeated coordinates. `results.csv` records the
format and its host conversion/serialization time separately from GPU kernel
time. COO timing includes both the output-initialization and atomic-compute
kernels by summing their GPU durations for each SpMV dispatch.

Each recorded result contains its workload definition, measurement method,
environment, reproduction commands, performance data, and interpretation:

- [gfx1101 SpMV mapping sweep](spmv-mapping-sweep-gfx1101.md) compares
  thread-, wave-, and block-per-row mappings across SuiteSparse and controlled
  synthetic inputs.
- [gfx1101 SpMM tile-size sweep](spmm-tile-size-sweep-gfx1101.md) compares
  thread-per-output and wave-per-row-tile mappings across RHS widths and tile
  sizes.
