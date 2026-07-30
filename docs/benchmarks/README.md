# SparseWave benchmarks

SparseWave's benchmark suite validates generated CSR kernels on AMD GPUs,
measures steady-state execution, can compare equivalent rocSPARSE operations,
and reports resources from the generated HSACO.

Each recorded result contains its workload definition, measurement method,
environment, reproduction commands, performance data, and interpretation:

- [gfx1101 SpMV mapping sweep](spmv-mapping-sweep-gfx1101.md) compares
  thread-, wave-, and block-per-row mappings across SuiteSparse and controlled
  synthetic inputs.
- [gfx1101 SpMM tile-size sweep](spmm-tile-size-sweep-gfx1101.md) compares
  thread-per-output and wave-per-row-tile mappings across RHS widths and tile
  sizes.
