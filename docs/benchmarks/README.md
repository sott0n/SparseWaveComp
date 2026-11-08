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
| SpMV | CSR | `--formats=csr`; `--mappings` selects the mappings to measure; `--position-chunk-sizes` selects consecutive positions per thread or block participant; `--position-wave-chunk-sizes` selects consecutive lane-width tiles per wave; `--position-reductions=atomic,segmented,workspace` selects the strategies supported by each position mapping | `thread-per-row`, `thread-per-position`, `wave-per-position`, `block-per-position`, `wave-per-row`, `block-per-row` |
| SpMV | COO | `--formats=coo`; position options are shared with CSR | `thread-per-position`, `wave-per-position`, `block-per-position` |
| SpMM | CSR | `--formats=csr`; `--tile-sizes` selects output-column tile widths; `--position-chunk-sizes` selects consecutive points per thread, block participant, or cooperative wave; `--position-wave-chunk-sizes` selects consecutive lane-width tiles per non-cooperative wave; `--position-orders=position-major,rhs-major` reorders the position and RHS-column axes; `--position-reductions=atomic,segmented,workspace` selects the strategies supported by each position mapping | `thread-per-output`, `wave-per-row-tile`, `thread-per-position`, `wave-per-position`, `block-per-position`, `wave-per-position-tile` |
| SpMM | BSR | `--formats=bsr`; `--bsr-block-sizes` selects square storage block sizes | `thread-per-output` |
| Elementwise add/multiply | CSR | `--nnz-per-row` controls both input row lengths; `--overlaps` controls their structural overlap and therefore the assembled output size | `thread-per-row` symbolic and numeric phases with a hierarchical GPU prefix scan |

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
time. Position-space timing sums every GPU stage in a dispatch, including
output initialization and any workspace stages.

BSR conversion groups entries into row-major dense blocks and coalesces
duplicate coordinates by addition. Matrix dimensions that are not divisible
by the selected block size are extended with zero rows and columns. The BSR
binary also retains the original Matrix Market coordinates, so CPU correctness
validation is independent from the converted blocks. Reports include the
number and density of nonzero blocks, the fraction of unused scalar slots
inside stored blocks, and the ratio of stored scalar slots to occupied input
coordinates.

SpMM correctness checks use f64 references and a per-output f32 rounding bound
based on the number and absolute sum of products, shared by CSR, BSR, and the
rocSPARSE baseline. This accommodates cancellation and reordered reductions
without globally increasing the tolerance. See the
[correctness follow-up](spmm-cooperative-position-gfx1101.md#correctness-follow-up)
for the formula and the motivating input. Validation runs outside GPU timing.

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

A TACO-style position split ablation can compare multiple thread and block
participant chunk factors
against the row, wave-position, and rocSPARSE baselines in one run:

```sh
python3 benchmark/run_spmv_benchmark.py \
  --rows=65536 \
  --columns=65536 \
  --nnz-per-row=4,32,256 \
  --distributions=uniform,skewed \
  --position-chunk-sizes=1,2,4,8 \
  --position-wave-chunk-sizes=1,2,4,8 \
  --position-reductions=atomic,segmented,workspace \
  --rocsparse
```

The `chunk` result column is the mapping-specific split factor. For thread and
block mappings it is the number of consecutive positions per participant. For
wave mapping it is the number of consecutive lane-width position tiles per
wave. A chunked
wave recovers the first compressed segment cooperatively, then carries segment
and boundary state across later tiles. `atomic` independently updates each
completed segment; `segmented` combines equal keys inside a thread chunk; and
`workspace` merges wave-boundary partials in a second stage. Reorder and
collapse are not included in this one-dimensional SpMV ablation because they
do not change its iteration order or shape.

Add `--position-ablation` to either runner to extract the best measured
configuration for each transformation stage. The runner prints the compact
stage table and writes `position-ablation.csv` beside the complete
`results.csv`. The summary reports speedup against both the untransformed
thread-position baseline and rocSPARSE when those cases were selected. See the
[integrated gfx1101 position-transformation ablation](position-transformation-ablation-gfx1101.md)
for the stage definitions and representative real-matrix results.

SpMM accepts the same generated row-length distributions. The following sweep
compares fixed row/output mappings, position split and reorder variants,
cooperative wave scheduling, and rocSPARSE from one deterministic input set:

```sh
python3 benchmark/run_spmm_benchmark.py \
  --rows=65536 \
  --columns=65536 \
  --nnz-per-row=4,32,256 \
  --distributions=uniform,skewed \
  --formats=csr \
  --mappings=thread-per-output,wave-per-row-tile,thread-per-position,wave-per-position,block-per-position,wave-per-position-tile \
  --rhs-columns=4,16,64 \
  --tile-sizes=1,4,8,16 \
  --position-chunk-sizes=1,4,8 \
  --position-wave-chunk-sizes=1,2,4,8 \
  --position-orders=position-major,rhs-major \
  --position-reductions=atomic,segmented \
  --rocsparse
```

`uniform` gives every row the requested NNZ count. `alternating` and `skewed`
preserve the same mean while concentrating work into every second or eighth
row, respectively. The generated CSR data is shared by every SparseWave
variant and the rocSPARSE baseline. Supplying `--matrix` instead selects the
existing Matrix Market workflow.

The CSR elementwise runner reports the complete dispatch as three logical
phases with separate symbolic, prefix, and numeric medians. The prefix phase
sums its block-local scan, block-total scan, and block-offset addition kernels.
Each block scan uses wave shuffle instructions for lane-local prefixes and LDS
only for the totals exchanged between waves. Generated input pairs use sorted
unique columns and controlled structural overlap:

```sh
python3 benchmark/run_elementwise_benchmark.py \
  --rows=65536 \
  --columns=65536 \
  --nnz-per-row=4,32 \
  --overlaps=25,50,100 \
  --operations=add,multiply \
  --block-sizes=64,128,256 \
  --report-resources \
  --warmup=10 \
  --iterations=50
```

Each phase share is calculated from the sum of measured per-dispatch phase
durations after warmup. In particular, `prefix share` exposes the total cost of
constructing compressed output offsets. With `--report-resources`, the runner
also writes `resources.csv` and prints VGPR, SGPR, spill, LDS, and scratch usage
for the symbolic, three prefix-scan, and numeric kernels.

Each recorded result contains its workload definition, measurement method,
environment, reproduction commands, performance data, and interpretation:

- [gfx1101 SpMV mapping sweep](spmv-mapping-sweep-gfx1101.md) compares
  thread-, wave-, and block-per-row mappings across SuiteSparse and controlled
  synthetic inputs.
- [gfx1101 SpMV position-mapping study](spmv-position-mapping-gfx1101.md)
  evaluates thread- and wave-per-position mappings, including segmented
  reduction costs and comparisons with row mappings and rocSPARSE.
- [gfx1101 SpMV position-split ablation](spmv-position-split-gfx1101.md)
  isolates thread chunk factors across row-length distributions and explains
  their performance using generated ISA and GPU resources.
- [gfx1101 SpMV segmented position reduction](spmv-position-segmented-gfx1101.md)
  compares per-position atomics with chunk-local row carry and aggregation,
  including an ablation across row distributions and chunk sizes.
- [gfx1101 SpMM tile-size sweep](spmm-tile-size-sweep-gfx1101.md) compares
  thread-per-output and wave-per-row-tile mappings across RHS widths and tile
  sizes.
- [gfx1101 position-space SpMM baseline](spmm-position-space-gfx1101.md)
  evaluates the reusable flattened position scheduler on SpMM and identifies
  the costs that remain before a load-balanced position schedule can compete.
- [gfx1101 position-space SpMM reorder](spmm-position-reorder-gfx1101.md)
  separates position/RHS-axis reorder from segmented reduction and compares
  the combined schedule with the existing row-owned mappings.
- [gfx1101 cooperative position-space SpMM](spmm-cooperative-position-gfx1101.md)
  evaluates wave cooperation across RHS columns, compares it with existing
  mappings and rocSPARSE, and explains its current costs using generated ISA.
- [gfx1101 generated SpMM scheduling ablation](spmm-generated-ablation-gfx1101.md)
  isolates split, reorder, and segmented reduction, then compares representative
  schedules across controlled row-length distributions and rocSPARSE.
- [gfx1101 wave position chunking ablation](wave-position-chunk-gfx1101.md)
  evaluates consecutive position tiles per wave across SpMV and SpMM, including
  its interaction with iteration order, GPU resources, and rocSPARSE.
- [gfx1101 block-local segmented position reduction](block-position-segmented-gfx1101.md)
- [gfx1101 block position chunking](block-position-chunk-gfx1101.md)
  measures the generic block reduction across SpMV and SpMM, and identifies
  when output-key locality repays its LDS scan and synchronization costs.
- [gfx1101 position-transformation ablation](position-transformation-ablation-gfx1101.md)
  integrates fixed mappings, split, reorder, reduction, wave/block scheduling,
  manual-best selection, and rocSPARSE on representative real matrices.
- [gfx1101 CSR elementwise block-size sweep](elementwise-block-size-sweep-gfx1101.md)
  measures symbolic, prefix-scan, and numeric phase behavior for sparse add
  and multiply and relates the timing tradeoff to generated GPU resources.
