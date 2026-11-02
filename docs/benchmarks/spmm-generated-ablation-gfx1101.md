# Generated SpMM scheduling ablation on gfx1101

This report evaluates the TACO-style position-space transformations against
the existing row/output-owned SpMM mappings and rocSPARSE. It asks two
separate questions:

1. Do split, reorder, and segmented reduction improve a position-space
   schedule?
2. Does the improved schedule compete with fixed row/output ownership on
   representative row-length distributions?

The answer to the first question is yes, with strong interactions between the
transformations. The answer to the second is no for the schedules tested here.
The fixed SparseWave mappings nevertheless beat rocSPARSE in 10 of the 27
generated workloads, identifying useful workload regions independently of the
position-space result.

## Environment

| Component | Value |
| --- | --- |
| Date | 2026-11-02 |
| GPU | AMD Radeon RX 7800 XT |
| Target chip | gfx1101 |
| Wavefront size | 32 |
| SparseWave commit | `0cef5f6b7d92f2ac15e78bf310f76ed978d5149e` |
| LLVM commit | `64b593c2371b7f7225b0ec190a37cd8b672e4c5d` |
| ROCm version | 6.4.4-129 |
| rocSPARSE version | 300400, Git revision `8fbfc797` |
| rocSPARSE algorithm | Default CSR SpMM |
| GPU performance level | Automatic |

SparseWave times are the sum of the output-initialization and compute kernels
reported by `rocprofv3`. rocSPARSE uses HIP events around the complete
operation. Each case used 10 warmup and 50 measured dispatches. GPU clocks
were not locked, so the results are intended for comparisons within these
runs, not cross-run absolute-latency claims. All 135 breadth-sweep results and
all 19 transformation-ablation results passed CPU-reference validation.

## Generated inputs

Every matrix has 65,536 rows and columns. The requested NNZ/row is the exact
mean in all three distributions:

| Distribution | Row lengths for mean `N` |
| --- | --- |
| Uniform | Every row has `N` entries |
| Alternating | Rows alternate between 1 and `2N - 1` entries |
| Skewed | Seven rows have 1 entry, followed by one row with `8N - 7` entries |

The sweep covers mean row lengths 4, 32, and 256 and RHS widths 8, 32, and
128. Every implementation consumes the same deterministic CSR values and
dense RHS for a workload.

## Transformation ablation

The detailed ablation uses the skewed, 32 NNZ/row, RHS-32 workload, a
256-thread block, and compares position schedules with the fixed mappings.
Times are `median / p95` in milliseconds.

| Mapping | Chunk | Order | Reduction | Time | Relative observation |
| --- | ---: | --- | --- | ---: | --- |
| Thread position | 1 | Position-major | Atomic | 97.45 / 99.20 | Untransformed thread-position baseline |
| Thread position | 4 | Position-major | Atomic | 79.91 / 80.80 | Split alone is 1.22x faster |
| Thread position | 4 | Position-major | Segmented | 70.24 / 70.88 | Segmentation adds 1.14x over the same split |
| Thread position | 4 | RHS-major | Segmented | 64.09 / 64.38 | Reorder adds 1.10x with segmentation enabled |
| Thread position | 8 | RHS-major | Segmented | 21.34 / 21.62 | Best thread-position result; 4.57x over baseline |
| Wave position tile | 1 | - | - | 199.81 / 330.24 | Unsplit cooperative baseline |
| Wave position tile | 4 | - | - | 7.81 / 7.83 | 25.59x over cooperative chunk 1 |
| Wave position tile | 8 | - | - | 6.84 / 6.89 | Best position result; 29.22x over chunk 1 |
| Thread output | - | - | - | 5.89 / 6.88 | Fixed output-owned mapping |
| Wave row tile 8 | - | - | - | 1.70 / 1.70 | Best fixed SparseWave result |
| rocSPARSE | - | - | - | 0.41 / 0.42 | Library baseline |

The transformations are not additive in isolation. For example, RHS-major
order with atomic reduction is slower than position-major order. Its benefit
appears only when a sufficiently large chunk and segmented aggregation make
more adjacent products contribute to the same output key. The best
thread-position combination is 4.57x faster than the untransformed
thread-position case, while cooperative chunking is 29.22x faster than its
chunk-1 case.

Despite these gains, the best position result remains 4.02x slower than the
best fixed SparseWave mapping and 16.79x slower than rocSPARSE for this
workload.

## Workload sweep

The breadth sweep fixes the best representative settings from the ablation:

- `wave-per-row-tile`: tile 8;
- `thread-per-position`: chunk 8, RHS-major, segmented;
- `wave-per-position-tile`: chunk 8;
- all SparseWave mappings: block size 256.

Each row reports the best of the four SparseWave mappings, its median, the
rocSPARSE median, and `rocSPARSE / SparseWave`. Values above 1.0 mean that
SparseWave is faster.

| Distribution | NNZ/row | RHS | Best SparseWave mapping | SparseWave | rocSPARSE | Speedup |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| Uniform | 4 | 8 | Thread output | 78.54 us | 88.74 us | **1.13x** |
| Uniform | 4 | 32 | Thread output | 210.65 us | 134.83 us | 0.64x |
| Uniform | 4 | 128 | Thread output | 800.97 us | 209.75 us | 0.26x |
| Uniform | 32 | 8 | Wave row tile | 77.28 us | 115.64 us | **1.50x** |
| Uniform | 32 | 32 | Wave row tile | 304.39 us | 163.51 us | 0.54x |
| Uniform | 32 | 128 | Wave row tile | 1.44 ms | 555.20 us | 0.39x |
| Uniform | 256 | 8 | Wave row tile | 316.95 us | 1.61 ms | **5.09x** |
| Uniform | 256 | 32 | Wave row tile | 1.31 ms | 876.97 us | 0.67x |
| Uniform | 256 | 128 | Wave row tile | 9.97 ms | 4.07 ms | 0.41x |
| Alternating | 4 | 8 | Thread output | 26.68 us | 30.20 us | **1.13x** |
| Alternating | 4 | 32 | Wave row tile | 289.39 us | 49.46 us | 0.17x |
| Alternating | 4 | 128 | Thread output | 820.41 us | 132.32 us | 0.16x |
| Alternating | 32 | 8 | Wave row tile | 91.86 us | 115.64 us | **1.26x** |
| Alternating | 32 | 32 | Wave row tile | 329.33 us | 164.40 us | 0.50x |
| Alternating | 32 | 128 | Wave row tile | 1.61 ms | 589.72 us | 0.37x |
| Alternating | 256 | 8 | Wave row tile | 452.95 us | 2.45 ms | **5.42x** |
| Alternating | 256 | 32 | Wave row tile | 1.37 ms | 1.68 ms | **1.23x** |
| Alternating | 256 | 128 | Wave row tile | 9.72 ms | 4.62 ms | 0.48x |
| Skewed | 4 | 8 | Thread output | 66.60 us | 51.38 us | 0.77x |
| Skewed | 4 | 32 | Wave row tile | 289.94 us | 70.16 us | 0.24x |
| Skewed | 4 | 128 | Thread output | 332.63 us | 187.88 us | 0.56x |
| Skewed | 32 | 8 | Wave row tile | 406.39 us | 228.23 us | 0.56x |
| Skewed | 32 | 32 | Wave row tile | 357.11 us | 386.05 us | **1.08x** |
| Skewed | 32 | 128 | Wave row tile | 1.70 ms | 1.05 ms | 0.62x |
| Skewed | 256 | 8 | Wave row tile | 931.10 us | 1.81 ms | **1.94x** |
| Skewed | 256 | 32 | Wave row tile | 1.40 ms | 3.01 ms | **2.15x** |
| Skewed | 256 | 128 | Wave row tile | 10.03 ms | 7.75 ms | 0.77x |

SparseWave wins 10 of 27 workloads. Seven wins occur at RHS 8, three at RHS
32, and none at RHS 128. The strongest region is a narrow RHS with long rows:
the wave-row mapping amortizes traversal and reduction within each row, while
rocSPARSE is particularly sensitive to the alternating long-row input in this
run. Wider RHS values favor rocSPARSE.

The best SparseWave mapping is wave-row tile in 20 workloads and thread-output
in seven. A position mapping is never the fastest SparseWave choice; even its
best representative result is 3.05x slower than the corresponding fixed
mapping, and the gap reaches approximately 83x on long-row workloads.

## Wave-segmented follow-up

The `wave-per-position` follow-up maps consecutive points in the collapsed
iteration space to adjacent lanes. A wave segmented-prefix reduction combines
adjacent equal output keys and only the final lane in each segment performs an
atomic update. This differs from `wave-per-position-tile`, where lanes
cooperate across RHS columns for one sparse position.

The follow-up used the same generated shapes, block size 256, row tile 8, and
cooperative chunk 8. It used 10 warmup and 20 measured dispatches per case and
compared both collapsed-axis orders in the same run as the fixed mappings,
cooperative mapping, and rocSPARSE. All 162 results passed CPU-reference
validation. It was measured on 2026-11-02 from a working tree based on
`a5d57b4e497619c8e4fe705725d5385ab85550cc` and includes the segmented-scan
correction described below; the remaining environment matches the table
above.

The table reports `position-major / RHS-major` median time. A value above 1.0
means that exposing adjacent row segments with RHS-major order is faster.

| Distribution | NNZ/row | RHS 8 | RHS 32 | RHS 128 |
| --- | ---: | ---: | ---: | ---: |
| Uniform | 4 | 1.59x | 1.10x | 0.88x |
| Uniform | 32 | 2.37x | 1.76x | 1.31x |
| Uniform | 256 | 11.36x | 9.69x | 6.10x |
| Alternating | 4 | 1.13x | 0.97x | 0.61x |
| Alternating | 32 | 4.09x | 3.26x | 2.36x |
| Alternating | 256 | 17.88x | 15.04x | 7.80x |
| Skewed | 4 | 0.54x | 1.35x | 0.79x |
| Skewed | 32 | 11.88x | 11.69x | 6.72x |
| Skewed | 256 | **25.58x** | 16.68x | 5.31x |

RHS-major is faster in 22 of 27 workloads. Its benefit grows with row length
because more adjacent lanes share one `(row, RHS column)` output key. Five
short-row cases favor position-major: the available segments are too short to
repay the reordered dense access pattern.

RHS-major wave segmentation beats the cooperative wave mapping in all nine
RHS-8 workloads, by 1.24x to 5.87x. Only eight of 32 lanes produce results in
the cooperative mapping at that width, whereas the segmented mapping uses the
lanes for sparse positions. At RHS 32 and 128, the cooperative mapping wins
all 18 workloads because its lanes cover dense columns and share sparse-side
work.

Neither wave-position order beats the fixed mappings. The best RHS-major case
is 2.99x slower than the corresponding fixed mapping, and the gap reaches
approximately 131x. It is also at least 7.69x slower than rocSPARSE. Wave
segmentation removes many atomics within a wave, but every segment crossing a
wave boundary still emits a partial atomic update, while the row-owned mapping
retains the entire accumulation and performs one final store.

| Mapping | Configuration | VGPR | SGPR | Spills | LDS | Scratch |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Wave position | Position-major | 25 | 40 | 0 | 0 | 0 |
| Wave position | RHS-major | 25 | 40 | 0 | 0 | 0 |
| Wave position tile | Chunk 8 | 34 | 50 | 0 | 0 | 0 |

The two orders generate the same resource counts, and neither spills or uses
LDS or scratch. The performance difference therefore comes from iteration
order and segment length rather than occupancy loss from register allocation.

The first full sweep also exposed a segmented-scan correctness defect. The
scan compared keys again at every shuffle offset, so a sequence such as
`A, B, ..., A, B` could combine equal but non-adjacent keys at a larger
offset. Uniform position-major SpMM consequently multiplied some outputs by
the row length. The corrected scan propagates segment-start markers with its
partial sums, preventing any later shuffle from crossing an earlier boundary.
The original failing uniform workload and representative SpMV wave-position
workloads pass after this correction.

## GPU resources

Resource usage is invariant across the generated shapes for a fixed compiled
configuration:

| Mapping | Configuration | VGPR | SGPR | Spills | LDS | Scratch |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Thread output | Block 256 | 20 | 20 | 0 | 0 | 0 |
| Wave row tile | Tile 8, block 256 | 55 | 54 | 0 | 0 | 0 |
| Thread position | Chunk 8, RHS-major, segmented | 32 | 41 | 0 | 0 | 0 |
| Wave position tile | Chunk 8 | 34 | 45 | 0 | 0 | 0 |

No tested kernel spills or uses LDS or scratch memory. Register pressure
therefore does not explain the position-schedule gap. The position schedules
recover output coordinates and perform global atomic updates after splitting
the sparse traversal. Split and segmented reduction reduce those costs, but
they do not recover the row-owned accumulation and single final store used by
the fixed wave-row mapping.

The wave-segmented follow-up shows that wave-local grouping alone is not
enough. The next research question is how to preserve a row segment across
wave boundaries or merge wave partials through a workspace without returning
to an atomic update for every partial. Applying generic position parallelism
directly to every sparse product is not competitive on this GPU.

## Reproduction

Run the transformation ablation with:

```sh
python3 benchmark/run_spmm_benchmark.py \
  --chip=gfx1101 \
  --rows=65536 \
  --columns=65536 \
  --nnz-per-row=32 \
  --distributions=skewed \
  --formats=csr \
  --mappings=thread-per-output,wave-per-row-tile,thread-per-position,wave-per-position-tile \
  --rhs-columns=32 \
  --block-sizes=256 \
  --tile-sizes=8,16 \
  --position-chunk-sizes=1,4,8 \
  --position-orders=position-major,rhs-major \
  --position-reductions=atomic,segmented \
  --warmup=10 \
  --iterations=50 \
  --rocsparse
```

Run the breadth sweep with:

```sh
python3 benchmark/run_spmm_benchmark.py \
  --chip=gfx1101 \
  --rows=65536 \
  --columns=65536 \
  --nnz-per-row=4,32,256 \
  --distributions=uniform,alternating,skewed \
  --formats=csr \
  --mappings=thread-per-output,wave-per-row-tile,thread-per-position,wave-per-position-tile \
  --rhs-columns=8,32,128 \
  --block-sizes=256 \
  --tile-sizes=8 \
  --position-chunk-sizes=8 \
  --position-orders=rhs-major \
  --position-reductions=segmented \
  --warmup=10 \
  --iterations=50 \
  --rocsparse
```

Run the wave-segmented follow-up with:

```sh
python3 benchmark/run_spmm_benchmark.py \
  --chip=gfx1101 \
  --rows=65536 \
  --columns=65536 \
  --nnz-per-row=4,32,256 \
  --distributions=uniform,alternating,skewed \
  --formats=csr \
  --mappings=thread-per-output,wave-per-row-tile,wave-per-position,wave-per-position-tile \
  --rhs-columns=8,32,128 \
  --block-sizes=256 \
  --tile-sizes=8 \
  --position-chunk-sizes=8 \
  --position-orders=position-major,rhs-major \
  --warmup=10 \
  --iterations=20 \
  --rocsparse
```
