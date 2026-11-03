# Wave position chunking ablation on gfx1101

This report evaluates wave chunking in SparseWave's generic position-space
scheduler. One physical wave processes one lane-width position tile when the
chunk size is 1. At larger chunk sizes, the same wave processes multiple
consecutive tiles and emits one workspace partial for each tile.

The experiment asks three questions:

1. Does reusing a wave across adjacent position tiles reduce scheduling and
   compressed-coordinate recovery cost?
2. How does the result interact with multidimensional iteration order?
3. Is the improvement sufficient to compete with row-owned SparseWave
   mappings or rocSPARSE?

The results are a snapshot for the revisions below, not a general performance
guarantee.

## Environment

| Component | Value |
| --- | --- |
| Date | 2026-11-03 |
| GPU | AMD Radeon RX 7800 XT |
| Target chip | gfx1101 |
| Wavefront size | 32 |
| SparseWave commit | `2418f5041b5e7690fcc1138a89feb0a5f6ec908c` |
| LLVM commit | `64b593c2371b7f7225b0ec190a37cd8b672e4c5d` |
| ROCm version | 6.4.4-129 |
| rocSPARSE version | 300400, Git revision `8fbfc797` |
| rocSPARSE algorithm | Default CSR SpMV/SpMM |
| Block size | 256 |
| Warmup dispatches | 10 |
| Measured dispatches | 20 |

SparseWave times are the sum of all operation kernels reported by
`rocprofv3`. The workspace strategy includes output initialization, partial
generation, and final workspace reduction. rocSPARSE uses HIP events around
the complete compute call. GPU clocks were not locked, so comparisons use
configurations measured within these runs. All 72 SpMV results and all 162
SpMM results passed CPU-reference validation.

## Mechanism

The scheduler assigns consecutive lane-width tiles to one physical wave:

```text
chunk 1: wave 0 -> positions [  0,  32)
         wave 1 -> positions [ 32,  64)

chunk 4: wave 0 -> positions [  0,  32), [ 32,  64),
                              [ 64,  96), [ 96, 128)
         wave 1 -> positions [128, 160), ...
```

For a monotonically increasing compressed position, the first tile performs
the cooperative segment search. Later tiles carry each lane's current segment
and next segment boundary, advancing them only when a position crosses that
boundary. Chunking therefore reduces physical waves and avoids restarting a
full segment search for every tile.

Each tile remains a separate reduction partition. This preserves the ordered
workspace reduction and isolates the scheduling transformation from reduction
semantics.

## Generated inputs

Every matrix has 65,536 rows and columns. The requested NNZ/row is the exact
mean in all distributions:

| Distribution | Row lengths for mean `N` |
| --- | --- |
| Uniform | Every row has `N` entries |
| Alternating | Rows alternate between 1 and `2N - 1` entries |
| Skewed | Seven rows have 1 entry, followed by one row with `8N - 7` entries |

Chunk sizes 1, 2, 4, and 8 were measured with workspace reduction. SpMV uses
one position axis. SpMM measures both `position,rhs` (position-major) and
`rhs,position` (RHS-major) collapsed-axis orders at RHS widths 8 and 32.

## SpMV results

Each entry is median / p95 in microseconds.

| Distribution | NNZ/row | Chunk 1 | Chunk 2 | Chunk 4 | Chunk 8 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Uniform | 4 | 40.04 / 117.16 | **36.74 / 37.08** | 37.88 / 38.48 | 37.34 / 37.88 |
| Uniform | 32 | 156.09 / 156.53 | 125.78 / 395.73 | 104.00 / 104.40 | **95.42 / 95.72** |
| Uniform | 256 | 1,322.50 / 1,326.16 | 1,030.45 / 1,034.23 | 844.51 / 848.90 | **771.09 / 774.74** |
| Alternating | 4 | 39.30 / 39.48 | **37.22 / 37.64** | 37.52 / 104.08 | 37.36 / 106.88 |
| Alternating | 32 | 183.29 / 184.56 | 147.54 / 148.04 | 118.26 / 119.68 | **106.18 / 106.64** |
| Alternating | 256 | 1,521.07 / 1,523.33 | 1,238.48 / 1,241.11 | 1,055.91 / 1,059.43 | **950.79 / 956.27** |
| Skewed | 4 | 39.48 / 39.80 | 37.26 / 37.88 | 37.56 / 38.20 | **37.24 / 38.00** |
| Skewed | 32 | 192.81 / 193.53 | 176.82 / 177.76 | 139.10 / 139.84 | **120.78 / 121.76** |
| Skewed | 256 | 2,210.43 / 2,216.81 | 1,927.31 / 1,943.69 | 1,756.59 / 1,763.49 | **1,675.99 / 1,684.65** |

Short rows improve by only 1.06x to 1.09x because chunk 1 already creates a
small amount of work and the larger loop state is not well amortized. Medium
and long rows improve by 1.32x to 1.73x, and every such workload selects chunk
8. Larger factors were not measured, so chunk 8 is the best tested value, not
an established optimum.

The best wave-chunk result beats rocSPARSE only for skewed 32-NNZ rows, by
1.36x. It also beats the best measured row-owned SparseWave mapping for that
workload and for uniform 32-NNZ rows. Across all nine workloads, however,
rocSPARSE is 1.99x faster than the best wave chunk by geometric mean. Long-row
row-owned mappings retain a full row accumulation and remain faster than the
position schedule.

Several p95 outliers appear in short workloads and uniform chunk 2 at 32
NNZ/row. They do not change the median trend, but show why tail latency must be
retained instead of selecting configurations from the median alone.

## SpMM results and iteration order

The table shows chunk 1 and chunk 8 as median / p95 in milliseconds. Chunks 2
and 4 were also measured. Position-major improves monotonically from 1 through
8 in every workload, while RHS-major selects chunk 1 in every workload.

| Distribution | NNZ/row | RHS | Position-major c1 | Position-major c8 | RHS-major c1 | RHS-major c8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Uniform | 4 | 8 | 0.30 / 0.30 | **0.19 / 0.19** | **0.28 / 0.28** | 0.33 / 0.33 |
| Uniform | 4 | 32 | 1.01 / 1.01 | **0.51 / 0.51** | **1.14 / 1.16** | 1.24 / 1.25 |
| Uniform | 32 | 8 | 4.78 / 4.79 | **1.43 / 1.43** | **2.06 / 2.06** | 2.29 / 2.29 |
| Uniform | 32 | 32 | 17.55 / 17.67 | **5.59 / 5.60** | **8.57 / 8.76** | 9.53 / 9.65 |
| Uniform | 256 | 8 | 214.10 / 214.75 | **53.43 / 54.35** | **18.93 / 19.05** | 20.30 / 20.46 |
| Uniform | 256 | 32 | 859.14 / 860.99 | **196.38 / 196.97** | **78.10 / 78.55** | 85.80 / 86.19 |
| Alternating | 4 | 8 | 0.34 / 0.35 | **0.18 / 0.18** | **0.28 / 0.28** | 0.33 / 0.33 |
| Alternating | 4 | 32 | 1.14 / 1.14 | **0.52 / 0.52** | **1.13 / 1.14** | 1.25 / 1.26 |
| Alternating | 32 | 8 | 8.09 / 8.13 | **1.77 / 1.78** | **2.09 / 2.10** | 2.31 / 2.31 |
| Alternating | 32 | 32 | 30.40 / 30.81 | **8.49 / 8.53** | **8.83 / 9.07** | 9.79 / 10.04 |
| Alternating | 256 | 8 | 389.85 / 390.69 | **79.24 / 80.76** | **20.42 / 20.51** | 21.60 / 21.73 |
| Alternating | 256 | 32 | 1,523.71 / 1,525.93 | **369.19 / 370.17** | **83.24 / 84.82** | 90.63 / 91.34 |
| Skewed | 4 | 8 | 0.48 / 0.48 | **0.19 / 0.19** | **0.28 / 0.28** | 0.33 / 0.33 |
| Skewed | 4 | 32 | 1.80 / 1.81 | **0.78 / 0.78** | **1.14 / 1.14** | 1.25 / 1.25 |
| Skewed | 32 | 8 | 25.39 / 25.59 | **6.54 / 6.57** | **2.11 / 2.12** | 2.35 / 2.36 |
| Skewed | 32 | 32 | 116.34 / 117.91 | **35.76 / 36.40** | **9.37 / 9.50** | 10.34 / 10.48 |
| Skewed | 256 | 8 | 1,183.00 / 1,184.90 | **249.10 / 250.20** | **25.83 / 25.97** | 27.29 / 27.44 |
| Skewed | 256 | 32 | 3,845.69 / 3,852.91 | **1,621.81 / 1,624.39** | **128.62 / 130.02** | 135.54 / 136.63 |

Position-major chunk 8 improves over chunk 1 by 3.09x geometric mean, with a
range of 1.58x to 4.92x. Its position coordinate is globally nondecreasing, so
the scheduler can carry segment boundaries across tiles.

RHS-major starts a new position traversal for every RHS column. The scheduler
must conservatively repeat coordinate recovery after each restart. Chunking
therefore adds loop state and reduces physical parallelism without reusing the
segment search. Chunk 8 is 1.06x to 1.17x slower than chunk 1 in this order.

Order remains more important than chunking on long rows. RHS-major chunk 1 is
faster than position-major chunk 8 in every 256-NNZ SpMM case because adjacent
lanes share an output key for longer segments. No measured SpMM position
configuration beats rocSPARSE; rocSPARSE is 3.73x to 85.48x faster than the
best measured wave position configuration.

## GPU resources and ISA size

Resource counts are invariant across the generated matrix shapes for a fixed
configuration. No measured kernel spills or uses LDS or scratch memory.

| Operation / order | Chunk | VGPR | SGPR | Kernel code bytes |
| --- | ---: | ---: | ---: | ---: |
| SpMV | 1 | 36 | 30 | 3,852 |
| SpMV | 2 | 46 | 56 | 5,208 |
| SpMV | 4 | 46 | 56 | 5,232 |
| SpMV | 8 | 46 | 56 | 5,232 |
| SpMM position-major | 1 | 39 | 42 | 5,656 |
| SpMM position-major | 2 | 51 | 72 | 7,200 |
| SpMM position-major | 4 | 51 | 72 | 7,212 |
| SpMM position-major | 8 | 51 | 72 | 7,200 |
| SpMM RHS-major | 1 | 31 | 42 | 5,404 |
| SpMM RHS-major | 2 | 49 | 63 | 6,732 |
| SpMM RHS-major | 4 | 46 | 63 | 6,768 |
| SpMM RHS-major | 8 | 46 | 63 | 6,748 |

Chunk sizes above 1 retain the same dynamic tile loop rather than unrolling a
separate body for every tile. This is why code size and register use are nearly
constant for chunks 2, 4, and 8. The one-time increase from chunk 1 stores the
carried segment, boundary, tile state, and workspace coordinates.

The performance result is consequently a balance between fewer waves and
coordinate-search reuse on one side, and reduced parallelism plus higher
register pressure on the other. The RHS-major result isolates the latter costs
when reuse is unavailable.

## Conclusion

Wave chunking is useful as a generic position-space transformation, but its
legality alone does not predict profitability. It should be selected together
with iteration order and knowledge of whether the compressed position remains
monotonic across a chunk. A future automatic strategy should therefore expose
these properties explicitly rather than choose a chunk size from row length
alone.

The result also reinforces the role of SparseWave as an experimental compiler:
the transformation produces a substantial isolated speedup, while the fair
rocSPARSE and row-owned comparisons show that it is not yet a generally
competitive final schedule.

## Reproduction

Run the SpMV sweep with:

```sh
python3 benchmark/run_spmv_benchmark.py \
  --chip=gfx1101 \
  --rows=65536 \
  --columns=65536 \
  --nnz-per-row=4,32,256 \
  --distributions=uniform,alternating,skewed \
  --formats=csr \
  --position-wave-chunk-sizes=1,2,4,8 \
  --position-reductions=workspace \
  --rocsparse \
  --block-sizes=256 \
  --warmup=10 \
  --iterations=20 \
  --keep-artifacts
```

Run the SpMM sweep with:

```sh
python3 benchmark/run_spmm_benchmark.py \
  --chip=gfx1101 \
  --rows=65536 \
  --columns=65536 \
  --nnz-per-row=4,32,256 \
  --distributions=uniform,alternating,skewed \
  --formats=csr \
  --mappings=wave-per-position \
  --rhs-columns=8,32 \
  --block-sizes=256 \
  --position-wave-chunk-sizes=1,2,4,8 \
  --position-orders=position-major,rhs-major \
  --position-reductions=workspace \
  --warmup=10 \
  --iterations=20 \
  --rocsparse \
  --keep-artifacts
```
