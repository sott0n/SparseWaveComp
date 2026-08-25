# Position-space SpMM reorder on gfx1101

This report evaluates a TACO-style reorder of SparseWave's collapsed CSR SpMM
iteration space. The experiment separates the effect of changing iteration
order from the effect of applying the existing operator-independent segmented
reduction.

## Transformation

The original position-major order keeps RHS columns adjacent:

```text
i = position * rhsColumns + rhsColumn
position  = i / rhsColumns
rhsColumn = i % rhsColumns
```

The reordered RHS-major form keeps CSR positions adjacent for a fixed output
column:

```text
i = rhsColumn * NNZ + position
position  = i % NNZ
rhsColumn = i / NNZ
```

The SpMM decomposition does not emit these quotient and remainder operations
itself. It creates a generic rank-2 keyed reduction:

```text
position_reduce
  lower = (0, 0)
  upper = (NNZ, rhsColumns)
  order = (position, rhsColumn) or (rhsColumn, position)
  body(position, rhsColumn) -> (outputKey, product)
```

The operator-independent position scheduler computes the collapsed worker
count and recovers logical coordinates from the order permutation. Rank-1
SpMV and the rank-2 SpMM domain therefore use the same reduction and scheduling
mechanism; only the operator-specific contribution body and selected axis
order differ.

CSR stores positions from the same row consecutively. RHS-major ordering
therefore makes contributions to `output[row, rhsColumn]` adjacent and allows
the generic position scheduler to combine equal-key contributions inside each
thread chunk before emitting an atomic update. Position-major remains the
default, and iteration order and reduction strategy are independent options so
their effects can be measured separately.

## Environment

| Component | Value |
| --- | --- |
| Date | 2026-08-24 |
| GPU | AMD Radeon RX 7800 XT |
| Target chip | gfx1101 |
| Wavefront size | 32 |
| SparseWave base commit | `a7362b7c1aa91b076482a9c627efb5e3a47089a8` plus this change |
| LLVM commit | `64b593c2371b7f7225b0ec190a37cd8b672e4c5d` |
| ROCm version | 6.4.4-129 |
| Matrix | SuiteSparse SNAP/ca-GrQc |
| Shape | 5,242 x 5,242, 28,980 NNZ |

All measured configurations passed CPU-reference output validation. GPU clocks
were not locked, so ratios below compare configurations within the same run;
absolute latency should not be compared across the two runs.

## Reorder ablation

This run used block size 64, chunk size 4, 20 warmup dispatches, and 200
measured dispatches.

| RHS columns | Order | Reduction | Median | p95 | GFLOP/s |
| ---: | --- | --- | ---: | ---: | ---: |
| 8 | position-major | atomic | 256.03 us | 274.17 us | 1.81 |
| 8 | position-major | segmented | 223.37 us | 238.13 us | 2.08 |
| 8 | rhs-major | atomic | 138.18 us | 143.04 us | 3.36 |
| 8 | rhs-major | segmented | **75.28 us** | 77.60 us | **6.16** |
| 32 | position-major | atomic | 255.33 us | 264.05 us | 7.26 |
| 32 | position-major | segmented | 137.10 us | 139.04 us | 13.53 |
| 32 | rhs-major | atomic | 193.09 us | 197.01 us | 9.61 |
| 32 | rhs-major | segmented | **105.20 us** | 106.08 us | **17.63** |

In this run, RHS-major segmented is 3.40x faster than position-major atomic at
8 RHS columns and 2.43x faster at 32 columns. Repeated unlocked-clock runs had
substantial absolute-latency variation, but consistently selected RHS-major
segmented as the fastest tested position schedule. Exact speedup claims require
a future repeated, interleaved run with controlled GPU clocks.

The result demonstrates the intended compiler interaction: reorder does not
universally improve memory behavior, but it exposes equal-key segments that a
separate reusable reduction transformation can exploit.

## Existing-mapping comparison

A second run used 10 warmup and 100 measured dispatches. The table reports the
best tested configuration for each mapping at each RHS width.

| RHS columns | Mapping | Configuration | Median | p95 | GFLOP/s |
| ---: | --- | --- | ---: | ---: | ---: |
| 8 | thread-per-output | block 256 | 40.90 us | 43.00 us | 11.34 |
| 8 | wave-per-row-tile | block 256, tile 8 | **28.94 us** | 29.56 us | **16.02** |
| 8 | thread-per-position | block 64, chunk 4, RHS-major segmented | 116.82 us | 120.72 us | 3.97 |
| 32 | thread-per-output | block 256 | **40.56 us** | 41.84 us | **45.73** |
| 32 | wave-per-row-tile | block 256, tile 16 | 49.14 us | 74.04 us | 37.74 |
| 32 | thread-per-position | block 256, chunk 8, RHS-major segmented | 200.88 us | 352.05 us | 9.23 |

The reordered position schedule remains 4.04x slower than the best existing
mapping at 8 RHS columns and 4.95x slower at 32 columns in this run. Reorder
and segmented reduction remove a substantial part of the naive position-space
cost, but the
schedule still repeatedly recovers CSR coordinates and uses atomics at chunk
boundaries. It also gives up the sparse-value and contiguous dense-side reuse
available to row-owned mappings.

The RHS-major segmented kernel uses 32 VGPRs and 41 SGPRs for chunk sizes 4 and
8, with no spills, LDS, or scratch memory. Position-major atomic uses 27 VGPRs
and 36 SGPRs. The modest resource increase is not the primary remaining
bottleneck.

## Reproduction

Reorder ablation:

```sh
python3 benchmark/run_spmm_benchmark.py \
  --chip=gfx1101 \
  --matrix=/path/to/ca-GrQc.mtx \
  --formats=csr \
  --mappings=thread-per-position \
  --rhs-columns=8,32 \
  --block-sizes=64 \
  --position-chunk-sizes=4 \
  --position-orders=position-major,rhs-major \
  --position-reductions=atomic,segmented \
  --warmup=20 \
  --iterations=200
```

Existing-mapping comparison:

```sh
python3 benchmark/run_spmm_benchmark.py \
  --chip=gfx1101 \
  --matrix=/path/to/ca-GrQc.mtx \
  --formats=csr \
  --mappings=thread-per-output,wave-per-row-tile,thread-per-position \
  --rhs-columns=8,32 \
  --block-sizes=64,256 \
  --tile-sizes=8,16 \
  --position-chunk-sizes=4,8 \
  --position-orders=rhs-major \
  --position-reductions=segmented \
  --warmup=10 \
  --iterations=100
```
