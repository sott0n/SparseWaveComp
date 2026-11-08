# Position-space transformation ablation on gfx1101

This report combines SparseWave's position-space transformations into one
ablation workflow. The benchmark runner selects the fastest measured
configuration in each stage and writes the selection to
`position-ablation.csv`. The full `results.csv` remains the source for exact
pairwise comparisons.

The stages separate fixed ownership, the untransformed position baseline,
split, reorder, local segmented reduction, wave and block schedules, the best
manually enumerated SparseWave configuration, and rocSPARSE. SpMV omits
reorder because its position domain is one-dimensional.

## Environment

| Component | Value |
| --- | --- |
| Date | 2026-11-08 |
| GPU | AMD Radeon RX 7800 XT |
| Target chip | gfx1101 |
| Wavefront size | 32 |
| SparseWave base commit | `80f3c56c06aa81a8bc49a8d3c2a2a3c1f46b107d` |
| LLVM commit | `64b593c2371b7f7225b0ec190a37cd8b672e4c5d` |
| ROCm version | 6.4.4-129 |
| rocSPARSE version | 300400, Git revision `8fbfc797` |
| Warmup / measured dispatches | 5 / 20 |
| GPU performance level | Automatic |

SparseWave times are the complete dispatch time reported by `rocprofv3`.
rocSPARSE uses HIP events around the complete operation. All configurations
passed CPU-reference validation. GPU clocks were not locked, so the results
support comparisons within these runs rather than cross-run absolute-latency
claims.

The inputs reproduce the pattern and dimensions of
[SNAP/ca-GrQc](https://sparse.tamu.edu/SNAP/ca-GrQc) and
[SNAP/ca-AstroPh](https://sparse.tamu.edu/SNAP/ca-AstroPh). The official SNAP
edge lists were reindexed densely and stored as Matrix Market `pattern
general` inputs. They contain 5,242 rows and 28,980 entries for ca-GrQc and
18,772 rows and 396,160 entries for ca-AstroPh.

Each stage reports its best measured configuration. Consequently, adjacent
rows may select different chunk or mapping parameters; the table is a
best-of-stage frontier, not an assumption that transformation speedups are
strictly additive.

## SpMV

Both matrices use a 128-thread block. Times are median / p95.

| Matrix | Stage | Selected configuration | Time | vs position baseline | vs rocSPARSE |
| --- | --- | --- | ---: | ---: | ---: |
| ca-GrQc | Position baseline | Thread, chunk 1, atomic | 121.42 / 130.48 us | 1.00x | 0.29x |
| ca-GrQc | Split | Thread, chunk 8, atomic | 37.50 / 113.92 us | 3.24x | 0.93x |
| ca-GrQc | Local segmented | Thread, chunk 4, segmented | 13.40 / 13.60 us | 9.06x | 2.61x |
| ca-GrQc | Wave position | Wave, chunk 1, atomic | **10.40 / 10.60 us** | **11.68x** | **3.36x** |
| ca-GrQc | Block position | Block, chunk 1, segmented | 14.28 / 17.16 us | 8.50x | 2.45x |
| ca-GrQc | Fixed mapping | Wave per row | 15.12 / 15.52 us | 8.03x | 2.31x |
| ca-GrQc | rocSPARSE | Default | 34.96 / 35.76 us | 3.47x | 1.00x |
| ca-AstroPh | Position baseline | Thread, chunk 1, atomic | 3,012.65 / 3,170.41 us | 1.00x | 0.02x |
| ca-AstroPh | Split | Thread, chunk 8, atomic | 608.49 / 753.05 us | 4.95x | 0.09x |
| ca-AstroPh | Local segmented | Thread, chunk 8, segmented | 127.02 / 130.92 us | 23.72x | 0.43x |
| ca-AstroPh | Wave position | Wave, chunk 2, atomic | 85.26 / 85.76 us | 35.33x | 0.64x |
| ca-AstroPh | Block position | Block, chunk 8, segmented | 73.32 / 74.72 us | 41.09x | 0.74x |
| ca-AstroPh | Fixed mapping | Wave per row | **41.80 / 42.84 us** | **72.07x** | **1.31x** |
| ca-AstroPh | rocSPARSE | Default | 54.58 / 56.44 us | 55.20x | 1.00x |

ca-GrQc is the first measured real-matrix region where the generic position
schedule is the best SparseWave strategy. Wave position is 1.45x faster than
wave per row and 3.36x faster than rocSPARSE. Thread-local segmented reduction
also beats both references, showing that the result is not tied to one GPU
mapping.

The same conclusion does not transfer to ca-AstroPh. Split and aggregation
recover 23.72x over the untransformed position baseline, and block position
recovers 41.09x, but fixed wave-per-row ownership remains 1.75x faster than the
best position schedule. The longer rows make cooperative row ownership
effective without paying cross-worker atomic updates.

## SpMM

The ca-GrQc SpMM run uses a 128-thread block. Fixed wave-row tiles cover sizes
4 and 8; position chunks cover 1, 4, and 8. The table retains the stages that
isolate split, reorder, and local segmented reduction.

| RHS columns | Stage | Selected configuration | Median | vs position baseline | vs rocSPARSE |
| ---: | --- | --- | ---: | ---: | ---: |
| 8 | Position baseline | Thread, chunk 1, position-major, atomic | 53.28 us | 1.00x | 0.42x |
| 8 | Split | Thread, chunk 4, position-major, atomic | 83.36 us | 0.64x | 0.27x |
| 8 | Reorder | Thread, chunk 8, RHS-major, atomic | 55.48 us | 0.96x | 0.40x |
| 8 | Local segmented | Thread, chunk 8, RHS-major, segmented | **24.56 us** | **2.17x** | **0.90x** |
| 8 | Fixed mapping | Wave-row tile 8 | 28.20 us | 1.89x | 0.79x |
| 8 | rocSPARSE | Default | 22.16 us | 2.40x | 1.00x |
| 32 | Position baseline | Thread, chunk 1, position-major, atomic | 133.72 us | 1.00x | 0.13x |
| 32 | Split | Thread, chunk 4, position-major, atomic | 144.52 us | 0.93x | 0.12x |
| 32 | Reorder | Thread, chunk 8, RHS-major, atomic | 149.98 us | 0.89x | 0.12x |
| 32 | Local segmented | Thread, chunk 8, RHS-major, segmented | 75.90 us | 1.76x | 0.24x |
| 32 | Fixed mapping | Wave-row tile 4 | **41.42 us** | **3.23x** | **0.43x** |
| 32 | rocSPARSE | Default | 17.84 us | 7.50x | 1.00x |

At RHS 8, split or reorder alone does not help, but their combination with
segmented reduction reduces the position baseline by 2.17x. It becomes the
best SparseWave configuration, 1.15x faster than the fixed mapping and within
11% of rocSPARSE. At RHS 32, the cost of the expanded output-key space and
atomic partials dominates; fixed wave-row tiling is 1.83x faster than the best
position stage, and rocSPARSE remains substantially faster.

## GPU resources

None of the selected kernels spills registers or uses scratch memory. The
SpMV fixed wave-row kernel uses 14 VGPRs and 20 SGPRs; the selected thread
segmented, wave-position, and block-position kernels use 20/30, 32/32, and
24/42 VGPRs/SGPRs respectively. Only block segmented reduction uses LDS, with
112 bytes for a 128-thread block.

For SpMM RHS 8, the selected thread-segmented kernel uses 34 VGPRs and 53
SGPRs, while fixed wave-row tile 8 uses 55 VGPRs and 54 SGPRs. The position
result therefore does not come from accepting spills or scratch traffic; it
changes dynamic traversal and output aggregation while using fewer vector
registers than the tiled fixed mapping.

## Interpretation

- Position-space load balancing is not a universal replacement for row
  ownership. It wins on the smaller irregular ca-GrQc SpMV input and loses to
  fixed ownership on ca-AstroPh.
- Split is an enabling transformation rather than an independently reliable
  optimization. Its benefit appears when adjacent contributions can be
  aggregated before reaching global memory.
- The best mapping changes with the operator and dense RHS width. SpMM needs
  enough output-key locality for reorder and segmented reduction to repay the
  larger collapsed iteration domain.
- The experiment identifies an AMD GPU performance region for the reusable
  research mechanism while retaining the losing cases required for a fair
  comparison.

## Reproduction

Add `--position-ablation` to a normal SpMV or SpMM benchmark invocation. The
runner prints the selected stages and writes `position-ablation.csv` beside
`results.csv` and `metadata.json`.

Run the SpMV sweep once for each Matrix Market input:

```sh
python3 benchmark/run_spmv_benchmark.py \
  --chip=gfx1101 \
  --matrix=/path/to/matrix.mtx \
  --formats=csr \
  --mappings=thread-per-row,thread-per-position,wave-per-position,block-per-position,wave-per-row,block-per-row \
  --position-chunk-sizes=1,4,8 \
  --position-wave-chunk-sizes=1,2,4 \
  --position-reductions=atomic,segmented,workspace \
  --block-sizes=64,128 \
  --warmup=5 \
  --iterations=20 \
  --rocsparse \
  --position-ablation
```

Run the ca-GrQc SpMM sweep with:

```sh
python3 benchmark/run_spmm_benchmark.py \
  --chip=gfx1101 \
  --matrix=/path/to/ca-GrQc.mtx \
  --formats=csr \
  --mappings=thread-per-output,wave-per-row-tile,thread-per-position,wave-per-position,block-per-position,wave-per-position-tile \
  --rhs-columns=8,32 \
  --tile-sizes=4,8 \
  --position-chunk-sizes=1,4,8 \
  --position-wave-chunk-sizes=1,2 \
  --position-orders=position-major,rhs-major \
  --position-reductions=atomic,segmented,workspace \
  --block-sizes=128 \
  --warmup=5 \
  --iterations=20 \
  --rocsparse \
  --position-ablation
```
