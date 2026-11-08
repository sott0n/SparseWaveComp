# SpMV position-ablation suite on gfx1101

This report applies the integrated position-transformation ablation to the six
matrices in `benchmark/suites/spmv-position.json`. It evaluates whether the
position-space schedule remains effective beyond the two matrices used during
its development and compares it with the best measured fixed SparseWave
mapping and rocSPARSE.

## Environment

| Component | Value |
| --- | --- |
| Date | 2026-11-08 |
| GPU | AMD Radeon RX 7800 XT |
| Target chip | gfx1101 |
| Wavefront size | 32 |
| SparseWave commit | `4bb8607ea6be51172fafa0ef87c8f4da122ae82f` |
| LLVM commit | `64b593c2371b7f7225b0ec190a37cd8b672e4c5d` |
| ROCm version | 6.4.4-129 |
| rocSPARSE version | 300400, Git revision `8fbfc797` |
| Warmup / measured dispatches | 5 / 20 |
| GPU performance level | Automatic |

SparseWave times are complete dispatch times reported by `rocprofv3`.
rocSPARSE uses HIP events around the complete operation. Every one of the 258
measured configurations passed CPU-reference validation. The complete suite
was run twice independently. GPU clocks were not locked, so this report uses
the repeated direction of each comparison rather than treating small timing
differences as stable.

## Inputs

| Matrix | Rows | NNZ | Mean row NNZ | Maximum row NNZ | Coverage |
| --- | ---: | ---: | ---: | ---: | --- |
| minnesota | 2,642 | 6,606 | 2.50 | 5 | Short road-network rows |
| ca-GrQc | 5,242 | 28,980 | 5.53 | 81 | Short irregular rows |
| wiki-Vote | 8,297 | 103,689 | 12.50 | 893 | Short heavy-tail rows |
| ca-AstroPh | 18,772 | 396,160 | 21.10 | 504 | Medium irregular rows |
| email-Enron | 36,692 | 367,662 | 10.02 | 1,383 | Medium heavy-tail rows |
| kron_g500-logn16 | 65,536 | 4,912,469 | 74.96 | 17,998 | Long heavy-tail rows |

The four SNAP edge lists were obtained from the original SNAP distribution and
converted to Matrix Market without changing their entry count. ca-GrQc and
ca-AstroPh node identifiers were densely reindexed; wiki-Vote retained its
original 8,297-coordinate space; email-Enron used its zero-based identifiers.
minnesota and kron_g500-logn16 used Matrix Market copies from the Network Data
Repository because the SuiteSparse file server reset TLS connections from the
measurement host. Their dimensions, field, symmetry, and NNZ match the
SuiteSparse entries linked from the checked-in manifest.

## Stage ablation

The following table is the first complete suite run. Each stage selects its
fastest block size, chunk size, and supported reduction strategy. Times are
medians in microseconds.

| Matrix | Position baseline | Split | Local segmented | Wave position | Block position | Best fixed | rocSPARSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| minnesota | 22.06 | 31.14 | 14.54 | **9.32** | 9.36 | 4.96 | 14.18 |
| ca-GrQc | 33.42 | 29.54 | 14.16 | **9.58** | 12.84 | 6.28 | 35.88 |
| wiki-Vote | 394.03 | 170.98 | 62.14 | **17.92** | 20.34 | 14.20 | 22.84 |
| ca-AstroPh | 601.99 | 240.44 | 45.78 | **26.76** | 31.26 | 17.58 | 68.48 |
| email-Enron | 1,124.36 | 312.23 | 62.26 | 34.12 | **29.56** | 23.86 | 21.72 |
| kron_g500-logn16 | 111,749.85 | 14,849.50 | 2,062.46 | **233.82** | 247.19 | 160.92 | 197.16 |

Split alone is not sufficient. The untransformed thread-position schedule
performs one atomic update per contribution, and increasing its chunk only
reduces worker count. Local segmented reduction produces the first large
recovery by combining adjacent equal output keys. Wave and block mappings then
aggregate across participants and reduce the position baseline by 2.37x on
minnesota and by up to 477.92x on kron_g500-logn16.

## Repeatability and comparisons

`vs fixed` and `vs rocSPARSE` are reference time divided by position time, so
a value greater than 1 means that the selected position schedule is faster.
The selected configuration can change when close variants exchange order;
both independent runs are shown.

| Matrix | Best position, run 1 | Best position, run 2 | vs fixed, runs 1 / 2 | vs rocSPARSE, runs 1 / 2 |
| --- | --- | --- | ---: | ---: |
| minnesota | Wave atomic, chunk 1, 9.32 us | Block atomic, chunk 1, 9.42 us | 0.53x / 0.76x | 1.52x / 3.13x |
| ca-GrQc | Wave atomic, chunk 1, 9.58 us | Wave atomic, chunk 1, 10.24 us | 0.66x / 0.61x | 3.75x / 3.50x |
| wiki-Vote | Wave atomic, chunk 1, 17.92 us | Block segmented, chunk 4, 16.00 us | 0.79x / 0.78x | 1.27x / 2.75x |
| ca-AstroPh | Wave atomic, chunk 4, 26.76 us | Wave atomic, chunk 4, 27.04 us | 0.66x / 0.98x | 2.56x / 2.39x |
| email-Enron | Block segmented, chunk 8, 29.56 us | Block segmented, chunk 4, 27.50 us | 0.81x / 0.86x | 0.73x / 0.79x |
| kron_g500-logn16 | Wave workspace, chunk 4, 233.82 us | Wave workspace, chunk 4, 233.02 us | 0.69x / 0.69x | 0.84x / 0.86x |

The position mechanism beats rocSPARSE on minnesota, ca-GrQc, wiki-Vote, and
ca-AstroPh in both runs. It loses on email-Enron and kron_g500-logn16. The best
fixed SparseWave row mapping is faster on all six matrices in both runs,
although ca-AstroPh is within 2.3% in the second run.

The absolute rocSPARSE and fixed-mapping times vary most on the smaller
matrices under automatic GPU clocks. Position latency is more stable for
ca-GrQc, ca-AstroPh, email-Enron, and kron_g500-logn16, while minnesota and
wiki-Vote can change which close position variant wins. The repeated win/loss
regions are therefore stronger evidence than any individual ratio.

## GPU resources

Resources below correspond to the run-1 position selections.

| Matrix | Mapping | VGPR | SGPR | LDS | Spill / scratch |
| --- | --- | ---: | ---: | ---: | ---: |
| minnesota | Wave atomic, chunk 1 | 32 | 32 | 0 B | 0 |
| ca-GrQc | Wave atomic, chunk 1 | 32 | 32 | 0 B | 0 |
| wiki-Vote | Wave atomic, chunk 1 | 32 | 32 | 0 B | 0 |
| ca-AstroPh | Wave atomic, chunk 4 | 42 | 50 | 0 B | 0 |
| email-Enron | Block segmented, chunk 8 | 24 | 44 | 112 B | 0 |
| kron_g500-logn16 | Wave workspace, chunk 4 | 46 | 57 | 0 B | 0 |

None of the selected kernels spills registers or uses scratch memory. The
email-Enron block reduction is the only selection that uses LDS. The losses to
fixed mapping are therefore not caused by spilling; they come from recovering
row keys and combining partial results after flattening traversal into position
space.

## Interpretation

- The transformation implementation is effective: split, segmented
  aggregation, and hierarchical mapping recover between 2.37x and 477.92x
  over the naive position baseline.
- Position scheduling is competitive with an external library over a useful
  region, beating rocSPARSE on four of six matrices in both runs.
- Position scheduling is not yet the best SparseWave strategy for this suite.
  Direct row ownership avoids key recovery and cross-worker output merging
  when a row can already keep a wave or block occupied.
- Long heavy-tail rows are the clearest losing region. kron_g500-logn16 favors
  block-per-row, while email-Enron also favors row ownership despite its lower
  mean row length because its maximum row is 1,383 entries.
- The current result supports keeping coordinate/row and position schedules as
  alternatives. It does not support replacing fixed ownership with position
  traversal universally.

The next useful analysis is to add row-length variance, percentiles, and empty
row fraction to the suite summary and correlate them with the repeated
strategy choices. That evidence can later drive an inspectable compiler
selection rule rather than a matrix-name-specific policy.

## Reproduction

Place the Matrix Market files under the paths specified by
`benchmark/suites/spmv-position.json`, then run:

```sh
python3 benchmark/run_spmv_ablation_suite.py \
  --matrix-root=/path/to/suitesparse \
  --output-dir=/path/to/results \
  -- --chip=gfx1101
```

This workspace contained a separate older tool build under `build/llvm`, so
the measurements explicitly selected the current binary with
`--sparsewave-opt=/home/sott0n/SparseWaveComp/build/bin/sparsewave-opt`.
Ordinary builds only need that override when their current binary is not at the
runner's configured default path.
