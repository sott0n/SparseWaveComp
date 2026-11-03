# Block position chunking on gfx1101

This report evaluates consecutive participant chunks in SparseWave's generic
block position schedule. Each block participant scans a configurable number of
adjacent points. Segments completed inside that local chunk are emitted
immediately, while its trailing keyed partial enters the existing wave and
block hierarchy. The mechanism operates on `sparsewave.position_reduce` and is
independent of the operator and storage format that produced the contribution.

```text
collapsed position domain
          |
          v
participant chunks: [p0 p1 p2 p3] [p4 p5 p6 p7] ...
          | local segmented reduction
          v
one trailing partial per active participant
          | wave shuffle + block boundary merge
          v
atomic output updates
```

If a chunk crosses key boundaries, every completed segment is atomically
flushed before the trailing partial is returned. This preserves correctness
without assuming that a chunk belongs to one sparse row. Long equal-key runs
still benefit because their participant partials reach the hierarchical
reduction without intermediate atomics.

## SpMV results

The generated CSR inputs contain 8,192 rows and use a 128-thread block. The
table compares the existing chunk-1 schedule with the best of chunks 2, 4, and
8. Times are median kernel latency over 50 measured dispatches after 10 warmup
dispatches.

| Distribution | NNZ/row | Chunk 1 | Best chunk | Best median | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| uniform | 4 | 38.58 us | 1 | 38.58 us | 1.00x |
| uniform | 32 | 179.12 us | 8 | 68.94 us | 2.60x |
| uniform | 128 | 496.20 us | 8 | 129.92 us | 3.82x |
| skewed | 4 | 37.83 us | 2 | 36.56 us | 1.03x |
| skewed | 32 | 93.55 us | 8 | 46.88 us | 2.00x |
| skewed | 128 | 249.89 us | 4 | 52.82 us | 4.73x |

Chunking is not beneficial for short uniform rows because chunk 1 already
provides enough parallel workers and the local loop adds control flow. The
benefit grows with key locality: medium and long rows reduce the number of
blocks, compressed-segment searches, and atomic partials. Chunk 8 is not
uniformly best; the skewed 128-NNZ case selects chunk 4 and chunk 8 has a much
larger p95.

## SpMM results

The SpMM sweep uses 4,096 uniform CSR rows, RHS-major position order, and the
same 128-thread block. Times are medians over 20 measured dispatches after five
warmup dispatches.

| NNZ/row | RHS columns | Chunk 1 | Best chunk | Best median | Speedup |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 4 | 82.40 us | 8 | 38.28 us | 2.15x |
| 32 | 16 | 274.00 us | 4 | 132.14 us | 2.07x |
| 128 | 4 | 249.86 us | 8 | 96.14 us | 2.60x |
| 128 | 16 | 967.41 us | 4 | 404.68 us | 2.39x |

This confirms that block chunking applies to a multidimensional keyed
reduction rather than only SpMV. As with SpMV, the best chunk depends on the
iteration order and workload. Chunk 8 shows tail-latency outliers in some
cases, so these results support an explicit experimental control rather than a
universal default change.

## GPU resources

All measured kernels use 112 bytes of LDS with no scratch memory or register
spills. SpMV uses 24 VGPRs and 42--46 SGPRs. SpMM uses 38 VGPRs and 63--65
SGPRs. The stable VGPR and LDS counts show that the speedup comes primarily
from changing dynamic work and aggregation, not from allocating a larger
block workspace.

## Reproduction

Run the SpMV sweep with:

```sh
python3 benchmark/run_spmv_benchmark.py \
  --formats=csr \
  --mappings=block-per-position \
  --rows=8192 \
  --columns=8192 \
  --nnz-per-row=4,32,128 \
  --distributions=uniform,skewed \
  --position-chunk-sizes=1,2,4,8 \
  --position-reductions=segmented \
  --block-sizes=128 \
  --warmup=10 \
  --iterations=50 \
  --chip=gfx1101 \
  --wave-size=32
```

Run the SpMM sweep with:

```sh
python3 benchmark/run_spmm_benchmark.py \
  --formats=csr \
  --mappings=block-per-position \
  --rows=4096 \
  --columns=4096 \
  --nnz-per-row=32,128 \
  --distributions=uniform \
  --rhs-columns=4,16 \
  --position-orders=rhs-major \
  --position-chunk-sizes=1,2,4,8 \
  --position-reductions=segmented \
  --block-sizes=128 \
  --warmup=5 \
  --iterations=20 \
  --chip=gfx1101 \
  --wave-size=32
```

Both measurements used an AMD Radeon RX 7800 XT (`gfx1101`), Wave32, and the
LLVM/ROCm revisions recorded by the benchmark artifacts.
