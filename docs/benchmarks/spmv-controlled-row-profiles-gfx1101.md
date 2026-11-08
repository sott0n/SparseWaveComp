# Controlled SpMV row profiles on gfx1101

This study isolates how CSR row-length distribution changes the relative
behavior of row-owned and position-space SpMV schedules. Every workload has
the same dimensions, NNZ, values, and dense-vector access sequence. Only the
placement of row boundaries changes.

## Environment

| Component | Value |
| --- | --- |
| Date | 2026-11-08 |
| GPU | AMD Radeon RX 7800 XT |
| Target chip | gfx1101 |
| Wavefront size | 32 |
| SparseWave commit | `7292095e2e7b9bdc21a11f3030a4504e907478d1` |
| LLVM commit | `64b593c2371b7f7225b0ec190a37cd8b672e4c5d` |
| ROCm version | 6.4.4-129 |
| rocSPARSE version | 300400, Git revision `8fbfc797` |
| Warmup / measured dispatches | 5 / 20 |
| GPU performance level | Automatic |

SparseWave times are complete dispatch times reported by `rocprofv3`.
rocSPARSE uses HIP events around the complete operation. The full suite was
run twice independently. Each run measured 43 configurations for each of
eight profiles, and all 688 measured configurations across both runs passed
CPU-reference validation.

## Controlled inputs

Every profile has 64,000 rows, 65,536 columns, mean row length 16, and
1,024,000 nonzeros. Column index `p` is `p mod 65536` and every stored value is
one, so the dense-vector access sequence is identical across profiles. Row
lengths are shuffled with seed 17.

| Profile | Row-length histogram | CV | Empty rows | p99 | Maximum |
| --- | --- | ---: | ---: | ---: | ---: |
| uniform-16 | `16:1` | 0.00 | 0% | 16 | 16 |
| compact-8-24 | `8:1,24:1` | 0.50 | 0% | 24 | 24 |
| broad-1-31 | `1:1,31:1` | 0.94 | 0% | 31 | 31 |
| empty-quarter | `0:1,16:2,32:1` | 0.71 | 25% | 32 | 32 |
| empty-half | `0:1,32:1` | 1.00 | 50% | 32 | 32 |
| tail-10-percent | `8:9,88:1` | 1.50 | 0% | 88 | 88 |
| tail-2-percent | `8:98,408:2` | 3.50 | 0% | 408 | 408 |
| tail-1-percent | `8:99,808:1` | 4.97 | 0% | 8 | 808 |

The tail-1-percent p99 is eight because exactly 99% of its rows have length
eight. Its maximum captures the rare tail that p99 intentionally excludes.

## Results

The table reports the fastest position-space schedule in each independent
run. `T-seg`, `W-atomic`, and `B-seg` mean thread-local segmented, wave atomic,
and block-local segmented reduction. `cN` is the position chunk size. A ratio
greater than one means the selected position schedule is faster than the
named reference.

| Profile | Best position, runs 1 / 2 | Position us, runs 1 / 2 | vs fixed, runs 1 / 2 | vs rocSPARSE, runs 1 / 2 |
| --- | --- | ---: | ---: | ---: |
| uniform-16 | T-seg c4 / T-seg c8 | 44.58 / 42.40 | 2.43x / 1.85x | 1.37x / 0.62x |
| compact-8-24 | T-seg c8 / T-seg c8 | 44.20 / 44.10 | 0.84x / 0.83x | 1.48x / 0.59x |
| broad-1-31 | W-atomic c4 / W-atomic c4 | 49.52 / 49.24 | 0.74x / 2.13x | 1.27x / 1.29x |
| empty-quarter | T-seg c8 / T-seg c8 | 46.28 / 46.20 | 0.77x / 0.77x | 0.56x / 0.62x |
| empty-half | W-atomic c4 / W-atomic c4 | 46.06 / 46.12 | 0.74x / 0.72x | 0.58x / 0.57x |
| tail-10-percent | W-atomic c4 / B-seg c8 | 47.96 / 46.26 | 1.37x / 1.67x | 1.51x / 0.63x |
| tail-2-percent | B-seg c8 / B-seg c8 | 45.68 / 46.04 | 1.01x / 0.99x | 0.78x / 0.77x |
| tail-1-percent | B-seg c8 / B-seg c4 | 44.72 / 53.92 | 1.11x / 0.93x | 1.61x / 1.33x |

Position-space latency and its selected scheduling level are substantially
more repeatable than the reference ratios. Seven profiles select the same
thread, wave, or block strategy in both runs. Tail-10-percent moves between
two close hierarchical variants, while tail-1-percent is the only profile
whose best position latency changes by more than 10%.

## Transformation behavior

The untransformed one-position-per-thread atomic baseline becomes increasingly
expensive as a smaller number of rows owns more positions. Its median grows
from about 0.61 ms for compact-8-24 to 3.35 ms for tail-2-percent and 6.33 ms
for tail-1-percent. Split, local aggregation, and hierarchical mapping recover
that loss: the best transformed position schedules remain between 42 and
54 us across all profiles and both runs.

The preferred position mechanism changes with the row-boundary structure:

- Uniform and compact boundaries favor thread-local segmented reduction.
- Broad short rows and the 50%-empty profile favor wave atomic reduction.
- Rare long tails favor block-local segmented reduction.
- A moderate 10% tail lies near the wave/block boundary.

This ordering is not described by CV alone. Empty-row fraction, tail
frequency, tail magnitude, and the alignment between chunk boundaries and row
boundaries all change how much local aggregation is available and how often a
partial row crosses a participant group.

## Comparison with fixed mappings

Position scheduling wins against the best measured fixed mapping in both runs
for uniform-16 and tail-10-percent. It loses consistently for compact-8-24,
empty-quarter, and empty-half. Tail-2-percent is effectively tied, while
broad-1-31 and tail-1-percent cross the break-even point between runs.

The fixed wave/row measurement itself varies strongly for uniform-16 and
broad-1-31 even though their inputs are deterministic. For example,
broad-1-31 changes from 36.56 us to 105.08 us while its best position result
remains near 49 us. GPU clocks were not locked, so these unstable ratios are
not suitable for a compiler threshold. The stable losing regions and the
repeatable position-strategy transitions are stronger evidence than an
individual fixed-mapping speedup.

## Comparison with rocSPARSE

The rocSPARSE reference also has large run-to-run changes for several short
kernels. Position scheduling beats it in both runs for broad-1-31 and
tail-1-percent, and loses in both runs for the two empty-row profiles and
tail-2-percent. The other profiles change direction. These results identify
cases for a frequency-controlled follow-up, but they do not establish a
general rocSPARSE advantage.

## Conclusion

The controlled study confirms that the position-space transformation is a
useful compiler mechanism rather than a universal replacement for row
ownership. Its split and reduction transformations keep execution time nearly
flat while the naive position baseline degrades by an order of magnitude with
rare long rows. It also exposes distinct thread, wave, and block reduction
regions that are hidden by mean row length alone.

The data does not justify an automatic selector yet. A future inspectable
decision needs multiple profile features and stable target measurements, not
a single CV threshold. The next step is targeted ISA, resource, and clock-
controlled analysis of the stable winning and losing regions and of the
variable fixed mappings.

## Reproduction

Run the checked-in controlled profile manifest without a Matrix Market input:

```sh
python3 benchmark/run_spmv_ablation_suite.py \
  --manifest=benchmark/suites/spmv-row-profiles.json \
  --output-dir=/path/to/profile-results \
  -- --chip=gfx1101 \
  --sparsewave-opt=/home/sott0n/SparseWaveComp/build/bin/sparsewave-opt
```

The `--sparsewave-opt` override is only needed when the current binary is not
at the runner's configured default path.
