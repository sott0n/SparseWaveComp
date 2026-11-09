# Position-space scheduling research summary

This document closes SparseWave's first research cycle: implement TACO-style
position-space transformations as reusable MLIR mechanisms, map them to AMD
GPU execution, and determine where they help or lose against row ownership
and rocSPARSE.

## Research question

Compressed sparse formats store nonzeros in a linear position space, while a
conventional GPU mapping assigns work from logical rows. Flattening work by
stored position can balance irregular row lengths, but it removes direct row
ownership and introduces coordinate recovery and cross-worker reduction.

The study asks:

> When does position-space load balancing repay recovery, synchronization,
> and partial-output aggregation on an AMD GPU?

## Compiler mechanism

SparseWave separates contribution semantics from iteration transformation and
GPU scheduling:

```text
operator contribution
        ↓
keyed sparsewave.position_reduce
        ↓
analyze axes → split → reorder → collapse
        ↓
thread / wave / block position_parallel
        ↓
local segmented / shuffle / LDS / workspace reduction
        ↓
SCF + GPU + ROCDL
```

Thin operator and storage adaptors provide the output key, value contribution,
and compressed-segment recovery. The position IR and scheduling passes are
shared by SpMV and SpMM rather than encoding an `operator × format × mapping`
matrix.

| Mechanism | Purpose | Main finding |
| --- | --- | --- |
| Position space and collapse | Flatten stored work into a schedulable logical domain | Balances NNZ ownership but loses direct output ownership |
| Split and chunking | Assign consecutive positions to one participant | Amortizes scheduling and recovery only when enough parallelism remains |
| Reorder | Change multidimensional position-axis nesting | Valuable when it makes equal output keys adjacent |
| Local segmented reduction | Combine equal keys within a thread chunk | Turns split into a useful optimization by reducing atomics |
| Wave scheduling | Share recovery and reduction through shuffle operations | Effective for compact and moderately irregular segments |
| Block segmentation | Merge adjacent keys through shuffle and LDS | Effective for long key runs; fixed scan and synchronization must be repaid |
| Workspace reduction | Carry partials across wave boundaries | Handles large reductions but adds extra storage and dispatch work |

## Experimental evidence

All performance studies validate GPU results against CPU references and retain
explicit fixed mappings. Equivalent CSR operations include rocSPARSE as an
external reference.

The integrated SpMV ablation shows why the transformations must be evaluated
together. On controlled profiles with the same 64,000 rows and 1,024,000 NNZ,
the naive one-position-per-thread atomic schedule grows from approximately
0.61 ms to 6.33 ms as tails become rarer and longer. Split, aggregation, and
hierarchical scheduling keep the best transformed position result between 42
and 54 us.

The same controlled study identifies distinct scheduling regions:

- uniform and compact row boundaries favor thread-local segmentation;
- broad short rows favor wave-level aggregation;
- rare long tails favor block-local segmentation;
- empty rows, tail frequency, and chunk-boundary alignment matter in addition
  to coefficient of variation.

Position scheduling is not universally faster than direct row ownership. It
wins in some controlled and real-matrix regions, loses consistently in others,
and reaches a break-even region for rare long tails. The
[controlled row-profile study](spmv-controlled-row-profiles-gfx1101.md)
records the complete repeated comparison.

SpMM demonstrates why operator dimensionality matters. Reordering from
position-major to RHS-major makes equal `(row, rhs)` output keys adjacent, so a
separate segmented-reduction transformation can combine them. On ca-GrQc with
eight RHS columns, the integrated position schedule reaches 24.56 us versus
28.20 us for the best fixed SparseWave mapping and 22.16 us for rocSPARSE. At
32 RHS columns it loses to both references because the expanded key space and
dense-side reuse favor row-owned tiling. The
[integrated transformation ablation](position-transformation-ablation-gfx1101.md)
contains the complete SpMV and SpMM stages.

## AMD GPU findings

Position-space profitability is controlled by dynamic key locality, not by
legality alone. The generated ISA explains the fixed cost:

- direct wave-row SpMV uses 940 B and 181 static instructions;
- wave-position atomic uses 4,800 B, 917 instructions, 42 VGPRs, and 50 SGPRs;
- block-position segmented uses 6,372 B, 1,290 instructions, and 56 B of LDS;
- none of these kernels spills registers or uses scratch memory.

The additional work is compressed-segment search, boundary tracking, shuffle
or LDS aggregation, and partial-output merging. A position schedule wins only
when improved work distribution and fewer global updates repay those costs.

Short-kernel timing under automatic GPU clocks also has discrete slow states.
Alternating fixed-first and position-first execution does not eliminate them,
so individual unlocked-clock speedups are insufficient evidence for a mapping
threshold. Stable win/loss regions and static mechanism analysis are more
reliable than one absolute ratio.

## Outcome

The first research cycle meets its architectural goal:

- split, reorder, collapse, parallel mapping, and keyed reduction are explicit
  MLIR transformations;
- the scheduler is reused across SpMV and SpMM;
- thread, wave, block, segmented, and workspace variants have IR and runtime
  correctness coverage;
- ablations isolate each transformation rather than reporting only a final
  kernel;
- fixed SparseWave mappings and rocSPARSE expose both winning and losing
  regions;
- AMD GPU resources and ISA explain the principal implementation costs.

This is a completed mechanism study, not a claim that position traversal is a
universal final schedule.

## Remaining work

- Replace remaining thin operator/storage adaptors with upstream sparse
  iteration when that path can represent the required mixed sparse/dense
  programs end to end.
- Reduce the static and dynamic cost of compressed-segment recovery.
- Establish a privileged or otherwise isolated GPU measurement environment
  before deriving fine-grained profitability thresholds.
- Defer automatic strategy selection until more than one research mechanism
  has comparable evidence.
- Select the next Phase 3 research question independently from position-space
  tuning; row-length-aware grouping is one candidate, not a committed design.

## Supporting studies

- [Position mapping](spmv-position-mapping-gfx1101.md)
- [Position split](spmv-position-split-gfx1101.md)
- [Thread-local segmented reduction](spmv-position-segmented-gfx1101.md)
- [Wave position chunking](wave-position-chunk-gfx1101.md)
- [Block-local segmented reduction](block-position-segmented-gfx1101.md)
- [Block position chunking](block-position-chunk-gfx1101.md)
- [SpMM reorder](spmm-position-reorder-gfx1101.md)
- [Integrated transformation ablation](position-transformation-ablation-gfx1101.md)
- [Real-matrix SpMV suite](spmv-position-suite-gfx1101.md)
- [Controlled row profiles](spmv-controlled-row-profiles-gfx1101.md)
