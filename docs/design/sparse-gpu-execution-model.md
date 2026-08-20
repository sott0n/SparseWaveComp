# Sparse GPU Execution Model

## Purpose

SparseWave needs to add operators, storage formats, and GPU schedules without
implementing every `operator × format × mapping × reduction` combination as a
separate lowering pattern. This document defines the responsibility boundaries
that the current CSR SpMV and SpMM lowering will be refactored toward.

The first refactor must preserve the current generated GPU IR and runtime
behavior. It is an extraction of reusable lowering mechanisms, not a new
optimization.

## Current State

`LinalgSparseToSparseWave.cpp` recognizes Linalg contractions, requires an
identity-mapped CSR SparseTensor operand, extracts its position, coordinate,
and value buffers, and creates `sparsewave.spmv` or `sparsewave.spmm`.

`SparseWaveToGPU.cpp` then contains one complete rewrite pattern for every
supported mapping:

- `ThreadPerRowSpMVPattern`
- `WavePerRowSpMVPattern`
- `BlockPerRowSpMVPattern`
- `ThreadPerOutputSpMMPattern`
- `WavePerRowTileSpMMPattern`

Each pattern currently builds all of the following:

1. the GPU launch shape;
2. the mapping from block, thread, wave, and lane IDs to a logical work unit;
3. CSR row-bound lookup and position-space traversal;
4. operator-specific dense loads and arithmetic;
5. sequential, shuffle, or LDS-assisted reduction;
6. output ownership, bounds checks, and stores.

This is acceptable for the first vertical slice, but it makes common mechanisms
hard to reuse and schedule choices hard to compose.

## Layer Boundaries

The compiler will use four layers:

```text
Linalg computation + SparseTensor encoding
                    |
                    v
     Sparse execution and scheduling
                    |
                    v
       GPU IR construction helpers
                    |
                    v
         AMDGPU / ROCDL backend
```

### Computation semantics

Linalg and SparseTensor remain the target-independent input. This layer
describes what is computed and which tensors are sparse, without choosing CSR
loads, wave mappings, or AMDGPU instructions.

The existing `sparsewave.spmv` and `sparsewave.spmm` operations are bridge
operations for the current CSR implementation. Their explicit CSR operands
must not become the model used for every future format and operator.

### Sparse execution and scheduling

This layer separates the choices that define a sparse GPU kernel:

- sparse iteration domain and traversal;
- storage-format access;
- logical work units;
- thread, wave, or block distribution;
- accumulator shape;
- reduction and result ownership.

TACO-style position-space transformations will eventually operate at this
layer. Concepts that must survive multiple passes or be independently
transformed need an explicit IR representation before that work begins.

### GPU IR construction

This layer emits `gpu`, `scf`, `arith`, and `memref` operations. Initially, the
reusable mechanisms will be C++ builders used by the existing rewrite
patterns. GPU launch arithmetic, shuffle construction, barriers, and
workgroup-memory staging belong here.

These implementation details do not need SparseWave dialect operations unless
a later compiler transformation must inspect or rewrite them.

### AMDGPU backend

Wave32/Wave64 selection, ROCDL conversion, code-object generation, and target
resource constraints remain downstream. Target-independent sparse iteration
must not directly depend on ROCDL operations.

## Reusable Components

The interfaces below are conceptual contracts. The first implementation should
prefer small data structures and builder functions over a deep C++ class
hierarchy.

### Format access

A format accessor translates a logical sparse iteration into storage
operations.

For CSR, it provides:

- the position range `[rowOffsets[row], rowOffsets[row + 1])`;
- the column coordinate at a position;
- the nonzero value at a position.

COO and BSR will provide different traversal bounds and coordinate decoding
behind the same role. Format access does not choose GPU threads or define the
operator arithmetic.

### Work distribution

A work-distribution builder maps a logical work-unit count to a GPU launch and
describes the current participant:

- launch grid and block dimensions;
- thread, lane, and wave identifiers;
- logical work-unit identifier;
- active-participant predicate;
- first position and position stride;
- result-owner predicate.

Examples of logical work units are one row, one output element, and one
`(row, output-column tile)` pair. The mapping must not contain SpMV- or
SpMM-specific multiply/add operations.

### Sparse traversal

Sparse traversal builds iteration over the positions supplied by the format
accessor. It accepts the first position and stride selected by work
distribution and exposes each logical coordinate and stored value to an
operator-specific computation.

This boundary allows the same CSR row traversal to serve SpMV, SpMM, and
SDDMM, while a position-space schedule can replace how the position range is
partitioned.

### Computation and accumulators

The computation component defines operator semantics inside one traversal
step:

- SpMV loads one vector element and updates one scalar accumulator;
- SpMM loads dense RHS elements and updates a tile of accumulators;
- SDDMM will combine dense operands for a sparse output position.

Accumulator count, initialization, per-position updates, and final values are
part of this contract. Register tiling is represented as multiple accumulator
values, not as a storage-format concern.

### Reduction

A reduction builder combines participant-local accumulators and returns both
the reduced values and the participants allowed to consume them.

The initial strategies are:

- sequential accumulation in one thread;
- XOR-shuffle reduction within one wave;
- shuffle plus LDS staging and a barrier across the waves of one block.

The reduction interface must support one or multiple accumulators so that SpMV
and tiled SpMM can use the same wave-reduction implementation. Atomic and
segmented reductions can be added without changing format access.

### Output

Output handling maps final accumulators to dense or sparse result locations. It
owns tail checks and store predicates but not the reduction mechanism.

This separation is required for dense SpMV/SpMM output, sparse SDDMM output,
and future sparse assembly paths to share traversal and distribution.

## Composition Examples

The current mappings can be described by composing the components:

| Kernel | Work unit | Distribution | Traversal | Accumulator | Reduction |
| --- | --- | --- | --- | --- | --- |
| SpMV thread-per-row | CSR row | one thread | sequential positions | scalar | sequential |
| SpMV thread-per-position | stored CSR position chunk | one thread | position split + first-row recovery and row carry | scalar product or same-row partial sum | per-position or segment-end atomic add |
| SpMV wave-per-position | stored CSR position range | one wave | position-space partition + coordinate recovery | segmented partial sums | wave segmented scan + atomic add |
| SpMV wave-per-row | CSR row | one wave | lane-strided positions | scalar | wave shuffle |
| SpMV block-per-row | CSR row | one block | thread-strided positions | scalar | wave shuffle + LDS |
| SpMM thread-per-output | output element | one thread | sequential row positions | scalar | sequential |
| SpMM wave-per-row-tile | row and column tile | one wave | lane-strided positions | tile | wave shuffle |

For example, wave-per-row SpMV becomes:

```text
CSR format access
  + row work unit
  + wave distribution
  + lane-strided position traversal
  + sparse-value × vector-value update
  + wave-shuffle reduction
  + lane-zero dense store
```

Changing the distribution or reduction must not require copying the CSR
accessor or the SpMV multiply/add computation.

## Implementation Sequence

The behavior-preserving refactor will proceed in small commits:

1. Extract wave-reduction construction and support a range of accumulators.
2. Extract common GPU participant and work-distribution arithmetic.
3. Extract CSR row bounds and position traversal.
4. Express current SpMV and SpMM patterns as compositions of these builders.
5. Validate the boundaries with CSR SDDMM and COO before promoting stable
   scheduling concepts into SparseWave IR.

At every step, existing IR tests, end-to-end correctness tests, benchmark
options, and generated mapping behavior remain the compatibility baseline.

## Design Constraints

- Operator semantics, storage access, and GPU distribution must be independently
  replaceable.
- A component must be shared by at least two consumers before introducing a
  general public abstraction.
- Helpers must expose generated MLIR values and predicates rather than hide
  control flow in opaque runtime code.
- Explicit mapping, block-size, tile-size, and wave-size options remain
  available for reproducible experiments.
- AMD-specific shuffle, LDS, and resource decisions remain below
  target-independent sparse scheduling.
- New dialect operations require a transformation use case; code deduplication
  alone is not sufficient justification.

## Validation

The refactor is complete when:

- all existing SpMV and SpMM IR and GPU runtime tests pass;
- the five current mappings retain equivalent launch, traversal, reduction,
  and store behavior;
- wave reduction is shared by scalar and tiled accumulators;
- at least one traversal or distribution component is reused by more than one
  operator;
- adding a new operator does not require another copy of complete CSR GPU
  lowering.
