#ifndef SPARSEWAVE_DIALECT_SPARSEWAVE_IR_SPARSEWAVEOPS_TD
#define SPARSEWAVE_DIALECT_SPARSEWAVE_IR_SPARSEWAVEOPS_TD

include "sparsewave/Dialect/SparseWave/IR/SparseWaveDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def SparseWave_YieldOp : SparseWave_Op<"yield", [Pure, Terminator,
    ParentOneOf<["SDDMMOp", "CSRRowwiseMapOp", "PositionForOp",
                 "PositionReorderOp", "PositionCollapseOp",
                 "PositionParallelOp", "PositionReduceOp"]>]> {
  let summary = "Yield from a SparseWave region";
  let arguments = (ins Variadic<AnyType>:$results);
  let builders = [OpBuilder<(ins), [{ build($_builder, $_state, ValueRange()); }]>];
  let assemblyFormat = [{ attr-dict ($results^ `:` type($results))? }];
}

def SparseWave_SpMVOp : SparseWave_Op<"spmv"> {
  let summary = "Multiply a CSR matrix by a dense vector";
  let description = [{
    `sparsewave.spmv` computes `output = matrix * vector`, where `matrix` is
    represented in compressed sparse row (CSR) form by row offsets, column
    indices, and nonzero values.

    All operands are one-dimensional memrefs. Row offsets and column indices
    use the same integer or index element type. Values, vector, and output use
    the same floating-point element type.

    Example:

    ```mlir
    sparsewave.spmv %row_offsets, %column_indices, %values, %vector, %output
        : memref<5xi32>, memref<8xi32>, memref<8xf32>,
          memref<4xf32>, memref<4xf32>
    ```
  }];

  let arguments = (ins Arg<AnyMemRef, "CSR row offsets", [MemRead]>:$rowOffsets,
                       Arg<AnyMemRef, "CSR column indices",
                           [MemRead]>:$columnIndices,
                       Arg<AnyMemRef, "CSR nonzero values", [MemRead]>:$values,
                       Arg<AnyMemRef, "dense input vector", [MemRead]>:$vector,
                       Arg<AnyMemRef, "dense output vector",
                           [MemWrite]>:$output);

  let assemblyFormat = [{
    $rowOffsets `,` $columnIndices `,` $values `,` $vector `,` $output
    attr-dict `:` type($rowOffsets) `,` type($columnIndices) `,`
    type($values) `,` type($vector) `,` type($output)
  }];

  let hasVerifier = 1;
}

def SparseWave_COOSpMVOp : SparseWave_Op<"coo_spmv"> {
  let summary = "Multiply a COO matrix by a dense vector";
  let description = [{
    `sparsewave.coo_spmv` computes `output = matrix * vector`, where `matrix`
    is represented in coordinate (COO) form by parallel row-index,
    column-index, and nonzero-value arrays.

    The output must be zero-initialized before accumulating products. The GPU
    lowering performs this initialization and atomically accumulates one
    product per nonzero entry.

    Example:

    ```mlir
    sparsewave.coo_spmv %row_indices, %column_indices, %values, %vector, %output
        : memref<8xi32>, memref<8xi32>, memref<8xf32>,
          memref<4xf32>, memref<4xf32>
    ```
  }];

  let arguments = (ins Arg<AnyMemRef, "COO row indices", [MemRead]>:$rowIndices,
                       Arg<AnyMemRef, "COO column indices",
                           [MemRead]>:$columnIndices,
                       Arg<AnyMemRef, "COO nonzero values", [MemRead]>:$values,
                       Arg<AnyMemRef, "dense input vector", [MemRead]>:$vector,
                       Arg<AnyMemRef, "dense output vector",
                           [MemRead, MemWrite]>:$output);

  let assemblyFormat = [{
    $rowIndices `,` $columnIndices `,` $values `,` $vector `,` $output
    attr-dict `:` type($rowIndices) `,` type($columnIndices) `,`
    type($values) `,` type($vector) `,` type($output)
  }];

  let hasVerifier = 1;
}

def SparseWave_SpMMOp : SparseWave_Op<"spmm"> {
  let summary = "Multiply a CSR matrix by a dense matrix";
  let description = [{
    `sparsewave.spmm` computes `output = sparse_matrix * dense_matrix`.
    The sparse left-hand side is represented in compressed sparse row (CSR)
    form. Each row of `output` has the same number of columns as
    `dense_matrix`.

    Example:

    ```mlir
    sparsewave.spmm %row_offsets, %column_indices, %values, %rhs, %output
        : memref<5xi32>, memref<8xi32>, memref<8xf32>,
          memref<4x3xf32>, memref<4x3xf32>
    ```
  }];

  let arguments = (ins Arg<AnyMemRef, "CSR row offsets", [MemRead]>:$rowOffsets,
                       Arg<AnyMemRef, "CSR column indices",
                           [MemRead]>:$columnIndices,
                       Arg<AnyMemRef, "CSR nonzero values", [MemRead]>:$values,
                       Arg<AnyMemRef, "dense right-hand-side matrix",
                           [MemRead]>:$rhs,
                       Arg<AnyMemRef, "dense output matrix",
                           [MemWrite]>:$output);

  let assemblyFormat = [{
    $rowOffsets `,` $columnIndices `,` $values `,` $rhs `,` $output
    attr-dict `:` type($rowOffsets) `,` type($columnIndices) `,`
    type($values) `,` type($rhs) `,` type($output)
  }];

  let hasVerifier = 1;
}

def SparseWave_BSRSpMMOp : SparseWave_Op<"bsr_spmm"> {
  let summary = "Multiply a BSR matrix by a dense matrix";
  let description = [{
    `sparsewave.bsr_spmm` computes `output = sparse_matrix * dense_matrix`.
    The sparse left-hand side is represented in block sparse row (BSR) form
    using square blocks.

    `blockRowOffsets` and `blockColumnIndices` address blocks rather than
    scalar entries. `blockValues` is a flat row-major buffer whose logical
    shape is `[nnzb][block_size][block_size]`, where `nnzb` is the number of
    stored blocks. Matrix row and column counts must be divisible by
    `block_size` in this initial BSR contract.

    Example:

    ```mlir
    sparsewave.bsr_spmm %block_row_offsets, %block_column_indices,
        %block_values, %rhs, %output block_size = 2
        : memref<3xi32>, memref<3xi32>, memref<12xf32>,
          memref<4x3xf32>, memref<4x3xf32>
    ```
  }];

  let arguments = (ins Arg<AnyMemRef, "BSR block-row offsets",
                           [MemRead]>:$blockRowOffsets,
                       Arg<AnyMemRef, "BSR block-column indices",
                           [MemRead]>:$blockColumnIndices,
                       Arg<AnyMemRef, "packed BSR block values",
                           [MemRead]>:$blockValues,
                       Arg<AnyMemRef, "dense right-hand-side matrix",
                           [MemRead]>:$rhs,
                       Arg<AnyMemRef, "dense output matrix",
                           [MemWrite]>:$output,
                       I64Attr:$blockSize);

  let assemblyFormat = [{
    $blockRowOffsets `,` $blockColumnIndices `,` $blockValues `,` $rhs `,`
    $output `block_size` `=` $blockSize attr-dict `:`
    type($blockRowOffsets) `,` type($blockColumnIndices) `,`
    type($blockValues) `,` type($rhs) `,` type($output)
  }];

  let hasVerifier = 1;
}

def SparseWave_SDDMMOp : SparseWave_Op<"sddmm",
    [SingleBlockImplicitTerminator<"YieldOp">]> {
  let summary = "Sample a dense matrix product using a CSR matrix";
  let description = [{
    `sparsewave.sddmm` computes a sampled dense-dense matrix multiplication.
    For every CSR entry at `(row, column)`, its scalar region receives the
    stored sample value and the computed dot product, then yields the output
    value for that position. A value-weighted SDDMM is expressed as

    ```
    ^bb0(%sample: f32, %dot: f32):
      %weighted = arith.mulf %sample, %dot : f32
      sparsewave.yield %weighted : f32
    ```

    Row offsets and column indices are preserved, so the output values have
    the same sparse structure as the input CSR matrix.

    Example:

    ```mlir
    sparsewave.sddmm %row_offsets, %column_indices, %values,
        %lhs, %rhs, %output_values {
      ^bb0(%sample: f32, %dot: f32):
        %weighted = arith.mulf %sample, %dot : f32
        sparsewave.yield %weighted : f32
    }
        : memref<5xi32>, memref<8xi32>, memref<8xf32>,
          memref<4x3xf32>, memref<3x4xf32>, memref<8xf32>
    ```
  }];

  let arguments = (ins Arg<AnyMemRef, "CSR row offsets", [MemRead]>:$rowOffsets,
                       Arg<AnyMemRef, "CSR column indices",
                           [MemRead]>:$columnIndices,
                       Arg<AnyMemRef, "CSR nonzero values", [MemRead]>:$values,
                       Arg<AnyMemRef, "dense left-hand-side matrix",
                           [MemRead]>:$lhs,
                       Arg<AnyMemRef, "dense right-hand-side matrix",
                           [MemRead]>:$rhs,
                       Arg<AnyMemRef, "CSR output values",
                           [MemWrite]>:$outputValues);
  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = [{
    $rowOffsets `,` $columnIndices `,` $values `,` $lhs `,` $rhs `,`
    $outputValues $body attr-dict `:` type($rowOffsets) `,`
    type($columnIndices) `,` type($values) `,` type($lhs) `,` type($rhs) `,`
    type($outputValues)
  }];

  let hasVerifier = 1;
}

def SparseWave_CSRRowReduceOp : SparseWave_Op<"csr_row_reduce"> {
  let summary = "Reduce the stored values in each CSR row";
  let description = [{
    `sparsewave.csr_row_reduce` reduces the stored values in each CSR row into
    a dense one-dimensional output. `kind = "sum"` uses zero as the identity;
    `kind = "max"` uses negative infinity. Consequently, empty rows reduce to
    the identity value.

    Example:

    ```mlir
    sparsewave.csr_row_reduce %row_offsets, %column_indices, %values, %output
        kind = "max"
        : memref<5xi32>, memref<8xi32>, memref<8xf32>, memref<4xf32>
    ```
  }];

  let arguments = (ins Arg<AnyMemRef, "CSR row offsets", [MemRead]>:$rowOffsets,
                       Arg<AnyMemRef, "CSR column indices",
                           [MemRead]>:$columnIndices,
                       Arg<AnyMemRef, "CSR nonzero values", [MemRead]>:$values,
                       Arg<AnyMemRef, "dense row reduction output",
                           [MemWrite]>:$output,
                       StrAttr:$kind);

  let assemblyFormat = [{
    $rowOffsets `,` $columnIndices `,` $values `,` $output
    `kind` `=` $kind attr-dict `:` type($rowOffsets) `,`
    type($columnIndices) `,` type($values) `,` type($output)
  }];

  let hasVerifier = 1;
}

def SparseWave_CSRRowwiseMapOp : SparseWave_Op<"csr_rowwise_map",
    [SingleBlockImplicitTerminator<"YieldOp">]> {
  let summary = "Map CSR values with a dense value broadcast by row";
  let description = [{
    `sparsewave.csr_rowwise_map` applies a scalar region to every stored CSR
    value. The region receives the stored value and the corresponding element
    of a dense row vector, and yields the replacement stored value. The CSR
    structure is preserved.

    Example:

    ```mlir
    sparsewave.csr_rowwise_map %row_offsets, %column_indices, %values,
        %row_max, %output_values {
      ^bb0(%value: f32, %maximum: f32):
        %shifted = arith.subf %value, %maximum : f32
        %mapped = math.exp %shifted : f32
        sparsewave.yield %mapped : f32
    } : memref<5xi32>, memref<8xi32>, memref<8xf32>,
        memref<4xf32>, memref<8xf32>
    ```
  }];

  let arguments = (ins Arg<AnyMemRef, "CSR row offsets", [MemRead]>:$rowOffsets,
                       Arg<AnyMemRef, "CSR column indices",
                           [MemRead]>:$columnIndices,
                       Arg<AnyMemRef, "CSR input values", [MemRead]>:$values,
                       Arg<AnyMemRef, "dense row values", [MemRead]>:$rowValues,
                       Arg<AnyMemRef, "CSR output values",
                           [MemWrite]>:$outputValues);
  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = [{
    $rowOffsets `,` $columnIndices `,` $values `,` $rowValues `,`
    $outputValues $body attr-dict `:` type($rowOffsets) `,`
    type($columnIndices) `,` type($values) `,` type($rowValues) `,`
    type($outputValues)
  }];

  let hasVerifier = 1;
}

def SparseWave_CSRElementwiseOp : SparseWave_Op<"csr_elementwise"> {
  let summary = "Apply an elementwise operation to two CSR matrices";
  let description = [{
    `sparsewave.csr_elementwise` coiterates two CSR matrices with sorted,
    unique column indices in each row and assembles a CSR result.

    `kind = "add"` traverses the union of the input coordinates. A coordinate
    present in both inputs receives the sum of both values, while a coordinate
    present in only one input preserves that value. `kind = "multiply"`
    traverses the intersection and emits the product at coordinates present in
    both inputs. Structural entries are retained even when the computed value
    is zero.

    The caller preallocates the three output buffers. For addition, the column
    and value capacity must be at least the sum of both input NNZ counts. For
    multiplication, it must be at least the smaller input NNZ count. The
    operation writes the actual result NNZ to the one-element `outputNnz`
    buffer, and the final row offset equals this value. Output buffers must not
    alias either input.

    Example:

    ```mlir
    sparsewave.csr_elementwise
        %lhs_offsets, %lhs_columns, %lhs_values,
        %rhs_offsets, %rhs_columns, %rhs_values,
        %out_offsets, %out_columns, %out_values, %out_nnz kind = "add"
        : memref<5xi32>, memref<8xi32>, memref<8xf32>,
          memref<5xi32>, memref<7xi32>, memref<7xf32>,
          memref<5xi32>, memref<15xi32>, memref<15xf32>, memref<1xi32>
    ```
  }];

  let arguments = (ins
      Arg<AnyMemRef, "left CSR row offsets", [MemRead]>:$lhsRowOffsets,
      Arg<AnyMemRef, "left CSR column indices", [MemRead]>:$lhsColumnIndices,
      Arg<AnyMemRef, "left CSR nonzero values", [MemRead]>:$lhsValues,
      Arg<AnyMemRef, "right CSR row offsets", [MemRead]>:$rhsRowOffsets,
      Arg<AnyMemRef, "right CSR column indices", [MemRead]>:$rhsColumnIndices,
      Arg<AnyMemRef, "right CSR nonzero values", [MemRead]>:$rhsValues,
      Arg<AnyMemRef, "output CSR row offsets", [MemWrite]>:$outputRowOffsets,
      Arg<AnyMemRef, "output CSR column indices",
          [MemWrite]>:$outputColumnIndices,
      Arg<AnyMemRef, "output CSR values", [MemWrite]>:$outputValues,
      Arg<AnyMemRef, "actual output NNZ", [MemWrite]>:$outputNnz,
      StrAttr:$kind);

  let assemblyFormat = [{
    $lhsRowOffsets `,` $lhsColumnIndices `,` $lhsValues `,`
    $rhsRowOffsets `,` $rhsColumnIndices `,` $rhsValues `,`
    $outputRowOffsets `,` $outputColumnIndices `,` $outputValues `,` $outputNnz
    `kind` `=` $kind attr-dict `:`
    type($lhsRowOffsets) `,` type($lhsColumnIndices) `,` type($lhsValues) `,`
    type($rhsRowOffsets) `,` type($rhsColumnIndices) `,` type($rhsValues) `,`
    type($outputRowOffsets) `,` type($outputColumnIndices) `,`
    type($outputValues) `,` type($outputNnz)
  }];

  let hasVerifier = 1;
}

def SparseWave_PositionSplitOp : SparseWave_Op<"position_split", [Pure]> {
  let summary = "Split a sparse position into outer and inner coordinates";
  let description = [{
    `sparsewave.position_split` derives two scheduling coordinates from one
    nonnegative sparse position. For a positive constant `factor`, the
    operation computes:

    ```text
    outer = position / factor
    inner = position % factor
    position = outer * factor + inner
    ```

    The operation records a target-independent TACO-style split without
    assigning either derived coordinate to hardware. Later scheduling passes
    can independently reorder or map `outer` and `inner`; lowering
    materializes the mapping as integer arithmetic.

    Example:

    ```mlir
    %outer, %inner = sparsewave.position_split %position by 32 : index
    ```
  }];

  let arguments = (ins Index:$position, I64Attr:$factor);
  let results = (outs Index:$outer, Index:$inner);

  let assemblyFormat = [{
    $position `by` $factor attr-dict `:` type($position)
  }];

  let hasVerifier = 1;
}

def SparseWave_PositionForOp : SparseWave_Op<"position_for",
    [SingleBlockImplicitTerminator<"YieldOp">, RecursiveMemoryEffects]> {
  let summary = "Execute one logical worker's split position chunk";
  let description = [{
    `sparsewave.position_for` materializes the sequential inner iteration of a
    TACO-style position split. The `worker_id` identifies the split `outer`
    coordinate, while the region receives the original `position` and the
    corresponding `inner` coordinate:

    ```text
    position = lower + worker_id * factor + inner
    ```

    Positions are restricted to the half-open range `[lower, upper)`. The last
    worker may therefore execute fewer than `factor` iterations. Worker IDs
    beyond the number of required chunks execute an empty region, allowing a
    later mapping pass to launch a rounded-up number of hardware workers.

    The operation describes a target-independent logical worker. It does not
    assign that worker to a GPU thread, wave, or block.

    Example:

    ```mlir
    sparsewave.position_for %worker_id in %lower to %upper by 32 : index {
    ^bb0(%position: index, %inner: index):
      // Process one stored sparse position.
      sparsewave.yield
    }
    ```
  }];

  let arguments = (ins Index:$lower, Index:$upper, Index:$workerId,
                       I64Attr:$factor);
  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = [{
    $workerId `in` $lower `to` $upper `by` $factor `:` type($lower) $body
    attr-dict
  }];

  let hasVerifier = 1;
}

def SparseWave_PositionReorderOp : SparseWave_Op<"position_reorder",
    [AttrSizedOperandSegments, SingleBlockImplicitTerminator<"YieldOp">,
     RecursiveMemoryEffects]> {
  let summary = "Execute logical position axes in a specified nesting order";
  let description = [{
    `sparsewave.position_reorder` represents a rectangular nest of logical
    position-space axes and makes their execution order explicit. The lower
    and upper bounds are listed in logical axis order. The `order` permutation
    lists those axes from outermost to innermost, while the region arguments
    remain in logical axis order.

    For example, `order = [1, 0]` executes axis 1 as the outer loop and axis 0
    as the inner loop, but passes `(axis0, axis1)` to the body:

    ```mlir
    sparsewave.position_reorder lower (%zero, %zero)
        upper (%rows, %columns) order = [1, 0] {
    ^bb0(%row: index, %column: index):
      // Executes column-major while preserving logical argument order.
      sparsewave.yield
    }
    ```

    Bounds describe half-open ranges and every loop has unit stride. This is a
    target-independent scheduling operation; it does not map an axis to GPU
    hardware.
  }];

  let arguments = (ins Variadic<Index>:$lower, Variadic<Index>:$upper,
                       DenseI64ArrayAttr:$order);
  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = [{
    `lower` `(` $lower `)` `upper` `(` $upper `)` `order` `=` $order $body
    attr-dict
  }];

  let hasVerifier = 1;
}

def SparseWave_PositionCollapseOp : SparseWave_Op<"position_collapse",
    [AttrSizedOperandSegments, SingleBlockImplicitTerminator<"YieldOp">,
     RecursiveMemoryEffects]> {
  let summary = "Execute one worker from a collapsed logical position space";
  let description = [{
    `sparsewave.position_collapse` combines a rectangular set of logical axes
    into one linear worker space. The `worker_id` identifies one point in that
    collapsed space. The operation recovers the original logical coordinates
    and passes them to the region in logical axis order.

    The `order` permutation lists axes from outermost to innermost in the
    linearization. For example, bounds with extents `(3, 4)` and
    `order = [0, 1]` map worker 5 to logical coordinates `(1, 1)`.

    ```mlir
    sparsewave.position_collapse %worker_id in lower (%zero, %zero)
        upper (%rows, %columns) order = [0, 1] {
    ^bb0(%row: index, %column: index):
      // Process one point from the collapsed iteration space.
      sparsewave.yield
    }
    ```

    A worker outside the collapsed space, or any space containing an empty
    axis, executes an empty region. This permits a later mapping pass to use a
    rounded-up number of GPU workers. The operation is target-independent and
    does not select a thread, wave, or block mapping.
  }];

  let arguments = (ins Index:$workerId, Variadic<Index>:$lower,
                       Variadic<Index>:$upper, DenseI64ArrayAttr:$order);
  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = [{
    $workerId `in` `lower` `(` $lower `)` `upper` `(` $upper `)`
    `order` `=` $order $body attr-dict
  }];

  let hasVerifier = 1;
}

def SparseWave_PositionParallelOp : SparseWave_Op<"position_parallel",
    [SingleBlockImplicitTerminator<"YieldOp">, RecursiveMemoryEffects]> {
  let summary = "Map logical position workers to parallel execution units";
  let description = [{
    `sparsewave.position_parallel` maps a one-dimensional set of logical
    workers to GPU threads, waves, or blocks. The region receives a logical
    `worker_id` followed by an ID and count for the participants cooperating
    within that worker:

    ```text
    mapping   participant_id   participant_count
    thread    0                1
    wave      lane ID          wave size
    block     thread ID        block size
    ```

    This uniform region interface lets a schedule express both independent
    thread work and cooperative wave/block work without referring to concrete
    GPU IDs. Target lowering selects the launch dimensions and materializes
    the participant values.

    ```mlir
    sparsewave.position_parallel %worker_count mapping = "wave"
        block_size = 128 {
    ^bb0(%worker_id: index, %lane: index, %wave_size: index):
      sparsewave.position_collapse %worker_id in lower (%zero, %zero)
          upper (%rows, %columns) order = [0, 1] {
      ^bb1(%row: index, %column: index):
        // Cooperatively process one collapsed point.
        sparsewave.yield
      }
      sparsewave.yield
    }
    ```

    `worker_count` may be zero. A rounded-up launch can contain extra hardware
    units, but only logical workers in `[0, worker_count)` execute the region.
    The optional `block_size` records a schedule-specific workgroup size. When
    omitted, target lowering uses its configured default.
  }];

  let arguments = (ins Index:$workerCount, StrAttr:$mapping,
                       OptionalAttr<I64Attr>:$blockSize);
  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = [{
    $workerCount `mapping` `=` $mapping
    (`block_size` `=` $blockSize^)? $body attr-dict
  }];

  let extraClassDeclaration = [{
    FailureOr<PositionMapping> getMappingKind() {
      return symbolizePositionMapping(getMapping());
    }
  }];

  let hasVerifier = 1;
}

def SparseWave_PositionReduceOp : SparseWave_Op<"position_reduce",
    [AttrSizedOperandSegments, SingleBlockImplicitTerminator<"YieldOp">,
     RecursiveMemoryEffects]> {
  let summary = "Reduce values produced over a logical position domain";
  let description = [{
    `sparsewave.position_reduce` describes an operator-independent keyed
    reduction over a rectangular logical position domain. Lower and upper
    bounds are listed in logical axis order. The `order` permutation lists the
    axes from outermost to innermost when the domain is collapsed for work
    distribution, while region arguments remain in logical axis order. The
    region yields an index key and a floating-point value. Values with the same
    key are reduced into the corresponding element of the output memref.

    The operation expresses reduction semantics and target-independent axis
    order without choosing how collapsed workers are distributed. A scheduling
    pass may map them to independent threads or cooperating waves and select
    the corresponding atomic or segmented reduction implementation.

    ```mlir
    sparsewave.position_reduce lower (%zero, %zero)
        upper (%positions, %columns) order = [1, 0]
        into %output kind = "sum" {
    ^bb0(%position: index, %column: index):
      %key, %value = // derive a keyed contribution from both axes
      sparsewave.yield %key, %value : index, f32
    } : memref<?xf32>
    ```

    `kind = "sum"` zero-initializes the output before accumulating values.
  }];

  let arguments = (ins Variadic<Index>:$lower, Variadic<Index>:$upper,
                       DenseI64ArrayAttr:$order,
                       Arg<AnyMemRef, "dense reduction output",
                           [MemWrite]>:$output,
                       StrAttr:$kind);
  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = [{
    `lower` `(` $lower `)` `upper` `(` $upper `)` `order` `=` $order
    `into` $output `kind` `=` $kind $body attr-dict `:` type($output)
  }];

  let hasVerifier = 1;
}

def SparseWave_PositionSpaceOp : SparseWave_Op<"position_space", [Pure]> {
  let summary = "Partition a linear sparse position space among workers";
  let description = [{
    `sparsewave.position_space` represents a one-dimensional half-open range of
    stored sparse positions and assigns one balanced contiguous partition to a
    logical worker. The operation returns that worker's `[begin, end)` range.

    `worker_id` is in `[0, worker_count)`. Partitions cover `[lower, upper)`
    without overlap and differ in size by at most one position. This operation
    only describes the logical partition. An enclosing `position_parallel`
    operation independently maps its workers to threads, waves, or blocks.

    For `span = upper - lower`, `base = span / worker_count`, and
    `remainder = span % worker_count`, the returned bounds are:

    ```text
    begin = lower + worker_id * base + min(worker_id, remainder)
    end = begin + base + (worker_id < remainder ? 1 : 0)
    ```

    Example:

    ```mlir
    %begin, %end = sparsewave.position_space %lower to %upper
        partition %worker_id of %worker_count : index
    ```
  }];

  let arguments = (ins Index:$lower, Index:$upper, Index:$workerId,
                       Index:$workerCount);
  let results = (outs Index:$begin, Index:$end);

  let assemblyFormat = [{
    $lower `to` $upper `partition` $workerId `of` $workerCount
    attr-dict `:` type($lower)
  }];

  let hasVerifier = 1;
}

def SparseWave_CSRRowAtPositionOp : SparseWave_Op<"csr_row_at_position"> {
  let summary = "Recover a CSR row from a stored position";
  let description = [{
    `sparsewave.csr_row_at_position` maps a stored position in flattened CSR
    storage back to its logical row. Keeping row recovery separate from the
    column-index load allows a sequential position schedule to recover the
    first row once and carry it across subsequent positions.

    Example:

    ```mlir
    %row = sparsewave.csr_row_at_position %row_offsets at %position
        : memref<?xi32>
    ```
  }];

  let arguments = (ins
      Arg<AnyMemRef, "CSR row offsets", [MemRead]>:$rowOffsets,
      Index:$position);
  let results = (outs Index:$row);

  let assemblyFormat = [{
    $rowOffsets `at` $position attr-dict `:` type($rowOffsets)
  }];

  let hasVerifier = 1;
}

def SparseWave_CSRCoordinatesOp : SparseWave_Op<"csr_coordinates"> {
  let summary = "Recover CSR coordinates from a stored position";
  let description = [{
    `sparsewave.csr_coordinates` maps a stored position in the flattened CSR
    column/value arrays back to its logical `(row, column)` coordinates. The
    position must be in the half-open range represented by the CSR storage.

    This operation separates coordinate recovery from an operator such as
    SpMV or SpMM, allowing position-space schedules to reuse the same CSR
    traversal mechanism.

    Example:

    ```mlir
    %row, %column = sparsewave.csr_coordinates
        %row_offsets, %column_indices at %position
        : memref<?xi32>, memref<?xi32>
    ```
  }];

  let arguments = (ins
      Arg<AnyMemRef, "CSR row offsets", [MemRead]>:$rowOffsets,
      Arg<AnyMemRef, "CSR column indices", [MemRead]>:$columnIndices,
      Index:$position);
  let results = (outs Index:$row, Index:$column);

  let assemblyFormat = [{
    $rowOffsets `,` $columnIndices `at` $position attr-dict `:`
    type($rowOffsets) `,` type($columnIndices)
  }];

  let hasVerifier = 1;
}

#endif // SPARSEWAVE_DIALECT_SPARSEWAVE_IR_SPARSEWAVEOPS_TD
