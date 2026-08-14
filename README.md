# SparseWave

SparseWave is an MLIR-based sparse compiler targeting AMD GPUs.

## SparseWave dialect

The `sparsewave` dialect represents sparse computations before GPU mapping.
Its first operation is `sparsewave.spmv`, a CSR sparse matrix-vector
multiplication with explicit row-offset, column-index, value, input-vector, and
output-vector memrefs. The operation verifier checks the rank, element types,
row count, and nonzero count whenever those properties are statically known.

## SparseWave GPU lowering

The `convert-sparsewave-to-gpu` pass maps `sparsewave.spmv` to a
one-dimensional `gpu.launch`. The `mapping` option accepts `thread-per-row`,
`thread-per-position`, `wave-per-position`, `wave-per-row`, and
`block-per-row`.
Thread-per-position partitions the flattened CSR nonzero positions across GPU
threads, recovers each position's row and column, and atomically accumulates
the output. It is the correctness baseline for position-space scheduling.
Wave-per-position assigns contiguous position ranges to waves, uses a
segmented scan to combine products from the same row, and performs one atomic
update per row segment instead of per nonzero.
Wave-per-row assigns one Wave32 to each CSR row and reduces lane partial sums
with shuffle instructions. Block-per-row assigns an entire block to each row,
reduces within each wave, then combines the wave sums through workgroup memory
and a barrier. The `block-size` option controls the number of threads per
block, defaults to 256, and must be between 1 and 1024.

The same pass maps `sparsewave.spmm` with an independent strategy and block
size. `thread-per-output` assigns one GPU thread to each dense output element.
`wave-per-row-tile` assigns one Wave32 to a CSR row and a tile of output
columns. Each lane loads a sparse value once, reuses it across the tile, and
the wave reduces one partial sum per output column. `spmm-tile-size` controls
the tile width and defaults to 4. `spmm-block-size` defaults to 256 and must be
between 1 and 1024; the wave mapping requires a multiple of 32.

```sh
sparsewave-opt input.mlir \
  --convert-sparsewave-to-gpu='mapping=thread-per-row block-size=128'
```

For wave-per-position, wave-per-row, and block-per-row, the block size must be
a multiple of 32:

```sh
sparsewave-opt input.mlir \
  --convert-sparsewave-to-gpu='mapping=wave-per-row block-size=128 wave-size=32'
```

```sh
sparsewave-opt input.mlir \
  --convert-sparsewave-to-gpu='mapping=block-per-row block-size=256 wave-size=32'
```

The `sparsewave-to-amdgpu-pipeline` composes SparseWave lowering, GPU kernel
outlining, and the AMD GPU backend:

```sh
sparsewave-opt input.mlir \
  --pass-pipeline='builtin.module(sparsewave-to-amdgpu-pipeline{chip=gfx1101 wavefront-size=32 spmv-mapping=wave-per-row spmv-block-size=128 spmm-mapping=wave-per-row-tile spmm-block-size=64 spmm-tile-size=4})'
```

The integrated pipeline exposes the lowering options as `spmv-mapping`,
`spmv-block-size`, `spmm-mapping`, `spmm-block-size`, and `spmm-tile-size`. They are
intentionally absent from the backend-only pipeline.

## AMD GPU pipeline

The lower-level `sparsewave-amdgpu-pipeline` accepts already outlined GPU
kernels and runs only the AMD GPU backend. This keeps the backend reusable for
GPU IR produced outside the SparseWave dialect.

Both pipelines use the same AMD GPU target options. The target chip is
required; the other target options have defaults:

```sh
sparsewave-opt input.mlir \
  --pass-pipeline='builtin.module(sparsewave-amdgpu-pipeline{chip=gfx942 wavefront-size=64 opt-level=3})'
```

The pipeline also accepts `triple`, `features`, `abi-version`,
`index-bitwidth`, `kernel-bare-ptr-calling-convention`, and
`host-bare-ptr-calling-convention`. These options are propagated consistently
to ROCDL target metadata and device and host lowering. Device lowering fails if
an unrealized conversion cast or a non-LLVM/ROCDL operation remains.

By default, the pipeline compiles each `gpu.module` to an HSA code object and
stores it in a `gpu.binary`. The ROCm toolkit is found through `ROCM_PATH`,
`ROCM_ROOT`, `ROCM_HOME`, or LLVM's configured default. It can also be set
explicitly with `rocm-path`:

```sh
sparsewave-opt input.mlir \
  --pass-pipeline='builtin.module(sparsewave-amdgpu-pipeline{chip=gfx942 rocm-path=/opt/rocm})'
```

The `binary-format` option accepts `none`, `llvm`, `isa`, `bin`, and `fatbin`;
its default is `bin`. Use `none` to inspect the lowered LLVM/ROCDL device IR
without serializing it.

When a binary format is selected, the pipeline also lowers host-side GPU
operations and operands to the LLVM dialect. The remaining `gpu.binary` and
`gpu.launch_func` operations are translated to calls through MLIR's GPU runtime
wrapper ABI, including module loading, kernel lookup, launch, and
synchronization.

## HSACO bundles

`sparsewave-bundle` compiles MLIR without a host runtime wrapper and writes an
[lrrt-compatible bundle](https://github.com/sott0n/light-rocm-runtime/blob/main/docs/manifest-schema.md):

```text
bundle/
  manifest.json
  kernels.hsaco
```

For example, a fixed-shape SpMM can be compiled with:

```sh
sparsewave-bundle spmm.mlir --output bundle --chip gfx1101 \
  --operation spmm --mapping wave-per-row-tile \
  --block-size 64 --tile-size 4 --wavefront-size 32
```

Initial support is limited to fixed-shape CSR i32/FP32 SpMM on gfx1101 with
Wave32, four RHS columns, `wave-per-row-tile`, block size 64, and tile size 4.
Pass the output row count as `n` to `lrrt::Bundle::launch(n, args)`; the emitted
grid is the HSA total work-item count. Dynamic LDS is zero, while fixed LDS
remains in the HSACO metadata. Unsupported configurations are rejected.

```sh
sparsewave-bundle --verify bundle
```

Generation and `--verify` check the HSACO hash, target, symbol, kernarg layout,
block, grid, and LDS contract. The regular `gpu.binary` and host-wrapper path
remains the default.

The runtime integration tests are enabled automatically when `mlir-runner`,
`libmlir_rocm_runtime.so`, `libmlir_runner_utils.so`, an ROCm architecture
enumerator, and `/dev/kfd` are available. Configure LLVM with
`MLIR_ENABLE_ROCM_RUNNER=ON` to build the ROCm runtime wrapper. The test
suite compiles HSACOs, loads them through HIP, launches kernels, and checks
both direct buffer writes and a CSR SpMV result produced by the GPU.

## Checkout

LLVM and MLIR are pinned through the `externals/llvm-project` submodule. Clone
SparseWave with:

```sh
git clone --recurse-submodules https://github.com/sott0n/SparseWaveComp.git
```

For an existing checkout:

```sh
git submodule update --init --recursive
```

## Development checks

Install the pre-commit hooks once per checkout:

```sh
python3 -m pip install pre-commit
pre-commit install
```

Run all formatting and lint checks manually with:

```sh
pre-commit run --all-files
```

The hooks apply LLVM-style `clang-format` to C and C++ sources, lint CMake
files, and check common whitespace, YAML, merge-conflict, and large-file
issues.

## Benchmarks

See [benchmarks and recorded results](docs/benchmarks/README.md).

## Design

See the [sparse GPU execution model](docs/design/sparse-gpu-execution-model.md)
for the responsibility boundaries guiding reusable sparse lowering.

## Bundled LLVM build

Build SparseWave together with its pinned LLVM and MLIR revision:

```sh
cmake -G Ninja -S externals/llvm-project/llvm -B build/llvm \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_EXTERNAL_PROJECTS=sparsewave \
  -DLLVM_EXTERNAL_SPARSEWAVE_SOURCE_DIR="$PWD" \
  -DLLVM_TARGETS_TO_BUILD="AMDGPU;Native" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build/llvm --target sparsewave-opt check-sparsewave
```

## Standalone development build

Point CMake at an existing MLIR build:

```sh
cmake -G Ninja -S . -B build \
  -DMLIR_DIR=/path/to/llvm-build/lib/cmake/mlir \
  -DLLVM_EXTERNAL_LIT=/path/to/llvm-build/bin/llvm-lit
cmake --build build --target sparsewave-opt check-sparsewave
```
