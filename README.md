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
`wave-per-row`, and `block-per-row`. Wave-per-row assigns one Wave32 to each
CSR row and reduces lane partial sums with shuffle instructions. Block-per-row
assigns an entire block to each row, reduces within each wave, then combines
the wave sums through workgroup memory and a barrier. The `block-size` option
controls the number of threads per block, defaults to 256, and must be between
1 and 1024.

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

For wave-per-row and block-per-row, the block size must be a multiple of 32:

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

## SpMV benchmark

The SpMV benchmark compares the `thread-per-row`, `wave-per-row`, and
`block-per-row` mappings using synthetic CSR matrices. The `uniform`,
`alternating`, and `skewed` distributions compare equal-average row lengths
with increasingly uneven work:
`alternating` switches between 1 and `2 * NNZ/row - 1`, while `skewed` assigns
one long row for every seven single-entry rows. Row counts must be multiples of
2 for `alternating` and 8 for `skewed`, ensuring that all distributions retain
the requested average. The benchmark initializes the inputs on the host, copies
them to device memory once, and uses `rocprofv3` kernel tracing so that
compilation, JIT, binary loading, and memory transfers are excluded from the
reported kernel times.

```sh
python3 benchmark/run_spmv_benchmark.py \
  --chip=gfx1101 \
  --nnz-per-row=1,2,4,8,16,32,64,128,256 \
  --distributions=uniform,alternating,skewed \
  --block-sizes=64,128,256,512
```

The benchmark performs 10 warmup dispatches and measures 50 dispatches by
default. It prints a comparison table and writes `results.csv` and
`metadata.json` under `build/benchmark/results/<timestamp>`. Generated MLIR and
raw profiler traces are temporary unless `--keep-artifacts` is specified.

Pass a Matrix Market coordinate file to benchmark a real sparse matrix instead
of the synthetic distributions:

```sh
python3 benchmark/run_spmv_benchmark.py \
  --chip=gfx1101 \
  --matrix=/path/to/matrix.mtx \
  --block-sizes=64,128,256,512
```

Coordinate matrices with `real`, `integer`, or `pattern` entries are supported
in `general` and `symmetric` form. Symmetric off-diagonal entries are expanded
before conversion to CSR. The benchmark uses the stored matrix values and an
all-ones input vector, and validates every output row against a host-computed
reference. Matrix data is converted to a temporary compact CSR binary and
loaded by the benchmark runner utility, avoiding source-size and compilation
overhead proportional to NNZ.

Use `--rows`, `--columns`, `--nnz-per-row`, `--distributions`,
`--block-sizes`, `--warmup`, and `--iterations` to change the workload. The
benchmark currently requires Wave32 because the `wave-per-row` and
`block-per-row` lowerings only support Wave32.

## SpMM benchmark

The SpMM benchmark uses a Matrix Market sparse left-hand side and varies the
number of columns in the dense right-hand side. It compares
`thread-per-output` with `wave-per-row-tile`:

```sh
python3 benchmark/run_spmm_benchmark.py \
  --chip=gfx1101 \
  --matrix=/path/to/matrix.mtx \
  --rhs-columns=8,16,32,64,128 \
  --block-sizes=64,128,256,512 \
  --tile-sizes=1,2,4,8,16
```

The sparse matrix is converted to the same temporary CSR binary used by the
SpMV benchmark. The runner initializes a dense RHS, computes the expected
matrix result on the host, and validates every output element. Ten warmup
dispatches and 50 measured dispatches are used by default. `rocprofv3` kernel
tracing excludes compilation, loading, allocation, copies, and validation from
the reported kernel times.

The `thread-per-output` baseline is measured once for each block size, while
`wave-per-row-tile` is measured for every requested tile size. The comparison
table reports median and p95 kernel time, billions of sparse value/RHS
products per second, and GFLOP/s.
`results.csv` and `metadata.json`
are written under `build/benchmark/spmm-results/<timestamp>`. Generated MLIR,
the CSR binary, and profiler traces remain temporary unless
`--keep-artifacts` is specified.

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
