# SparseWave

SparseWave is an MLIR-based sparse compiler targeting AMD GPUs.

The project is currently bootstrapping its compiler infrastructure. The first
milestone provides an `mlir-opt`-style driver and a lit-based regression test
suite. AMD GPU lowering and the runtime will be added incrementally.

## AMD GPU pipeline

The `sparsewave-amdgpu-pipeline` is the entry point for the AMD GPU backend.
The target chip is required; the other target options have defaults:

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

The runtime integration test is enabled automatically when `mlir-runner`,
`libmlir_rocm_runtime.so`, `libmlir_runner_utils.so`, an ROCm architecture
enumerator, and `/dev/kfd` are available. Configure LLVM with
`MLIR_ENABLE_ROCM_RUNNER=ON` to build the ROCm runtime wrapper. The test
compiles an HSACO, loads it through HIP, launches a kernel, and checks the
values written by the GPU.

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
