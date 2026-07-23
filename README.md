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
`index-bitwidth`, and `kernel-bare-ptr-calling-convention`. These options are
propagated consistently to ROCDL target metadata and device lowering. The
pipeline currently lowers device code to the LLVM and ROCDL dialects.
Host-side GPU launch lowering and AMDGPU binary generation will be added
incrementally.

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
