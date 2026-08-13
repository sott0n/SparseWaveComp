# SparseWave PyTorch examples

These examples use PyTorch 2.13's `torch.export` path.

Create the environment and run the CSR SpMM export example:

```sh
cd examples/pytorch
uv sync
uv run python csr_spmm.py
```

Lower the imported graph to SparseWave IR:

```sh
uv run python csr_spmm.py --mlir-output /tmp/csr_spmm.mlir
../../build/bin/sparsewave-opt --allow-unregistered-dialect \
  --convert-torch-to-sparsewave /tmp/csr_spmm.mlir
```

Compile and run the CSR SpMM through the HIP Runtime:

```sh
uv run python csr_spmm.py --runtime-mlir-output /tmp/csr_spmm_runtime.mlir
../../build/bin/sparsewave-opt --allow-unregistered-dialect \
  --pass-pipeline='builtin.module(sparsewave-to-amdgpu-pipeline{chip=gfx1101 wavefront-size=32 rocm-path=/opt/rocm spmm-block-size=64})' \
  /tmp/csr_spmm_runtime.mlir | \
../../build/llvm/bin/mlir-runner \
  --shared-libs=../../build/llvm/lib/libmlir_rocm_runtime.so \
  --shared-libs=../../build/llvm/lib/libmlir_runner_utils.so \
  --entry-point-result=void
```

Run its frontend tests directly with the same environment:

```sh
uv run python ../../test/TorchExport/csr_spmm_test.py csr_spmm.py
```
