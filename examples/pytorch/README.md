# SparseWave PyTorch examples

These examples use PyTorch 2.13's `torch.export` path.

Create the environment and run the CSR SpMM export example:

```sh
cd examples/pytorch
uv sync
uv run python csr_spmm.py
uv run python sparse_attention.py
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

Compile and run SparseAttention through the HIP Runtime:

```sh
uv run python sparse_attention.py \
  --runtime-mlir-output /tmp/sparse_attention_runtime.mlir
../../build/bin/sparsewave-pytorch-opt --allow-unregistered-dialect \
  --convert-torch-sparse-attention-to-sparsewave \
  /tmp/sparse_attention_runtime.mlir | \
../../build/bin/sparsewave-opt --allow-unregistered-dialect \
  --pass-pipeline='builtin.module(sparsewave-to-amdgpu-pipeline{chip=gfx1101 wavefront-size=32 rocm-path=/opt/rocm sddmm-block-size=64 row-reduction-block-size=64 rowwise-map-block-size=64 spmm-block-size=64})' | \
../../build/llvm/bin/mlir-runner \
  --shared-libs=../../build/llvm/lib/libmlir_rocm_runtime.so \
  --shared-libs=../../build/llvm/lib/libmlir_runner_utils.so \
  --entry-point-result=void
```

To produce the corresponding lrrt application bundle directly from raw Torch
MLIR, run:

```sh
uv run python sparse_attention.py --mlir-output /tmp/sparse_attention.mlir
../../build/bin/sparsewave-bundle /tmp/sparse_attention.mlir \
  --output /tmp/sparse_attention.bundle --chip gfx1101 \
  --rocm-path /opt/rocm --operation sparse-attention
```

The manifest exposes `sparse_attention_scores`, `sparse_attention_row_max`,
`sparse_attention_exp`, `sparse_attention_row_sum`,
`sparse_attention_normalize`, and `sparse_attention_output`. The lrrt Executor
chooses their launch order and allocates the declared `scores`, `rowMaximum`,
and `rowSum` intermediates. It passes output rows as `n` to all kernels except
`sparse_attention_output`, which uses the number of output elements. The key
input for `sparse_attention_scores` is specialized in transposed layout.

Run its frontend tests directly with the same environment:

```sh
uv run python ../../test/TorchExport/csr_spmm_test.py csr_spmm.py
uv run python ../../test/TorchExport/sparse_attention_test.py sparse_attention.py
```
