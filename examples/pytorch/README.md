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

Run its frontend tests directly with the same environment:

```sh
uv run python ../../test/TorchExport/csr_spmm_test.py csr_spmm.py
```
