#!/usr/bin/env python3

import argparse
import math
from pathlib import Path

import torch

from sparsewave.torch_export import import_torch_program, render_generic_torch_mlir


class SparseAttention(torch.nn.Module):
    """Scaled dot-product attention restricted by a CSR mask."""

    def __init__(self, head_dimension):
        super().__init__()
        self.scale = 1.0 / math.sqrt(head_dimension)

    def forward(self, mask, query, key, value):
        scores = torch.sparse.sampled_addmm(
            mask,
            query,
            key.transpose(0, 1),
            beta=0.0,
            alpha=self.scale,
        )
        scores = scores.to_sparse_coo()
        probabilities = torch.sparse.softmax(scores, dim=1)
        return torch.sparse.mm(probabilities, value)


def export_sparse_attention(mask, query, key, value):
    """Capture CSR-masked SparseAttention using TorchDynamo."""
    if mask.layout != torch.sparse_csr:
        raise ValueError("mask must use the torch.sparse_csr layout")
    tensors = (mask, query, key, value)
    if any(tensor.dtype != torch.float32 for tensor in tensors):
        raise ValueError("mask, query, key, and value must use torch.float32")
    if mask.crow_indices().dtype != torch.int32:
        raise ValueError("mask CSR indices must use torch.int32")
    if mask.col_indices().dtype != torch.int32:
        raise ValueError("mask CSR indices must use torch.int32")
    if any(tensor.dim() != 2 for tensor in tensors):
        raise ValueError("mask, query, key, and value must be rank-two tensors")
    if mask.shape[0] != query.shape[0]:
        raise ValueError("mask rows must equal query rows")
    if mask.shape[1] != key.shape[0] or key.shape[0] != value.shape[0]:
        raise ValueError("mask columns must equal key and value rows")
    if query.shape[1] != key.shape[1]:
        raise ValueError("query and key head dimensions must match")
    if query.shape[1] == 0:
        raise ValueError("head dimension must be nonzero")

    module = SparseAttention(query.shape[1])
    return torch.export.export(module, tensors, strict=True)


def make_example_inputs():
    row_offsets = torch.tensor([0, 2, 3], dtype=torch.int32)
    column_indices = torch.tensor([0, 2, 1], dtype=torch.int32)
    mask = torch.sparse_csr_tensor(
        row_offsets,
        column_indices,
        torch.zeros(3, dtype=torch.float32),
        size=(2, 3),
        check_invariants=True,
    )
    query = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    key = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=torch.float32
    )
    value = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.float32
    )
    return mask, query, key, value


def main():
    parser = argparse.ArgumentParser(
        description="Export PyTorch SparseAttention for SparseWave."
    )
    parser.add_argument(
        "--mlir-output",
        type=Path,
        help="write generic Torch MLIR to this path",
    )
    args = parser.parse_args()

    inputs = make_example_inputs()
    exported = export_sparse_attention(*inputs)
    result = exported.module()(*inputs)
    expected = SparseAttention(inputs[1].shape[1])(*inputs)
    torch.testing.assert_close(result, expected)

    torch_mlir = render_generic_torch_mlir(import_torch_program(exported))
    if args.mlir_output:
        args.mlir_output.write_text(torch_mlir)

    print(exported.graph_module.graph)
    print("Torch MLIR:")
    print(torch_mlir)
    print("result:")
    print(result)


if __name__ == "__main__":
    main()
