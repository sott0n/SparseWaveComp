#!/usr/bin/env python3

import argparse
from pathlib import Path

import torch

from sparsewave.torch_export import import_torch_program, render_generic_torch_mlir


class CSRSpMM(torch.nn.Module):
    """Sparse CSR matrix times a dense matrix."""

    def forward(self, matrix, rhs):
        return torch.sparse.mm(matrix, rhs)


def export_csr_spmm(matrix, rhs):
    """Capture CSR SpMM as an ExportedProgram using TorchDynamo."""
    if matrix.layout != torch.sparse_csr:
        raise ValueError("matrix must use the torch.sparse_csr layout")
    if matrix.dtype != torch.float32 or rhs.dtype != torch.float32:
        raise ValueError("matrix and rhs must use torch.float32")
    if matrix.crow_indices().dtype != torch.int32:
        raise ValueError("matrix CSR indices must use torch.int32")
    if matrix.col_indices().dtype != torch.int32:
        raise ValueError("matrix CSR indices must use torch.int32")
    if matrix.dim() != 2 or rhs.dim() != 2:
        raise ValueError("matrix and rhs must be rank-two tensors")
    if matrix.shape[1] != rhs.shape[0]:
        raise ValueError("matrix columns must equal rhs rows")
    return torch.export.export(CSRSpMM(), (matrix, rhs), strict=True)


def make_example_inputs():
    row_offsets = torch.tensor([0, 2, 3], dtype=torch.int32)
    column_indices = torch.tensor([0, 2, 1], dtype=torch.int32)
    values = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    matrix = torch.sparse_csr_tensor(
        row_offsets,
        column_indices,
        values,
        size=(2, 3),
        check_invariants=True,
    )
    rhs = torch.tensor(
        [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]], dtype=torch.float32
    )
    return matrix, rhs


def render_runtime_module(imported):
    """Add the HIP runner harness for the fixed example inputs."""
    torch_module = render_generic_torch_mlir(imported)
    body_start = torch_module.find("{")
    body_end = torch_module.rfind("}")
    if body_start < 0 or body_end <= body_start:
        raise ValueError("expected a builtin module from torch-mlir")
    torch_program = torch_module[body_start + 1 : body_end].strip()
    template = Path(__file__).with_name("csr_spmm_runtime.mlir.in").read_text()
    return template.replace("@TORCH_PROGRAM@", torch_program)


def main():
    parser = argparse.ArgumentParser(
        description="Export a PyTorch CSR SpMM graph for SparseWave."
    )
    parser.add_argument(
        "--mlir-output",
        type=Path,
        help="write generic Torch MLIR to this path",
    )
    parser.add_argument(
        "--runtime-mlir-output",
        type=Path,
        help="write Torch MLIR with the HIP runtime example harness",
    )
    args = parser.parse_args()

    matrix, rhs = make_example_inputs()
    exported = export_csr_spmm(matrix, rhs)
    imported = import_torch_program(
        exported,
        function_name="spmm" if args.runtime_mlir_output else "main",
    )
    result = exported.module()(matrix, rhs)
    expected = CSRSpMM()(matrix, rhs)
    torch.testing.assert_close(result, expected)

    torch_mlir = render_generic_torch_mlir(imported)
    if args.mlir_output:
        args.mlir_output.write_text(torch_mlir)
    if args.runtime_mlir_output:
        args.runtime_mlir_output.write_text(render_runtime_module(imported))

    print(exported.graph_module.graph)
    print("Torch MLIR:")
    print(torch_mlir)
    print("result:")
    print(result)


if __name__ == "__main__":
    main()
