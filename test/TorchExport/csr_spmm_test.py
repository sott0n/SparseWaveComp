# REQUIRES: pytorch-2.13
# RUN: %python %s %S/../../examples/pytorch/csr_spmm.py

import importlib.util
from pathlib import Path
import sys
import unittest

import torch


EXAMPLE = Path(sys.argv[1]).resolve()
REPOSITORY = EXAMPLE.parents[2]
sys.path.insert(0, str(REPOSITORY / "python"))
sys.argv = [sys.argv[0]]

SPEC = importlib.util.spec_from_file_location("csr_spmm_example", EXAMPLE)
CSR_SPMM = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CSR_SPMM
SPEC.loader.exec_module(CSR_SPMM)


class CSRSpMMExportTest(unittest.TestCase):
    def test_export_captures_sparse_mm(self):
        matrix, rhs = CSR_SPMM.make_example_inputs()
        exported = CSR_SPMM.export_csr_spmm(matrix, rhs)

        call_targets = [
            node.target
            for node in exported.graph_module.graph.nodes
            if node.op == "call_function"
        ]
        self.assertEqual(call_targets, [torch.ops.aten._sparse_mm.default])

        actual = exported.module()(matrix, rhs)
        expected = CSR_SPMM.CSRSpMM()(matrix, rhs)
        torch.testing.assert_close(actual, expected)

    def test_exported_program_accepts_new_values_with_the_same_shape(self):
        matrix, rhs = CSR_SPMM.make_example_inputs()
        exported = CSR_SPMM.export_csr_spmm(matrix, rhs)

        second_matrix = torch.sparse_csr_tensor(
            matrix.crow_indices(),
            matrix.col_indices(),
            torch.tensor([2.0, -1.0, 0.5], dtype=torch.float32),
            size=matrix.shape,
            check_invariants=True,
        )
        second_rhs = rhs + 1.0
        actual = exported.module()(second_matrix, second_rhs)
        expected = CSR_SPMM.CSRSpMM()(second_matrix, second_rhs)
        torch.testing.assert_close(actual, expected)

    def test_frontend_rejects_non_csr_input(self):
        matrix, rhs = CSR_SPMM.make_example_inputs()
        with self.assertRaisesRegex(ValueError, "sparse_csr"):
            CSR_SPMM.export_csr_spmm(matrix.to_dense(), rhs)

    def test_frontend_rejects_incompatible_shapes(self):
        matrix, _ = CSR_SPMM.make_example_inputs()
        rhs = torch.zeros((4, 2), dtype=torch.float32)
        with self.assertRaisesRegex(ValueError, "matrix columns"):
            CSR_SPMM.export_csr_spmm(matrix, rhs)


if __name__ == "__main__":
    unittest.main()
