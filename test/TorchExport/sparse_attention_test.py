# REQUIRES: pytorch-2.13
# RUN: %python %s %S/../../examples/pytorch/sparse_attention.py

import importlib.util
from pathlib import Path
import sys
import unittest

import torch


EXAMPLE = Path(sys.argv[1]).resolve()
sys.argv = [sys.argv[0]]
SPEC = importlib.util.spec_from_file_location("sparse_attention_example", EXAMPLE)
SPARSE_ATTENTION = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SPARSE_ATTENTION
SPEC.loader.exec_module(SPARSE_ATTENTION)


class SparseAttentionExportTest(unittest.TestCase):
    def test_export_captures_sparse_attention(self):
        inputs = SPARSE_ATTENTION.make_example_inputs()
        exported = SPARSE_ATTENTION.export_sparse_attention(*inputs)

        call_targets = [
            node.target
            for node in exported.graph_module.graph.nodes
            if node.op == "call_function"
        ]
        self.assertEqual(
            call_targets,
            [
                torch.ops.aten.transpose.int,
                torch.ops.aten.sparse_sampled_addmm.default,
                torch.ops.aten.to_sparse.default,
                torch.ops.aten._sparse_softmax.int,
                torch.ops.aten._sparse_mm.default,
            ],
        )

        actual = exported.module()(*inputs)
        expected = SPARSE_ATTENTION.SparseAttention(inputs[1].shape[1])(*inputs)
        torch.testing.assert_close(actual, expected)

    def test_exported_program_accepts_new_values_with_the_same_shape(self):
        mask, query, key, value = SPARSE_ATTENTION.make_example_inputs()
        exported = SPARSE_ATTENTION.export_sparse_attention(mask, query, key, value)

        new_inputs = (mask, query + 0.5, key - 0.25, value * 2.0)
        actual = exported.module()(*new_inputs)
        expected = SPARSE_ATTENTION.SparseAttention(query.shape[1])(*new_inputs)
        torch.testing.assert_close(actual, expected)

    def test_frontend_rejects_incompatible_shapes(self):
        mask, query, key, value = SPARSE_ATTENTION.make_example_inputs()
        with self.assertRaisesRegex(ValueError, "head dimensions"):
            SPARSE_ATTENTION.export_sparse_attention(mask, query, torch.zeros((3, 4)), value)

    def test_frontend_rejects_non_csr_mask(self):
        mask, query, key, value = SPARSE_ATTENTION.make_example_inputs()
        with self.assertRaisesRegex(ValueError, "sparse_csr"):
            SPARSE_ATTENTION.export_sparse_attention(
                mask.to_dense(), query, key, value
            )


if __name__ == "__main__":
    unittest.main()
