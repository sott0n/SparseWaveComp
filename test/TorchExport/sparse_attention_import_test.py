# REQUIRES: pytorch-2.13
# RUN: %python %s %S/../../examples/pytorch/sparse_attention.py

import importlib.util
from pathlib import Path
import sys
import unittest


EXAMPLE = Path(sys.argv[1]).resolve()
REPOSITORY = EXAMPLE.parents[2]
sys.path.insert(0, str(REPOSITORY / "python"))
sys.argv = [sys.argv[0]]

from sparsewave.torch_export import import_torch_program, render_generic_torch_mlir


SPEC = importlib.util.spec_from_file_location("sparse_attention_example", EXAMPLE)
SPARSE_ATTENTION = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SPARSE_ATTENTION
SPEC.loader.exec_module(SPARSE_ATTENTION)


class SparseAttentionImportTest(unittest.TestCase):
    def test_import_preserves_sparse_attention_graph(self):
        inputs = SPARSE_ATTENTION.make_example_inputs()
        imported = import_torch_program(
            SPARSE_ATTENTION.export_sparse_attention(*inputs)
        )
        module = render_generic_torch_mlir(imported)

        operations = [
            '"torch.aten.transpose.int"',
            'name = "torch.aten.sparse_sampled_addmm"',
            'name = "torch.aten.to_sparse"',
            'name = "torch.aten._sparse_softmax.int"',
            'name = "torch.aten._sparse_mm"',
        ]
        positions = [module.index(operation) for operation in operations]
        self.assertEqual(positions, sorted(positions))
        self.assertIn("#sparse_tensor.encoding", module)
        self.assertIn("posWidth = 32", module)
        self.assertIn("crdWidth = 32", module)
        self.assertIn("!torch.vtensor<[2,2],f32>", module)


if __name__ == "__main__":
    unittest.main()
