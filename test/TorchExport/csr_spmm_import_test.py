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

from sparsewave.torch_export import import_torch_program, render_generic_torch_mlir


SPEC = importlib.util.spec_from_file_location("csr_spmm_example", EXAMPLE)
CSR_SPMM = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CSR_SPMM
SPEC.loader.exec_module(CSR_SPMM)


class CSRSpMMImportTest(unittest.TestCase):
    def test_import_preserves_csr_encoding_and_sparse_mm(self):
        matrix, rhs = CSR_SPMM.make_example_inputs()
        exported = CSR_SPMM.export_csr_spmm(matrix, rhs)
        module = str(import_torch_program(exported))

        self.assertIn("!torch.vtensor<[2,3],f32,#sparse>", module)
        self.assertIn("d0 : dense, d1 : compressed", module)
        self.assertIn("posWidth = 32", module)
        self.assertIn("crdWidth = 32", module)
        self.assertIn('torch.operator "torch.aten._sparse_mm"', module)
        self.assertIn("!torch.vtensor<[2,2],f32>", module)

    def test_import_rejects_non_exported_program(self):
        with self.assertRaisesRegex(TypeError, "ExportedProgram"):
            import_torch_program(torch.nn.Identity())

    def test_generic_assembly_is_available_to_sparsewave(self):
        matrix, rhs = CSR_SPMM.make_example_inputs()
        exported = CSR_SPMM.export_csr_spmm(matrix, rhs)
        imported = import_torch_program(exported)

        mlir = render_generic_torch_mlir(imported)

        self.assertIn('"torch.operator"', mlir)
        self.assertIn('name = "torch.aten._sparse_mm"', mlir)
        self.assertIn("#sparse_tensor.encoding", mlir)
        self.assertNotIn("#sparse =", mlir)


if __name__ == "__main__":
    unittest.main()
