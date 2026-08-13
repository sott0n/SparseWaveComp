# REQUIRES: pytorch-2.13
# RUN: %python %s %S/../../examples/pytorch/csr_spmm.py %t
# RUN: sparsewave-opt --allow-unregistered-dialect %t \
# RUN:   --convert-torch-to-sparsewave | FileCheck %s

import importlib.util
from pathlib import Path
import sys


EXAMPLE = Path(sys.argv[1]).resolve()
OUTPUT = Path(sys.argv[2]).resolve()
REPOSITORY = EXAMPLE.parents[2]
sys.path.insert(0, str(REPOSITORY / "python"))
sys.argv = [sys.argv[0]]

from sparsewave.torch_export import import_torch_program, render_generic_torch_mlir


SPEC = importlib.util.spec_from_file_location("csr_spmm_example", EXAMPLE)
CSR_SPMM = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CSR_SPMM
SPEC.loader.exec_module(CSR_SPMM)

matrix, rhs = CSR_SPMM.make_example_inputs()
exported = CSR_SPMM.export_csr_spmm(matrix, rhs)
module = import_torch_program(exported)
OUTPUT.write_text(render_generic_torch_mlir(module))


# CHECK-LABEL: func.func @main(
# CHECK-SAME: memref<3xi32>
# CHECK-SAME: memref<?xi32>
# CHECK-SAME: memref<?xf32>
# CHECK-SAME: memref<3x2xf32>
# CHECK-SAME: memref<2x2xf32>
# CHECK: sparsewave.spmm
# CHECK-NOT: torch.
