# REQUIRES: pytorch-2.13
# RUN: %python %s %S/../../examples/pytorch/sparse_attention.py %t
# RUN: sparsewave-pytorch-opt --allow-unregistered-dialect %t \
# RUN:   --convert-torch-sparse-attention-to-sparsewave | FileCheck %s

import importlib.util
from pathlib import Path
import sys


EXAMPLE = Path(sys.argv[1]).resolve()
OUTPUT = Path(sys.argv[2]).resolve()
REPOSITORY = EXAMPLE.parents[2]
sys.path.insert(0, str(REPOSITORY / "python"))
sys.argv = [sys.argv[0]]

from sparsewave.torch_export import import_torch_program, render_generic_torch_mlir


SPEC = importlib.util.spec_from_file_location("sparse_attention_example", EXAMPLE)
SPARSE_ATTENTION = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SPARSE_ATTENTION
SPEC.loader.exec_module(SPARSE_ATTENTION)

inputs = SPARSE_ATTENTION.make_example_inputs()
exported = SPARSE_ATTENTION.export_sparse_attention(*inputs)
module = import_torch_program(exported)
OUTPUT.write_text(render_generic_torch_mlir(module))


# The raw buffer ABI consumes a transposed key and caller-provided score,
# row-maximum, row-sum, and output buffers.
# CHECK-LABEL: func.func @main(
# CHECK-SAME: memref<3xi32>
# CHECK-SAME: memref<?xi32>
# CHECK-SAME: memref<?xf32>
# CHECK-SAME: memref<2x2xf32>
# CHECK-SAME: memref<2x3xf32>
# CHECK-SAME: memref<3x2xf32>
# CHECK-SAME: memref<?xf32>
# CHECK-SAME: memref<2xf32>
# CHECK-SAME: memref<2xf32>
# CHECK-SAME: memref<2x2xf32>
# CHECK: sparsewave.sddmm
# CHECK: arith.mulf
# CHECK: arith.mulf
# CHECK: arith.addf
# CHECK: sparsewave.csr_row_reduce
# CHECK-SAME: kind = "max"
# CHECK: sparsewave.csr_rowwise_map
# CHECK: arith.subf
# CHECK: math.exp
# CHECK: sparsewave.csr_row_reduce
# CHECK-SAME: kind = "sum"
# CHECK: sparsewave.csr_rowwise_map
# CHECK: arith.divf
# CHECK: sparsewave.spmm
# CHECK-NOT: torch.
