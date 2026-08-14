# REQUIRES: pytorch-2.13, rocm-toolkit
# RUN: %python %s %S/../../examples/pytorch/csr_spmm.py %t.mlir
# RUN: rm -rf %t.bundle
# RUN: sparsewave-bundle %t.mlir --output %t.bundle --chip gfx1101 \
# RUN:   --rocm-path %rocm_path --operation spmm \
# RUN:   --mapping wave-per-row-tile --block-size 64 --tile-size 4 \
# RUN:   --wavefront-size 32
# RUN: sparsewave-bundle --verify %t.bundle
# RUN: FileCheck %s --input-file=%t.bundle/manifest.json

import importlib.util
from pathlib import Path
import sys

import torch


EXAMPLE = Path(sys.argv[1]).resolve()
OUTPUT = Path(sys.argv[2]).resolve()
REPOSITORY = EXAMPLE.parents[2]
sys.path.insert(0, str(REPOSITORY / "python"))
sys.argv = [sys.argv[0]]

from sparsewave.torch_export import (
    import_torch_program,
    render_generic_torch_mlir,
)


SPEC = importlib.util.spec_from_file_location("csr_spmm_example", EXAMPLE)
CSR_SPMM = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CSR_SPMM
SPEC.loader.exec_module(CSR_SPMM)

row_offsets = torch.tensor([0, 2, 3, 5, 6], dtype=torch.int32)
column_indices = torch.tensor([0, 3, 1, 2, 6, 7], dtype=torch.int32)
values = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=torch.float32)
matrix = torch.sparse_csr_tensor(
    row_offsets,
    column_indices,
    values,
    size=(4, 8),
    check_invariants=True,
)
rhs = torch.arange(32, dtype=torch.float32).reshape(8, 4)
exported = CSR_SPMM.export_csr_spmm(matrix, rhs)
module = import_torch_program(exported, function_name="main")
OUTPUT.write_text(render_generic_torch_mlir(module))


# CHECK: "args": [
# CHECK: "name": "rowOffsets"
# CHECK: "name": "columnIndices"
# CHECK: "name": "values"
# CHECK: "name": "rhs"
# CHECK: "name": "output"
# CHECK: "grid": [
# CHECK-NEXT: "ceil_div(n, 2) * 64",
# CHECK: "kernarg_size": 40
# CHECK: "fixed_output_rows": 4
# CHECK: "hsaco_sha256": "{{[0-9a-f]+}}"
# CHECK: "symbol": "main_kernel"
# CHECK: "manifest_version": 1
# CHECK: "target": "gfx1101"
