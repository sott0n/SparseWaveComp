# REQUIRES: pytorch-2.13, rocm-toolkit
# RUN: %python %s %S/../../examples/pytorch/sparse_attention.py %t.mlir
# RUN: rm -rf %t.bundle
# RUN: sparsewave-bundle %t.mlir --output %t.bundle --chip gfx1101 \
# RUN:   --rocm-path %rocm_path --operation sparse-attention \
# RUN:   --block-size 64 --wavefront-size 32
# RUN: sparsewave-bundle --verify %t.bundle
# RUN: not grep -q execution_order %t.bundle/manifest.json
# RUN: FileCheck %s --input-file=%t.bundle/manifest.json

import importlib.util
from pathlib import Path
import sys


EXAMPLE = Path(sys.argv[1]).resolve()
OUTPUT = Path(sys.argv[2]).resolve()
REPOSITORY = EXAMPLE.parents[2]
sys.path.insert(0, str(REPOSITORY / "python"))
sys.argv = [sys.argv[0]]

from sparsewave.torch_export import (
    import_torch_program,
    render_generic_torch_mlir,
)


SPEC = importlib.util.spec_from_file_location("sparse_attention_example", EXAMPLE)
SPARSE_ATTENTION = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SPARSE_ATTENTION
SPEC.loader.exec_module(SPARSE_ATTENTION)

inputs = SPARSE_ATTENTION.make_example_inputs()
exported = SPARSE_ATTENTION.export_sparse_attention(*inputs)
module = import_torch_program(exported, function_name="main")
OUTPUT.write_text(render_generic_torch_mlir(module))


# CHECK: "code_object": "kernels/sparse_attention_exp.hsaco"
# CHECK: "name": "sparse_attention_exp"
# CHECK: "code_object": "kernels/sparse_attention_normalize.hsaco"
# CHECK: "name": "sparse_attention_normalize"
# CHECK: "code_object": "kernels/sparse_attention_output.hsaco"
# CHECK: "name": "sparse_attention_output"
# CHECK: "launch_n": "output_elements"
# CHECK: "code_object": "kernels/sparse_attention_row_max.hsaco"
# CHECK: "name": "sparse_attention_row_max"
# CHECK: "code_object": "kernels/sparse_attention_row_sum.hsaco"
# CHECK: "name": "sparse_attention_row_sum"
# CHECK: "code_object": "kernels/sparse_attention_scores.hsaco"
# CHECK: "name": "sparse_attention_scores"
# CHECK: "launch_n": "output_rows"
# CHECK: "manifest_version": 1
# CHECK: "application": "sparse-attention"
# CHECK: "dynamic_dimensions": [
# CHECK: "name": "nnz"
# CHECK: "head_dimension": 2
# CHECK: "key_value_rows": 3
# CHECK: "output_rows": 2
# CHECK: "value_columns": 2
# CHECK: "target": "gfx1101"
