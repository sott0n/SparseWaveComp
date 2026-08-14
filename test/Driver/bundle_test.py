# RUN: %python %s %S/../../tools/sparsewave-bundle/sparsewave-bundle.py

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock
import sys


SPEC = importlib.util.spec_from_file_location("sparsewave_bundle", sys.argv[1])
BUNDLE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUNDLE)


METADATA_TEXT = """---
amdhsa.kernels:
  - .args:
      - .address_space: generic
        .offset: 0
        .size: 8
        .value_kind: global_buffer
      - .address_space: generic
        .offset: 8
        .size: 8
        .value_kind: global_buffer
      - .address_space: generic
        .offset: 16
        .size: 8
        .value_kind: global_buffer
      - .address_space: generic
        .offset: 24
        .size: 8
        .value_kind: global_buffer
      - .address_space: generic
        .offset: 32
        .size: 8
        .value_kind: global_buffer
    .group_segment_fixed_size: 256
    .kernarg_segment_align: 8
    .kernarg_segment_size: 40
    .max_flat_workgroup_size: 64
    .name: spmm_kernel
    .reqd_workgroup_size:
      - 64
      - 1
      - 1
    .symbol: spmm_kernel.kd
    .wavefront_size: 32
amdhsa.target: amdgcn-amd-amdhsa-unknown-gfx1101
...
"""


def metadata():
    return {
        "target": BUNDLE.target_from_metadata(METADATA_TEXT),
        "kernels": BUNDLE.kernels_from_metadata(METADATA_TEXT),
    }


def options():
    return types.SimpleNamespace(
        triple="amdgcn-amd-amdhsa",
        chip="gfx1101",
        features="",
        abi_version="600",
        wavefront_size=32,
        operation="spmm",
        mapping="wave-per-row-tile",
        block_size=64,
        tile_size=4,
        rocm_path="",
        sparsewave_pytorch_opt=Path("sparsewave-pytorch-opt"),
    )


def ir_description():
    return {
        "application": "spmm",
        "name": "spmm",
        "args": [
            {"name": name, "type": "ptr"}
            for name in BUNDLE.SPMM_ARGUMENT_NAMES
        ],
        "output_rows": 4,
        "rhs_columns": 4,
    }


def attention_source():
    return """func.func @main(
  %arg0: memref<3xi32>, %arg1: memref<?xi32>, %arg2: memref<?xf32>,
  %arg3: memref<2x2xf32>, %arg4: memref<2x3xf32>,
  %arg5: memref<3x2xf32>, %arg6: memref<?xf32>,
  %arg7: memref<2xf32>, %arg8: memref<2xf32>,
  %arg9: memref<2x2xf32>) {
  sparsewave.sddmm %arg0, %arg1, %arg2, %arg3, %arg4, %arg6
      {sparsewave.kernel_name = "sparse_attention_scores"}
  return
}
"""


def attention_metadata(name, argument_count):
    return {
        "target": "gfx1101",
        "kernels": [
            {
                "name": name,
                "descriptor_symbol": f"{name}.kd",
                "args": [
                    {
                        "offset": index * 8,
                        "size": 8,
                        "value_kind": "global_buffer",
                    }
                    for index in range(argument_count)
                ],
                "kernarg_size": argument_count * 8,
                "block": [64, 1, 1],
                "fixed_group_segment_bytes": 128,
                "wavefront_size": 32,
            }
        ],
    }


class BundleTest(unittest.TestCase):
    def test_lrrt_manifest_uses_ir_types_and_hsaco_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            hsaco = Path(directory) / "kernels.hsaco"
            hsaco.write_bytes(b"\x7fELFpayload")
            manifest = BUNDLE.manifest_for(
                options(), hsaco, metadata(), ir_description()
            )

        self.assertEqual(manifest["manifest_version"], 1)
        self.assertEqual(manifest["target"], "gfx1101")
        self.assertNotIn("hsaco", manifest)
        kernel = manifest["kernels"][0]
        self.assertEqual(kernel["name"], "spmm")
        self.assertEqual(kernel["symbol"], "spmm_kernel")
        self.assertEqual(kernel["code_object"], "kernels.hsaco")
        self.assertEqual(kernel["kernarg_size"], 40)
        self.assertEqual(kernel["block"], [64, 1, 1])
        self.assertEqual(kernel["grid"], ["ceil_div(n, 2) * 64", 1, 1])
        self.assertEqual(kernel["shared_memory_bytes"], 0)
        self.assertEqual(kernel["workspace_bytes"], 0)
        self.assertEqual(
            kernel["args"],
            [
                {"name": name, "type": "ptr", "offset": index * 8, "size": 8}
                for index, name in enumerate(BUNDLE.SPMM_ARGUMENT_NAMES)
            ],
        )
        self.assertEqual(
            kernel["sparsewave"]["fixed_group_segment_bytes"], 256
        )
        self.assertEqual(kernel["sparsewave"]["launch_n"], "output_rows")

    def test_fixed_spmm_uses_positional_abi_names(self):
        source = """func.func @spmm(
  %rowOffsets: memref<5xi32>,
  %columnIndices: memref<8xi32>,
  %values: memref<8xf32>,
  %rhs: memref<8x4xf32>,
  %output: memref<4x4xf32>) {
  sparsewave.spmm %rowOffsets, %columnIndices, %values, %rhs, %output
      : memref<5xi32>, memref<8xi32>, memref<8xf32>,
        memref<8x4xf32>, memref<4x4xf32>
  return
}
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "spmm.mlir"
            path.write_text(source, encoding="utf-8")
            parsed = BUNDLE.parse_fixed_spmm_ir(path)
        self.assertEqual(parsed, ir_description())

    def test_frontend_dynamic_nnz_and_ssa_names_are_accepted(self):
        source = """func.func @main(
  %arg0: memref<5xi32>,
  %arg1: memref<?xi32>,
  %arg2: memref<?xf32>,
  %arg3: memref<8x4xf32>,
  %arg4: memref<4x4xf32>) {
  sparsewave.spmm %arg0, %arg1, %arg2, %arg3, %arg4
      : memref<5xi32>, memref<?xi32>, memref<?xf32>,
        memref<8x4xf32>, memref<4x4xf32>
  return
}
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frontend.mlir"
            path.write_text(source, encoding="utf-8")
            parsed = BUNDLE.parse_fixed_spmm_ir(path)
        self.assertEqual(parsed, ir_description())

    def test_raw_torch_input_is_lowered_inside_producer(self):
        source = """func.func @main(
    %arg0: !torch.vtensor<[4,8],f32>,
    %arg1: !torch.vtensor<[8,4],f32>)
    -> !torch.vtensor<[4,4],f32> {
  %0 = \"torch.operator\"(%arg0, %arg1) <{name = \"torch.aten._sparse_mm\"}>
      : (!torch.vtensor<[4,8],f32>, !torch.vtensor<[8,4],f32>)
      -> !torch.vtensor<[4,4],f32>
  return %0 : !torch.vtensor<[4,4],f32>
}
"""
        lowered = """func.func @main(
    %arg0: memref<5xi32>, %arg1: memref<?xi32>,
    %arg2: memref<?xf32>, %arg3: memref<8x4xf32>,
    %arg4: memref<4x4xf32>) {
  sparsewave.spmm %arg0, %arg1, %arg2, %arg3, %arg4
      : memref<5xi32>, memref<?xi32>, memref<?xf32>,
        memref<8x4xf32>, memref<4x4xf32>
  return
}
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "torch.mlir"
            path.write_text(source, encoding="utf-8")
            args = options()
            args.input = path
            args.sparsewave_opt = Path("sparsewave-opt")
            completed = types.SimpleNamespace(stdout=lowered)
            with mock.patch.object(BUNDLE, "run", return_value=completed) as run:
                normalized, command = BUNDLE.prepare_bundle_input(args)
        self.assertEqual(normalized, lowered)
        self.assertEqual(
            command,
            [
                "sparsewave-opt",
                "--allow-unregistered-dialect",
                str(path),
                "--convert-torch-to-sparsewave",
            ],
        )
        run.assert_called_once_with(command, text=True)

    def test_raw_torch_attention_uses_attention_frontend(self):
        source = '"torch.aten.sparse_sampled_addmm"() : () -> ()\n'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "attention.mlir"
            path.write_text(source, encoding="utf-8")
            args = options()
            args.input = path
            completed = types.SimpleNamespace(stdout=attention_source())
            with mock.patch.object(
                BUNDLE, "run", return_value=completed
            ) as run:
                normalized, command = BUNDLE.prepare_bundle_input(args)
        self.assertEqual(normalized, attention_source())
        self.assertEqual(
            command,
            [
                "sparsewave-pytorch-opt",
                "--allow-unregistered-dialect",
                str(path),
                "--convert-torch-sparse-attention-to-sparsewave",
            ],
        )
        run.assert_called_once_with(command, text=True)

    def test_sparse_attention_manifest_has_stable_multi_kernel_abi(self):
        ir = BUNDLE.parse_sparse_attention_ir_text(attention_source())
        self.assertEqual(ir["specialization"]["output_rows"], 2)
        self.assertFalse(ir["buffers"]["scores"]["tensor"]["specialized"])
        self.assertEqual(
            ir["buffers"]["scores"]["tensor"]["dynamic_dimensions"],
            [{"index": 0, "name": "nnz"}],
        )

        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory)
            artifacts = {}
            for name, specification in BUNDLE.ATTENTION_KERNELS.items():
                path = bundle / f"{name}.hsaco"
                path.write_bytes(b"\x7fELF" + name.encode())
                artifacts[name] = {
                    "path": path,
                    "relative_path": f"kernels/{name}.hsaco",
                    "metadata": attention_metadata(
                        name, len(specification["args"])
                    ),
                }
            manifest = BUNDLE.sparse_attention_manifest_for(
                options(), artifacts, ir
            )

        self.assertEqual(
            [kernel["name"] for kernel in manifest["kernels"]],
            sorted(BUNDLE.ATTENTION_KERNELS),
        )
        self.assertNotIn("execution_order", manifest)
        for kernel in manifest["kernels"]:
            self.assertEqual(kernel["symbol"], kernel["name"])
            self.assertEqual(
                kernel["code_object"], f"kernels/{kernel['name']}.hsaco"
            )
            self.assertEqual(kernel["grid"], ["ceil_div(n, 64) * 64", 1, 1])
            self.assertTrue(all(arg["type"] == "ptr" for arg in kernel["args"]))

    def test_verify_resolves_every_attention_artifact(self):
        ir = BUNDLE.parse_sparse_attention_ir_text(attention_source())
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory)
            (bundle / "kernels").mkdir()
            artifacts = {}
            metadata_by_path = {}
            for name, specification in BUNDLE.ATTENTION_KERNELS.items():
                relative = f"kernels/{name}.hsaco"
                path = bundle / relative
                path.write_bytes(b"\x7fELF" + name.encode())
                kernel_metadata = attention_metadata(
                    name, len(specification["args"])
                )
                artifacts[name] = {
                    "path": path,
                    "relative_path": relative,
                    "metadata": kernel_metadata,
                }
                metadata_by_path[path] = kernel_metadata
            manifest = BUNDLE.sparse_attention_manifest_for(
                options(), artifacts, ir
            )
            (bundle / "manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            with mock.patch.object(
                BUNDLE,
                "inspect_hsaco",
                side_effect=lambda _, path: metadata_by_path[path],
            ) as inspect:
                verified = BUNDLE.verify_bundle(bundle, Path("llvm-readobj"))
        self.assertEqual(len(verified["kernels"]), 6)
        self.assertEqual(inspect.call_count, 6)

    def test_pipeline_preserves_bundle_compilation_contract(self):
        rendered = BUNDLE.pipeline(options())
        self.assertIn("lower-host=false", rendered)
        self.assertIn("kernel-bare-ptr-calling-convention=true", rendered)
        self.assertIn("sink-launch-index-computations=true", rendered)
        self.assertIn("prepare-gpu-bare-ptr-abi=true", rendered)

    def test_unsupported_launch_contract_is_rejected(self):
        cases = {
            "chip": "gfx1100",
            "wavefront_size": 64,
            "mapping": "thread-per-output",
            "block_size": 128,
            "tile_size": 8,
        }
        for field, value in cases.items():
            with self.subTest(field=field):
                invalid = options()
                setattr(invalid, field, value)
                with self.assertRaisesRegex(
                    ValueError, "initial lrrt bundle support requires"
                ):
                    BUNDLE.validate_supported_options(invalid)

    def test_verify_checks_lrrt_and_hsaco_consistency(self):
        cases = (
            ("manifest version", lambda m: m.update(manifest_version=2), "version"),
            ("target", lambda m: m.update(target="gfx942"), "target"),
            (
                "digest",
                lambda m: m["kernels"][0]["sparsewave"].update(
                    hsaco_sha256="wrong"
                ),
                "hsaco_sha256",
            ),
            (
                "symbol",
                lambda m: m["kernels"][0].update(symbol="wrong"),
                "missing from HSACO",
            ),
            (
                "kernarg size",
                lambda m: m["kernels"][0].update(kernarg_size=48),
                "kernarg_size",
            ),
            (
                "argument offset",
                lambda m: m["kernels"][0]["args"][1].update(offset=16),
                "offset",
            ),
            (
                "argument size",
                lambda m: m["kernels"][0]["args"][1].update(size=4),
                "size",
            ),
            (
                "block",
                lambda m: m["kernels"][0].update(block=[32, 1, 1]),
                "block",
            ),
            (
                "fixed LDS as dynamic",
                lambda m: m["kernels"][0].update(shared_memory_bytes=256),
                "dynamic shared memory",
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory)
            hsaco = bundle / "kernels.hsaco"
            hsaco.write_bytes(b"\x7fELFpayload")
            valid = BUNDLE.manifest_for(
                options(), hsaco, metadata(), ir_description()
            )
            with mock.patch.object(
                BUNDLE, "inspect_hsaco", return_value=metadata()
            ):
                (bundle / "manifest.json").write_text(
                    json.dumps(valid), encoding="utf-8"
                )
                BUNDLE.verify_bundle(bundle, Path("llvm-readobj"))
                for name, mutate, error in cases:
                    with self.subTest(name=name):
                        invalid = copy.deepcopy(valid)
                        mutate(invalid)
                        (bundle / "manifest.json").write_text(
                            json.dumps(invalid), encoding="utf-8"
                        )
                        with self.assertRaisesRegex(ValueError, error):
                            BUNDLE.verify_bundle(
                                bundle, Path("llvm-readobj")
                            )


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]])
