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
    )


def ir_description():
    return {
        "name": "spmm",
        "args": [
            {"name": name, "type": "ptr"}
            for name in BUNDLE.SPMM_ARGUMENT_NAMES
        ],
        "output_rows": 4,
        "rhs_columns": 4,
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

    def test_fixed_spmm_types_and_names_come_from_ir(self):
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

    def test_pipeline_preserves_bundle_compilation_contract(self):
        rendered = BUNDLE.pipeline(options())
        self.assertIn("lower-host=false", rendered)
        self.assertIn("kernel-bare-ptr-calling-convention=true", rendered)
        self.assertIn("sink-launch-index-computations=true", rendered)

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
