# RUN: %python %s %S/../../tools/sparsewave-bundle/sparsewave-bundle.py

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import sys


SPEC = importlib.util.spec_from_file_location("sparsewave_bundle", sys.argv[1])
BUNDLE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUNDLE)


METADATA = """---
amdhsa.kernels:
  - .args:
      - .address_space: global
        .name: values
        .offset: 0
        .size: 8
        .value_kind: global_buffer
    .group_segment_fixed_size: 256
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .max_flat_workgroup_size: 64
    .name: spmm_kernel
    .symbol: spmm_kernel.kd
    .wavefront_size: 32
amdhsa.target: amdgcn-amd-amdhsa-unknown-gfx-test
...
"""


class BundleTest(unittest.TestCase):
    def test_metadata_drives_manifest_kernel_fields(self):
        output = json.dumps([{"NoteSections": [{"AMDGPU Metadata": METADATA}]}])
        self.assertEqual(
            BUNDLE.kernels_from_metadata(output),
            [
                {
                    "name": "spmm_kernel",
                    "symbol": "spmm_kernel.kd",
                    "kernarg": {
                        "size": 8,
                        "alignment": 8,
                        "arguments": [
                            {
                                "address_space": "global",
                                "name": "values",
                                "offset": 0,
                                "size": 8,
                                "value_kind": "global_buffer",
                            }
                        ],
                    },
                    "block": [64, 1, 1],
                    "shared_memory_bytes": 256,
                    "wavefront_size": 32,
                }
            ],
        )

    def test_verify_rejects_manifest_hsaco_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory)
            hsaco = bundle / "kernels.hsaco"
            hsaco.write_bytes(b"\x7fELFpayload")
            manifest = {
                "schema_version": 1,
                "hsaco": {"file": "kernels.hsaco", "sha256": "wrong"},
                "kernels": [],
            }
            (bundle / "manifest.json").write_text(json.dumps(manifest))
            with mock.patch.object(
                BUNDLE,
                "inspect_hsaco",
                return_value={"architecture": "gfx-test", "kernels": []},
            ):
                with self.assertRaisesRegex(ValueError, "digest"):
                    BUNDLE.verify_bundle(bundle, Path("llvm-readobj"))

    def test_verify_rejects_kernel_metadata_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory)
            hsaco = bundle / "kernels.hsaco"
            hsaco.write_bytes(b"\x7fELFpayload")
            manifest = {
                "schema_version": 1,
                "hsaco": {
                    "file": "kernels.hsaco",
                    "sha256": __import__("hashlib").sha256(
                        hsaco.read_bytes()
                    ).hexdigest(),
                },
                "target": {"architecture": "gfx-test"},
                "kernels": [],
            }
            (bundle / "manifest.json").write_text(json.dumps(manifest))
            with mock.patch.object(
                BUNDLE,
                "inspect_hsaco",
                return_value={
                    "architecture": "gfx-test",
                    "kernels": [{"symbol": "changed"}],
                },
            ):
                with self.assertRaisesRegex(ValueError, "kernel metadata"):
                    BUNDLE.verify_bundle(bundle, Path("llvm-readobj"))


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]])
