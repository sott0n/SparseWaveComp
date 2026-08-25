# RUN: %python %s %S/../../benchmark/run_spmm_benchmark.py %t sparsewave-opt

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import types
import unittest


SCRIPT = Path(sys.argv[1]).resolve()
TEMPORARY_ROOT = Path(sys.argv[2]).resolve()
SPARSEWAVE_OPT = sys.argv[3]
sys.argv = [sys.argv[0]]
sys.path.insert(0, str(SCRIPT.parent))

SPEC = importlib.util.spec_from_file_location("spmm_benchmark", SCRIPT)
BENCHMARK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCHMARK)


class SpMMBenchmarkTest(unittest.TestCase):
    def setUp(self):
        TEMPORARY_ROOT.mkdir(parents=True, exist_ok=True)
        matrix_path = TEMPORARY_ROOT / "spmm.mtx"
        matrix_path.write_text(
            """%%MatrixMarket matrix coordinate real general
2 3 3
1 1 1
1 3 2
2 2 3
""",
            encoding="utf-8",
        )
        self.matrix_path = matrix_path
        self.matrix = BENCHMARK.common.read_matrix_market(matrix_path)

    def test_rendered_matrix_market_input_lowers(self):
        rendered = BENCHMARK.render_mlir(
            SCRIPT.parent / "spmm.mlir.in",
            self.matrix,
            rhs_columns=4,
            dispatches=3,
        )
        self.assertNotIn("@RHS_COLUMNS@", rendered)
        self.assertIn("arith.constant 4 : index", rendered)
        self.assertIn("@loadSpMMBenchmarkInputs", rendered)
        source = TEMPORARY_ROOT / "spmm.mlir"
        source.write_text(rendered, encoding="utf-8")
        subprocess.run(
            [
                SPARSEWAVE_OPT,
                str(source),
                (
                    "--convert-sparsewave-to-gpu="
                    "spmm-mapping=wave-per-row-tile spmm-block-size=64 "
                    "wave-size=32 spmm-tile-size=4"
                ),
            ],
            check=True,
            capture_output=True,
            text=True,
        )

    def test_matrix_market_converts_to_padded_bsr(self):
        bsr = BENCHMARK.common.convert_to_bsr(self.matrix, block_size=2)
        self.assertEqual(bsr["rows"], 2)
        self.assertEqual(bsr["columns"], 4)
        self.assertEqual(bsr["block_row_offsets"], [0, 2])
        self.assertEqual(bsr["block_column_indices"], [0, 1])
        self.assertEqual(
            bsr["block_values"],
            [1.0, 0.0, 0.0, 3.0, 2.0, 0.0, 0.0, 0.0],
        )
        self.assertEqual(bsr["nnzb"], 2)
        self.assertEqual(bsr["block_density"], 1.0)
        self.assertEqual(bsr["internal_zero_fraction"], 0.625)
        self.assertEqual(bsr["storage_overhead"], 8.0 / 3.0)

        binary_path = TEMPORARY_ROOT / "spmm.bsr"
        BENCHMARK.common.write_bsr_binary(binary_path, bsr)
        self.assertEqual(
            binary_path.read_bytes()[:8], BENCHMARK.common.BSR_BINARY_MAGIC
        )

    def test_bsr_conversion_coalesces_duplicate_coordinates(self):
        matrix = dict(self.matrix)
        matrix["nnz"] = 4
        matrix["row_indices"] = [0, 0, 0, 1]
        matrix["column_indices"] = [0, 0, 2, 1]
        matrix["values"] = [1.0, 4.0, 2.0, 3.0]
        bsr = BENCHMARK.common.convert_to_bsr(matrix, block_size=2)
        self.assertEqual(bsr["block_values"][0], 5.0)
        self.assertEqual(bsr["storage_overhead"], 8.0 / 3.0)

    def test_rendered_bsr_input_lowers(self):
        bsr = BENCHMARK.common.convert_to_bsr(self.matrix, block_size=2)
        rendered = BENCHMARK.render_bsr_mlir(
            SCRIPT.parent / "bsr_spmm.mlir.in",
            bsr,
            rhs_columns=4,
            dispatches=3,
        )
        self.assertNotIn("@BSR_BLOCK_SIZE@", rendered)
        self.assertIn("block_size = 2", rendered)
        self.assertIn("@loadBSRSpMMBenchmarkInputs", rendered)
        source = TEMPORARY_ROOT / "bsr-spmm.mlir"
        source.write_text(rendered, encoding="utf-8")
        subprocess.run(
            [
                SPARSEWAVE_OPT,
                str(source),
                "--convert-sparsewave-to-gpu=spmm-block-size=64",
            ],
            check=True,
            capture_output=True,
            text=True,
        )

    def test_result_counts_every_sparse_dense_product(self):
        args = types.SimpleNamespace(
            chip="gfx1101",
            matrix=self.matrix_path,
            matrix_data=self.matrix,
            wave_size=32,
            warmup=1,
            iterations=4,
        )
        result = BENCHMARK.result_row(
            args,
            mapping="thread-per-output",
            block_size=64,
            tile_size=None,
            position_chunk_size=None,
            rhs_columns=8,
            timing={
                "min_us": 1.0,
                "median_us": 2.0,
                "p95_us": 3.0,
            },
            resources={
                "vgpr_count": 24,
                "sgpr_count": 20,
                "vgpr_spill_count": 0,
                "sgpr_spill_count": 0,
                "lds_bytes": 0,
                "scratch_bytes": 0,
                "kernel_wave_size": 32,
                "max_workgroup_size": 64,
            },
        )
        self.assertEqual(result["rhs_cols"], 8)
        self.assertEqual(result["format"], "csr")
        self.assertIsNone(result["storage_block_size"])
        self.assertIsNone(result["tile_size"])
        self.assertEqual(result["gproducts_per_sec"], 0.012)
        self.assertEqual(result["gflops"], 0.024)
        self.assertEqual(result["vgpr_count"], 24)
        self.assertEqual(result["max_workgroup_size"], 64)

    def test_embedded_gpu_binary_and_resources_are_parsed(self):
        compiled = TEMPORARY_ROOT / "compiled.mlir"
        compiled.write_text(
            r'gpu.binary @kernel [#gpu.object<bin = "\7FELF\00\5C">]'
            "\n",
            encoding="utf-8",
        )
        hsaco = TEMPORARY_ROOT / "kernel.hsaco"
        BENCHMARK.common.extract_gpu_binary(compiled, hsaco)
        self.assertEqual(hsaco.read_bytes(), b"\x7fELF\x00\\")

        metadata = """---
amdhsa.kernels:
  - .args: []
    .group_segment_fixed_size: 256
    .max_flat_workgroup_size: 128
    .name:           spmm_kernel
    .private_segment_fixed_size: 16
    .sgpr_count:     20
    .sgpr_spill_count: 1
    .vgpr_count:     32
    .vgpr_spill_count: 2
    .wavefront_size: 32
...
"""
        readobj_output = json.dumps(
            [{"NoteSections": [{"AMDGPU Metadata": metadata}]}]
        )
        resources = BENCHMARK.common.parse_kernel_resources(
            readobj_output, "spmm_kernel"
        )
        self.assertEqual(
            resources,
            {
                "vgpr_count": 32,
                "sgpr_count": 20,
                "vgpr_spill_count": 2,
                "sgpr_spill_count": 1,
                "lds_bytes": 256,
                "scratch_bytes": 16,
                "kernel_wave_size": 32,
                "max_workgroup_size": 128,
            },
        )
        BENCHMARK.common.validate_gpu_resources(resources, 32, 128)
        with self.assertRaisesRegex(ValueError, "wave size"):
            BENCHMARK.common.validate_gpu_resources(resources, 64, 128)
        with self.assertRaisesRegex(ValueError, "block size"):
            BENCHMARK.common.validate_gpu_resources(resources, 32, 256)

    def test_benchmark_cases_run_each_baseline_once(self):
        cases = list(
            BENCHMARK.benchmark_cases(
                [64],
                [4],
                [4],
                ["position-major", "rhs-major"],
                ["atomic", "segmented"],
                list(BENCHMARK.MAPPINGS),
            )
        )
        self.assertEqual(
            cases,
            [
                ("thread-per-output", 64, None, None, None, None),
                ("wave-per-row-tile", 64, 4, None, None, None),
                (
                    "thread-per-position",
                    64,
                    None,
                    4,
                    "position-major",
                    "atomic",
                ),
                (
                    "thread-per-position",
                    64,
                    None,
                    4,
                    "position-major",
                    "segmented",
                ),
                (
                    "thread-per-position",
                    64,
                    None,
                    4,
                    "rhs-major",
                    "atomic",
                ),
                (
                    "thread-per-position",
                    64,
                    None,
                    4,
                    "rhs-major",
                    "segmented",
                ),
            ],
        )

    def test_position_mapping_uses_initialization_and_compute_kernels(self):
        self.assertEqual(
            BENCHMARK.sparsewave_kernel_layout("thread-per-position"),
            (2, 1),
        )
        self.assertEqual(
            BENCHMARK.sparsewave_kernel_layout("thread-per-output"),
            (1, 0),
        )

    def test_position_spmm_pipeline_lowers(self):
        rendered = BENCHMARK.render_mlir(
            SCRIPT.parent / "spmm.mlir.in",
            self.matrix,
            rhs_columns=4,
            dispatches=3,
        )
        source = TEMPORARY_ROOT / "position-spmm.mlir"
        source.write_text(rendered, encoding="utf-8")
        subprocess.run(
            [
                SPARSEWAVE_OPT,
                str(source),
                (
                    "--pass-pipeline=builtin.module("
                    "decompose-position-spmm{iteration-order=rhs-major},"
                    "schedule-sparsewave-position{mapping=thread "
                    "block-size=64 thread-chunk-size=4 "
                    "thread-reduction=segmented},"
                    "convert-sparsewave-to-gpu{"
                    "spmm-mapping=thread-per-output "
                    "spmm-block-size=64 position-block-size=64})"
                ),
            ],
            check=True,
            capture_output=True,
            text=True,
        )

    def test_wave_mapping_validation(self):
        args = types.SimpleNamespace(
            sparsewave_opt=Path(__file__),
            mlir_runner=Path(__file__),
            rocm_runtime=Path(__file__),
            runner_utils=Path(__file__),
            benchmark_utils=Path(__file__),
            rocprofv3=Path(__file__),
            llvm_readobj=Path(__file__),
            rocsparse=False,
            rocsparse_benchmark=Path(__file__),
            block_sizes=[64, 1024],
            wave_size=32,
            tile_sizes=[1, 4, 32],
            position_chunk_sizes=[1, 4, 8],
            position_orders=list(BENCHMARK.POSITION_ORDERS),
            position_reductions=list(BENCHMARK.POSITION_REDUCTIONS),
            mappings=list(BENCHMARK.MAPPINGS),
            bsr_block_sizes=[2, 4, 8],
            formats=["csr", "bsr"],
            matrix_data=self.matrix,
        )
        BENCHMARK.validate_paths(args)
        args.block_sizes = [48]
        with self.assertRaisesRegex(ValueError, "multiples of wave size"):
            BENCHMARK.validate_paths(args)
        args.block_sizes = [64]
        args.tile_sizes = [33]
        with self.assertRaisesRegex(ValueError, "must not exceed 32"):
            BENCHMARK.validate_paths(args)

        args.formats = ["bsr"]
        args.mappings = ["thread-per-position"]
        args.wave_size = 64
        args.tile_sizes = [33]
        BENCHMARK.validate_paths(args)

    def test_format_options_are_validated(self):
        self.assertEqual(BENCHMARK.parse_formats("csr,bsr"), ["csr", "bsr"])
        with self.assertRaisesRegex(argparse.ArgumentTypeError, "unknown"):
            BENCHMARK.parse_formats("ell")

    def test_mapping_options_are_validated(self):
        self.assertEqual(
            BENCHMARK.parse_mappings("thread-per-position"),
            ["thread-per-position"],
        )
        with self.assertRaisesRegex(argparse.ArgumentTypeError, "unknown"):
            BENCHMARK.parse_mappings("block-per-row")

    def test_position_options_are_validated(self):
        self.assertEqual(
            BENCHMARK.parse_position_orders("position-major,rhs-major"),
            ["position-major", "rhs-major"],
        )
        self.assertEqual(
            BENCHMARK.parse_position_reductions("atomic,segmented"),
            ["atomic", "segmented"],
        )
        with self.assertRaisesRegex(argparse.ArgumentTypeError, "unknown"):
            BENCHMARK.parse_position_orders("diagonal")
        with self.assertRaisesRegex(argparse.ArgumentTypeError, "unknown"):
            BENCHMARK.parse_position_reductions("tree")


if __name__ == "__main__":
    unittest.main()
