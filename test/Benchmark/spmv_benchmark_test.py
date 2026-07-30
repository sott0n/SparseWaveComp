# RUN: %python %s %S/../../benchmark/run_spmv_benchmark.py %t sparsewave-opt

import csv
import importlib.util
from pathlib import Path
import struct
import subprocess
import sys
import types
import unittest


SCRIPT = Path(sys.argv[1]).resolve()
TEMPORARY_ROOT = Path(sys.argv[2]).resolve()
SPARSEWAVE_OPT = sys.argv[3]
sys.argv = [sys.argv[0]]
sys.path.insert(0, str(SCRIPT.parent))

SPEC = importlib.util.spec_from_file_location("spmv_benchmark", SCRIPT)
BENCHMARK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCHMARK)


class SpMVBenchmarkTest(unittest.TestCase):
    def setUp(self):
        TEMPORARY_ROOT.mkdir(parents=True, exist_ok=True)

    def test_rendered_distributions_lower_all_mappings(self):
        template = SCRIPT.parent / "spmv.mlir.in"
        for distribution in BENCHMARK.DISTRIBUTIONS:
            rendered = BENCHMARK.render_mlir(
                template,
                rows=16,
                columns=32,
                nnz_per_row=8,
                dispatches=3,
                distribution=distribution,
            )
            self.assertNotIn("@ROWS@", rendered)
            self.assertNotIn("@INITIALIZE_INPUTS@", rendered)
            source = TEMPORARY_ROOT / f"spmv-{distribution}.mlir"
            source.write_text(rendered, encoding="utf-8")
            for mapping in BENCHMARK.MAPPINGS:
                subprocess.run(
                    [
                        SPARSEWAVE_OPT,
                        str(source),
                        (
                            "--convert-sparsewave-to-gpu="
                            f"mapping={mapping} block-size=128 wave-size=32"
                        ),
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )

    def test_matrix_market_general_converts_to_csr(self):
        matrix_path = TEMPORARY_ROOT / "general.mtx"
        matrix_path.write_text(
            """%%MatrixMarket matrix coordinate real general
% a rectangular matrix with an empty row
3 4 3
1 4 2.5
1 2 -1
3 1 4
""",
            encoding="utf-8",
        )
        matrix = BENCHMARK.read_matrix_market(matrix_path)
        self.assertEqual(matrix["rows"], 3)
        self.assertEqual(matrix["columns"], 4)
        self.assertEqual(matrix["row_offsets"], [0, 2, 2, 3])
        self.assertEqual(matrix["column_indices"], [1, 3, 0])
        self.assertEqual(matrix["values"], [-1.0, 2.5, 4.0])
        self.assertEqual(matrix["min_row_nnz"], 0)
        self.assertEqual(matrix["max_row_nnz"], 2)

    def test_matrix_market_symmetric_pattern_expands_off_diagonal(self):
        matrix_path = TEMPORARY_ROOT / "symmetric.mtx"
        matrix_path.write_text(
            """%%MatrixMarket matrix coordinate pattern symmetric
3 3 3
1 1
2 1
3 2
""",
            encoding="utf-8",
        )
        matrix = BENCHMARK.read_matrix_market(matrix_path)
        self.assertEqual(matrix["nnz"], 5)
        self.assertEqual(matrix["row_offsets"], [0, 2, 4, 5])
        self.assertEqual(matrix["column_indices"], [0, 1, 0, 2, 1])
        self.assertEqual(matrix["values"], [1.0] * 5)

    def test_matrix_market_entry_count_is_validated(self):
        matrix_path = TEMPORARY_ROOT / "truncated.mtx"
        matrix_path.write_text(
            """%%MatrixMarket matrix coordinate integer general
2 2 2
1 1 1
""",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "file contains 1"):
            BENCHMARK.read_matrix_market(matrix_path)

    def test_matrix_csr_binary_header_and_dimensions(self):
        matrix_path = TEMPORARY_ROOT / "binary.mtx"
        matrix_path.write_text(
            """%%MatrixMarket matrix coordinate real general
2 3 2
1 2 2
2 3 3
""",
            encoding="utf-8",
        )
        matrix = BENCHMARK.read_matrix_market(matrix_path)
        binary_path = TEMPORARY_ROOT / "matrix.csr"
        BENCHMARK.write_csr_binary(binary_path, matrix)
        binary = binary_path.read_bytes()
        self.assertEqual(binary[:8], BENCHMARK.CSR_BINARY_MAGIC)
        self.assertEqual(struct.unpack("<QQQ", binary[8:32]), (2, 3, 2))

    def test_rendered_matrix_market_input_lowers_all_mappings(self):
        template = SCRIPT.parent / "spmv.mlir.in"
        matrix_path = TEMPORARY_ROOT / "lower.mtx"
        matrix_path.write_text(
            """%%MatrixMarket matrix coordinate real general
2 3 2
1 2 2
2 3 3
""",
            encoding="utf-8",
        )
        matrix = BENCHMARK.read_matrix_market(matrix_path)
        rendered = BENCHMARK.render_mlir(
            template,
            rows=matrix["rows"],
            columns=matrix["columns"],
            nnz_per_row=None,
            dispatches=3,
            distribution="matrix-market",
            matrix=matrix,
        )
        self.assertIn("@loadSpMVBenchmarkInputs", rendered)
        self.assertNotIn("@EXTERNAL_DECLARATIONS@", rendered)
        source = TEMPORARY_ROOT / "spmv-matrix-market.mlir"
        source.write_text(rendered, encoding="utf-8")
        for mapping in BENCHMARK.MAPPINGS:
            subprocess.run(
                [
                    SPARSEWAVE_OPT,
                    str(source),
                    (
                        "--convert-sparsewave-to-gpu="
                        f"mapping={mapping} block-size=128 wave-size=32"
                    ),
                ],
                check=True,
                capture_output=True,
                text=True,
            )

    def test_distributions_preserve_mean_for_complete_periods(self):
        for distribution in BENCHMARK.DISTRIBUTIONS:
            shape = BENCHMARK.workload_shape(
                rows=16, nnz_per_row=8, distribution=distribution
            )
            self.assertEqual(shape["nnz"], 128)
            self.assertEqual(shape["mean_row_nnz"], 8)

        alternating = BENCHMARK.workload_shape(16, 8, "alternating")
        self.assertEqual(alternating["min_row_nnz"], 1)
        self.assertEqual(alternating["max_row_nnz"], 15)
        skewed = BENCHMARK.workload_shape(16, 8, "skewed")
        self.assertEqual(skewed["min_row_nnz"], 1)
        self.assertEqual(skewed["max_row_nnz"], 57)

    def test_synthetic_matrix_matches_generated_workload(self):
        matrix = BENCHMARK.synthetic_matrix(
            rows=8,
            columns=16,
            nnz_per_row=4,
            distribution="skewed",
        )
        self.assertEqual(matrix["row_offsets"], [0, 1, 2, 3, 4, 5, 6, 7, 32])
        self.assertEqual(matrix["nnz"], 32)
        self.assertEqual(len(matrix["column_indices"]), 32)
        self.assertEqual(matrix["values"], [1.0] * 32)

    def test_block_size_list_parser(self):
        self.assertEqual(
            BENCHMARK.parse_positive_int_list("64,128,256,512"),
            [64, 128, 256, 512],
        )
        with self.assertRaisesRegex(
            BENCHMARK.argparse.ArgumentTypeError, "duplicate values"
        ):
            BENCHMARK.parse_positive_int_list("128,128")

    def test_distribution_parser_rejects_duplicates(self):
        with self.assertRaisesRegex(
            BENCHMARK.argparse.ArgumentTypeError, "duplicate distributions"
        ):
            BENCHMARK.parse_distributions("uniform,uniform")

    def test_distribution_periods_are_required(self):
        BENCHMARK.validate_distribution_rows(
            16, ["uniform", "alternating", "skewed"]
        )
        with self.assertRaisesRegex(ValueError, "skewed: multiple of 8"):
            BENCHMARK.validate_distribution_rows(7, ["skewed"])
        with self.assertRaisesRegex(ValueError, "alternating: multiple of 2"):
            BENCHMARK.validate_distribution_rows(15, ["alternating"])

    def test_trace_parser_discards_warmup(self):
        trace = TEMPORARY_ROOT / "kernel_trace.csv"
        with trace.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                ["Kernel_Name", "Start_Timestamp", "End_Timestamp"]
            )
            writer.writerow(["other_kernel", 0, 9000])
            writer.writerow(["spmv_kernel", 0, 100000])
            writer.writerow(["spmv_kernel", 0, 1000])
            writer.writerow(["spmv_kernel", 0, 2000])
            writer.writerow(["spmv_kernel", 0, 3000])
            writer.writerow(["spmv_kernel", 0, 4000])

        result = BENCHMARK.parse_kernel_trace(
            trace, kernel_name="spmv_kernel", warmup=1, iterations=4
        )
        self.assertEqual(result["min_us"], 1.0)
        self.assertEqual(result["median_us"], 2.5)
        self.assertEqual(result["p95_us"], 4.0)

    def test_result_includes_gpu_resources(self):
        args = types.SimpleNamespace(
            chip="gfx1101",
            wave_size=32,
            warmup=1,
            iterations=4,
        )
        workload = {
            "matrix": "",
            "distribution": "uniform",
            "nnz_per_row": 4,
            "shape": {
                "rows": 8,
                "columns": 16,
                "nnz": 32,
                "min_row_nnz": 4,
                "max_row_nnz": 4,
                "mean_row_nnz": 4,
            },
        }
        resources = {
            "vgpr_count": 24,
            "sgpr_count": 20,
            "vgpr_spill_count": 0,
            "sgpr_spill_count": 0,
            "lds_bytes": 128,
            "scratch_bytes": 0,
            "kernel_wave_size": 32,
            "max_workgroup_size": 64,
        }
        result = BENCHMARK.result_row(
            args,
            mapping="block-per-row",
            block_size=64,
            workload=workload,
            timing={
                "min_us": 1.0,
                "median_us": 2.0,
                "p95_us": 3.0,
            },
            resources=resources,
        )
        self.assertEqual(result["gnnz_per_sec"], 0.016)
        self.assertEqual(result["gflops"], 0.032)
        self.assertEqual(result["vgpr_count"], 24)
        self.assertEqual(result["lds_bytes"], 128)
        self.assertEqual(result["max_workgroup_size"], 64)

    def test_empty_gpu_resources_cover_result_columns(self):
        resources = BENCHMARK.common.empty_gpu_resources()
        self.assertEqual(
            set(resources), set(BENCHMARK.common.GPU_RESOURCE_FIELDS)
        )
        self.assertTrue(all(value is None for value in resources.values()))

    def test_rocsparse_output_and_timings_are_parsed(self):
        runner = BENCHMARK.common.parse_rocsparse_output(
            """rocsparse_version=304000
rocsparse_git_rev=8fbfc797
preprocess_us=12.5
timings_us=100.0,4.0,2.0,3.0
[0]
"""
        )
        self.assertEqual(runner["version"], "304000")
        self.assertEqual(runner["git_rev"], "8fbfc797")
        self.assertEqual(runner["preprocess_us"], 12.5)
        timing = BENCHMARK.common.summarize_timings(
            runner["timings_us"], warmup=1, iterations=3
        )
        self.assertEqual(timing["min_us"], 2.0)
        self.assertEqual(timing["median_us"], 3.0)
        self.assertEqual(timing["p95_us"], 4.0)

    def test_rocsparse_report_uses_library_baseline(self):
        rows = [
            {
                "implementation": "sparsewave",
                "block_size": 64,
                "mapping": "wave-per-row",
                "median_us": 2.0,
            },
            {
                "implementation": "rocsparse",
                "block_size": None,
                "mapping": "default",
                "median_us": 4.0,
            },
        ]
        reported = BENCHMARK.common.add_speedups(
            BENCHMARK.MATRIX_ROCSPARSE_REPORT, rows
        )
        self.assertEqual(reported[0]["speedup"], 2.0)
        self.assertEqual(reported[1]["speedup"], 1.0)

    def test_report_spec_computes_mapping_speedup(self):
        rows = [
            {
                "block_size": 64,
                "mapping": "thread-per-row",
                "median_us": 4.0,
            },
            {
                "block_size": 64,
                "mapping": "wave-per-row",
                "median_us": 2.0,
            },
        ]
        reported = BENCHMARK.common.add_speedups(
            BENCHMARK.MATRIX_REPORT, rows
        )
        self.assertEqual(reported[0]["speedup"], 1.0)
        self.assertEqual(reported[1]["speedup"], 2.0)


if __name__ == "__main__":
    unittest.main()
