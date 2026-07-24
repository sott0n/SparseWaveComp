# RUN: %python %s %S/../../benchmark/run_spmv_benchmark.py %t sparsewave-opt

import csv
import importlib.util
from pathlib import Path
import subprocess
import sys
import unittest


SCRIPT = Path(sys.argv[1]).resolve()
TEMPORARY_ROOT = Path(sys.argv[2]).resolve()
SPARSEWAVE_OPT = sys.argv[3]
sys.argv = [sys.argv[0]]

SPEC = importlib.util.spec_from_file_location("spmv_benchmark", SCRIPT)
BENCHMARK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCHMARK)


class SpMVBenchmarkTest(unittest.TestCase):
    def setUp(self):
        TEMPORARY_ROOT.mkdir(parents=True, exist_ok=True)

    def test_rendered_distributions_lower_both_mappings(self):
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
            self.assertNotIn("@ROW_LENGTH@", rendered)
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


if __name__ == "__main__":
    unittest.main()
