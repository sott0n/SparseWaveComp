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

    def test_rendered_mlir_lowers_both_mappings(self):
        template = SCRIPT.parent / "spmv.mlir.in"
        rendered = BENCHMARK.render_mlir(
            template, rows=16, columns=32, nnz_per_row=8, dispatches=3
        )
        self.assertNotIn("@ROWS@", rendered)
        self.assertIn("arith.constant 128 : index", rendered)
        source = TEMPORARY_ROOT / "spmv.mlir"
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
