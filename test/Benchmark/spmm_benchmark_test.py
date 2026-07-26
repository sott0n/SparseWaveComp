# RUN: %python %s %S/../../benchmark/run_spmm_benchmark.py %t sparsewave-opt

import importlib.util
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
                    "spmm-mapping=thread-per-output spmm-block-size=64"
                ),
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
            rhs_columns=8,
            timing={
                "min_us": 1.0,
                "median_us": 2.0,
                "p95_us": 3.0,
            },
        )
        self.assertEqual(result["rhs_cols"], 8)
        self.assertEqual(result["gproducts_per_sec"], 0.012)
        self.assertEqual(result["gflops"], 0.024)

    def test_block_size_validation_allows_non_wave_multiples(self):
        args = types.SimpleNamespace(
            sparsewave_opt=Path(__file__),
            mlir_runner=Path(__file__),
            rocm_runtime=Path(__file__),
            runner_utils=Path(__file__),
            benchmark_utils=Path(__file__),
            rocprofv3=Path(__file__),
            block_sizes=[48, 1024],
            matrix_data=self.matrix,
        )
        BENCHMARK.validate_paths(args)
        args.block_sizes = [1025]
        with self.assertRaisesRegex(ValueError, "must not exceed 1024"):
            BENCHMARK.validate_paths(args)


if __name__ == "__main__":
    unittest.main()
