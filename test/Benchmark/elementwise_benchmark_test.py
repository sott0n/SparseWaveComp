# RUN: %python %s %S/../../benchmark/run_elementwise_benchmark.py %t sparsewave-opt

import csv
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

SPEC = importlib.util.spec_from_file_location("elementwise_benchmark", SCRIPT)
BENCHMARK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCHMARK)


class ElementwiseBenchmarkTest(unittest.TestCase):
    def setUp(self):
        TEMPORARY_ROOT.mkdir(parents=True, exist_ok=True)

    def test_rendered_operations_lower_with_named_phases(self):
        template = SCRIPT.parent / "elementwise.mlir.in"
        for operation in BENCHMARK.OPERATIONS:
            rendered = BENCHMARK.render_mlir(
                template,
                rows=8,
                columns=32,
                nnz_per_row=4,
                overlap_percent=50,
                operation=operation,
                dispatches=3,
            )
            self.assertNotIn("@EXPECTED_VALUE@", rendered)
            source = TEMPORARY_ROOT / f"elementwise-{operation}.mlir"
            source.write_text(rendered, encoding="utf-8")
            result = subprocess.run(
                [
                    SPARSEWAVE_OPT,
                    str(source),
                    "--convert-sparsewave-to-gpu=elementwise-block-size=64",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn("elementwise_symbolic", result.stdout)
            self.assertIn("elementwise_prefix", result.stdout)
            self.assertIn("elementwise_numeric", result.stdout)

    def test_workload_shape_tracks_union_and_intersection(self):
        addition = BENCHMARK.workload_shape(8, 4, 50, "add")
        self.assertEqual(addition["overlap_nnz"], 2)
        self.assertEqual(addition["input_nnz"], 32)
        self.assertEqual(addition["output_nnz_per_row"], 6)
        self.assertEqual(addition["output_nnz"], 48)

        multiplication = BENCHMARK.workload_shape(8, 4, 50, "multiply")
        self.assertEqual(multiplication["output_nnz_per_row"], 2)
        self.assertEqual(multiplication["output_nnz"], 16)

    def test_nonintegral_overlap_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "is not integral"):
            BENCHMARK.workload_shape(8, 3, 50, "add")

    def test_phase_trace_reports_total_and_fraction(self):
        trace = TEMPORARY_ROOT / "kernel_trace.csv"
        durations = (
            ("elementwise_symbolic", 100_000),
            ("elementwise_prefix_local", 30_000),
            ("elementwise_prefix_blocks", 40_000),
            ("elementwise_prefix_add", 30_000),
            ("elementwise_numeric", 100_000),
            ("elementwise_symbolic", 1_000),
            ("elementwise_prefix_local", 500),
            ("elementwise_prefix_blocks", 1_000),
            ("elementwise_prefix_add", 500),
            ("elementwise_numeric", 3_000),
            ("elementwise_symbolic", 2_000),
            ("elementwise_prefix_local", 1_000),
            ("elementwise_prefix_blocks", 2_000),
            ("elementwise_prefix_add", 1_000),
            ("elementwise_numeric", 6_000),
        )
        with trace.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                ["Kernel_Name", "Start_Timestamp", "End_Timestamp"]
            )
            for kernel, duration in durations:
                writer.writerow([kernel, 0, duration])

        timings = BENCHMARK.common.parse_kernel_phase_trace(
            trace,
            BENCHMARK.PHASE_KERNELS,
            warmup=1,
            iterations=2,
        )
        self.assertEqual(timings["total"]["median_us"], 9.0)
        self.assertEqual(timings["prefix"]["median_us"], 3.0)
        self.assertAlmostEqual(timings["prefix"]["fraction"], 1.0 / 3.0)
        self.assertAlmostEqual(
            sum(timings[phase]["fraction"] for phase in BENCHMARK.PHASE_KERNELS),
            1.0,
        )

    def test_invalid_zero_overlap_multiply_is_rejected(self):
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
            block_sizes=[64],
            wave_size=32,
            rows=8,
            columns=32,
            nnz_per_row=[4],
            overlaps=[0],
            operations=["multiply"],
            report_resources=False,
        )
        with self.assertRaisesRegex(ValueError, "at least one overlapping"):
            BENCHMARK.validate_paths(args)

    def test_resource_rows_preserve_phase_and_kernel(self):
        args = types.SimpleNamespace(chip="gfx1101", wave_size=32)
        resources = {
            kernel: {
                "vgpr_count": index + 10,
                "sgpr_count": index + 20,
                "vgpr_spill_count": 0,
                "sgpr_spill_count": 0,
                "lds_bytes": index * 4,
                "scratch_bytes": 0,
                "kernel_wave_size": 32,
                "max_workgroup_size": 64,
            }
            for index, (_, kernel) in enumerate(BENCHMARK.RESOURCE_KERNELS)
        }

        rows = BENCHMARK.resource_rows(
            args,
            operation="add",
            nnz_per_row=4,
            overlap_percent=50,
            block_size=64,
            resources=resources,
        )

        self.assertEqual(
            [(row["phase"], row["kernel"]) for row in rows],
            list(BENCHMARK.RESOURCE_KERNELS),
        )
        self.assertEqual(rows[0]["vgpr_count"], 10)
        self.assertEqual(rows[-1]["max_workgroup_size"], 64)

    def test_gpu_binary_extraction_selects_named_phase(self):
        compiled = TEMPORARY_ROOT / "phase-binaries.mlir"
        compiled.write_text(
            r'''gpu.binary @elementwise_symbolic [#gpu.object<bin = "\7FELF\01">]
gpu.binary @elementwise_numeric [#gpu.object<bin = "\7FELF\02">]
''',
            encoding="utf-8",
        )
        hsaco = TEMPORARY_ROOT / "numeric.hsaco"

        BENCHMARK.common.extract_gpu_binary(
            compiled, hsaco, binary_name="elementwise_numeric"
        )

        self.assertEqual(hsaco.read_bytes(), b"\x7fELF\x02")


if __name__ == "__main__":
    unittest.main()
