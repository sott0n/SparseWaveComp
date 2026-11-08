# RUN: %python %s %S/../../benchmark/run_spmv_ablation_suite.py %t

import csv
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import unittest
from unittest import mock


SCRIPT = Path(sys.argv[1]).resolve()
TEMPORARY_ROOT = Path(sys.argv[2]).resolve()
sys.argv = [sys.argv[0]]
sys.path.insert(0, str(SCRIPT.parent))

SPEC = importlib.util.spec_from_file_location("spmv_ablation_suite", SCRIPT)
SUITE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SUITE
SPEC.loader.exec_module(SUITE)


class SpMVAblationSuiteTest(unittest.TestCase):
    def setUp(self):
        TEMPORARY_ROOT.mkdir(parents=True, exist_ok=True)

    def write_manifest(self, name="suite.json", matrices=None, arguments=None):
        path = TEMPORARY_ROOT / name
        path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "benchmark_arguments": arguments or ["--rocsparse"],
                    "matrices": matrices
                    or [
                        {
                            "name": "first",
                            "path": "group/first.mtx",
                            "category": "short rows",
                            "source": "https://example.com/first",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        return path

    def test_manifest_resolves_inputs_below_matrix_root(self):
        matrix_root = TEMPORARY_ROOT / "matrices"
        manifest = SUITE.load_manifest(self.write_manifest(), matrix_root)
        self.assertEqual(manifest.benchmark_arguments, ("--rocsparse",))
        self.assertEqual(
            manifest.matrices[0].path,
            matrix_root / "group" / "first.mtx",
        )
        self.assertEqual(manifest.matrices[0].category, "short rows")

    def test_manifest_rejects_reserved_runner_arguments(self):
        path = self.write_manifest(
            name="reserved.json", arguments=["--matrix=other.mtx"]
        )
        with self.assertRaisesRegex(ValueError, "reserved option '--matrix'"):
            SUITE.load_manifest(path, TEMPORARY_ROOT)

    def test_manifest_accepts_generated_row_profile(self):
        path = self.write_manifest(
            name="profile.json",
            matrices=[
                {
                    "name": "controlled",
                    "category": "controlled profile",
                    "profile": {
                        "rows": 8,
                        "columns": 16,
                        "histogram": "0:1,4:3",
                        "seed": 7,
                    },
                }
            ],
        )
        manifest = SUITE.load_manifest(path, matrix_root=None)
        profile = manifest.matrices[0]
        self.assertIsNone(profile.path)
        self.assertEqual(profile.source, "generated row-length histogram")
        self.assertEqual(
            profile.input_arguments,
            (
                "--rows=8",
                "--columns=16",
                "--row-length-histogram=0:1,4:3",
                "--row-order-seed=7",
            ),
        )

    def test_generated_profile_validation(self):
        path = self.write_manifest(
            name="invalid-profile.json",
            matrices=[
                {
                    "name": "invalid",
                    "category": "invalid",
                    "profile": {
                        "rows": 7,
                        "columns": 16,
                        "histogram": "0:1,4:3",
                    },
                }
            ],
        )
        with self.assertRaisesRegex(ValueError, "divisible"):
            SUITE.load_manifest(path, matrix_root=None)

    def test_matrix_selection_preserves_requested_order(self):
        matrices = (
            SUITE.MatrixSpec("first", Path("first.mtx"), "a", "first"),
            SUITE.MatrixSpec("second", Path("second.mtx"), "b", "second"),
        )
        selected = SUITE.select_matrices(matrices, "second,first")
        self.assertEqual([matrix.name for matrix in selected], ["second", "first"])
        with self.assertRaisesRegex(ValueError, "unknown suite matrices"):
            SUITE.select_matrices(matrices, "missing")

    def test_suite_runs_each_matrix_and_aggregates_results(self):
        matrix_root = TEMPORARY_ROOT / "suite-inputs"
        matrix_path = matrix_root / "first" / "first.mtx"
        matrix_path.parent.mkdir(parents=True, exist_ok=True)
        matrix_path.write_text("matrix placeholder\n", encoding="utf-8")
        entries = [
            {
                "name": "first",
                "path": "first/first.mtx",
                "category": "short",
                "source": "https://example.com/first",
            },
            {
                "name": "second",
                "category": "controlled",
                "profile": {
                    "rows": 8,
                    "columns": 16,
                    "histogram": "0:1,4:3",
                    "seed": 7,
                },
            },
        ]
        manifest = self.write_manifest(
            name="aggregate.json",
            matrices=entries,
            arguments=["--block-sizes=64", "--rocsparse"],
        )
        output = TEMPORARY_ROOT / "aggregate-output"
        shutil.rmtree(output, ignore_errors=True)

        def fake_run(command, check):
            self.assertTrue(check)
            output_option = next(
                argument for argument in command if argument.startswith("--output-dir=")
            )
            case_output = Path(output_option.split("=", maxsplit=1)[1])
            case_output.mkdir()
            columns = SUITE.SUITE_COLUMNS[4:]
            base = {column: "" for column in columns}
            rows = []
            for stage, mapping, median in (
                ("fixed mapping", "wave-per-row", "12.000000"),
                ("position baseline", "thread-per-position", "20.000000"),
                ("wave position", "wave-per-position", "8.000000"),
                ("rocSPARSE", "default", "10.000000"),
            ):
                rows.append(
                    {
                        **base,
                        "operation": "spmv",
                        "stage": stage,
                        "implementation": (
                            "rocsparse" if stage == "rocSPARSE" else "sparsewave"
                        ),
                        "mapping": mapping,
                        "block_size": "64" if stage != "rocSPARSE" else "",
                        "position_chunk_size": (
                            "1" if "position" in mapping else ""
                        ),
                        "position_reduction": (
                            "atomic" if "position" in mapping else ""
                        ),
                        "median_us": median,
                        "p95_us": median,
                    }
                )
            with (case_output / "position-ablation.csv").open(
                "w", newline="", encoding="utf-8"
            ) as stream:
                writer = csv.DictWriter(stream, fieldnames=columns)
                writer.writeheader()
                writer.writerows(rows)
            profile = {
                "rows": "8",
                "nnz": "32",
                "min_row_nnz": "0",
                "max_row_nnz": "12",
                "mean_row_nnz": "4.000000",
                "stddev_row_nnz": "3.000000",
                "cv_row_nnz": "0.750000",
                "median_row_nnz": "3.000000",
                "p90_row_nnz": "8",
                "p99_row_nnz": "12",
                "empty_row_fraction": "0.125000",
            }
            with (case_output / "results.csv").open(
                "w", newline="", encoding="utf-8"
            ) as stream:
                writer = csv.DictWriter(stream, fieldnames=profile)
                writer.writeheader()
                writer.writerow(profile)

        with mock.patch.object(SUITE.subprocess, "run", side_effect=fake_run) as run:
            result = SUITE.main(
                [
                    f"--manifest={manifest}",
                    f"--matrix-root={matrix_root}",
                    f"--output-dir={output}",
                    "--",
                    "--iterations=3",
                ]
            )

        self.assertEqual(result, 0)
        self.assertEqual(run.call_count, 2)
        first_command = run.call_args_list[0].args[0]
        second_command = run.call_args_list[1].args[0]
        self.assertIn("--position-ablation", first_command)
        self.assertIn("--block-sizes=64", first_command)
        self.assertIn("--iterations=3", first_command)
        self.assertIn("--row-length-histogram=0:1,4:3", second_command)
        self.assertNotIn("--matrix=None", second_command)

        with (output / "summary.csv").open(newline="", encoding="utf-8") as stream:
            summary = list(csv.DictReader(stream))
        self.assertEqual([row["matrix"] for row in summary], ["first", "second"])
        self.assertEqual(summary[0]["best_position_stage"], "wave position")
        self.assertEqual(summary[0]["vs_fixed_mapping"], "1.500000")
        self.assertEqual(summary[0]["vs_rocsparse"], "1.250000")
        self.assertEqual(summary[0]["cv_row_nnz"], "0.750000")
        self.assertEqual(summary[0]["p99_row_nnz"], "12")

        with (output / "position-ablation.csv").open(
            newline="", encoding="utf-8"
        ) as stream:
            aggregate = list(csv.DictReader(stream))
        self.assertEqual(len(aggregate), 8)
        self.assertEqual(aggregate[0]["suite_matrix"], "first")
        self.assertEqual(aggregate[-1]["suite_matrix"], "second")


if __name__ == "__main__":
    unittest.main()
