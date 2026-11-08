#!/usr/bin/env python3

import argparse
import csv
from dataclasses import dataclass
import datetime
import json
from pathlib import Path
import re
import subprocess
import sys

import benchmark_utils as common


SCHEMA_VERSION = 1
RESERVED_BENCHMARK_OPTIONS = {
    "--columns",
    "--matrix",
    "--output-dir",
    "--position-ablation",
    "--row-length-histogram",
    "--row-order-seed",
    "--rows",
}
POSITION_STAGES = {
    "position baseline",
    "split",
    "local segmented",
    "wave position",
    "block position",
}
SUITE_COLUMNS = (
    "suite_matrix",
    "category",
    "source",
    "matrix_path",
    "operation",
    "matrix",
    "distribution",
    "nnz_per_row",
    "rhs_cols",
    "stage",
    "implementation",
    "format",
    "mapping",
    "block_size",
    "tile_size",
    "position_chunk_size",
    "position_order",
    "position_reduction",
    "median_us",
    "p95_us",
    "vs_position_baseline",
    "vs_rocsparse",
)
SUMMARY_COLUMNS = (
    "matrix",
    "category",
    "rows",
    "nnz",
    *common.ROW_PROFILE_FIELDS,
    "best_position_stage",
    "mapping",
    "block_size",
    "position_chunk_size",
    "position_reduction",
    "median_us",
    "vs_fixed_mapping",
    "vs_rocsparse",
)


@dataclass(frozen=True)
class MatrixSpec:
    name: str
    path: Path | None
    category: str
    source: str
    input_arguments: tuple[str, ...] = ()


@dataclass(frozen=True)
class SuiteManifest:
    benchmark_arguments: tuple[str, ...]
    matrices: tuple[MatrixSpec, ...]


def _require_string(value, context):
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} must be a nonempty string")
    return value


def _require_positive_integer(value, context):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{context} must be a positive integer")
    return value


def load_generated_profile(profile, context):
    if not isinstance(profile, dict):
        raise ValueError(f"{context} must be an object")
    expected = {"rows", "columns", "histogram", "seed"}
    unknown = set(profile) - expected
    if unknown:
        raise ValueError(
            f"{context} has unknown fields: {', '.join(sorted(unknown))}"
        )
    rows = _require_positive_integer(profile.get("rows"), f"{context}.rows")
    columns = _require_positive_integer(
        profile.get("columns"), f"{context}.columns"
    )
    histogram_value = _require_string(
        profile.get("histogram"), f"{context}.histogram"
    )
    try:
        histogram = common.parse_row_length_histogram(histogram_value)
    except argparse.ArgumentTypeError as error:
        raise ValueError(f"{context}.histogram: {error}") from error
    if max(length for length, _ in histogram) > columns:
        raise ValueError(
            f"{context}.histogram row lengths must not exceed columns"
        )
    total_weight = sum(weight for _, weight in histogram)
    if rows % total_weight != 0:
        raise ValueError(
            f"{context}.rows must be divisible by histogram weight "
            f"{total_weight}"
        )
    seed = profile.get("seed", 0)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError(f"{context}.seed must be a nonnegative integer")
    return (
        f"--rows={rows}",
        f"--columns={columns}",
        "--row-length-histogram="
        f"{common.format_row_length_histogram(histogram)}",
        f"--row-order-seed={seed}",
    )


def load_manifest(path, matrix_root):
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{path}: invalid JSON: {error}") from error
    if not isinstance(document, dict):
        raise ValueError(f"{path}: expected a JSON object")
    if document.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"{path}: expected schema_version {SCHEMA_VERSION}"
        )

    arguments = document.get("benchmark_arguments", [])
    if not isinstance(arguments, list) or not all(
        isinstance(argument, str) for argument in arguments
    ):
        raise ValueError(f"{path}: benchmark_arguments must be a string list")
    validate_forwarded_arguments(arguments, f"{path}: benchmark_arguments")

    entries = document.get("matrices")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{path}: matrices must be a nonempty list")
    matrices = []
    names = set()
    for index, entry in enumerate(entries):
        context = f"{path}: matrices[{index}]"
        if not isinstance(entry, dict):
            raise ValueError(f"{context} must be an object")
        name = _require_string(entry.get("name"), f"{context}.name")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
            raise ValueError(f"{context}.name is not a safe directory name")
        if name in names:
            raise ValueError(f"{path}: duplicate matrix name '{name}'")
        names.add(name)
        has_path = "path" in entry
        has_profile = "profile" in entry
        if has_path == has_profile:
            raise ValueError(
                f"{context} must define exactly one of path or profile"
            )
        if has_path:
            if matrix_root is None:
                raise ValueError(
                    f"{context}.path requires the --matrix-root option"
                )
            relative_path = Path(
                _require_string(entry.get("path"), f"{context}.path")
            )
            if relative_path.is_absolute() or ".." in relative_path.parts:
                raise ValueError(f"{context}.path must stay below matrix root")
            matrix_path = matrix_root / relative_path
            input_arguments = (f"--matrix={matrix_path}",)
            source = _require_string(entry.get("source"), f"{context}.source")
        else:
            matrix_path = None
            input_arguments = load_generated_profile(
                entry.get("profile"), f"{context}.profile"
            )
            source = _require_string(
                entry.get("source", "generated row-length histogram"),
                f"{context}.source",
            )
        matrices.append(
            MatrixSpec(
                name=name,
                path=matrix_path,
                category=_require_string(
                    entry.get("category"), f"{context}.category"
                ),
                source=source,
                input_arguments=input_arguments,
            )
        )
    return SuiteManifest(tuple(arguments), tuple(matrices))


def validate_forwarded_arguments(arguments, context="benchmark arguments"):
    for argument in arguments:
        option = argument.split("=", maxsplit=1)[0]
        if option in RESERVED_BENCHMARK_OPTIONS:
            raise ValueError(
                f"{context} cannot set reserved option '{option}'"
            )


def select_matrices(matrices, names):
    if not names:
        return matrices
    requested = [name.strip() for name in names.split(",") if name.strip()]
    if len(requested) != len(set(requested)):
        raise ValueError("--matrices contains duplicate names")
    by_name = {matrix.name: matrix for matrix in matrices}
    unknown = [name for name in requested if name not in by_name]
    if unknown:
        raise ValueError(f"unknown suite matrices: {', '.join(unknown)}")
    return tuple(by_name[name] for name in requested)


def read_ablation(path):
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"{path}: position ablation is empty")
    if any(row.get("operation") != "spmv" for row in rows):
        raise ValueError(f"{path}: expected only SpMV ablation rows")
    return rows


def read_matrix_profile(path):
    with path.open(newline="", encoding="utf-8") as stream:
        row = next(csv.DictReader(stream), None)
    if row is None:
        raise ValueError(f"{path}: benchmark results are empty")
    fields = ("rows", "nnz", *common.ROW_PROFILE_FIELDS)
    missing = [field for field in fields if field not in row]
    if missing:
        raise ValueError(f"{path}: missing matrix profile fields {missing}")
    return {field: row[field] for field in fields}


def summarize_matrix(matrix, rows, profile):
    position_rows = [row for row in rows if row["stage"] in POSITION_STAGES]
    if not position_rows:
        raise ValueError(f"{matrix.name}: missing position stages")
    best = min(position_rows, key=lambda row: float(row["median_us"]))
    fixed = next((row for row in rows if row["stage"] == "fixed mapping"), None)
    rocsparse = next((row for row in rows if row["stage"] == "rocSPARSE"), None)
    median = float(best["median_us"])
    return {
        "matrix": matrix.name,
        "category": matrix.category,
        **profile,
        "best_position_stage": best["stage"],
        "mapping": best["mapping"],
        "block_size": best["block_size"],
        "position_chunk_size": best["position_chunk_size"],
        "position_reduction": best["position_reduction"],
        "median_us": f"{median:.6f}",
        "vs_fixed_mapping": (
            f"{float(fixed['median_us']) / median:.6f}" if fixed else ""
        ),
        "vs_rocsparse": (
            f"{float(rocsparse['median_us']) / median:.6f}"
            if rocsparse
            else ""
        ),
    }


def write_csv(path, columns, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows):
    print("SpMV position-ablation suite")
    for row in rows:
        comparisons = []
        if row["vs_fixed_mapping"]:
            comparisons.append(f"{float(row['vs_fixed_mapping']):.2f}x fixed")
        if row["vs_rocsparse"]:
            comparisons.append(f"{float(row['vs_rocsparse']):.2f}x rocSPARSE")
        suffix = f" ({', '.join(comparisons)})" if comparisons else ""
        print(
            f"  {row['matrix']}: {row['best_position_stage']}, "
            f"{float(row['median_us']):.2f} us{suffix}"
        )


def parse_arguments(argv):
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Run the SpMV position-transformation ablation over a manifest "
            "of Matrix Market or generated row-profile inputs."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=repository / "benchmark" / "suites" / "spmv-position.json",
    )
    parser.add_argument("--matrix-root", type=Path)
    parser.add_argument(
        "--matrices",
        help="Comma-separated manifest matrix names; defaults to the full suite.",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--spmv-runner",
        type=Path,
        default=repository / "benchmark" / "run_spmv_benchmark.py",
    )
    parser.add_argument(
        "benchmark_arguments",
        nargs=argparse.REMAINDER,
        help="Additional SpMV runner arguments after '--'.",
    )
    args = parser.parse_args(argv)
    if args.benchmark_arguments[:1] == ["--"]:
        args.benchmark_arguments = args.benchmark_arguments[1:]
    validate_forwarded_arguments(args.benchmark_arguments)
    return args


def main(argv=None):
    args = parse_arguments(argv)
    manifest_path = args.manifest.resolve()
    matrix_root = args.matrix_root.resolve() if args.matrix_root else None
    manifest = load_manifest(manifest_path, matrix_root)
    matrices = select_matrices(manifest.matrices, args.matrices)
    missing = [
        matrix.path
        for matrix in matrices
        if matrix.path is not None and not matrix.path.is_file()
    ]
    if missing:
        formatted = "\n".join(f"  {path}" for path in missing)
        raise ValueError(f"missing Matrix Market inputs:\n{formatted}")

    repository = Path(__file__).resolve().parents[1]
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    output_directory = (
        args.output_dir
        or repository / "build" / "benchmark" / "spmv-suite" / timestamp
    ).resolve()
    output_directory.mkdir(parents=True, exist_ok=False)

    suite_rows = []
    summaries = []
    commands = []
    forwarded = (*manifest.benchmark_arguments, *args.benchmark_arguments)
    for matrix in matrices:
        case_directory = output_directory / matrix.name
        command = [
            sys.executable,
            str(args.spmv_runner.resolve()),
            *matrix.input_arguments,
            f"--output-dir={case_directory}",
            "--position-ablation",
            *forwarded,
        ]
        subprocess.run(command, check=True)
        commands.append(command)
        rows = read_ablation(case_directory / "position-ablation.csv")
        profile = read_matrix_profile(case_directory / "results.csv")
        suite_rows.extend(
            {
                "suite_matrix": matrix.name,
                "category": matrix.category,
                "source": matrix.source,
                "matrix_path": str(matrix.path) if matrix.path else "",
                **row,
            }
            for row in rows
        )
        summaries.append(summarize_matrix(matrix, rows, profile))

    write_csv(output_directory / "position-ablation.csv", SUITE_COLUMNS, suite_rows)
    write_csv(output_directory / "summary.csv", SUMMARY_COLUMNS, summaries)
    (output_directory / "metadata.json").write_text(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "matrix_root": str(matrix_root) if matrix_root else None,
                "matrices": [matrix.name for matrix in matrices],
                "commands": commands,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print_summary(summaries)
    print()
    print(f"Results: {output_directory}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
