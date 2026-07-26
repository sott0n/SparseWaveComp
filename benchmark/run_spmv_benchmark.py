#!/usr/bin/env python3

import argparse
from pathlib import Path
import statistics
import subprocess
import sys

import benchmark_utils as common

MAPPINGS = ("thread-per-row", "wave-per-row", "block-per-row")
DISTRIBUTIONS = ("uniform", "alternating", "skewed")
DISTRIBUTION_PERIODS = {
    "uniform": 1,
    "alternating": 2,
    "skewed": 8,
}
RESULT_COLUMNS = (
    "chip",
    "matrix",
    "mapping",
    "block_size",
    "wave_size",
    "rows",
    "cols",
    "nnz",
    "nnz_per_row",
    "min_row_nnz",
    "max_row_nnz",
    "mean_row_nnz",
    "distribution",
    "warmup",
    "iterations",
    "min_us",
    "median_us",
    "p95_us",
    "gnnz_per_sec",
    "gflops",
    "correct",
)
RESULT_FLOAT_FIELDS = (
    "min_us",
    "median_us",
    "p95_us",
    "gnnz_per_sec",
    "gflops",
    "mean_row_nnz",
)
MATRIX_REPORT = common.BenchmarkReport(
    details=("matrix", "rows", "columns", "nnz"),
    dimensions=("block_size", "mapping"),
    metrics=("median_us", "p95_us", "gnnz_per_sec"),
    baseline=("mapping", "thread-per-row"),
)
SYNTHETIC_REPORT = common.BenchmarkReport(
    details=("rows", "columns", "distributions"),
    dimensions=("distribution", "nnz_per_row", "block_size", "mapping"),
    metrics=("median_us", "p95_us", "gnnz_per_sec"),
    baseline=("mapping", "thread-per-row"),
)
CSR_BINARY_MAGIC = common.CSR_BINARY_MAGIC
positive_int = common.positive_int
nonnegative_int = common.nonnegative_int
parse_positive_int_list = common.parse_positive_int_list
read_matrix_market = common.read_matrix_market
write_csr_binary = common.write_csr_binary
parse_kernel_trace = common.parse_kernel_trace


def parse_distributions(value):
    values = [item.strip() for item in value.split(",") if item.strip()]
    invalid = [item for item in values if item not in DISTRIBUTIONS]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one distribution")
    if invalid:
        raise argparse.ArgumentTypeError(
            f"unknown distributions {invalid}; expected {DISTRIBUTIONS}"
        )
    if len(values) != len(set(values)):
        raise argparse.ArgumentTypeError(
            f"duplicate distributions are not allowed: {values}"
        )
    return values


def row_length(nnz_per_row, distribution, row):
    if distribution == "uniform":
        return nnz_per_row
    if distribution == "alternating":
        return 1 if row % 2 == 0 else 2 * nnz_per_row - 1
    if distribution == "skewed":
        return 8 * nnz_per_row - 7 if row % 8 == 7 else 1
    raise ValueError(f"unknown distribution: {distribution}")


def workload_shape(rows, nnz_per_row, distribution):
    lengths = [
        row_length(nnz_per_row, distribution, row) for row in range(rows)
    ]
    return {
        "nnz": sum(lengths),
        "min_row_nnz": min(lengths),
        "max_row_nnz": max(lengths),
        "mean_row_nnz": statistics.mean(lengths),
    }


def create_workloads(args):
    if args.matrix_data is not None:
        return [
            {
                "key": args.matrix_data["name"],
                "matrix": str(args.matrix_data["path"]),
                "distribution": "matrix-market",
                "nnz_per_row": "",
                "shape": args.matrix_data,
            }
        ]

    workloads = []
    for distribution in args.distributions:
        for nnz_per_row in args.nnz_per_row:
            shape = workload_shape(args.rows, nnz_per_row, distribution)
            shape.update({"rows": args.rows, "columns": args.columns})
            workloads.append(
                {
                    "key": f"{distribution}/nnz-{nnz_per_row}",
                    "matrix": "",
                    "distribution": distribution,
                    "nnz_per_row": nnz_per_row,
                    "shape": shape,
                }
            )
    return workloads


def validate_distribution_rows(rows, distributions):
    invalid = [
        (distribution, DISTRIBUTION_PERIODS[distribution])
        for distribution in distributions
        if rows % DISTRIBUTION_PERIODS[distribution] != 0
    ]
    if invalid:
        requirements = ", ".join(
            f"{distribution}: multiple of {period}"
            for distribution, period in invalid
        )
        raise ValueError(
            "row count must complete each distribution period to preserve "
            f"the requested mean NNZ/row ({requirements})"
        )


def row_length_mlir(nnz_per_row, distribution):
    if distribution == "uniform":
        return f"%rowLength = arith.constant {nnz_per_row} : index"
    if distribution == "alternating":
        period = 2
        long_slot = 1
        long_length = 2 * nnz_per_row - 1
    elif distribution == "skewed":
        period = 8
        long_slot = 7
        long_length = 8 * nnz_per_row - 7
    else:
        raise ValueError(f"unknown distribution: {distribution}")
    return "\n".join(
        (
            f"%distributionPeriod = arith.constant {period} : index",
            f"%longSlot = arith.constant {long_slot} : index",
            (
                "%distributionSlot = "
                "arith.remui %row, %distributionPeriod : index"
            ),
            (
                "%isLongRow = "
                "arith.cmpi eq, %distributionSlot, %longSlot : index"
            ),
            "%shortRowLength = arith.constant 1 : index",
            f"%longRowLength = arith.constant {long_length} : index",
            (
                "%rowLength = arith.select %isLongRow, "
                "%longRowLength, %shortRowLength : index"
            ),
        )
    )


def indent_mlir(value, spaces):
    indentation = " " * spaces
    return value.replace("\n", f"\n{indentation}")


def synthetic_initialization_mlir(nnz_per_row, distribution):
    row_length_code = indent_mlir(
        row_length_mlir(nnz_per_row, distribution), 4
    )
    return f"""%c0I32 = arith.constant 0 : i32
  memref.store %c0I32, %hostRowOffsets[%c0] : memref<?xi32>
  scf.for %row = %c0 to %cRows step %c1
      iter_args(%offset = %c0) -> index {{
    {row_length_code}
    %nextOffset = arith.addi %offset, %rowLength : index
    %nextOffsetI32 = arith.index_cast %nextOffset : index to i32
    %nextRow = arith.addi %row, %c1 : index
    memref.store %nextOffsetI32, %hostRowOffsets[%nextRow] : memref<?xi32>
    %rowLengthI64 = arith.index_cast %rowLength : index to i64
    %expected = arith.uitofp %rowLengthI64 : i64 to f32
    memref.store %expected, %hostExpected[%row] : memref<?xf32>
    scf.yield %nextOffset : index
  }}
  scf.for %position = %c0 to %cNnz step %c1 {{
    %column = arith.remui %position, %cColumns : index
    %columnI32 = arith.index_cast %column : index to i32
    memref.store %columnI32, %hostColumnIndices[%position] : memref<?xi32>
    memref.store %c1F32, %hostValues[%position] : memref<?xf32>
  }}
  scf.for %column = %c0 to %cColumns step %c1 {{
    memref.store %c1F32, %hostVector[%column] : memref<?xf32>
  }}"""


def render_mlir(
    template_path,
    rows,
    columns,
    nnz_per_row,
    dispatches,
    distribution="uniform",
    matrix=None,
):
    if matrix is None:
        shape = workload_shape(rows, nnz_per_row, distribution)
        initialization = synthetic_initialization_mlir(
            nnz_per_row, distribution
        )
        declarations = ""
    else:
        shape = matrix
        initialization = (
            "func.call @loadSpMVBenchmarkInputs("
            "%hostRowOffsets, %hostColumnIndices, %hostValues, "
            "%hostVector, %hostExpected) : "
            "(memref<?xi32>, memref<?xi32>, memref<?xf32>, "
            "memref<?xf32>, memref<?xf32>) -> ()"
        )
        declarations = (
            "func.func private @loadSpMVBenchmarkInputs("
            "memref<?xi32>, memref<?xi32>, memref<?xf32>, "
            "memref<?xf32>, memref<?xf32>) "
            "attributes {llvm.emit_c_interface}"
        )
    replacements = {
        "@ROWS@": str(rows),
        "@ROWS_PLUS_ONE@": str(rows + 1),
        "@COLUMNS@": str(columns),
        "@NNZ@": str(shape["nnz"]),
        "@DISPATCHES@": str(dispatches),
        "@INITIALIZE_INPUTS@": indent_mlir(initialization, 2),
        "@EXTERNAL_DECLARATIONS@": declarations,
    }
    return common.render_template(template_path, replacements)


def result_row(args, mapping, block_size, workload, timing):
    shape = workload["shape"]
    nnz = shape["nnz"]
    median_seconds = timing["median_us"] / 1_000_000.0
    return {
        "chip": args.chip,
        "matrix": workload["matrix"],
        "mapping": mapping,
        "block_size": block_size,
        "wave_size": args.wave_size,
        "rows": shape["rows"],
        "cols": shape["columns"],
        "nnz": nnz,
        "nnz_per_row": workload["nnz_per_row"],
        "min_row_nnz": shape["min_row_nnz"],
        "max_row_nnz": shape["max_row_nnz"],
        "mean_row_nnz": shape["mean_row_nnz"],
        "distribution": workload["distribution"],
        "warmup": args.warmup,
        "iterations": args.iterations,
        "min_us": timing["min_us"],
        "median_us": timing["median_us"],
        "p95_us": timing["p95_us"],
        "gnnz_per_sec": nnz / median_seconds / 1_000_000_000.0,
        "gflops": 2.0 * nnz / median_seconds / 1_000_000_000.0,
        "correct": True,
    }


def build_metadata(args, repository, commands):
    metadata = common.base_metadata(args, repository, "spmv")
    metadata.update(
        {
            "rows": args.rows,
            "columns": args.columns,
            "nnz_per_row": (
                args.nnz_per_row if args.matrix_data is None else None
            ),
            "distributions": (
                args.distributions if args.matrix_data is None else None
            ),
            "matrix": str(args.matrix) if args.matrix else None,
            "matrix_field": (
                args.matrix_data["field"]
                if args.matrix_data is not None
                else None
            ),
            "matrix_symmetry": (
                args.matrix_data["symmetry"]
                if args.matrix_data is not None
                else None
            ),
            "commands": commands,
        }
    )
    return metadata


def print_results(args, results):
    report = MATRIX_REPORT if args.matrix_data is not None else SYNTHETIC_REPORT
    common.print_benchmark_report(
        args,
        "SparseWave SpMV benchmark",
        report,
        results,
    )


def validate_paths(args):
    common.validate_required_paths(
        args, needs_benchmark_utils=args.matrix_data is not None
    )
    if args.matrix_data is None:
        validate_distribution_rows(args.rows, args.distributions)
    if args.wave_size != 32:
        raise ValueError(
            "wave-per-row and block-per-row benchmarking currently require "
            "Wave32"
        )
    common.validate_block_sizes(args, require_wave_multiple=True)
    maximum_i32 = (1 << 31) - 1
    columns = (
        args.matrix_data["columns"]
        if args.matrix_data is not None
        else args.columns
    )
    if columns > maximum_i32:
        raise ValueError("column count must fit in the i32 column-index type")
    if args.matrix_data is not None:
        invalid_nnz = args.matrix_data["nnz"] > maximum_i32
    else:
        invalid_nnz = any(
            workload_shape(args.rows, value, distribution)["nnz"] > maximum_i32
            for value in args.nnz_per_row
            for distribution in args.distributions
        )
    if invalid_nnz:
        raise ValueError("NNZ must fit in the i32 CSR row-offset type")


def parse_arguments(argv):
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Benchmark SparseWave SpMV mappings on an AMD GPU."
    )
    parser.add_argument(
        "--matrix",
        type=Path,
        help="Matrix Market coordinate file to benchmark instead of synthetic inputs.",
    )
    parser.add_argument("--rows", type=positive_int, default=65536)
    parser.add_argument("--columns", type=positive_int, default=65536)
    parser.add_argument(
        "--nnz-per-row",
        type=parse_positive_int_list,
        default=parse_positive_int_list("1,2,4,8,16,32,64,128,256"),
    )
    parser.add_argument(
        "--distributions",
        type=parse_distributions,
        default=parse_distributions("uniform,alternating,skewed"),
    )
    common.add_common_arguments(parser, repository)
    args = parser.parse_args(argv)

    common.configure_common_arguments(args)
    if args.matrix is not None:
        args.matrix = args.matrix.resolve()
        args.matrix_data = read_matrix_market(args.matrix)
        args.rows = args.matrix_data["rows"]
        args.columns = args.matrix_data["columns"]
    else:
        args.matrix_data = None
    return args


def main(argv=None):
    args = parse_arguments(argv)
    validate_paths(args)
    repository = Path(__file__).resolve().parents[1]
    template = repository / "benchmark" / "spmv.mlir.in"
    results = []
    commands = []
    with common.BenchmarkWorkspace(
        args,
        repository,
        result_directory="results",
        temporary_prefix="sparsewave-spmv-benchmark-",
    ) as workspace:
        csr_binary = None
        if args.matrix_data is not None:
            csr_binary = workspace.artifact_root / "matrix.csr"
            write_csr_binary(csr_binary, args.matrix_data)

        for workload in create_workloads(args):
            shape = workload["shape"]
            source_text = render_mlir(
                template,
                shape["rows"],
                shape["columns"],
                workload["nnz_per_row"],
                args.warmup + args.iterations,
                workload["distribution"],
                args.matrix_data,
            )
            for block_size in args.block_sizes:
                for mapping in MAPPINGS:
                    case_directory = (
                        workspace.artifact_root
                        / workload["key"]
                        / f"block-{block_size}"
                        / mapping
                    )
                    timing, compile_command, profile_command = common.run_case(
                        args,
                        source_text,
                        case_directory,
                        "spmv",
                        mapping,
                        block_size,
                        csr_binary,
                    )
                    results.append(
                        result_row(
                            args,
                            mapping,
                            block_size,
                            workload,
                            timing,
                        )
                    )
                    commands.append(
                        {
                            "matrix": workload["matrix"],
                            "distribution": workload["distribution"],
                            "nnz_per_row": workload["nnz_per_row"],
                            "block_size": block_size,
                            "mapping": mapping,
                            "compile": compile_command,
                            "profile": profile_command,
                        }
                    )

        common.write_results(
            workspace.output_directory / "results.csv",
            RESULT_COLUMNS,
            RESULT_FLOAT_FIELDS,
            results,
        )
        common.write_metadata(
            workspace.output_directory / "metadata.json",
            build_metadata(args, repository, commands),
        )
        print_results(args, results)
        print()
        print(f"Results: {workspace.output_directory}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
