#!/usr/bin/env python3

import argparse
from pathlib import Path
import sys

import benchmark_utils as common


OPERATIONS = ("add", "multiply")
PHASE_KERNELS = {
    "symbolic": "elementwise_symbolic",
    "prefix": "elementwise_prefix",
    "numeric": "elementwise_numeric",
}
RESULT_COLUMNS = (
    "chip",
    "operation",
    "block_size",
    "wave_size",
    "rows",
    "columns",
    "nnz_per_row",
    "overlap_percent",
    "output_nnz",
    "warmup",
    "iterations",
    "min_us",
    "median_us",
    "p95_us",
    "symbolic_median_us",
    "symbolic_fraction",
    "prefix_median_us",
    "numeric_median_us",
    "numeric_fraction",
    "prefix_fraction",
    "correct",
)
RESULT_FLOAT_FIELDS = (
    "min_us",
    "median_us",
    "p95_us",
    "symbolic_median_us",
    "symbolic_fraction",
    "prefix_median_us",
    "numeric_median_us",
    "numeric_fraction",
    "prefix_fraction",
)
REPORT = common.BenchmarkReport(
    details=("rows", "columns"),
    dimensions=("operation", "nnz_per_row", "overlap_percent", "block_size"),
    metrics=(
        "median_us",
        "symbolic_median_us",
        "symbolic_fraction",
        "prefix_median_us",
        "prefix_fraction",
        "numeric_median_us",
        "numeric_fraction",
    ),
)


def parse_operations(value):
    values = [item.strip() for item in value.split(",") if item.strip()]
    invalid = [item for item in values if item not in OPERATIONS]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one operation")
    if invalid:
        raise argparse.ArgumentTypeError(
            f"unknown operations {invalid}; expected {OPERATIONS}"
        )
    if len(values) != len(set(values)):
        raise argparse.ArgumentTypeError(
            f"duplicate operations are not allowed: {values}"
        )
    return values


def parse_overlap_percentages(value):
    try:
        values = [int(item.strip()) for item in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"expected comma-separated integer percentages: {value}"
        ) from error
    if not values:
        raise argparse.ArgumentTypeError("expected at least one overlap")
    invalid = [item for item in values if item < 0 or item > 100]
    if invalid:
        raise argparse.ArgumentTypeError(
            f"overlap percentages must be between 0 and 100: {invalid}"
        )
    if len(values) != len(set(values)):
        raise argparse.ArgumentTypeError(
            f"duplicate overlaps are not allowed: {values}"
        )
    return values


def workload_shape(rows, nnz_per_row, overlap_percent, operation):
    overlap_product = nnz_per_row * overlap_percent
    if overlap_product % 100:
        raise ValueError(
            f"{overlap_percent}% of {nnz_per_row} NNZ/row is not integral"
        )
    overlap_nnz = overlap_product // 100
    output_nnz_per_row = (
        2 * nnz_per_row - overlap_nnz
        if operation == "add"
        else overlap_nnz
    )
    return {
        "overlap_nnz": overlap_nnz,
        "input_nnz": rows * nnz_per_row,
        "output_nnz_per_row": output_nnz_per_row,
        "output_nnz": rows * output_nnz_per_row,
    }


def expected_value_mlir(operation):
    if operation == "multiply":
        return "%expectedValue = arith.constant 2.0 : f32"
    return "\n    ".join(
        (
            "%isShared = arith.cmpi ult, %localPosition, %cOverlapNnz : index",
            (
                "%isLhsOnly = arith.cmpi ult, %localPosition, "
                "%cInputNnzPerRow : index"
            ),
            (
                "%nonSharedValue = arith.select %isLhsOnly, "
                "%c1F32, %c2F32 : f32"
            ),
            (
                "%expectedValue = arith.select %isShared, "
                "%c3F32, %nonSharedValue : f32"
            ),
        )
    )


def render_mlir(template_path, rows, columns, nnz_per_row, overlap_percent,
                operation, dispatches):
    shape = workload_shape(rows, nnz_per_row, overlap_percent, operation)
    replacements = {
        "@KIND@": operation,
        "@ROWS@": rows,
        "@ROWS_PLUS_ONE@": rows + 1,
        "@COLUMNS@": columns,
        "@INPUT_NNZ_PER_ROW@": nnz_per_row,
        "@INPUT_NNZ@": shape["input_nnz"],
        "@OVERLAP_NNZ@": shape["overlap_nnz"],
        "@OUTPUT_NNZ_PER_ROW@": shape["output_nnz_per_row"],
        "@OUTPUT_NNZ@": shape["output_nnz"],
        "@DISPATCHES@": dispatches,
        "@EXPECTED_VALUE@": expected_value_mlir(operation),
    }
    return common.render_template(template_path, replacements)


def result_row(args, operation, nnz_per_row, overlap_percent, block_size,
               shape, timings):
    total = timings["total"]
    return {
        "chip": args.chip,
        "operation": operation,
        "block_size": block_size,
        "wave_size": args.wave_size,
        "rows": args.rows,
        "columns": args.columns,
        "nnz_per_row": nnz_per_row,
        "overlap_percent": overlap_percent,
        "output_nnz": shape["output_nnz"],
        "warmup": args.warmup,
        "iterations": args.iterations,
        "min_us": total["min_us"],
        "median_us": total["median_us"],
        "p95_us": total["p95_us"],
        "symbolic_median_us": timings["symbolic"]["median_us"],
        "symbolic_fraction": timings["symbolic"]["fraction"],
        "prefix_median_us": timings["prefix"]["median_us"],
        "numeric_median_us": timings["numeric"]["median_us"],
        "numeric_fraction": timings["numeric"]["fraction"],
        "prefix_fraction": timings["prefix"]["fraction"],
        "correct": True,
    }


def validate_paths(args):
    common.validate_required_paths(args, needs_benchmark_utils=False)
    common.validate_block_sizes(args, require_wave_multiple=False)
    if args.rocsparse:
        raise ValueError("rocSPARSE elementwise baselines are not implemented")
    if args.columns < 2 * max(args.nnz_per_row):
        raise ValueError(
            "column count must be at least twice the maximum NNZ/row"
        )
    maximum_i32 = (1 << 31) - 1
    for operation in args.operations:
        for nnz_per_row in args.nnz_per_row:
            for overlap in args.overlaps:
                shape = workload_shape(
                    args.rows, nnz_per_row, overlap, operation
                )
                if operation == "multiply" and shape["overlap_nnz"] == 0:
                    raise ValueError(
                        "multiply requires at least one overlapping NNZ/row"
                    )
                if max(shape["input_nnz"], shape["output_nnz"]) > maximum_i32:
                    raise ValueError("input and output NNZ must fit in i32")


def parse_arguments(argv):
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Benchmark SparseWave CSR elementwise output assembly."
    )
    parser.add_argument("--rows", type=common.positive_int, default=65536)
    parser.add_argument("--columns", type=common.positive_int, default=65536)
    parser.add_argument(
        "--nnz-per-row",
        type=common.parse_positive_int_list,
        default=common.parse_positive_int_list("4,32"),
    )
    parser.add_argument(
        "--overlaps",
        type=parse_overlap_percentages,
        default=parse_overlap_percentages("25,50,100"),
        help="Comma-separated structural overlap percentages.",
    )
    parser.add_argument(
        "--operations",
        type=parse_operations,
        default=parse_operations("add,multiply"),
    )
    common.add_common_arguments(parser, repository)
    args = parser.parse_args(argv)
    common.configure_common_arguments(args)
    args.matrix_data = None
    return args


def main(argv=None):
    args = parse_arguments(argv)
    validate_paths(args)
    repository = Path(__file__).resolve().parents[1]
    template = repository / "benchmark" / "elementwise.mlir.in"
    results = []
    commands = []
    with common.BenchmarkWorkspace(
        args,
        repository,
        result_directory="elementwise-results",
        temporary_prefix="sparsewave-elementwise-benchmark-",
    ) as workspace:
        for operation in args.operations:
            for nnz_per_row in args.nnz_per_row:
                for overlap in args.overlaps:
                    shape = workload_shape(
                        args.rows, nnz_per_row, overlap, operation
                    )
                    source_text = render_mlir(
                        template,
                        args.rows,
                        args.columns,
                        nnz_per_row,
                        overlap,
                        operation,
                        args.warmup + args.iterations,
                    )
                    for block_size in args.block_sizes:
                        case_directory = (
                            workspace.artifact_root
                            / operation
                            / f"nnz-{nnz_per_row}"
                            / f"overlap-{overlap}"
                            / f"block-{block_size}"
                        )
                        case_directory.mkdir(parents=True)
                        source = case_directory / "input.mlir"
                        compiled = case_directory / "compiled.mlir"
                        trace_directory = case_directory / "trace"
                        trace_directory.mkdir()
                        source.write_text(source_text, encoding="utf-8")
                        compile_command = common.compile_mlir(
                            args,
                            source,
                            compiled,
                            "elementwise",
                            None,
                            block_size,
                        )
                        profile_command = common.profile_mlir(
                            args,
                            compiled,
                            trace_directory,
                            None,
                            "elementwise",
                        )
                        trace = common.discover_trace(trace_directory)
                        if args.gpu_name == args.chip:
                            args.gpu_name = common.discover_gpu_name(
                                trace_directory, args.chip
                            )
                        timings = common.parse_kernel_phase_trace(
                            trace,
                            PHASE_KERNELS,
                            args.warmup,
                            args.iterations,
                        )
                        results.append(
                            result_row(
                                args,
                                operation,
                                nnz_per_row,
                                overlap,
                                block_size,
                                shape,
                                timings,
                            )
                        )
                        commands.extend((compile_command, profile_command))

        common.write_results(
            workspace.output_directory / "results.csv",
            RESULT_COLUMNS,
            RESULT_FLOAT_FIELDS,
            results,
        )
        metadata = common.base_metadata(args, repository, "elementwise")
        metadata.update(
            {
                "rows": args.rows,
                "columns": args.columns,
                "nnz_per_row": args.nnz_per_row,
                "overlap_percentages": args.overlaps,
                "operations": args.operations,
                "commands": commands,
            }
        )
        common.write_metadata(
            workspace.output_directory / "metadata.json", metadata
        )
        common.print_benchmark_report(
            args, "SparseWave CSR elementwise benchmark", REPORT, results
        )
        print(f"\nResults: {workspace.output_directory}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
