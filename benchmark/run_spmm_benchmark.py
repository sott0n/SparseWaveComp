#!/usr/bin/env python3

import argparse
from pathlib import Path
import subprocess
import sys

import benchmark_utils as common


BASELINE_MAPPING = "thread-per-output"
TILED_MAPPING = "wave-per-row-tile"
RESULT_COLUMNS = (
    "chip",
    "matrix",
    "mapping",
    "block_size",
    "wave_size",
    "tile_size",
    "rows",
    "input_cols",
    "rhs_cols",
    "nnz",
    "min_row_nnz",
    "max_row_nnz",
    "mean_row_nnz",
    "warmup",
    "iterations",
    "min_us",
    "median_us",
    "p95_us",
    "gproducts_per_sec",
    "gflops",
    "correct",
)
RESULT_FLOAT_FIELDS = (
    "min_us",
    "median_us",
    "p95_us",
    "gproducts_per_sec",
    "gflops",
    "mean_row_nnz",
)
REPORT = common.BenchmarkReport(
    details=("matrix", "rows", "input_columns", "nnz", "rhs_columns"),
    dimensions=("rhs_cols", "block_size", "tile_size", "mapping"),
    metrics=("median_us", "p95_us", "gproducts_per_sec", "gflops"),
)


def render_mlir(template_path, matrix, rhs_columns, dispatches):
    return common.render_template(
        template_path,
        {
            "@ROWS@": matrix["rows"],
            "@ROWS_PLUS_ONE@": matrix["rows"] + 1,
            "@INPUT_COLUMNS@": matrix["columns"],
            "@RHS_COLUMNS@": rhs_columns,
            "@NNZ@": matrix["nnz"],
            "@DISPATCHES@": dispatches,
        },
    )


def benchmark_cases(block_sizes, tile_sizes):
    for block_size in block_sizes:
        yield BASELINE_MAPPING, block_size, None
        for tile_size in tile_sizes:
            yield TILED_MAPPING, block_size, tile_size


def result_row(
    args, mapping, block_size, tile_size, rhs_columns, timing
):
    matrix = args.matrix_data
    products = matrix["nnz"] * rhs_columns
    median_seconds = timing["median_us"] / 1_000_000.0
    return {
        "chip": args.chip,
        "matrix": str(args.matrix),
        "mapping": mapping,
        "block_size": block_size,
        "wave_size": args.wave_size,
        "tile_size": tile_size,
        "rows": matrix["rows"],
        "input_cols": matrix["columns"],
        "rhs_cols": rhs_columns,
        "nnz": matrix["nnz"],
        "min_row_nnz": matrix["min_row_nnz"],
        "max_row_nnz": matrix["max_row_nnz"],
        "mean_row_nnz": matrix["mean_row_nnz"],
        "warmup": args.warmup,
        "iterations": args.iterations,
        "min_us": timing["min_us"],
        "median_us": timing["median_us"],
        "p95_us": timing["p95_us"],
        "gproducts_per_sec": products / median_seconds / 1_000_000_000.0,
        "gflops": 2.0 * products / median_seconds / 1_000_000_000.0,
        "correct": True,
    }


def build_metadata(args, repository, commands):
    metadata = common.base_metadata(args, repository, "spmm")
    metadata.update(
        {
            "rhs_columns": args.rhs_columns,
            "tile_sizes": args.tile_sizes,
            "matrix": str(args.matrix),
            "matrix_field": args.matrix_data["field"],
            "matrix_symmetry": args.matrix_data["symmetry"],
            "rows": args.matrix_data["rows"],
            "input_columns": args.matrix_data["columns"],
            "nnz": args.matrix_data["nnz"],
            "commands": commands,
        }
    )
    return metadata


def print_results(args, results):
    common.print_benchmark_report(
        args,
        "SparseWave SpMM benchmark",
        REPORT,
        results,
    )


def validate_paths(args):
    common.validate_required_paths(args, needs_benchmark_utils=True)
    common.validate_block_sizes(args, require_wave_multiple=True)
    if args.wave_size != 32:
        raise ValueError("wave-per-row-tile currently requires wave size 32")
    if any(tile_size > 32 for tile_size in args.tile_sizes):
        raise ValueError("tile sizes must not exceed 32")
    maximum_i32 = (1 << 31) - 1
    if args.matrix_data["columns"] > maximum_i32:
        raise ValueError(
            "matrix column count must fit in the i32 column-index type"
        )
    if (
        args.matrix_data["rows"] > maximum_i32
        or args.matrix_data["nnz"] > maximum_i32
    ):
        raise ValueError("matrix rows and NNZ must fit in the i32 CSR type")


def parse_arguments(argv):
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Benchmark SparseWave SpMM mappings on an AMD GPU."
    )
    parser.add_argument(
        "--matrix",
        type=Path,
        required=True,
        help="Matrix Market coordinate matrix used as the sparse LHS.",
    )
    parser.add_argument(
        "--rhs-columns",
        type=common.parse_positive_int_list,
        default=common.parse_positive_int_list("8,16,32,64,128"),
        help="Comma-separated dense RHS column counts.",
    )
    tile_group = parser.add_mutually_exclusive_group()
    tile_group.add_argument(
        "--tile-sizes",
        type=common.parse_positive_int_list,
        default=common.parse_positive_int_list("1,2,4,8,16"),
        help=(
            "Comma-separated output-column tile sizes measured for "
            "wave-per-row-tile."
        ),
    )
    tile_group.add_argument(
        "--tile-size",
        dest="tile_sizes",
        type=lambda value: [common.positive_int(value)],
        help=argparse.SUPPRESS,
    )
    common.add_common_arguments(parser, repository)
    args = parser.parse_args(argv)
    common.configure_common_arguments(args)
    args.matrix = args.matrix.resolve()
    args.matrix_data = common.read_matrix_market(args.matrix)
    return args


def main(argv=None):
    args = parse_arguments(argv)
    validate_paths(args)
    repository = Path(__file__).resolve().parents[1]
    template = repository / "benchmark" / "spmm.mlir.in"
    results = []
    commands = []
    with common.BenchmarkWorkspace(
        args,
        repository,
        result_directory="spmm-results",
        temporary_prefix="sparsewave-spmm-benchmark-",
    ) as workspace:
        csr_binary = workspace.artifact_root / "matrix.csr"
        common.write_csr_binary(csr_binary, args.matrix_data)
        for rhs_columns in args.rhs_columns:
            source_text = render_mlir(
                template,
                args.matrix_data,
                rhs_columns,
                args.warmup + args.iterations,
            )
            for mapping, block_size, tile_size in benchmark_cases(
                args.block_sizes, args.tile_sizes
            ):
                case_directory = (
                    workspace.artifact_root
                    / f"rhs-{rhs_columns}"
                    / f"block-{block_size}"
                    / mapping
                )
                pipeline_options = ()
                if tile_size is not None:
                    case_directory /= f"tile-{tile_size}"
                    pipeline_options = (f"spmm-tile-size={tile_size}",)
                timing, compile_command, profile_command = common.run_case(
                    args,
                    source_text,
                    case_directory,
                    "spmm",
                    mapping,
                    block_size,
                    csr_binary,
                    pipeline_options=pipeline_options,
                )
                results.append(
                    result_row(
                        args,
                        mapping,
                        block_size,
                        tile_size,
                        rhs_columns,
                        timing,
                    )
                )
                commands.append(
                    {
                        "rhs_columns": rhs_columns,
                        "block_size": block_size,
                        "mapping": mapping,
                        "tile_size": tile_size,
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
    except (
        OSError,
        RuntimeError,
        ValueError,
        subprocess.CalledProcessError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
