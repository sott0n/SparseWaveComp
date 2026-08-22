#!/usr/bin/env python3

import argparse
from pathlib import Path
import subprocess
import sys
import time

import benchmark_utils as common


BASELINE_MAPPING = "thread-per-output"
TILED_MAPPING = "wave-per-row-tile"
POSITION_MAPPING = "thread-per-position"
MAPPINGS = (BASELINE_MAPPING, TILED_MAPPING, POSITION_MAPPING)
FORMATS = ("csr", "bsr")
RESULT_COLUMNS = (
    "chip",
    "matrix",
    "implementation",
    "format",
    "mapping",
    "algorithm",
    "block_size",
    "wave_size",
    "tile_size",
    "position_chunk_size",
    "storage_block_size",
    "rows",
    "input_cols",
    "rhs_cols",
    "nnz",
    "nnzb",
    "block_density",
    "internal_zero_fraction",
    "storage_overhead",
    "min_row_nnz",
    "max_row_nnz",
    "mean_row_nnz",
    "warmup",
    "iterations",
    "preprocess_us",
    "conversion_us",
    "min_us",
    "median_us",
    "p95_us",
    "gproducts_per_sec",
    "gflops",
    *common.GPU_RESOURCE_FIELDS,
    "correct",
)
RESULT_FLOAT_FIELDS = (
    "preprocess_us",
    "conversion_us",
    "min_us",
    "median_us",
    "p95_us",
    "gproducts_per_sec",
    "gflops",
    "mean_row_nnz",
    "block_density",
    "internal_zero_fraction",
    "storage_overhead",
)
REPORT = common.BenchmarkReport(
    details=("matrix", "rows", "input_columns", "nnz", "rhs_columns"),
    dimensions=(
        "rhs_cols",
        "format",
        "storage_block_size",
        "block_size",
        "tile_size",
        "position_chunk_size",
        "mapping",
    ),
    metrics=("median_us", "p95_us", "gproducts_per_sec", "gflops"),
)
ROCSPARSE_REPORT = common.BenchmarkReport(
    details=("matrix", "rows", "input_columns", "nnz", "rhs_columns"),
    dimensions=(
        "rhs_cols",
        "implementation",
        "format",
        "storage_block_size",
        "block_size",
        "tile_size",
        "position_chunk_size",
        "mapping",
    ),
    metrics=("median_us", "p95_us", "gproducts_per_sec", "gflops"),
    baseline=("implementation", "rocsparse"),
    baseline_group=("rhs_cols",),
)
RESOURCE_REPORT = common.BenchmarkReport(
    details=(),
    dimensions=(
        "rhs_cols",
        "format",
        "storage_block_size",
        "block_size",
        "tile_size",
        "position_chunk_size",
        "mapping",
    ),
    metrics=common.GPU_RESOURCE_METRICS,
)
FORMAT_REPORT = common.BenchmarkReport(
    details=(),
    dimensions=("format", "storage_block_size"),
    metrics=(
        "nnzb",
        "block_density",
        "internal_zero_fraction",
        "storage_overhead",
    ),
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


def render_bsr_mlir(template_path, matrix, rhs_columns, dispatches):
    block_size = matrix["block_size"]
    return common.render_template(
        template_path,
        {
            "@ROWS@": matrix["rows"],
            "@BLOCK_ROWS_PLUS_ONE@": matrix["rows"] // block_size + 1,
            "@INPUT_COLUMNS@": matrix["columns"],
            "@RHS_COLUMNS@": rhs_columns,
            "@NNZB@": matrix["nnzb"],
            "@BLOCK_VALUE_COUNT@": len(matrix["block_values"]),
            "@BSR_BLOCK_SIZE@": block_size,
            "@DISPATCHES@": dispatches,
        },
    )


def parse_formats(value):
    values = [item.strip() for item in value.split(",") if item.strip()]
    invalid = [item for item in values if item not in FORMATS]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one sparse format")
    if invalid:
        raise argparse.ArgumentTypeError(
            f"unknown sparse formats {invalid}; expected {FORMATS}"
        )
    if len(values) != len(set(values)):
        raise argparse.ArgumentTypeError(
            f"duplicate sparse formats are not allowed: {values}"
        )
    return values


def parse_mappings(value):
    values = [item.strip() for item in value.split(",") if item.strip()]
    invalid = [item for item in values if item not in MAPPINGS]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one SpMM mapping")
    if invalid:
        raise argparse.ArgumentTypeError(
            f"unknown SpMM mappings {invalid}; expected {MAPPINGS}"
        )
    if len(values) != len(set(values)):
        raise argparse.ArgumentTypeError(
            f"duplicate SpMM mappings are not allowed: {values}"
        )
    return values


def benchmark_cases(block_sizes, tile_sizes, position_chunk_sizes, mappings):
    for block_size in block_sizes:
        if BASELINE_MAPPING in mappings:
            yield BASELINE_MAPPING, block_size, None, None
        if TILED_MAPPING in mappings:
            for tile_size in tile_sizes:
                yield TILED_MAPPING, block_size, tile_size, None
        if POSITION_MAPPING in mappings:
            for chunk_size in position_chunk_sizes:
                yield POSITION_MAPPING, block_size, None, chunk_size


def sparsewave_kernel_layout(mapping):
    if mapping == POSITION_MAPPING:
        return 2, 1
    return 1, 0


def result_row(
    args,
    mapping,
    block_size,
    tile_size,
    position_chunk_size,
    rhs_columns,
    timing,
    resources,
    sparse_format="csr",
    bsr_data=None,
    conversion_us=None,
    implementation="sparsewave",
    preprocess_us=None,
):
    matrix = args.matrix_data
    products = matrix["nnz"] * rhs_columns
    median_seconds = timing["median_us"] / 1_000_000.0
    return {
        "chip": args.chip,
        "matrix": str(args.matrix),
        "implementation": implementation,
        "format": sparse_format,
        "mapping": mapping,
        "algorithm": "default" if implementation == "rocsparse" else "",
        "block_size": block_size,
        "wave_size": (
            args.wave_size if implementation == "sparsewave" else None
        ),
        "tile_size": tile_size,
        "position_chunk_size": position_chunk_size,
        "storage_block_size": (
            bsr_data["block_size"] if bsr_data is not None else None
        ),
        "rows": matrix["rows"],
        "input_cols": matrix["columns"],
        "rhs_cols": rhs_columns,
        "nnz": matrix["nnz"],
        "nnzb": bsr_data["nnzb"] if bsr_data is not None else None,
        "block_density": (
            bsr_data["block_density"] if bsr_data is not None else None
        ),
        "internal_zero_fraction": (
            bsr_data["internal_zero_fraction"]
            if bsr_data is not None
            else None
        ),
        "storage_overhead": (
            bsr_data["storage_overhead"] if bsr_data is not None else None
        ),
        "min_row_nnz": matrix["min_row_nnz"],
        "max_row_nnz": matrix["max_row_nnz"],
        "mean_row_nnz": matrix["mean_row_nnz"],
        "warmup": args.warmup,
        "iterations": args.iterations,
        "preprocess_us": preprocess_us,
        "conversion_us": conversion_us,
        "min_us": timing["min_us"],
        "median_us": timing["median_us"],
        "p95_us": timing["p95_us"],
        "gproducts_per_sec": products / median_seconds / 1_000_000_000.0,
        "gflops": 2.0 * products / median_seconds / 1_000_000_000.0,
        **resources,
        "correct": True,
    }


def build_metadata(args, repository, commands):
    metadata = common.base_metadata(args, repository, "spmm")
    metadata.update(
        {
            "rhs_columns": args.rhs_columns,
            "tile_sizes": args.tile_sizes,
            "position_chunk_sizes": args.position_chunk_sizes,
            "mappings": args.mappings,
            "formats": args.formats,
            "bsr_block_sizes": args.bsr_block_sizes,
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
        ROCSPARSE_REPORT if args.rocsparse else REPORT,
        results,
    )
    print()
    common.print_gpu_resource_report(
        args,
        "SparseWave SpMM GPU resources",
        RESOURCE_REPORT,
        results,
    )
    if "bsr" in args.formats:
        print()
        common.print_benchmark_report(
            args,
            "SparseWave SpMM storage formats",
            FORMAT_REPORT,
            [
                row
                for row in results
                if row["implementation"] == "sparsewave"
                and row["rhs_cols"] == args.rhs_columns[0]
                and row["block_size"] == args.block_sizes[0]
                and row["tile_size"] is None
                and row["mapping"] == BASELINE_MAPPING
            ],
        )


def validate_paths(args):
    common.validate_required_paths(
        args,
        needs_benchmark_utils=True,
        needs_resource_inspector=True,
    )
    uses_tiled_mapping = (
        "csr" in args.formats and TILED_MAPPING in args.mappings
    )
    common.validate_block_sizes(args, require_wave_multiple=uses_tiled_mapping)
    if uses_tiled_mapping and args.wave_size != 32:
        raise ValueError("wave-per-row-tile currently requires wave size 32")
    if uses_tiled_mapping and any(
        tile_size > 32 for tile_size in args.tile_sizes
    ):
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
    if "bsr" in args.formats and any(
        block_size > 1024 for block_size in args.bsr_block_sizes
    ):
        raise ValueError("BSR block sizes must not exceed 1024")


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
        "--formats",
        type=parse_formats,
        default=["csr", "bsr"],
        help="Comma-separated sparse storage formats: csr,bsr.",
    )
    parser.add_argument(
        "--bsr-block-sizes",
        type=common.parse_positive_int_list,
        default=common.parse_positive_int_list("2,4,8"),
        help="Comma-separated square BSR storage block sizes.",
    )
    parser.add_argument(
        "--mappings",
        type=parse_mappings,
        default=list(MAPPINGS),
        help=(
            "Comma-separated CSR mappings: thread-per-output,"
            "wave-per-row-tile,thread-per-position."
        ),
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
    parser.add_argument(
        "--position-chunk-sizes",
        type=common.parse_positive_int_list,
        default=common.parse_positive_int_list("1,4,8"),
        help=(
            "Comma-separated consecutive position-column pairs processed "
            "by each thread-per-position worker."
        ),
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
        conversion_start = time.perf_counter_ns()
        common.write_csr_binary(csr_binary, args.matrix_data)
        csr_conversion_us = (time.perf_counter_ns() - conversion_start) / 1000.0
        bsr_inputs = []
        if "bsr" in args.formats:
            for storage_block_size in args.bsr_block_sizes:
                conversion_start = time.perf_counter_ns()
                bsr_data = common.convert_to_bsr(
                    args.matrix_data, storage_block_size
                )
                bsr_binary = (
                    workspace.artifact_root
                    / f"matrix-b{storage_block_size}.bsr"
                )
                common.write_bsr_binary(bsr_binary, bsr_data)
                conversion_us = (
                    time.perf_counter_ns() - conversion_start
                ) / 1000.0
                bsr_inputs.append((bsr_data, bsr_binary, conversion_us))

        for rhs_columns in args.rhs_columns:
            if "csr" in args.formats:
                source_text = render_mlir(
                    template,
                    args.matrix_data,
                    rhs_columns,
                    args.warmup + args.iterations,
                )
                for mapping, block_size, tile_size, chunk_size in benchmark_cases(
                    args.block_sizes,
                    args.tile_sizes,
                    args.position_chunk_sizes,
                    args.mappings,
                ):
                    case_directory = (
                        workspace.artifact_root
                        / f"rhs-{rhs_columns}"
                        / "csr"
                        / f"block-{block_size}"
                        / mapping
                    )
                    pipeline_options = ()
                    if tile_size is not None:
                        case_directory /= f"tile-{tile_size}"
                        pipeline_options = (f"spmm-tile-size={tile_size}",)
                    if chunk_size is not None:
                        case_directory /= f"chunk-{chunk_size}"
                        pipeline_options = (
                            f"spmm-position-chunk-size={chunk_size}",
                        )
                    kernels_per_dispatch, compute_binary_index = (
                        sparsewave_kernel_layout(mapping)
                    )
                    timing, compile_command, profile_command = common.run_case(
                        args,
                        source_text,
                        case_directory,
                        "spmm",
                        mapping,
                        block_size,
                        csr_binary,
                        pipeline_options=pipeline_options,
                        kernels_per_dispatch=kernels_per_dispatch,
                    )
                    resources, resource_command = common.inspect_gpu_resources(
                        args,
                        case_directory / "compiled.mlir",
                        case_directory / "kernel.hsaco",
                        "spmm_kernel",
                        args.wave_size,
                        block_size,
                        binary_index=compute_binary_index,
                    )
                    results.append(
                        result_row(
                            args,
                            mapping,
                            block_size,
                            tile_size,
                            chunk_size,
                            rhs_columns,
                            timing,
                            resources,
                            conversion_us=csr_conversion_us,
                        )
                    )
                    commands.append(
                        {
                            "rhs_columns": rhs_columns,
                            "format": "csr",
                            "block_size": block_size,
                            "mapping": mapping,
                            "tile_size": tile_size,
                            "position_chunk_size": chunk_size,
                            "compile": compile_command,
                            "resources": resource_command,
                            "profile": profile_command,
                        }
                    )

            for bsr_data, bsr_binary, conversion_us in bsr_inputs:
                source_text = render_bsr_mlir(
                    repository / "benchmark" / "bsr_spmm.mlir.in",
                    bsr_data,
                    rhs_columns,
                    args.warmup + args.iterations,
                )
                for block_size in args.block_sizes:
                    case_directory = (
                        workspace.artifact_root
                        / f"rhs-{rhs_columns}"
                        / f"bsr-{bsr_data['block_size']}"
                        / f"block-{block_size}"
                        / BASELINE_MAPPING
                    )
                    timing, compile_command, profile_command = common.run_case(
                        args,
                        source_text,
                        case_directory,
                        "spmm",
                        BASELINE_MAPPING,
                        block_size,
                        bsr_binary,
                        sparse_format="bsr",
                        kernel_name="bsr_spmm_kernel",
                    )
                    resources, resource_command = common.inspect_gpu_resources(
                        args,
                        case_directory / "compiled.mlir",
                        case_directory / "kernel.hsaco",
                        "bsr_spmm_kernel",
                        args.wave_size,
                        block_size,
                    )
                    results.append(
                        result_row(
                            args,
                            BASELINE_MAPPING,
                            block_size,
                            None,
                            None,
                            rhs_columns,
                            timing,
                            resources,
                            sparse_format="bsr",
                            bsr_data=bsr_data,
                            conversion_us=conversion_us,
                        )
                    )
                    commands.append(
                        {
                            "rhs_columns": rhs_columns,
                            "format": "bsr",
                            "storage_block_size": bsr_data["block_size"],
                            "block_size": block_size,
                            "mapping": BASELINE_MAPPING,
                            "compile": compile_command,
                            "resources": resource_command,
                            "profile": profile_command,
                        }
                    )
            if args.rocsparse:
                timing, preprocess_us, profile_command = (
                    common.run_rocsparse_case(
                        args,
                        workspace.artifact_root
                        / f"rhs-{rhs_columns}"
                        / "rocsparse"
                        / "default",
                        "spmm",
                        csr_binary,
                        rhs_columns=rhs_columns,
                    )
                )
                results.append(
                    result_row(
                        args,
                        "default",
                        None,
                        None,
                        None,
                        rhs_columns,
                        timing,
                        common.empty_gpu_resources(),
                        implementation="rocsparse",
                        preprocess_us=preprocess_us,
                    )
                )
                commands.append(
                    {
                        "rhs_columns": rhs_columns,
                        "implementation": "rocsparse",
                        "algorithm": "default",
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
