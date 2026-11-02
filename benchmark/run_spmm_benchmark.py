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
WAVE_POSITION_MAPPING = "wave-per-position"
COOPERATIVE_POSITION_MAPPING = "wave-per-position-tile"
MAPPINGS = (
    BASELINE_MAPPING,
    TILED_MAPPING,
    POSITION_MAPPING,
    WAVE_POSITION_MAPPING,
    COOPERATIVE_POSITION_MAPPING,
)
POSITION_ORDERS = ("position-major", "rhs-major")
THREAD_POSITION_REDUCTIONS = common.POSITION_THREAD_REDUCTIONS
WAVE_POSITION_REDUCTIONS = common.POSITION_WAVE_REDUCTIONS
POSITION_REDUCTIONS = common.POSITION_REDUCTIONS
FORMATS = ("csr", "bsr")
DISTRIBUTIONS = common.ROW_DISTRIBUTIONS
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
    "position_order",
    "position_reduction",
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
    "nnz_per_row",
    "distribution",
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
        "position_order",
        "position_reduction",
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
        "position_order",
        "position_reduction",
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
        "position_order",
        "position_reduction",
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
SYNTHETIC_REPORT = common.BenchmarkReport(
    details=("rows", "columns", "distributions", "rhs_columns"),
    dimensions=(
        "distribution",
        "nnz_per_row",
        "rhs_cols",
        "format",
        "storage_block_size",
        "block_size",
        "tile_size",
        "position_chunk_size",
        "position_order",
        "position_reduction",
        "mapping",
    ),
    metrics=("median_us", "p95_us", "gproducts_per_sec", "gflops"),
    baseline=("mapping", BASELINE_MAPPING),
    baseline_group=(
        "distribution",
        "nnz_per_row",
        "rhs_cols",
        "format",
        "storage_block_size",
        "block_size",
    ),
)
SYNTHETIC_ROCSPARSE_REPORT = common.BenchmarkReport(
    details=("rows", "columns", "distributions", "rhs_columns"),
    dimensions=(
        "distribution",
        "nnz_per_row",
        "rhs_cols",
        "implementation",
        "format",
        "storage_block_size",
        "block_size",
        "tile_size",
        "position_chunk_size",
        "position_order",
        "position_reduction",
        "mapping",
    ),
    metrics=("median_us", "p95_us", "gproducts_per_sec", "gflops"),
    baseline=("implementation", "rocsparse"),
    baseline_group=("distribution", "nnz_per_row", "rhs_cols"),
)
SYNTHETIC_RESOURCE_REPORT = common.BenchmarkReport(
    details=(),
    dimensions=(
        "distribution",
        "nnz_per_row",
        *RESOURCE_REPORT.dimensions,
    ),
    metrics=common.GPU_RESOURCE_METRICS,
)
SYNTHETIC_FORMAT_REPORT = common.BenchmarkReport(
    details=(),
    dimensions=(
        "distribution",
        "nnz_per_row",
        *FORMAT_REPORT.dimensions,
    ),
    metrics=FORMAT_REPORT.metrics,
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


def parse_position_orders(value):
    values = [item.strip() for item in value.split(",") if item.strip()]
    invalid = [item for item in values if item not in POSITION_ORDERS]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one position order")
    if invalid:
        raise argparse.ArgumentTypeError(
            f"unknown position orders {invalid}; expected {POSITION_ORDERS}"
        )
    if len(values) != len(set(values)):
        raise argparse.ArgumentTypeError(
            f"duplicate position orders are not allowed: {values}"
        )
    return values


def parse_position_reductions(value):
    return common.parse_position_reductions(value)


def parse_distributions(value):
    return common.parse_row_distributions(value)


def create_workloads(args):
    if args.matrix_data is not None:
        yield {
            "key": args.matrix_data["name"],
            "matrix": str(args.matrix_data["path"]),
            "distribution": "matrix-market",
            "nnz_per_row": None,
            "shape": args.matrix_data,
        }
        return

    for distribution in args.distributions:
        for nnz_per_row in args.nnz_per_row:
            yield {
                "key": f"{distribution}/nnz-{nnz_per_row}",
                "matrix": "",
                "distribution": distribution,
                "nnz_per_row": nnz_per_row,
                "shape": common.synthetic_csr_matrix(
                    args.rows,
                    args.columns,
                    nnz_per_row,
                    distribution,
                ),
            }


def benchmark_cases(
    block_sizes,
    tile_sizes,
    position_chunk_sizes,
    position_orders,
    position_reductions,
    mappings,
):
    for block_size in block_sizes:
        if BASELINE_MAPPING in mappings:
            yield BASELINE_MAPPING, block_size, None, None, None, None
        if TILED_MAPPING in mappings:
            for tile_size in tile_sizes:
                yield TILED_MAPPING, block_size, tile_size, None, None, None
        if WAVE_POSITION_MAPPING in mappings:
            for order in position_orders:
                for reduction in common.position_reductions_for_mapping(
                    WAVE_POSITION_MAPPING, position_reductions
                ):
                    yield (
                        WAVE_POSITION_MAPPING,
                        block_size,
                        None,
                        None,
                        order,
                        reduction,
                    )
        if COOPERATIVE_POSITION_MAPPING in mappings:
            for chunk_size in position_chunk_sizes:
                yield (
                    COOPERATIVE_POSITION_MAPPING,
                    block_size,
                    None,
                    chunk_size,
                    None,
                    None,
                )
        if POSITION_MAPPING in mappings:
            for chunk_size in position_chunk_sizes:
                for order in position_orders:
                    for reduction in common.position_reductions_for_mapping(
                        POSITION_MAPPING, position_reductions
                    ):
                        yield (
                            POSITION_MAPPING,
                            block_size,
                            None,
                            chunk_size,
                            order,
                            reduction,
                        )


def sparsewave_kernel_layout(mapping, position_reduction=None):
    position_layout = common.position_kernel_layout(
        mapping, position_reduction
    )
    if position_layout is not None:
        return position_layout
    if mapping == COOPERATIVE_POSITION_MAPPING:
        return 2, 1
    return 1, 0


def result_row(
    args,
    workload,
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
    position_order=None,
    position_reduction=None,
):
    matrix = workload["shape"]
    products = matrix["nnz"] * rhs_columns
    median_seconds = timing["median_us"] / 1_000_000.0
    return {
        "chip": args.chip,
        "matrix": workload["matrix"],
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
        "position_order": position_order,
        "position_reduction": position_reduction,
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
        "nnz_per_row": workload["nnz_per_row"],
        "distribution": workload["distribution"],
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
            "position_orders": args.position_orders,
            "position_reductions": args.position_reductions,
            "mappings": args.mappings,
            "formats": args.formats,
            "bsr_block_sizes": args.bsr_block_sizes,
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
            "rows": args.rows,
            "input_columns": args.columns,
            "nnz_per_row": (
                args.nnz_per_row if args.matrix_data is None else None
            ),
            "distributions": (
                args.distributions if args.matrix_data is None else None
            ),
            "commands": commands,
        }
    )
    return metadata


def print_results(args, results):
    if args.matrix_data is not None:
        report = ROCSPARSE_REPORT if args.rocsparse else REPORT
        resource_report = RESOURCE_REPORT
        format_report = FORMAT_REPORT
    else:
        if args.rocsparse:
            report = SYNTHETIC_ROCSPARSE_REPORT
        elif BASELINE_MAPPING in args.mappings:
            report = SYNTHETIC_REPORT
        else:
            report = common.BenchmarkReport(
                details=SYNTHETIC_REPORT.details,
                dimensions=SYNTHETIC_REPORT.dimensions,
                metrics=SYNTHETIC_REPORT.metrics,
            )
        resource_report = SYNTHETIC_RESOURCE_REPORT
        format_report = SYNTHETIC_FORMAT_REPORT
    common.print_benchmark_report(
        args,
        "SparseWave SpMM benchmark",
        report,
        results,
    )
    print()
    common.print_gpu_resource_report(
        args,
        "SparseWave SpMM GPU resources",
        resource_report,
        results,
    )
    if "bsr" in args.formats:
        print()
        common.print_benchmark_report(
            args,
            "SparseWave SpMM storage formats",
            format_report,
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
    uses_cooperative_mapping = (
        "csr" in args.formats
        and COOPERATIVE_POSITION_MAPPING in args.mappings
    )
    uses_wave_position_mapping = (
        "csr" in args.formats and WAVE_POSITION_MAPPING in args.mappings
    )
    common.validate_block_sizes(
        args,
        require_wave_multiple=(
            uses_tiled_mapping
            or uses_wave_position_mapping
            or uses_cooperative_mapping
        ),
    )
    if (uses_tiled_mapping or uses_wave_position_mapping) and (
        args.wave_size != 32
    ):
        raise ValueError(
            "wave-per-row-tile and wave-per-position currently require "
            "wave size 32"
        )
    if uses_tiled_mapping and any(
        tile_size > 32 for tile_size in args.tile_sizes
    ):
        raise ValueError("tile sizes must not exceed 32")
    if POSITION_MAPPING in args.mappings and not any(
        reduction in THREAD_POSITION_REDUCTIONS
        for reduction in args.position_reductions
    ):
        raise ValueError(
            "thread-per-position requires an atomic or segmented position "
            "reduction"
        )
    if WAVE_POSITION_MAPPING in args.mappings and not any(
        reduction in WAVE_POSITION_REDUCTIONS
        for reduction in args.position_reductions
    ):
        raise ValueError(
            "wave-per-position requires an atomic or workspace position "
            "reduction"
        )
    if args.matrix_data is None:
        common.validate_distribution_rows(args.rows, args.distributions)
    maximum_i32 = (1 << 31) - 1
    columns = (
        args.matrix_data["columns"]
        if args.matrix_data is not None
        else args.columns
    )
    if columns > maximum_i32:
        raise ValueError(
            "matrix column count must fit in the i32 column-index type"
        )
    if args.matrix_data is not None:
        rows = args.matrix_data["rows"]
        invalid_nnz = args.matrix_data["nnz"] > maximum_i32
    else:
        rows = args.rows
        invalid_nnz = any(
            common.synthetic_workload_shape(
                args.rows, nnz_per_row, distribution
            )["nnz"]
            > maximum_i32
            for nnz_per_row in args.nnz_per_row
            for distribution in args.distributions
        )
    if rows > maximum_i32 or invalid_nnz:
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
        help=(
            "Matrix Market coordinate matrix used instead of generated "
            "row-length distributions."
        ),
    )
    parser.add_argument("--rows", type=common.positive_int, default=65536)
    parser.add_argument("--columns", type=common.positive_int, default=65536)
    parser.add_argument(
        "--nnz-per-row",
        type=common.parse_positive_int_list,
        default=common.parse_positive_int_list("4,32,256"),
        help="Comma-separated mean NNZ/row values for generated matrices.",
    )
    parser.add_argument(
        "--distributions",
        type=parse_distributions,
        default=parse_distributions("uniform,alternating,skewed"),
        help=(
            "Comma-separated generated row-length distributions: "
            "uniform,alternating,skewed."
        ),
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
            "wave-per-row-tile,thread-per-position,wave-per-position,"
            "wave-per-position-tile."
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
            "Comma-separated consecutive position-space points processed by "
            "each thread-per-position or cooperative wave worker."
        ),
    )
    parser.add_argument(
        "--position-orders",
        type=parse_position_orders,
        default=parse_position_orders("position-major,rhs-major"),
        help=(
            "Comma-separated collapsed iteration orders for position-space "
            "mappings: position-major,rhs-major."
        ),
    )
    parser.add_argument(
        "--position-reductions",
        type=parse_position_reductions,
        default=parse_position_reductions("atomic,segmented,workspace"),
        help=(
            "Comma-separated position reduction strategies. "
            "Thread mapping supports atomic,segmented; wave mapping supports "
            "atomic,workspace."
        ),
    )
    common.add_common_arguments(parser, repository)
    args = parser.parse_args(argv)
    common.configure_common_arguments(args)
    if args.matrix is not None:
        args.matrix = args.matrix.resolve()
        args.matrix_data = common.read_matrix_market(args.matrix)
        args.rows = args.matrix_data["rows"]
        args.columns = args.matrix_data["columns"]
    else:
        args.matrix_data = None
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
        for workload in create_workloads(args):
            matrix = workload["shape"]
            workload_directory = workspace.artifact_root / workload["key"]
            workload_directory.mkdir(parents=True, exist_ok=True)
            csr_binary = workload_directory / "matrix.csr"
            conversion_start = time.perf_counter_ns()
            common.write_csr_binary(csr_binary, matrix)
            csr_conversion_us = (
                time.perf_counter_ns() - conversion_start
            ) / 1000.0
            bsr_inputs = []
            if "bsr" in args.formats:
                for storage_block_size in args.bsr_block_sizes:
                    conversion_start = time.perf_counter_ns()
                    bsr_data = common.convert_to_bsr(
                        matrix, storage_block_size
                    )
                    bsr_binary = (
                        workload_directory
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
                        matrix,
                        rhs_columns,
                        args.warmup + args.iterations,
                    )
                    for (
                        mapping,
                        block_size,
                        tile_size,
                        chunk_size,
                        position_order,
                        position_reduction,
                    ) in benchmark_cases(
                        args.block_sizes,
                        args.tile_sizes,
                        args.position_chunk_sizes,
                        args.position_orders,
                        args.position_reductions,
                        args.mappings,
                    ):
                        case_directory = (
                            workload_directory
                            / f"rhs-{rhs_columns}"
                            / "csr"
                            / f"block-{block_size}"
                            / mapping
                        )
                        pipeline_options = ()
                        if tile_size is not None:
                            case_directory /= f"tile-{tile_size}"
                            pipeline_options = (
                                f"spmm-tile-size={tile_size}",
                            )
                        if chunk_size is not None:
                            case_directory /= f"chunk-{chunk_size}"
                            pipeline_options = (
                                f"spmm-position-chunk-size={chunk_size}",
                            )
                        if position_order is not None:
                            case_directory /= f"order-{position_order}"
                            pipeline_options += (
                                f"spmm-position-order={position_order}",
                            )
                        if position_reduction is not None:
                            case_directory /= (
                                f"reduction-{position_reduction}"
                            )
                            pipeline_options += (
                                common.position_reduction_pipeline_option(
                                    mapping, position_reduction
                                ),
                            )
                        kernels_per_dispatch, compute_binary_index = (
                            sparsewave_kernel_layout(
                                mapping, position_reduction
                            )
                        )
                        timing, compile_command, profile_command = (
                            common.run_case(
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
                        )
                        resources, resource_command = (
                            common.inspect_gpu_resources(
                                args,
                                case_directory / "compiled.mlir",
                                case_directory / "kernel.hsaco",
                                "spmm_kernel",
                                args.wave_size,
                                block_size,
                                binary_index=compute_binary_index,
                            )
                        )
                        results.append(
                            result_row(
                                args,
                                workload,
                                mapping,
                                block_size,
                                tile_size,
                                chunk_size,
                                rhs_columns,
                                timing,
                                resources,
                                conversion_us=csr_conversion_us,
                                position_order=position_order,
                                position_reduction=position_reduction,
                            )
                        )
                        commands.append(
                            {
                                "workload": workload["key"],
                                "rhs_columns": rhs_columns,
                                "format": "csr",
                                "block_size": block_size,
                                "mapping": mapping,
                                "tile_size": tile_size,
                                "position_chunk_size": chunk_size,
                                "position_order": position_order,
                                "position_reduction": position_reduction,
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
                            workload_directory
                            / f"rhs-{rhs_columns}"
                            / f"bsr-{bsr_data['block_size']}"
                            / f"block-{block_size}"
                            / BASELINE_MAPPING
                        )
                        timing, compile_command, profile_command = (
                            common.run_case(
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
                        )
                        resources, resource_command = (
                            common.inspect_gpu_resources(
                                args,
                                case_directory / "compiled.mlir",
                                case_directory / "kernel.hsaco",
                                "bsr_spmm_kernel",
                                args.wave_size,
                                block_size,
                            )
                        )
                        results.append(
                            result_row(
                                args,
                                workload,
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
                                "workload": workload["key"],
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
                            workload_directory
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
                            workload,
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
                            "workload": workload["key"],
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
