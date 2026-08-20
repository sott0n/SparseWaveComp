#!/usr/bin/env python3

import argparse
from pathlib import Path
import statistics
import subprocess
import sys
import time

import benchmark_utils as common

MAPPINGS = (
    "thread-per-row",
    "thread-per-position",
    "wave-per-position",
    "wave-per-row",
    "block-per-row",
)
FORMATS = ("csr", "coo")
FORMAT_MAPPINGS = {
    "csr": MAPPINGS,
    "coo": ("thread-per-nonzero",),
}
DISTRIBUTIONS = ("uniform", "alternating", "skewed")
POSITION_REDUCTIONS = ("atomic", "segmented")
DISTRIBUTION_PERIODS = {
    "uniform": 1,
    "alternating": 2,
    "skewed": 8,
}
RESULT_COLUMNS = (
    "chip",
    "matrix",
    "implementation",
    "format",
    "mapping",
    "position_chunk_size",
    "position_reduction",
    "algorithm",
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
    "preprocess_us",
    "conversion_us",
    "min_us",
    "median_us",
    "p95_us",
    "gnnz_per_sec",
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
    "gnnz_per_sec",
    "gflops",
    "mean_row_nnz",
)
MATRIX_REPORT = common.BenchmarkReport(
    details=("matrix", "rows", "columns", "nnz"),
    dimensions=(
        "format",
        "block_size",
        "mapping",
        "position_chunk_size",
        "position_reduction",
    ),
    metrics=("median_us", "p95_us", "gnnz_per_sec"),
    baseline=("mapping", "thread-per-row"),
    baseline_group=("block_size",),
)
MATRIX_ROCSPARSE_REPORT = common.BenchmarkReport(
    details=("matrix", "rows", "columns", "nnz"),
    dimensions=(
        "implementation",
        "format",
        "block_size",
        "mapping",
        "position_chunk_size",
        "position_reduction",
    ),
    metrics=("median_us", "p95_us", "gnnz_per_sec"),
    baseline=("implementation", "rocsparse"),
    baseline_group=(),
)
SYNTHETIC_REPORT = common.BenchmarkReport(
    details=("rows", "columns", "distributions"),
    dimensions=(
        "distribution",
        "nnz_per_row",
        "format",
        "block_size",
        "mapping",
        "position_chunk_size",
        "position_reduction",
    ),
    metrics=("median_us", "p95_us", "gnnz_per_sec"),
    baseline=("mapping", "thread-per-row"),
    baseline_group=("distribution", "nnz_per_row", "block_size"),
)
SYNTHETIC_ROCSPARSE_REPORT = common.BenchmarkReport(
    details=("rows", "columns", "distributions"),
    dimensions=(
        "distribution",
        "nnz_per_row",
        "implementation",
        "format",
        "block_size",
        "mapping",
        "position_chunk_size",
        "position_reduction",
    ),
    metrics=("median_us", "p95_us", "gnnz_per_sec"),
    baseline=("implementation", "rocsparse"),
    baseline_group=("distribution", "nnz_per_row"),
)
MATRIX_RESOURCE_REPORT = common.BenchmarkReport(
    details=(),
    dimensions=(
        "format",
        "block_size",
        "mapping",
        "position_chunk_size",
        "position_reduction",
    ),
    metrics=common.GPU_RESOURCE_METRICS,
)
SYNTHETIC_RESOURCE_REPORT = common.BenchmarkReport(
    details=(),
    dimensions=(
        "distribution",
        "nnz_per_row",
        "format",
        "block_size",
        "mapping",
        "position_chunk_size",
        "position_reduction",
    ),
    metrics=MATRIX_RESOURCE_REPORT.metrics,
)
CSR_BINARY_MAGIC = common.CSR_BINARY_MAGIC
COO_BINARY_MAGIC = common.COO_BINARY_MAGIC
positive_int = common.positive_int
nonnegative_int = common.nonnegative_int
parse_positive_int_list = common.parse_positive_int_list
read_matrix_market = common.read_matrix_market
write_csr_binary = common.write_csr_binary
write_coo_binary = common.write_coo_binary
parse_kernel_trace = common.parse_kernel_trace


def sparsewave_kernel_layout(sparse_format, mapping):
    if sparse_format == "coo" or mapping in (
        "thread-per-position",
        "wave-per-position",
    ):
        return 2, 1
    return 1, 0


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


def parse_position_reductions(value):
    values = [item.strip() for item in value.split(",") if item.strip()]
    invalid = [item for item in values if item not in POSITION_REDUCTIONS]
    if not values:
        raise argparse.ArgumentTypeError(
            "expected at least one position reduction"
        )
    if invalid:
        raise argparse.ArgumentTypeError(
            f"unknown position reductions {invalid}; "
            f"expected {POSITION_REDUCTIONS}"
        )
    if len(values) != len(set(values)):
        raise argparse.ArgumentTypeError(
            f"duplicate position reductions are not allowed: {values}"
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


def synthetic_matrix(rows, columns, nnz_per_row, distribution):
    shape = workload_shape(rows, nnz_per_row, distribution)
    row_offsets = [0]
    for row in range(rows):
        row_offsets.append(
            row_offsets[-1] + row_length(nnz_per_row, distribution, row)
        )
    return {
        **shape,
        "rows": rows,
        "columns": columns,
        "row_offsets": row_offsets,
        "column_indices": [
            position % columns for position in range(shape["nnz"])
        ],
        "values": [1.0] * shape["nnz"],
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


def synthetic_initialization_mlir(nnz_per_row, distribution, sparse_format):
    row_length_code = indent_mlir(
        row_length_mlir(nnz_per_row, distribution), 4
    )
    expected = f"""scf.for %row = %c0 to %cRows step %c1 {{
    {row_length_code}
    %rowLengthI64 = arith.index_cast %rowLength : index to i64
    %expected = arith.uitofp %rowLengthI64 : i64 to f32
    memref.store %expected, %hostExpected[%row] : memref<?xf32>
  }}"""
    if sparse_format == "csr":
        sparse_indices = f"""%c0I32 = arith.constant 0 : i32
  memref.store %c0I32, %hostFirstIndices[%c0] : memref<?xi32>
  scf.for %row = %c0 to %cRows step %c1
      iter_args(%offset = %c0) -> index {{
    {row_length_code}
    %nextOffset = arith.addi %offset, %rowLength : index
    %nextOffsetI32 = arith.index_cast %nextOffset : index to i32
    %nextRow = arith.addi %row, %c1 : index
    memref.store %nextOffsetI32, %hostFirstIndices[%nextRow] : memref<?xi32>
    scf.yield %nextOffset : index
  }}"""
    elif sparse_format == "coo":
        sparse_indices = f"""scf.for %row = %c0 to %cRows step %c1
      iter_args(%offset = %c0) -> index {{
    {row_length_code}
    %rowI32 = arith.index_cast %row : index to i32
    %nextOffset = arith.addi %offset, %rowLength : index
    scf.for %position = %offset to %nextOffset step %c1 {{
      memref.store %rowI32, %hostFirstIndices[%position] : memref<?xi32>
    }}
    scf.yield %nextOffset : index
  }}"""
    else:
        raise ValueError(f"unknown sparse format: {sparse_format}")
    return f"""{sparse_indices}
  {expected}
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
    sparse_format="csr",
):
    if matrix is None:
        shape = workload_shape(rows, nnz_per_row, distribution)
        initialization = synthetic_initialization_mlir(
            nnz_per_row, distribution, sparse_format
        )
        declarations = ""
    else:
        shape = matrix
        loader = (
            "loadCSRSpMVBenchmarkInputs"
            if sparse_format == "csr"
            else "loadCOOSpMVBenchmarkInputs"
        )
        initialization = (
            f"func.call @{loader}("
            "%hostFirstIndices, %hostColumnIndices, %hostValues, "
            "%hostVector, %hostExpected) : "
            "(memref<?xi32>, memref<?xi32>, memref<?xf32>, "
            "memref<?xf32>, memref<?xf32>) -> ()"
        )
        declarations = (
            f"func.func private @{loader}("
            "memref<?xi32>, memref<?xi32>, memref<?xf32>, "
            "memref<?xf32>, memref<?xf32>) "
            "attributes {llvm.emit_c_interface}"
        )
    replacements = {
        "@ROWS@": str(rows),
        "@FIRST_INDEX_COUNT@": str(
            rows + 1 if sparse_format == "csr" else shape["nnz"]
        ),
        "@COLUMNS@": str(columns),
        "@NNZ@": str(shape["nnz"]),
        "@DISPATCHES@": str(dispatches),
        "@INITIALIZE_INPUTS@": indent_mlir(initialization, 2),
        "@SPMV_OPERATION@": (
            "sparsewave.spmv"
            if sparse_format == "csr"
            else "sparsewave.coo_spmv"
        ),
        "@EXTERNAL_DECLARATIONS@": declarations,
    }
    return common.render_template(template_path, replacements)


def result_row(
    args,
    mapping,
    block_size,
    workload,
    timing,
    resources,
    sparse_format="csr",
    implementation="sparsewave",
    preprocess_us=None,
    conversion_us=None,
    position_chunk_size=None,
    position_reduction=None,
):
    shape = workload["shape"]
    nnz = shape["nnz"]
    median_seconds = timing["median_us"] / 1_000_000.0
    return {
        "chip": args.chip,
        "matrix": workload["matrix"],
        "implementation": implementation,
        "format": sparse_format,
        "mapping": mapping,
        "position_chunk_size": position_chunk_size,
        "position_reduction": position_reduction,
        "algorithm": "default" if implementation == "rocsparse" else "",
        "block_size": block_size,
        "wave_size": (
            args.wave_size if implementation == "sparsewave" else None
        ),
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
        "preprocess_us": preprocess_us,
        "conversion_us": conversion_us,
        "min_us": timing["min_us"],
        "median_us": timing["median_us"],
        "p95_us": timing["p95_us"],
        "gnnz_per_sec": nnz / median_seconds / 1_000_000_000.0,
        "gflops": 2.0 * nnz / median_seconds / 1_000_000_000.0,
        **resources,
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
            "formats": args.formats,
            "position_chunk_sizes": args.position_chunk_sizes,
            "position_reductions": args.position_reductions,
            "commands": commands,
        }
    )
    return metadata


def print_results(args, results):
    if args.matrix_data is not None:
        report = MATRIX_ROCSPARSE_REPORT if args.rocsparse else MATRIX_REPORT
    else:
        report = (
            SYNTHETIC_ROCSPARSE_REPORT
            if args.rocsparse
            else SYNTHETIC_REPORT
        )
    common.print_benchmark_report(
        args,
        "SparseWave SpMV benchmark",
        report,
        results,
    )
    print()
    common.print_gpu_resource_report(
        args,
        "SparseWave SpMV GPU resources",
        (
            MATRIX_RESOURCE_REPORT
            if args.matrix_data is not None
            else SYNTHETIC_RESOURCE_REPORT
        ),
        results,
    )


def validate_paths(args):
    common.validate_required_paths(
        args,
        needs_benchmark_utils=args.matrix_data is not None,
        needs_resource_inspector=True,
    )
    if args.matrix_data is None:
        validate_distribution_rows(args.rows, args.distributions)
    if args.wave_size != 32:
        raise ValueError(
            "wave-per-position, wave-per-row, and block-per-row benchmarking "
            "currently require Wave32"
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
    if "coo" in args.formats and args.rows > maximum_i32:
        raise ValueError("row count must fit in the i32 COO row-index type")
    if args.matrix_data is not None:
        invalid_nnz = args.matrix_data["nnz"] > maximum_i32
    else:
        invalid_nnz = any(
            workload_shape(args.rows, value, distribution)["nnz"] > maximum_i32
            for value in args.nnz_per_row
            for distribution in args.distributions
        )
    if invalid_nnz:
        raise ValueError("NNZ must fit in the i32 sparse index type")


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
    parser.add_argument(
        "--formats",
        type=parse_formats,
        default=parse_formats("csr"),
        help="Comma-separated SparseWave storage formats: csr,coo.",
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
    parser.add_argument(
        "--position-chunk-sizes",
        type=parse_positive_int_list,
        default=parse_positive_int_list("1"),
        help=(
            "Comma-separated TACO-style split factors for the CSR "
            "thread-per-position mapping."
        ),
    )
    parser.add_argument(
        "--position-reductions",
        type=parse_position_reductions,
        default=parse_position_reductions("atomic"),
        help=(
            "Comma-separated reduction strategies for each CSR "
            "thread-per-position chunk: atomic,segmented."
        ),
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
        sparse_binaries = {}
        conversion_us = {}
        if args.matrix_data is not None:
            csr_binary = workspace.artifact_root / "matrix.csr"
            coo_binary = workspace.artifact_root / "matrix.coo"
            for sparse_format, binary, writer in (
                ("csr", csr_binary, write_csr_binary),
                ("coo", coo_binary, write_coo_binary),
            ):
                if sparse_format not in args.formats and not (
                    sparse_format == "csr" and args.rocsparse
                ):
                    continue
                start = time.perf_counter_ns()
                writer(binary, args.matrix_data)
                conversion_us[sparse_format] = (
                    time.perf_counter_ns() - start
                ) / 1000.0
                sparse_binaries[sparse_format] = binary

        for workload in create_workloads(args):
            shape = workload["shape"]
            workload_csr = sparse_binaries.get("csr")
            if args.rocsparse and workload_csr is None:
                workload_csr = (
                    workspace.artifact_root / workload["key"] / "matrix.csr"
                )
                workload_csr.parent.mkdir(parents=True)
                write_csr_binary(
                    workload_csr,
                    synthetic_matrix(
                        shape["rows"],
                        shape["columns"],
                        workload["nnz_per_row"],
                        workload["distribution"],
                    ),
                )
            for sparse_format in args.formats:
                source_text = render_mlir(
                    template,
                    shape["rows"],
                    shape["columns"],
                    workload["nnz_per_row"],
                    args.warmup + args.iterations,
                    workload["distribution"],
                    args.matrix_data,
                    sparse_format,
                )
                for block_size in args.block_sizes:
                    for mapping in FORMAT_MAPPINGS[sparse_format]:
                        chunk_sizes = (
                            args.position_chunk_sizes
                            if sparse_format == "csr"
                            and mapping == "thread-per-position"
                            else (None,)
                        )
                        for chunk_size in chunk_sizes:
                            reductions = (
                                args.position_reductions
                                if chunk_size is not None
                                else (None,)
                            )
                            for position_reduction in reductions:
                                kernels_per_dispatch, compute_binary_index = (
                                    sparsewave_kernel_layout(
                                        sparse_format, mapping
                                    )
                                )
                                case_directory = (
                                    workspace.artifact_root
                                    / workload["key"]
                                    / sparse_format
                                    / f"block-{block_size}"
                                    / mapping
                                )
                                pipeline_options = ()
                                if chunk_size is not None:
                                    case_directory /= f"chunk-{chunk_size}"
                                    case_directory /= position_reduction
                                    pipeline_options = (
                                        "spmv-position-chunk-size="
                                        f"{chunk_size}",
                                        "spmv-position-reduction="
                                        f"{position_reduction}",
                                    )
                                timing, compile_command, profile_command = (
                                    common.run_case(
                                        args,
                                        source_text,
                                        case_directory,
                                        "spmv",
                                        (
                                            mapping
                                            if sparse_format == "csr"
                                            else "thread-per-row"
                                        ),
                                        block_size,
                                        sparse_binaries.get(sparse_format),
                                        pipeline_options=pipeline_options,
                                        sparse_format=sparse_format,
                                        kernels_per_dispatch=(
                                            kernels_per_dispatch
                                        ),
                                    )
                                )
                                resources, resource_command = (
                                    common.inspect_gpu_resources(
                                        args,
                                        case_directory / "compiled.mlir",
                                        case_directory / "kernel.hsaco",
                                        "spmv_kernel",
                                        args.wave_size,
                                        block_size,
                                        binary_index=compute_binary_index,
                                    )
                                )
                                results.append(
                                    result_row(
                                        args,
                                        mapping,
                                        block_size,
                                        workload,
                                        timing,
                                        resources,
                                        sparse_format=sparse_format,
                                        conversion_us=conversion_us.get(
                                            sparse_format
                                        ),
                                        position_chunk_size=chunk_size,
                                        position_reduction=(
                                            position_reduction
                                        ),
                                    )
                                )
                                commands.append(
                                    {
                                        "matrix": workload["matrix"],
                                        "distribution": workload[
                                            "distribution"
                                        ],
                                        "nnz_per_row": workload[
                                            "nnz_per_row"
                                        ],
                                        "format": sparse_format,
                                        "block_size": block_size,
                                        "mapping": mapping,
                                        "position_chunk_size": chunk_size,
                                        "position_reduction": (
                                            position_reduction
                                        ),
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
                        / workload["key"]
                        / "rocsparse"
                        / "default",
                        "spmv",
                        workload_csr,
                    )
                )
                results.append(
                    result_row(
                        args,
                        "default",
                        None,
                        workload,
                        timing,
                        common.empty_gpu_resources(),
                        sparse_format="csr",
                        implementation="rocsparse",
                        preprocess_us=preprocess_us,
                        conversion_us=conversion_us.get("csr"),
                    )
                )
                commands.append(
                    {
                        "matrix": workload["matrix"],
                        "distribution": workload["distribution"],
                        "nnz_per_row": workload["nnz_per_row"],
                        "implementation": "rocsparse",
                        "format": "csr",
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
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
