#!/usr/bin/env python3

import argparse
import csv
import datetime
import json
import math
import os
from pathlib import Path
import statistics
import struct
import subprocess
import sys
import tempfile


MAPPINGS = ("thread-per-row", "wave-per-row")
DISTRIBUTIONS = ("uniform", "alternating", "skewed")
DISTRIBUTION_PERIODS = {
    "uniform": 1,
    "alternating": 2,
    "skewed": 8,
}
TRACE_COLUMNS = {
    "kernel": ("Kernel_Name", "Kernel Name", "Name"),
    "start": ("Start_Timestamp", "Start Timestamp", "Start"),
    "end": ("End_Timestamp", "End Timestamp", "End"),
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
CSR_BINARY_MAGIC = b"SWCSR001"


def positive_int(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer: {value}")
    return parsed


def nonnegative_int(value):
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError(
            f"expected a nonnegative integer: {value}"
        )
    return parsed


def parse_positive_int_list(value):
    try:
        values = [positive_int(item.strip()) for item in value.split(",")]
    except (argparse.ArgumentTypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    if not values:
        raise argparse.ArgumentTypeError("expected at least one NNZ/row value")
    if len(values) != len(set(values)):
        raise argparse.ArgumentTypeError(
            f"duplicate values are not allowed: {values}"
        )
    return values


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


def read_matrix_market(path):
    with path.open(encoding="utf-8") as stream:
        header = stream.readline().strip().split()
        if len(header) != 5 or header[:3] != [
            "%%MatrixMarket",
            "matrix",
            "coordinate",
        ]:
            raise ValueError(
                f"{path}: expected a Matrix Market coordinate matrix"
            )
        field = header[3].lower()
        symmetry = header[4].lower()
        if field not in ("real", "integer", "pattern"):
            raise ValueError(
                f"{path}: unsupported Matrix Market field '{field}'"
            )
        if symmetry not in ("general", "symmetric"):
            raise ValueError(
                f"{path}: unsupported Matrix Market symmetry '{symmetry}'"
            )

        size_line = next(
            (
                line
                for line in stream
                if line.strip() and not line.lstrip().startswith("%")
            ),
            None,
        )
        if size_line is None:
            raise ValueError(f"{path}: missing matrix dimensions")
        try:
            rows, columns, stored_nnz = map(int, size_line.split())
        except ValueError as error:
            raise ValueError(f"{path}: invalid matrix dimensions") from error
        if rows <= 0 or columns <= 0 or stored_nnz < 0:
            raise ValueError(f"{path}: matrix dimensions must be nonnegative")
        if symmetry == "symmetric" and rows != columns:
            raise ValueError(f"{path}: a symmetric matrix must be square")

        entries = []
        read_entries = 0
        for line_number, line in enumerate(stream, start=3):
            stripped = line.strip()
            if not stripped or stripped.startswith("%"):
                continue
            parts = stripped.split()
            expected_fields = 2 if field == "pattern" else 3
            if len(parts) != expected_fields:
                raise ValueError(
                    f"{path}:{line_number}: expected {expected_fields} fields"
                )
            try:
                row = int(parts[0]) - 1
                column = int(parts[1]) - 1
                if field == "pattern":
                    value = 1.0
                elif field == "integer":
                    value = float(int(parts[2]))
                else:
                    value = float(parts[2])
            except ValueError as error:
                raise ValueError(
                    f"{path}:{line_number}: invalid matrix entry"
                ) from error
            if not (0 <= row < rows and 0 <= column < columns):
                raise ValueError(
                    f"{path}:{line_number}: matrix index is out of bounds"
                )
            if not math.isfinite(value):
                raise ValueError(
                    f"{path}:{line_number}: matrix value must be finite"
                )
            try:
                struct.pack("<f", value)
            except (OverflowError, struct.error) as error:
                raise ValueError(
                    f"{path}:{line_number}: matrix value must fit in f32"
                ) from error
            entries.append((row, column, value))
            read_entries += 1
            if symmetry == "symmetric" and row != column:
                entries.append((column, row, value))

    if read_entries != stored_nnz:
        raise ValueError(
            f"{path}: header declares {stored_nnz} entries, "
            f"but the file contains {read_entries}"
        )

    entries.sort(key=lambda entry: (entry[0], entry[1]))
    row_offsets = [0] * (rows + 1)
    column_indices = []
    values = []
    for row, column, value in entries:
        row_offsets[row + 1] += 1
        column_indices.append(column)
        values.append(value)
    for row in range(rows):
        row_offsets[row + 1] += row_offsets[row]
    row_lengths = [
        row_offsets[row + 1] - row_offsets[row] for row in range(rows)
    ]
    return {
        "path": path,
        "name": path.stem,
        "field": field,
        "symmetry": symmetry,
        "rows": rows,
        "columns": columns,
        "nnz": len(entries),
        "min_row_nnz": min(row_lengths),
        "max_row_nnz": max(row_lengths),
        "mean_row_nnz": statistics.mean(row_lengths),
        "row_offsets": row_offsets,
        "column_indices": column_indices,
        "values": values,
    }


def write_csr_binary(path, matrix):
    maximum_i32 = (1 << 31) - 1
    if matrix["rows"] > maximum_i32 or matrix["nnz"] > maximum_i32:
        raise ValueError("matrix rows and NNZ must fit in the i32 CSR type")
    with path.open("wb") as stream:
        stream.write(CSR_BINARY_MAGIC)
        stream.write(
            struct.pack(
                "<QQQ",
                matrix["rows"],
                matrix["columns"],
                matrix["nnz"],
            )
        )
        stream.write(
            struct.pack(
                f"<{len(matrix['row_offsets'])}i", *matrix["row_offsets"]
            )
        )
        stream.write(
            struct.pack(
                f"<{len(matrix['column_indices'])}i",
                *matrix["column_indices"],
            )
        )
        stream.write(
            struct.pack(f"<{len(matrix['values'])}f", *matrix["values"])
        )


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
    rendered = template_path.read_text(encoding="utf-8")
    for token, value in replacements.items():
        rendered = rendered.replace(token, value)
    unresolved = [token for token in replacements if token in rendered]
    if unresolved:
        raise ValueError(f"unresolved MLIR template tokens: {unresolved}")
    return rendered


def find_column(fieldnames, candidates):
    for candidate in candidates:
        if candidate in fieldnames:
            return candidate
    raise ValueError(
        f"missing one of the required trace columns {candidates}; "
        f"found {fieldnames}"
    )


def parse_kernel_trace(path, kernel_name, warmup, iterations):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise ValueError(f"trace has no CSV header: {path}")
        kernel_column = find_column(reader.fieldnames, TRACE_COLUMNS["kernel"])
        start_column = find_column(reader.fieldnames, TRACE_COLUMNS["start"])
        end_column = find_column(reader.fieldnames, TRACE_COLUMNS["end"])
        durations = []
        for row in reader:
            if kernel_name not in row[kernel_column]:
                continue
            start = int(row[start_column])
            end = int(row[end_column])
            if end < start:
                raise ValueError(f"trace contains a negative duration: {row}")
            durations.append((end - start) / 1000.0)

    expected = warmup + iterations
    if len(durations) != expected:
        raise ValueError(
            f"expected {expected} '{kernel_name}' dispatches, "
            f"found {len(durations)} in {path}"
        )
    measured = sorted(durations[warmup:])
    p95_index = max(0, math.ceil(0.95 * len(measured)) - 1)
    return {
        "min_us": measured[0],
        "median_us": statistics.median(measured),
        "p95_us": measured[p95_index],
    }


def discover_trace(directory):
    candidates = sorted(directory.rglob("*kernel_trace*.csv"))
    if len(candidates) != 1:
        raise ValueError(
            f"expected one kernel trace CSV in {directory}, "
            f"found {len(candidates)}: {candidates}"
        )
    return candidates[0]


def discover_gpu_name(directory, chip):
    candidates = sorted(directory.rglob("*agent_info*.csv"))
    for path in candidates:
        with path.open(newline="", encoding="utf-8-sig") as stream:
            for row in csv.DictReader(stream):
                if row.get("Name") == chip:
                    return row.get("Product_Name") or chip
    return chip


def run_command(command, **kwargs):
    return subprocess.run(command, check=True, text=True, **kwargs)


def detect_chip(rocm_path):
    tools = (
        rocm_path / "llvm" / "bin" / "amdgpu-arch",
        rocm_path / "bin" / "amdgpu-arch",
        rocm_path / "bin" / "rocm_agent_enumerator",
    )
    for tool in tools:
        if not tool.is_file():
            continue
        output = run_command(
            [str(tool)], capture_output=True
        ).stdout.splitlines()
        chips = [line.strip() for line in output if line.strip().startswith("gfx")]
        if chips:
            return chips[0]
    raise RuntimeError("could not detect an AMD GPU target chip")


def git_output(repository, *arguments):
    try:
        return run_command(
            ["git", "-C", str(repository), *arguments],
            capture_output=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def rocm_version(rocm_path):
    for relative_path in (".info/version", ".info/version-dev"):
        path = rocm_path / relative_path
        if path.is_file():
            return path.read_text(encoding="utf-8").strip()
    return "unknown"


def compile_mlir(args, source, output, mapping, block_size):
    pipeline = (
        "builtin.module(convert-scf-to-cf,"
        "sparsewave-to-amdgpu-pipeline{"
        f"chip={args.chip} "
        f"wavefront-size={args.wave_size} "
        f"rocm-path={args.rocm_path} "
        f"spmv-mapping={mapping} "
        f"spmv-block-size={block_size}"
        "},reconcile-unrealized-casts)"
    )
    command = [
        str(args.sparsewave_opt),
        str(source),
        f"--pass-pipeline={pipeline}",
        "-o",
        str(output),
    ]
    run_command(command)
    return command


def profile_mlir(args, compiled, trace_directory, csr_binary=None):
    command = [
        str(args.rocprofv3),
        "--kernel-trace",
        "--output-format",
        "csv",
        "--output-directory",
        str(trace_directory),
        "--",
        str(args.mlir_runner),
        str(compiled),
        f"--shared-libs={args.rocm_runtime}",
        f"--shared-libs={args.runner_utils}",
    ]
    if csr_binary is not None:
        command.append(f"--shared-libs={args.benchmark_utils}")
    command.append("--entry-point-result=void")
    environment = os.environ.copy()
    if csr_binary is not None:
        environment["SPARSEWAVE_BENCHMARK_CSR"] = str(csr_binary)
    completed = subprocess.run(
        command, capture_output=True, text=True, env=environment
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"profiler command failed with exit code {completed.returncode}:\n"
            f"{completed.stderr}"
        )
    output_lines = [
        line.strip()
        for line in completed.stdout.splitlines()
        if line.strip()
    ]
    if not output_lines or output_lines[-1] != "[0]":
        raise RuntimeError(
            "SpMV correctness validation failed; expected zero mismatched rows, "
            f"got stdout: {completed.stdout!r}"
        )
    return command


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


def write_results(path, results):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        for result in results:
            formatted = dict(result)
            for field in (
                "min_us",
                "median_us",
                "p95_us",
                "gnnz_per_sec",
                "gflops",
                "mean_row_nnz",
            ):
                formatted[field] = f"{result[field]:.6f}"
            formatted["correct"] = str(result["correct"]).lower()
            writer.writerow(formatted)


def write_metadata(path, args, repository, commands):
    llvm_repository = repository / "externals" / "llvm-project"
    metadata = {
        "compiler_commit": git_output(repository, "rev-parse", "HEAD"),
        "llvm_commit": git_output(llvm_repository, "rev-parse", "HEAD"),
        "rocm_version": rocm_version(args.rocm_path),
        "gpu": args.gpu_name,
        "chip": args.chip,
        "wave_size": args.wave_size,
        "block_sizes": args.block_sizes,
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
            args.matrix_data["field"] if args.matrix_data is not None else None
        ),
        "matrix_symmetry": (
            args.matrix_data["symmetry"]
            if args.matrix_data is not None
            else None
        ),
        "warmup": args.warmup,
        "iterations": args.iterations,
        "commands": commands,
    }
    path.write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )


def print_results(args, results):
    print("SparseWave SpMV benchmark")
    print(f"  GPU:           {args.gpu_name}")
    print(f"  chip:          {args.chip}")
    print(f"  wave size:     {args.wave_size}")
    print(
        "  block sizes:   "
        + ", ".join(str(value) for value in args.block_sizes)
    )
    if args.matrix_data is not None:
        print(f"  matrix:        {args.matrix}")
        print(f"  rows:          {args.matrix_data['rows']}")
        print(f"  columns:       {args.matrix_data['columns']}")
        print(f"  NNZ:           {args.matrix_data['nnz']}")
    else:
        print(f"  rows:          {args.rows}")
        print(f"  columns:       {args.columns}")
    print(f"  warmup:        {args.warmup}")
    print(f"  iterations:    {args.iterations}")
    if args.matrix_data is None:
        print(f"  distributions: {', '.join(args.distributions)}")
    print()
    if args.matrix_data is not None:
        print(
            "block  thread median  wave median  thread GNNZ/s  "
            "wave GNNZ/s  speedup  winner"
        )
        by_case = {
            (result["block_size"], result["mapping"]): result
            for result in results
        }
        for block_size in args.block_sizes:
            thread = by_case[(block_size, "thread-per-row")]
            wave = by_case[(block_size, "wave-per-row")]
            speedup = thread["median_us"] / wave["median_us"]
            winner = "wave" if speedup > 1.0 else "thread"
            print(
                f"{block_size:5d}"
                f"  {thread['median_us']:11.2f} us"
                f"  {wave['median_us']:9.2f} us"
                f"  {thread['gnnz_per_sec']:13.2f}"
                f"  {wave['gnnz_per_sec']:11.2f}"
                f"  {speedup:7.2f}x"
                f"  {winner}"
            )
        return

    print(
        "distribution  NNZ/row  block  thread median  wave median  "
        "thread GNNZ/s  wave GNNZ/s  speedup  winner"
    )
    by_case = {
        (
            result["distribution"],
            result["nnz_per_row"],
            result["block_size"],
            result["mapping"],
        ): result
        for result in results
    }
    for distribution in args.distributions:
        for nnz_per_row in args.nnz_per_row:
            for block_size in args.block_sizes:
                thread = by_case[
                    (
                        distribution,
                        nnz_per_row,
                        block_size,
                        "thread-per-row",
                    )
                ]
                wave = by_case[
                    (
                        distribution,
                        nnz_per_row,
                        block_size,
                        "wave-per-row",
                    )
                ]
                speedup = thread["median_us"] / wave["median_us"]
                winner = "wave" if speedup > 1.0 else "thread"
                print(
                    f"{distribution:12s}"
                    f"  {nnz_per_row:7d}"
                    f"  {block_size:5d}"
                    f"  {thread['median_us']:11.2f} us"
                    f"  {wave['median_us']:9.2f} us"
                    f"  {thread['gnnz_per_sec']:13.2f}"
                    f"  {wave['gnnz_per_sec']:11.2f}"
                    f"  {speedup:7.2f}x"
                    f"  {winner}"
                )


def validate_paths(args):
    paths = [
        args.sparsewave_opt,
        args.mlir_runner,
        args.rocm_runtime,
        args.runner_utils,
        args.rocprofv3,
    ]
    if args.matrix_data is not None:
        paths.append(args.benchmark_utils)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"required benchmark tools are missing: {missing}")
    if args.matrix_data is None:
        validate_distribution_rows(args.rows, args.distributions)
    if args.wave_size != 32:
        raise ValueError("wave-per-row benchmarking currently requires Wave32")
    invalid_block_sizes = [
        value
        for value in args.block_sizes
        if value > 1024 or value % args.wave_size != 0
    ]
    if invalid_block_sizes:
        raise ValueError(
            "block sizes must not exceed 1024 and must be multiples of wave "
            f"size; invalid values: {invalid_block_sizes}"
        )
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
    parser.add_argument("--warmup", type=nonnegative_int, default=10)
    parser.add_argument("--iterations", type=positive_int, default=50)
    block_group = parser.add_mutually_exclusive_group()
    block_group.add_argument(
        "--block-sizes",
        type=parse_positive_int_list,
        default=parse_positive_int_list("64,128,256,512"),
    )
    block_group.add_argument(
        "--block-size",
        dest="block_sizes",
        type=lambda value: [positive_int(value)],
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--wave-size", type=positive_int, default=32)
    parser.add_argument("--chip")
    parser.add_argument("--rocm-path", type=Path, default=Path("/opt/rocm"))
    parser.add_argument(
        "--build-dir", type=Path, default=repository / "build" / "llvm"
    )
    parser.add_argument("--sparsewave-opt", type=Path)
    parser.add_argument("--mlir-runner", type=Path)
    parser.add_argument("--rocm-runtime", type=Path)
    parser.add_argument("--runner-utils", type=Path)
    parser.add_argument("--benchmark-utils", type=Path)
    parser.add_argument("--rocprofv3", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--keep-artifacts", action="store_true")
    args = parser.parse_args(argv)

    args.sparsewave_opt = (
        args.sparsewave_opt or args.build_dir / "bin" / "sparsewave-opt"
    )
    args.mlir_runner = args.mlir_runner or args.build_dir / "bin" / "mlir-runner"
    args.rocm_runtime = (
        args.rocm_runtime or args.build_dir / "lib" / "libmlir_rocm_runtime.so"
    )
    args.runner_utils = (
        args.runner_utils or args.build_dir / "lib" / "libmlir_runner_utils.so"
    )
    benchmark_utils_candidates = (
        args.build_dir
        / "tools"
        / "sparsewave"
        / "lib"
        / "libsparsewave_spmv_benchmark_utils.so",
        args.build_dir / "lib" / "libsparsewave_spmv_benchmark_utils.so",
    )
    args.benchmark_utils = args.benchmark_utils or next(
        (
            path
            for path in benchmark_utils_candidates
            if path.is_file()
        ),
        benchmark_utils_candidates[0],
    )
    args.rocprofv3 = (
        args.rocprofv3 or args.rocm_path / "bin" / "rocprofv3"
    )
    args.chip = args.chip or detect_chip(args.rocm_path)
    args.gpu_name = args.chip
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
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    output_directory = (
        args.output_dir
        or repository / "build" / "benchmark" / "results" / timestamp
    )
    output_directory.mkdir(parents=True, exist_ok=False)
    temporary_root = None
    if args.keep_artifacts:
        artifact_root = output_directory / "artifacts"
        artifact_root.mkdir()
    else:
        temporary_root = tempfile.TemporaryDirectory(
            prefix="sparsewave-spmv-benchmark-"
        )
        artifact_root = Path(temporary_root.name)

    results = []
    commands = []
    try:
        csr_binary = None
        if args.matrix_data is not None:
            csr_binary = artifact_root / "matrix.csr"
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
                        artifact_root
                        / workload["key"]
                        / f"block-{block_size}"
                        / mapping
                    )
                    case_directory.mkdir(parents=True)
                    source = case_directory / "input.mlir"
                    compiled = case_directory / "compiled.mlir"
                    trace_directory = case_directory / "trace"
                    trace_directory.mkdir()
                    source.write_text(source_text, encoding="utf-8")
                    compile_command = compile_mlir(
                        args,
                        source,
                        compiled,
                        mapping,
                        block_size,
                    )
                    profile_command = profile_mlir(
                        args, compiled, trace_directory, csr_binary
                    )
                    trace = discover_trace(trace_directory)
                    if args.gpu_name == args.chip:
                        args.gpu_name = discover_gpu_name(
                            trace_directory, args.chip
                        )
                    timing = parse_kernel_trace(
                        trace,
                        "spmv_kernel",
                        args.warmup,
                        args.iterations,
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

        write_results(output_directory / "results.csv", results)
        write_metadata(
            output_directory / "metadata.json", args, repository, commands
        )
        print_results(args, results)
        print()
        print(f"Results: {output_directory}")
    finally:
        if temporary_root is not None:
            temporary_root.cleanup()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
