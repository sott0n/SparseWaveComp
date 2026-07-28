import argparse
import csv
from dataclasses import dataclass
import datetime
import json
import math
import os
from pathlib import Path
import statistics
import struct
import subprocess
import tempfile


TRACE_COLUMNS = {
    "kernel": ("Kernel_Name", "Kernel Name", "Name"),
    "start": ("Start_Timestamp", "Start Timestamp", "Start"),
    "end": ("End_Timestamp", "End Timestamp", "End"),
}
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
        raise argparse.ArgumentTypeError("expected at least one value")
    if len(values) != len(set(values)):
        raise argparse.ArgumentTypeError(
            f"duplicate values are not allowed: {values}"
        )
    return values


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


def render_template(template_path, replacements):
    rendered = template_path.read_text(encoding="utf-8")
    for token, value in replacements.items():
        rendered = rendered.replace(token, str(value))
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


def add_common_arguments(parser, repository):
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


def configure_common_arguments(args):
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
        / "libsparsewave_benchmark_utils.so",
        args.build_dir / "lib" / "libsparsewave_benchmark_utils.so",
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


def validate_required_paths(args, needs_benchmark_utils):
    paths = [
        args.sparsewave_opt,
        args.mlir_runner,
        args.rocm_runtime,
        args.runner_utils,
        args.rocprofv3,
    ]
    if needs_benchmark_utils:
        paths.append(args.benchmark_utils)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"required benchmark tools are missing: {missing}")


def validate_block_sizes(args, require_wave_multiple):
    invalid = [
        value
        for value in args.block_sizes
        if value > 1024
        or (require_wave_multiple and value % args.wave_size != 0)
    ]
    if invalid:
        requirement = (
            " and must be multiples of wave size"
            if require_wave_multiple
            else ""
        )
        raise ValueError(
            f"block sizes must not exceed 1024{requirement}; "
            f"invalid values: {invalid}"
        )


def compile_mlir(
    args,
    source,
    output,
    operation,
    mapping,
    block_size,
    pipeline_options=(),
):
    lowering_options = [
        f"{operation}-mapping={mapping}",
        f"{operation}-block-size={block_size}",
        *pipeline_options,
    ]
    pipeline = (
        "builtin.module(convert-scf-to-cf,"
        "sparsewave-to-amdgpu-pipeline{"
        f"chip={args.chip} "
        f"wavefront-size={args.wave_size} "
        f"rocm-path={args.rocm_path} "
        + " ".join(lowering_options)
        + "},reconcile-unrealized-casts)"
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


def profile_mlir(
    args, compiled, trace_directory, csr_binary, operation
):
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
            f"{operation.upper()} correctness validation failed; expected "
            f"zero mismatches, got stdout: {completed.stdout!r}"
        )
    return command


def run_case(
    args,
    source_text,
    case_directory,
    operation,
    mapping,
    block_size,
    csr_binary,
    pipeline_options=(),
):
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
        operation,
        mapping,
        block_size,
        pipeline_options,
    )
    profile_command = profile_mlir(
        args, compiled, trace_directory, csr_binary, operation
    )
    trace = discover_trace(trace_directory)
    if args.gpu_name == args.chip:
        args.gpu_name = discover_gpu_name(trace_directory, args.chip)
    timing = parse_kernel_trace(
        trace,
        f"{operation}_kernel",
        args.warmup,
        args.iterations,
    )
    return timing, compile_command, profile_command


def write_results(path, columns, float_fields, results):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for result in results:
            formatted = dict(result)
            for field in float_fields:
                formatted[field] = f"{result[field]:.6f}"
            formatted["correct"] = str(result["correct"]).lower()
            writer.writerow(formatted)


def base_metadata(args, repository, operation):
    llvm_repository = repository / "externals" / "llvm-project"
    return {
        "operation": operation,
        "compiler_commit": git_output(repository, "rev-parse", "HEAD"),
        "llvm_commit": git_output(llvm_repository, "rev-parse", "HEAD"),
        "rocm_version": rocm_version(args.rocm_path),
        "gpu": args.gpu_name,
        "chip": args.chip,
        "wave_size": args.wave_size,
        "block_sizes": args.block_sizes,
        "warmup": args.warmup,
        "iterations": args.iterations,
    }


def write_metadata(path, metadata):
    path.write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )


def print_common_configuration(args, title):
    print(title)
    print(f"  GPU:           {args.gpu_name}")
    print(f"  chip:          {args.chip}")
    print(f"  wave size:     {args.wave_size}")
    print(
        "  block sizes:   "
        + ", ".join(str(value) for value in args.block_sizes)
    )


@dataclass(frozen=True)
class TableColumn:
    header: str
    key: str
    width: int
    format_spec: str = ""
    suffix: str = ""
    alignment: str = ">"

    def format_header(self):
        return f"{self.header:{self.alignment}{self.width}}"

    def format_value(self, row):
        raw_value = row[self.key]
        if raw_value is None:
            value = "-"
        else:
            value = format(raw_value, self.format_spec) + self.suffix
        return f"{value:{self.alignment}{self.width}}"


@dataclass(frozen=True)
class BenchmarkReport:
    details: tuple
    dimensions: tuple
    metrics: tuple
    baseline: object = None


REPORT_COLUMNS = {
    "distribution": TableColumn(
        "distribution", "distribution", 12, alignment="<"
    ),
    "nnz_per_row": TableColumn("NNZ/row", "nnz_per_row", 7, "d"),
    "rhs_cols": TableColumn("RHS cols", "rhs_cols", 8, "d"),
    "block_size": TableColumn("block", "block_size", 5, "d"),
    "tile_size": TableColumn("tile", "tile_size", 4, "d"),
    "mapping": TableColumn("mapping", "mapping", 18, alignment="<"),
    "median_us": TableColumn("median", "median_us", 11, ".2f", " us"),
    "p95_us": TableColumn("p95", "p95_us", 11, ".2f", " us"),
    "gnnz_per_sec": TableColumn(
        "GNNZ/s", "gnnz_per_sec", 8, ".2f"
    ),
    "gproducts_per_sec": TableColumn(
        "GProduct/s", "gproducts_per_sec", 10, ".2f"
    ),
    "gflops": TableColumn("GFLOP/s", "gflops", 7, ".2f"),
    "speedup": TableColumn("vs baseline", "speedup", 11, ".2f", "x"),
}


def report_detail(args, name):
    matrix = getattr(args, "matrix_data", None)
    if name == "matrix":
        return "matrix", args.matrix
    if name == "rows":
        return "rows", matrix["rows"] if matrix is not None else args.rows
    if name == "columns":
        value = matrix["columns"] if matrix is not None else args.columns
        return "columns", value
    if name == "input_columns":
        return "input columns", matrix["columns"]
    if name == "nnz":
        return "NNZ", matrix["nnz"]
    if name == "rhs_columns":
        return "RHS columns", ", ".join(
            str(value) for value in args.rhs_columns
        )
    if name == "distributions":
        return "distributions", ", ".join(args.distributions)
    raise ValueError(f"unknown benchmark report detail '{name}'")


def add_speedups(report, rows):
    if report.baseline is None:
        return rows
    baseline_key, baseline_value = report.baseline
    group_keys = tuple(
        key for key in report.dimensions if key != baseline_key
    )
    baselines = {
        tuple(row[key] for key in group_keys): row
        for row in rows
        if row[baseline_key] == baseline_value
    }
    reported_rows = []
    for row in rows:
        group = tuple(row[key] for key in group_keys)
        if group not in baselines:
            raise ValueError(
                f"missing report baseline {baseline_key}={baseline_value} "
                f"for dimensions {group}"
            )
        baseline = baselines[group]
        reported_rows.append(
            {
                **row,
                "speedup": baseline["median_us"] / row["median_us"],
            }
        )
    return reported_rows


def print_table(columns, rows):
    print("  ".join(column.format_header() for column in columns))
    for row in rows:
        print(
            "  ".join(column.format_value(row) for column in columns)
        )


def print_benchmark_report(args, title, report, rows):
    print_common_configuration(args, title)
    for detail in report.details:
        label, value = report_detail(args, detail)
        print(f"  {label + ':':15s}{value}")
    print(f"  {'warmup:':15s}{args.warmup}")
    print(f"  {'iterations:':15s}{args.iterations}")
    print()
    column_keys = report.dimensions + report.metrics
    if report.baseline is not None:
        column_keys += ("speedup",)
    unknown_columns = [
        key for key in column_keys if key not in REPORT_COLUMNS
    ]
    if unknown_columns:
        raise ValueError(f"unknown benchmark report columns: {unknown_columns}")
    columns = tuple(REPORT_COLUMNS[key] for key in column_keys)
    rows = add_speedups(report, rows)
    print_table(columns, rows)


class BenchmarkWorkspace:
    def __init__(self, args, repository, result_directory, temporary_prefix):
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.output_directory = (
            args.output_dir
            or repository / "build" / "benchmark" / result_directory / timestamp
        )
        self.output_directory.mkdir(parents=True, exist_ok=False)
        self.temporary_root = None
        if args.keep_artifacts:
            self.artifact_root = self.output_directory / "artifacts"
            self.artifact_root.mkdir()
        else:
            self.temporary_root = tempfile.TemporaryDirectory(
                prefix=temporary_prefix
            )
            self.artifact_root = Path(self.temporary_root.name)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.temporary_root is not None:
            self.temporary_root.cleanup()
