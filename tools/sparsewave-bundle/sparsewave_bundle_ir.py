import re


SUPPORTED_OPERATION = "spmm"
SPARSE_ATTENTION = "sparse-attention"
SUPPORTED_TARGET = "gfx1101"
SUPPORTED_MAPPING = "wave-per-row-tile"
SUPPORTED_BLOCK_SIZE = 64
SUPPORTED_TILE_SIZE = 4
SUPPORTED_WAVEFRONT_SIZE = 32
SPMM_ARGUMENT_NAMES = (
    "rowOffsets",
    "columnIndices",
    "values",
    "rhs",
    "output",
)
SCALAR_TYPE_SIZES = {
    "i32": 4,
    "i64": 8,
    "fp32": 4,
    "fp16": 2,
    "bf16": 2,
}
ATTENTION_KERNELS = {
    "sparse_attention_scores": {
        "role": "scores",
        "operation": "sddmm",
        "mapping": "thread-per-row",
        "launch_n": "output_rows",
        "args": (
            "rowOffsets",
            "columnIndices",
            "maskValues",
            "query",
            "transposedKey",
            "scores",
        ),
    },
    "sparse_attention_row_max": {
        "role": "row_max",
        "operation": "csr_row_reduce_max",
        "mapping": "thread-per-row",
        "launch_n": "output_rows",
        "args": ("rowOffsets", "scores", "rowMaximum"),
    },
    "sparse_attention_exp": {
        "role": "exp",
        "operation": "csr_rowwise_map",
        "mapping": "thread-per-row",
        "launch_n": "output_rows",
        "args": ("rowOffsets", "rowMaximum", "scores"),
    },
    "sparse_attention_row_sum": {
        "role": "row_sum",
        "operation": "csr_row_reduce_sum",
        "mapping": "thread-per-row",
        "launch_n": "output_rows",
        "args": ("rowOffsets", "scores", "rowSum"),
    },
    "sparse_attention_normalize": {
        "role": "normalize",
        "operation": "csr_rowwise_map",
        "mapping": "thread-per-row",
        "launch_n": "output_rows",
        "args": ("rowOffsets", "rowSum", "scores"),
    },
    "sparse_attention_output": {
        "role": "output",
        "operation": "spmm",
        "mapping": "thread-per-output",
        "launch_n": "output_elements",
        "args": (
            "rowOffsets",
            "columnIndices",
            "scores",
            "value",
            "output",
        ),
    },
}
ATTENTION_BUFFER_NAMES = (
    "rowOffsets",
    "columnIndices",
    "maskValues",
    "query",
    "transposedKey",
    "value",
    "scores",
    "rowMaximum",
    "rowSum",
    "output",
)


def fail(message):
    raise ValueError(message)


def abi_type_from_ir(type_text):
    if type_text.startswith("memref<") or type_text.startswith(
        "unranked_memref<"
    ):
        return "ptr"
    scalar_types = {
        "i32": "i32",
        "i64": "i64",
        "f32": "fp32",
        "f16": "fp16",
        "bf16": "bf16",
    }
    if type_text in scalar_types:
        return scalar_types[type_text]
    fail(f"unsupported kernel argument IR type '{type_text}'")


def parse_memref_tensor(type_text, dynamic_name=None):
    match = re.fullmatch(
        r"memref<((?:(?:\d+|\?)x)*)(i32|f32|f16|bf16)>", type_text
    )
    if match is None:
        fail(f"unsupported tensor ABI type '{type_text}'")
    dimensions = [value for value in match.group(1).split("x") if value]
    shape = [None if value == "?" else int(value) for value in dimensions]
    dynamic_dimensions = [
        {"index": index, "name": dynamic_name or f"dim{index}"}
        for index, value in enumerate(shape)
        if value is None
    ]
    tensor = {
        "dtype": {"f32": "fp32"}.get(match.group(2), match.group(2)),
        "rank": len(shape),
        "shape": shape,
        "specialized": not dynamic_dimensions,
    }
    if dynamic_dimensions:
        tensor["dynamic_dimensions"] = dynamic_dimensions
    return tensor


def parse_function_arguments_before(text, operation_position):
    functions = list(
        re.finditer(
            r"func\.func\s+@([A-Za-z_.$-][A-Za-z0-9_.$-]*)\s*"
            r"\((.*?)\)\s*\{",
            text[:operation_position],
            flags=re.DOTALL,
        )
    )
    if not functions:
        fail("expected a function containing the SparseWave application")
    function = functions[-1]
    arguments = re.findall(
        r"%([A-Za-z_.$-][A-Za-z0-9_.$-]*)\s*:\s*"
        r"(memref<[^>]+>|unranked_memref<[^>]+>|i32|i64|f32|f16|bf16)",
        function.group(2),
    )
    return function.group(1), arguments


def parse_fixed_spmm_ir_text(text):
    operation = re.search(r"^\s*sparsewave\.spmm\b", text, re.MULTILINE)
    if operation is None:
        fail("expected one fixed-shape function containing sparsewave.spmm")
    _, arguments = parse_function_arguments_before(text, operation.start())
    if len(arguments) != len(SPMM_ARGUMENT_NAMES):
        fail("fixed-shape SpMM bundle requires exactly five kernel arguments")
    types = [type_text for _, type_text in arguments]
    row_offsets = re.fullmatch(r"memref<(\d+)xi32>", types[0])
    column_indices = re.fullmatch(r"memref<(\d+|\?)xi32>", types[1])
    values = re.fullmatch(r"memref<(\d+|\?)xf32>", types[2])
    rhs = re.fullmatch(r"memref<(\d+)x(\d+)xf32>", types[3])
    output = re.fullmatch(r"memref<(\d+)x(\d+)xf32>", types[4])
    if not all((row_offsets, column_indices, values, rhs, output)):
        fail("bundle input must use fixed-shape CSR i32 and FP32 SpMM memrefs")

    rows = int(output.group(1))
    rhs_columns = int(rhs.group(2))
    if int(row_offsets.group(1)) != rows + 1:
        fail("rowOffsets extent must equal output rows plus one")
    column_indices_extent = column_indices.group(1)
    values_extent = values.group(1)
    if (
        column_indices_extent != "?"
        and values_extent != "?"
        and int(column_indices_extent) != int(values_extent)
    ):
        fail("columnIndices and values extents must match")
    if rows <= 0 or int(rhs.group(1)) <= 0:
        fail("output and RHS row extents must be positive")
    if rhs_columns != int(output.group(2)):
        fail("RHS and output column extents must match")
    if rhs_columns != SUPPORTED_TILE_SIZE:
        fail(
            "initial lrrt bundle support requires RHS/output columns equal "
            f"tile size {SUPPORTED_TILE_SIZE}"
        )

    return {
        "application": SUPPORTED_OPERATION,
        "name": SUPPORTED_OPERATION,
        "args": [
            {"name": name, "type": abi_type_from_ir(type_text)}
            for name, (_, type_text) in zip(SPMM_ARGUMENT_NAMES, arguments)
        ],
        "output_rows": rows,
        "rhs_columns": rhs_columns,
    }


def parse_sparse_attention_ir_text(text):
    marker = re.search(
        r'sparsewave\.kernel_name\s*=\s*"sparse_attention_scores"', text
    )
    if marker is None:
        fail("SparseAttention IR is missing stable kernel role metadata")
    _, arguments = parse_function_arguments_before(text, marker.start())
    if len(arguments) != len(ATTENTION_BUFFER_NAMES):
        fail("SparseAttention bundle requires exactly ten buffer arguments")

    types = [type_text for _, type_text in arguments]
    patterns = (
        r"memref<(\d+)xi32>",
        r"memref<\?xi32>",
        r"memref<\?xf32>",
        r"memref<(\d+)x(\d+)xf32>",
        r"memref<(\d+)x(\d+)xf32>",
        r"memref<(\d+)x(\d+)xf32>",
        r"memref<\?xf32>",
        r"memref<(\d+)xf32>",
        r"memref<(\d+)xf32>",
        r"memref<(\d+)x(\d+)xf32>",
    )
    matches = [
        re.fullmatch(pattern, type_text)
        for pattern, type_text in zip(patterns, types)
    ]
    if not all(matches):
        fail("bundle input must use the supported i32/FP32 SparseAttention ABI")

    query_rows, head_dimension = map(int, matches[3].groups())
    key_head_dimension, key_rows = map(int, matches[4].groups())
    value_rows, value_columns = map(int, matches[5].groups())
    output_rows, output_columns = map(int, matches[9].groups())
    if int(matches[0].group(1)) != output_rows + 1:
        fail("SparseAttention rowOffsets extent must equal output rows plus one")
    if (
        query_rows != output_rows
        or key_head_dimension != head_dimension
        or key_rows != value_rows
        or value_columns != output_columns
        or int(matches[7].group(1)) != output_rows
        or int(matches[8].group(1)) != output_rows
    ):
        fail("incompatible specialized SparseAttention buffer shapes")

    buffers = {}
    for name, type_text in zip(ATTENTION_BUFFER_NAMES, types):
        dynamic_name = (
            "nnz"
            if name in {"columnIndices", "maskValues", "scores"}
            else None
        )
        buffers[name] = {
            "name": name,
            "type": "ptr",
            "tensor": parse_memref_tensor(type_text, dynamic_name),
            "kind": (
                "intermediate"
                if name in {"scores", "rowMaximum", "rowSum"}
                else "output"
                if name == "output"
                else "input"
            ),
        }

    kernels = {}
    for name, specification in ATTENTION_KERNELS.items():
        kernels[name] = {
            **specification,
            "name": name,
            "args": [buffers[arg_name] for arg_name in specification["args"]],
        }
    return {
        "application": SPARSE_ATTENTION,
        "buffers": buffers,
        "kernels": kernels,
        "specialization": {
            "output_rows": output_rows,
            "key_value_rows": value_rows,
            "head_dimension": head_dimension,
            "value_columns": value_columns,
        },
    }


def parse_application_ir_text(text):
    if "sparse_attention_scores" in text:
        return parse_sparse_attention_ir_text(text)
    return parse_fixed_spmm_ir_text(text)


def parse_fixed_spmm_ir(path):
    return parse_fixed_spmm_ir_text(path.read_text(encoding="utf-8"))
