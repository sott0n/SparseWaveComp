#!/usr/bin/env python3

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys


MANIFEST_VERSION = 1
CODE_OBJECT = "kernels.hsaco"
SUPPORTED_TARGET = "gfx1101"
SUPPORTED_OPERATION = "spmm"
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


def fail(message):
    raise ValueError(message)


def find_tool(name):
    sibling = Path(__file__).resolve().with_name(name)
    if sibling.is_file():
        return sibling
    installed = shutil.which(name)
    if installed:
        return Path(installed)
    rocm_tool = Path("/opt/rocm/llvm/bin") / name
    return rocm_tool if rocm_tool.is_file() else Path(name)


def run(command, **kwargs):
    completed = subprocess.run(command, capture_output=True, **kwargs)
    if completed.returncode:
        stderr = completed.stderr
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        raise RuntimeError(
            f"command failed with exit code {completed.returncode}: "
            f"{' '.join(map(str, command))}\n{stderr}"
        )
    return completed


def extract_gpu_binary(compiled_text):
    matches = list(re.finditer(r'\bbin\s*=\s*"', compiled_text))
    if len(matches) != 1:
        fail(f"expected one embedded GPU binary, found {len(matches)}")

    binary = bytearray()
    cursor = matches[0].end()
    while cursor < len(compiled_text) and compiled_text[cursor] != '"':
        character = compiled_text[cursor]
        if character != "\\":
            binary.extend(character.encode("utf-8"))
            cursor += 1
            continue
        if cursor + 1 < len(compiled_text) and compiled_text[cursor + 1] == "\\":
            binary.append(ord("\\"))
            cursor += 2
            continue
        escaped = compiled_text[cursor + 1 : cursor + 3]
        if len(escaped) != 2 or not all(
            character in "0123456789abcdefABCDEF" for character in escaped
        ):
            fail(f"invalid byte escape in embedded GPU binary: {escaped!r}")
        binary.append(int(escaped, 16))
        cursor += 3

    if cursor == len(compiled_text):
        fail("unterminated embedded GPU binary")
    if not binary.startswith(b"\x7fELF"):
        fail("embedded GPU binary is not an ELF file")
    return bytes(binary)


def metadata_document(readobj_output):
    documents = json.loads(readobj_output)
    found = []

    def visit(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if key == "AMDGPU Metadata":
                    found.append(child)
                else:
                    visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(documents)
    if len(found) != 1:
        fail(f"expected one AMDGPU metadata document, found {len(found)}")
    return found[0]


def yaml_scalar(value):
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
        return value[1:-1]
    if re.fullmatch(r"-?(?:0x[0-9a-fA-F]+|[0-9]+)", value):
        return int(value, 0)
    if value in ("true", "false"):
        return value == "true"
    return value


def parse_fields(block, indent):
    fields = {}
    pattern = rf"^ {{{indent}}}\.([A-Za-z0-9_]+):[ \t]*(.*?)[ \t]*$"
    for match in re.finditer(pattern, block, flags=re.MULTILINE):
        fields[match.group(1)] = yaml_scalar(match.group(2))
    return fields


def parse_metadata_arguments(block):
    args_match = re.search(
        r"^(?:  - |    )\.args:[ \t]*(.*)$", block, flags=re.MULTILINE
    )
    if args_match is None:
        fail("kernel metadata is missing '.args'")
    if args_match.group(1).strip() == "[]":
        return []

    start = args_match.end()
    end_match = re.search(
        r"^    \.[A-Za-z0-9_]+:", block[start:], re.MULTILINE
    )
    args_text = (
        block[start : start + end_match.start()] if end_match else block[start:]
    )
    starts = list(
        re.finditer(
            r"^      - \.([A-Za-z0-9_]+):[ \t]*(.*?)[ \t]*$",
            args_text,
            re.MULTILINE,
        )
    )
    arguments = []
    for index, match in enumerate(starts):
        end = (
            starts[index + 1].start()
            if index + 1 < len(starts)
            else len(args_text)
        )
        argument = {match.group(1): yaml_scalar(match.group(2))}
        argument.update(parse_fields(args_text[match.end() : end], 8))
        arguments.append(argument)
    return arguments


def parse_block_size(block, fields):
    required = re.search(
        r"^    \.reqd_workgroup_size:[ \t]*\n"
        r"((?:      - [0-9]+[ \t]*\n?){3})",
        block,
        flags=re.MULTILINE,
    )
    if required:
        return [
            int(value)
            for value in re.findall(
                r"^      - ([0-9]+)", required.group(1), re.MULTILINE
            )
        ]
    return [fields["max_flat_workgroup_size"], 1, 1]


def target_from_metadata(document):
    target = re.search(
        r"^amdhsa\.target:[ \t]*(\S+)[ \t]*$", document, re.MULTILINE
    )
    if target is None:
        fail("AMDGPU metadata is missing 'amdhsa.target'")
    architecture = yaml_scalar(target.group(1))
    chip = re.search(r"(?:^|-)\b(gfx[0-9a-z]+)(?::[^-]+)?$", architecture)
    if chip is None:
        fail(f"cannot determine AMDGPU chip from target '{architecture}'")
    return chip.group(1)


def kernels_from_metadata(document):
    blocks = re.split(
        r"(?=^  - \.[A-Za-z0-9_]+:)", document, flags=re.MULTILINE
    )[1:]
    if not blocks:
        fail("AMDGPU metadata contains no kernels")

    kernels = []
    for block in blocks:
        fields = parse_fields(block, 4)
        required = (
            "name",
            "symbol",
            "kernarg_segment_size",
            "group_segment_fixed_size",
            "max_flat_workgroup_size",
            "wavefront_size",
        )
        missing = [f".{name}" for name in required if name not in fields]
        if missing:
            fail(f"kernel metadata is missing {', '.join(missing)}")
        kernels.append(
            {
                "name": fields["name"],
                "descriptor_symbol": fields["symbol"],
                "args": parse_metadata_arguments(block),
                "kernarg_size": fields["kernarg_segment_size"],
                "block": parse_block_size(block, fields),
                "fixed_group_segment_bytes": fields[
                    "group_segment_fixed_size"
                ],
                "wavefront_size": fields["wavefront_size"],
            }
        )
    return kernels


def inspect_hsaco(readobj, hsaco):
    output = run(
        [str(readobj), "--notes", "--elf-output-style=JSON", str(hsaco)],
        text=True,
    ).stdout
    document = metadata_document(output)
    return {
        "target": target_from_metadata(document),
        "kernels": kernels_from_metadata(document),
    }


def abi_type_from_ir(type_text):
    if type_text.startswith("memref<") or type_text.startswith("unranked_memref<"):
        return "ptr"
    scalar_types = {
        "i32": "i32",
        "f32": "fp32",
        "f16": "fp16",
        "bf16": "bf16",
    }
    if type_text in scalar_types:
        return scalar_types[type_text]
    fail(f"unsupported kernel argument IR type '{type_text}'")


def parse_fixed_spmm_ir_text(text):
    operation = re.search(r"^\s*sparsewave\.spmm\b", text, re.MULTILINE)
    if operation is None:
        fail("expected one fixed-shape function containing sparsewave.spmm")
    functions = list(
        re.finditer(
            r"func\.func\s+@([A-Za-z_.$-][A-Za-z0-9_.$-]*)\s*"
            r"\((.*?)\)\s*\{",
            text[: operation.start()],
            flags=re.DOTALL,
        )
    )
    if not functions:
        fail("expected one fixed-shape function containing sparsewave.spmm")
    function = functions[-1]
    arguments = re.findall(
        r"%([A-Za-z_.$-][A-Za-z0-9_.$-]*)\s*:\s*"
        r"(memref<[^>]+>|unranked_memref<[^>]+>|i32|f32|f16|bf16)",
        function.group(2),
    )
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
        "name": SUPPORTED_OPERATION,
        "args": [
            {"name": name, "type": abi_type_from_ir(type_text)}
            for name, (_, type_text) in zip(SPMM_ARGUMENT_NAMES, arguments)
        ],
        "output_rows": rows,
        "rhs_columns": rhs_columns,
    }


def parse_fixed_spmm_ir(path):
    return parse_fixed_spmm_ir_text(path.read_text(encoding="utf-8"))


def prepare_bundle_input(args):
    text = args.input.read_text(encoding="utf-8")
    if not re.search(r'(?:!torch\.|"torch\.)', text):
        return text, None

    command = [
        str(args.sparsewave_opt),
        "--allow-unregistered-dialect",
        str(args.input),
        "--convert-torch-to-sparsewave",
    ]
    return run(command, text=True).stdout, command


def validate_supported_options(args):
    expected = {
        "chip": (args.chip, SUPPORTED_TARGET),
        "operation": (args.operation, SUPPORTED_OPERATION),
        "mapping": (args.mapping, SUPPORTED_MAPPING),
        "block size": (args.block_size, SUPPORTED_BLOCK_SIZE),
        "tile size": (args.tile_size, SUPPORTED_TILE_SIZE),
        "wavefront size": (args.wavefront_size, SUPPORTED_WAVEFRONT_SIZE),
    }
    for label, (actual, supported) in expected.items():
        if actual != supported:
            fail(
                f"initial lrrt bundle support requires {label} {supported}, "
                f"got {actual}"
            )
    if args.triple != "amdgcn-amd-amdhsa" or args.features:
        fail(
            "initial lrrt bundle support requires the default AMDHSA triple "
            "and no features"
        )


def merge_arguments(ir_arguments, metadata_arguments):
    if len(ir_arguments) != len(metadata_arguments):
        fail(
            "compiler IR and HSACO kernel argument counts differ: "
            f"{len(ir_arguments)} != {len(metadata_arguments)}"
        )

    arguments = []
    for ir_argument, metadata_argument in zip(
        ir_arguments, metadata_arguments
    ):
        if "offset" not in metadata_argument or "size" not in metadata_argument:
            fail("HSACO kernel argument is missing offset or size")
        if ir_argument["type"] != "ptr":
            fail("initial lrrt SpMM bundle supports pointer arguments only")
        if metadata_argument.get("value_kind") != "global_buffer":
            fail(
                f"IR pointer argument '{ir_argument['name']}' does not map to "
                "an HSACO global_buffer argument"
            )
        if metadata_argument.get("size") != 8:
            fail(
                f"IR pointer argument '{ir_argument['name']}' must have an "
                "8-byte HSACO entry"
            )
        arguments.append(
            {
                **ir_argument,
                "offset": metadata_argument["offset"],
                "size": metadata_argument["size"],
            }
        )
    return arguments


def grid_expression(block_size, wavefront_size):
    waves_per_block = block_size // wavefront_size
    return f"ceil_div(n, {waves_per_block}) * {block_size}"


def manifest_for(args, hsaco, metadata, ir):
    if metadata["target"] != args.chip:
        fail(
            f"requested target {args.chip} does not match HSACO target "
            f"{metadata['target']}"
        )
    if len(metadata["kernels"]) != 1:
        fail(f"expected one HSACO kernel, found {len(metadata['kernels'])}")
    kernel = metadata["kernels"][0]
    if kernel["wavefront_size"] != args.wavefront_size:
        fail("HSACO wavefront size does not match the requested Wave32 target")
    if kernel["block"] != [args.block_size, 1, 1]:
        fail("HSACO required workgroup size does not match the requested block")

    digest = hashlib.sha256(hsaco.read_bytes()).hexdigest()
    return {
        "manifest_version": MANIFEST_VERSION,
        "target": metadata["target"],
        "kernels": [
            {
                "name": ir["name"],
                "symbol": kernel["name"],
                "code_object": CODE_OBJECT,
                "args": merge_arguments(ir["args"], kernel["args"]),
                "kernarg_size": kernel["kernarg_size"],
                "block": kernel["block"],
                "grid": [
                    grid_expression(args.block_size, args.wavefront_size),
                    1,
                    1,
                ],
                "shared_memory_bytes": 0,
                "workspace_bytes": 0,
                "sparsewave": {
                    "operation": args.operation,
                    "mapping": args.mapping,
                    "tile_size": args.tile_size,
                    "wavefront_size": args.wavefront_size,
                    "hsaco_sha256": digest,
                    "fixed_group_segment_bytes": kernel[
                        "fixed_group_segment_bytes"
                    ],
                    "launch_n": "output_rows",
                    "fixed_output_rows": ir["output_rows"],
                },
            }
        ],
    }


def validate_manifest_kernel(kernel, hsaco_kernel, digest):
    if kernel.get("name") != SUPPORTED_OPERATION:
        fail("initial SparseWave logical kernel name must be 'spmm'")
    if kernel.get("code_object") != CODE_OBJECT:
        fail(f"kernel code_object must be '{CODE_OBJECT}'")
    if kernel.get("symbol") != hsaco_kernel["name"]:
        fail("manifest kernel symbol does not match HSACO .name")
    if kernel.get("kernarg_size") != hsaco_kernel["kernarg_size"]:
        fail("manifest kernarg_size does not match HSACO metadata")
    if kernel.get("block") != hsaco_kernel["block"]:
        fail("manifest block does not match HSACO metadata")
    if kernel.get("shared_memory_bytes") != 0:
        fail("fixed HSACO group segment must not be dynamic shared memory")
    if kernel.get("workspace_bytes") != 0:
        fail("initial SparseWave bundle does not support workspace")

    manifest_args = kernel.get("args")
    metadata_args = hsaco_kernel["args"]
    if not isinstance(manifest_args, list) or len(manifest_args) != len(
        metadata_args
    ):
        fail("manifest and HSACO kernel argument counts differ")
    for index, (manifest_arg, metadata_arg) in enumerate(
        zip(manifest_args, metadata_args)
    ):
        if manifest_arg.get("name") != SPMM_ARGUMENT_NAMES[index]:
            fail("manifest argument name does not match compiler IR contract")
        if manifest_arg.get("offset") != metadata_arg.get("offset"):
            fail("manifest argument offset does not match HSACO metadata")
        if manifest_arg.get("size") != metadata_arg.get("size"):
            fail("manifest argument size does not match HSACO metadata")
        if manifest_arg.get("type") != "ptr":
            fail("initial SparseWave bundle arguments must have type 'ptr'")
        if metadata_arg.get("value_kind") != "global_buffer":
            fail("manifest ptr argument does not match HSACO global_buffer")

    producer = kernel.get("sparsewave")
    if not isinstance(producer, dict):
        fail("kernel is missing sparsewave producer metadata")
    expected_producer = {
        "operation": SUPPORTED_OPERATION,
        "mapping": SUPPORTED_MAPPING,
        "tile_size": SUPPORTED_TILE_SIZE,
        "wavefront_size": SUPPORTED_WAVEFRONT_SIZE,
        "hsaco_sha256": digest,
        "fixed_group_segment_bytes": hsaco_kernel[
            "fixed_group_segment_bytes"
        ],
        "launch_n": "output_rows",
    }
    for field, expected in expected_producer.items():
        if producer.get(field) != expected:
            fail(f"invalid sparsewave producer field '{field}'")
    if not isinstance(producer.get("fixed_output_rows"), int) or producer[
        "fixed_output_rows"
    ] <= 0:
        fail("invalid sparsewave producer field 'fixed_output_rows'")
    expected_grid = [
        grid_expression(SUPPORTED_BLOCK_SIZE, SUPPORTED_WAVEFRONT_SIZE),
        1,
        1,
    ]
    if kernel.get("grid") != expected_grid:
        fail("manifest grid is not the supported total-work-item expression")


def verify_bundle(bundle, readobj):
    manifest_path = bundle / "manifest.json"
    hsaco = bundle / CODE_OBJECT
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        type(manifest.get("manifest_version")) is not int
        or manifest["manifest_version"] != MANIFEST_VERSION
    ):
        fail(
            "unsupported manifest_version: "
            f"{manifest.get('manifest_version')}"
        )

    metadata = inspect_hsaco(readobj, hsaco)
    if manifest.get("target") != metadata["target"]:
        fail("manifest target does not match HSACO architecture")
    if manifest["target"] != SUPPORTED_TARGET:
        fail(f"initial SparseWave bundle target must be {SUPPORTED_TARGET}")

    manifest_kernels = manifest.get("kernels")
    if not isinstance(manifest_kernels, list) or len(manifest_kernels) != len(
        metadata["kernels"]
    ):
        fail("manifest and HSACO kernel counts differ")
    digest = hashlib.sha256(hsaco.read_bytes()).hexdigest()
    metadata_by_name = {kernel["name"]: kernel for kernel in metadata["kernels"]}
    for kernel in manifest_kernels:
        symbol = kernel.get("symbol")
        if symbol not in metadata_by_name:
            fail(f"manifest kernel symbol '{symbol}' is missing from HSACO")
        validate_manifest_kernel(kernel, metadata_by_name[symbol], digest)
    return manifest


def pipeline(args):
    options = {
        "triple": args.triple,
        "chip": args.chip,
        "features": args.features,
        "abi-version": args.abi_version,
        "wavefront-size": args.wavefront_size,
        "binary-format": "bin",
        "kernel-bare-ptr-calling-convention": "true",
        "lower-host": "false",
        "sink-launch-index-computations": "true",
        "prepare-gpu-bare-ptr-abi": "true",
        "spmm-mapping": args.mapping,
        "spmm-block-size": args.block_size,
        "spmm-tile-size": args.tile_size,
    }
    if args.rocm_path:
        options["rocm-path"] = args.rocm_path
    rendered = " ".join(
        f"{key}={value}" for key, value in options.items() if value != ""
    )
    return f"builtin.module(sparsewave-to-amdgpu-pipeline{{{rendered}}})"


def build_bundle(args):
    validate_supported_options(args)
    input_text, frontend_command = prepare_bundle_input(args)
    ir = parse_fixed_spmm_ir_text(input_text)
    output = args.output.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        fail(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    command = [str(args.sparsewave_opt)]
    compile_input = None
    if frontend_command:
        command.extend(["--allow-unregistered-dialect", "-"])
        compile_input = input_text
    else:
        command.append(str(args.input))
    command.append(f"--pass-pipeline={pipeline(args)}")
    compiled = run(command, input=compile_input, text=True).stdout
    hsaco = output / CODE_OBJECT
    hsaco.write_bytes(extract_gpu_binary(compiled))
    metadata = inspect_hsaco(args.llvm_readobj, hsaco)
    manifest = manifest_for(args, hsaco, metadata, ir)
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    verify_bundle(output, args.llvm_readobj)
    return [candidate for candidate in (frontend_command, command) if candidate]


def parser():
    result = argparse.ArgumentParser(
        description="Compile shape-specialized SpMM MLIR into an lrrt bundle."
    )
    result.add_argument("input", nargs="?", type=Path)
    result.add_argument("-o", "--output", type=Path)
    result.add_argument("--verify", type=Path, metavar="BUNDLE")
    result.add_argument(
        "--sparsewave-opt",
        type=Path,
        default=find_tool("sparsewave-opt"),
    )
    result.add_argument(
        "--llvm-readobj",
        type=Path,
        default=find_tool("llvm-readobj"),
    )
    result.add_argument("--operation", default=SUPPORTED_OPERATION)
    result.add_argument("--triple", default="amdgcn-amd-amdhsa")
    result.add_argument("--chip")
    result.add_argument("--features", default="")
    result.add_argument("--abi-version", default="600")
    result.add_argument("--wavefront-size", type=int, default=32)
    result.add_argument("--rocm-path", default="")
    result.add_argument("--mapping", default=SUPPORTED_MAPPING)
    result.add_argument("--block-size", type=int, default=SUPPORTED_BLOCK_SIZE)
    result.add_argument("--tile-size", type=int, default=SUPPORTED_TILE_SIZE)
    return result


def main():
    argument_parser = parser()
    args = argument_parser.parse_args()
    try:
        if args.verify:
            verify_bundle(args.verify, args.llvm_readobj)
            return
        if args.input is None or args.output is None or not args.chip:
            argument_parser.error(
                "bundle generation requires INPUT, --output, and --chip"
            )
        commands = build_bundle(args)
        for command in commands:
            print(
                "bundle compile: " + " ".join(map(str, command)),
                file=sys.stderr,
            )
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as error:
        print(f"sparsewave-bundle: error: {error}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
