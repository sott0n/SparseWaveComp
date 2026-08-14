#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys


sys.path.insert(0, str(Path(__file__).resolve().parent))

from sparsewave_bundle_ir import (  # noqa: E402
    ATTENTION_KERNELS,
    SPARSE_ATTENTION,
    SPMM_ARGUMENT_NAMES,
    SUPPORTED_BLOCK_SIZE,
    SUPPORTED_MAPPING,
    SUPPORTED_OPERATION,
    SUPPORTED_TARGET,
    SUPPORTED_TILE_SIZE,
    SUPPORTED_WAVEFRONT_SIZE,
    fail,
    parse_application_ir_text,
    parse_fixed_spmm_ir,
    parse_fixed_spmm_ir_text,
    parse_sparse_attention_ir_text,
)
from sparsewave_bundle_manifest import (  # noqa: E402
    CODE_OBJECT,
    manifest_for,
    sparse_attention_manifest_for,
)
import sparsewave_bundle_hsaco as bundle_hsaco  # noqa: E402
import sparsewave_bundle_verify as bundle_verify  # noqa: E402


# Keep these public for producer unit tests and existing Python users.
inspect_hsaco = bundle_hsaco.inspect_hsaco
kernels_from_metadata = bundle_hsaco.kernels_from_metadata
target_from_metadata = bundle_hsaco.target_from_metadata


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


def decode_gpu_binary(compiled_text, start):
    binary = bytearray()
    cursor = start
    while cursor < len(compiled_text) and compiled_text[cursor] != '"':
        character = compiled_text[cursor]
        if character != "\\":
            binary.extend(character.encode("utf-8"))
            cursor += 1
            continue
        if (
            cursor + 1 < len(compiled_text)
            and compiled_text[cursor + 1] == "\\"
        ):
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


def extract_gpu_binaries(compiled_text):
    binary_ops = list(
        re.finditer(
            r"\bgpu\.binary\s+@([A-Za-z_.$-][A-Za-z0-9_.$-]*)",
            compiled_text,
        )
    )
    binaries = {}
    for index, binary_op in enumerate(binary_ops):
        end = (
            binary_ops[index + 1].start()
            if index + 1 < len(binary_ops)
            else len(compiled_text)
        )
        match = re.search(
            r'\bbin\s*=\s*"', compiled_text[binary_op.end() : end]
        )
        if match is None:
            fail(f"gpu.binary @{binary_op.group(1)} has no embedded binary")
        name = binary_op.group(1)
        if name in binaries:
            fail(f"duplicate gpu.binary symbol '{name}'")
        start = binary_op.end() + match.end()
        binaries[name] = decode_gpu_binary(compiled_text, start)
    if not binaries:
        fail("compiled module contains no embedded GPU binaries")
    return binaries


def extract_gpu_binary(compiled_text):
    binaries = extract_gpu_binaries(compiled_text)
    if len(binaries) != 1:
        fail(f"expected one embedded GPU binary, found {len(binaries)}")
    return next(iter(binaries.values()))


def prepare_bundle_input(args):
    text = args.input.read_text(encoding="utf-8")
    if not re.search(r'(?:!torch\.|"torch\.)', text):
        return text, None

    if "torch.aten.sparse_sampled_addmm" in text:
        command = [
            str(args.sparsewave_pytorch_opt),
            "--allow-unregistered-dialect",
            str(args.input),
            "--convert-torch-sparse-attention-to-sparsewave",
        ]
    else:
        command = [
            str(args.sparsewave_opt),
            "--allow-unregistered-dialect",
            str(args.input),
            "--convert-torch-to-sparsewave",
        ]
    return run(command, text=True).stdout, command


def validate_supported_options(args, application=SUPPORTED_OPERATION):
    expected = {
        "chip": (args.chip, SUPPORTED_TARGET),
        "block size": (args.block_size, SUPPORTED_BLOCK_SIZE),
        "wavefront size": (args.wavefront_size, SUPPORTED_WAVEFRONT_SIZE),
    }
    if application == SUPPORTED_OPERATION:
        expected.update(
            {
                "mapping": (args.mapping, SUPPORTED_MAPPING),
                "tile size": (args.tile_size, SUPPORTED_TILE_SIZE),
            }
        )
    requested = args.operation.replace("_", "-")
    if requested not in ("auto", application):
        fail(
            f"input application is {application}, but --operation is "
            f"{args.operation}"
        )
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


def pipeline(args, application=SUPPORTED_OPERATION):
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
        "spmm-block-size": args.block_size,
    }
    if application == SPARSE_ATTENTION:
        options.update(
            {
                "spmm-mapping": "thread-per-output",
                "sddmm-block-size": args.block_size,
                "row-reduction-block-size": args.block_size,
                "rowwise-map-block-size": args.block_size,
            }
        )
    else:
        options.update(
            {
                "spmm-mapping": args.mapping,
                "spmm-tile-size": args.tile_size,
            }
        )
    if args.rocm_path:
        options["rocm-path"] = args.rocm_path
    rendered = " ".join(
        f"{key}={value}" for key, value in options.items() if value != ""
    )
    return f"builtin.module(sparsewave-to-amdgpu-pipeline{{{rendered}}})"


def verify_bundle(bundle, readobj):
    return bundle_verify.verify_bundle(bundle, readobj, inspect_hsaco)


def build_bundle(args):
    input_text, frontend_command = prepare_bundle_input(args)
    ir = parse_application_ir_text(input_text)
    application = ir["application"]
    validate_supported_options(args, application)
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
    command.append(f"--pass-pipeline={pipeline(args, application)}")
    compiled = run(command, input=compile_input, text=True).stdout
    binaries = extract_gpu_binaries(compiled)
    if application == SUPPORTED_OPERATION:
        if len(binaries) != 1:
            fail(f"expected one SpMM GPU binary, found {len(binaries)}")
        hsaco = output / CODE_OBJECT
        hsaco.write_bytes(next(iter(binaries.values())))
        metadata = inspect_hsaco(args.llvm_readobj, hsaco)
        manifest = manifest_for(args, hsaco, metadata, ir)
    else:
        if set(binaries) != set(ATTENTION_KERNELS):
            fail(
                "compiled SparseAttention GPU modules do not match the "
                "stable kernel set"
            )
        kernel_directory = output / "kernels"
        kernel_directory.mkdir()
        artifacts = {}
        for name in sorted(binaries):
            relative = f"kernels/{name}.hsaco"
            hsaco = output / relative
            hsaco.write_bytes(binaries[name])
            artifacts[name] = {
                "path": hsaco,
                "relative_path": relative,
                "metadata": inspect_hsaco(args.llvm_readobj, hsaco),
            }
        manifest = sparse_attention_manifest_for(args, artifacts, ir)
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    verify_bundle(output, args.llvm_readobj)
    return [
        candidate for candidate in (frontend_command, command) if candidate
    ]


def parser():
    result = argparse.ArgumentParser(
        description=(
            "Compile a shape-specialized SparseWave application into an "
            "lrrt bundle."
        )
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
    result.add_argument(
        "--sparsewave-pytorch-opt",
        type=Path,
        default=find_tool("sparsewave-pytorch-opt"),
    )
    result.add_argument("--operation", default="auto")
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
