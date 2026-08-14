#!/usr/bin/env python3

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys


SCHEMA_VERSION = 1


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
        if len(escaped) != 2 or not all(c in "0123456789abcdefABCDEF" for c in escaped):
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


def parse_arguments(block):
    args_match = re.search(
        r"^(?:  - |    )\.args:[ \t]*(.*)$", block, flags=re.MULTILINE
    )
    if args_match is None:
        fail("kernel metadata is missing '.args'")
    if args_match.group(1).strip() == "[]":
        return []
    start = args_match.end()
    end_match = re.search(r"^    \.[A-Za-z0-9_]+:", block[start:], re.MULTILINE)
    args_text = block[start : start + end_match.start()] if end_match else block[start:]
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
            for value in re.findall(r"^      - ([0-9]+)", required.group(1), re.MULTILINE)
        ]
    return [fields["max_flat_workgroup_size"], 1, 1]


def kernels_from_metadata(readobj_output):
    document = metadata_document(readobj_output)
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
            "kernarg_segment_align",
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
                "symbol": fields["symbol"],
                "kernarg": {
                    "size": fields["kernarg_segment_size"],
                    "alignment": fields["kernarg_segment_align"],
                    "arguments": parse_arguments(block),
                },
                "block": parse_block_size(block, fields),
                "shared_memory_bytes": fields["group_segment_fixed_size"],
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
    target = re.search(
        r"^amdhsa\.target:[ \t]*(\S+)[ \t]*$", document, re.MULTILINE
    )
    if target is None:
        fail("AMDGPU metadata is missing 'amdhsa.target'")
    return {
        "architecture": yaml_scalar(target.group(1)),
        "kernels": kernels_from_metadata(output),
    }


def manifest_for(args, hsaco, metadata):
    return {
        "schema_version": SCHEMA_VERSION,
        "hsaco": {
            "file": "kernels.hsaco",
            "sha256": hashlib.sha256(hsaco.read_bytes()).hexdigest(),
        },
        "target": {
            "triple": args.triple,
            "chip": args.chip,
            "features": args.features,
            "abi_version": args.abi_version,
            "wavefront_size": args.wavefront_size,
            "architecture": metadata["architecture"],
        },
        "compile": {
            "operation": args.operation,
            "mapping": selected_mapping(args),
            "block_size": args.block_size,
            "tile_size": args.tile_size if args.operation == "spmm" else None,
            "kernel_bare_ptr_calling_convention": True,
        },
        "kernels": metadata["kernels"],
    }


def verify_bundle(bundle, readobj):
    manifest_path = bundle / "manifest.json"
    hsaco = bundle / "kernels.hsaco"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        fail(f"unsupported manifest schema version: {manifest.get('schema_version')}")
    if manifest.get("hsaco", {}).get("file") != "kernels.hsaco":
        fail("manifest HSACO file must be 'kernels.hsaco'")
    expected_hash = hashlib.sha256(hsaco.read_bytes()).hexdigest()
    if manifest.get("hsaco", {}).get("sha256") != expected_hash:
        fail("manifest HSACO digest does not match kernels.hsaco")
    actual_metadata = inspect_hsaco(readobj, hsaco)
    if manifest.get("target", {}).get("architecture") != actual_metadata["architecture"]:
        fail("manifest target architecture does not match kernels.hsaco")
    if manifest.get("kernels") != actual_metadata["kernels"]:
        fail("manifest kernel metadata does not match kernels.hsaco")
    return manifest


def selected_mapping(args):
    if args.mapping:
        return args.mapping
    return "thread-per-output" if args.operation == "spmm" else "thread-per-row"


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
    }
    if args.rocm_path:
        options["rocm-path"] = args.rocm_path
    if args.operation == "spmm":
        options.update(
            {
                "spmm-mapping": selected_mapping(args),
                "spmm-block-size": args.block_size,
                "spmm-tile-size": args.tile_size,
            }
        )
    elif args.operation == "spmv":
        options.update(
            {
                "spmv-mapping": selected_mapping(args),
                "spmv-block-size": args.block_size,
            }
        )
    rendered = " ".join(
        f"{key}={value}" for key, value in options.items() if value != ""
    )
    return f"builtin.module(sparsewave-to-amdgpu-pipeline{{{rendered}}})"


def build_bundle(args):
    output = args.output.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        fail(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    command = [str(args.sparsewave_opt), str(args.input), f"--pass-pipeline={pipeline(args)}"]
    compiled = run(command, text=True).stdout
    hsaco = output / "kernels.hsaco"
    hsaco.write_bytes(extract_gpu_binary(compiled))
    metadata = inspect_hsaco(args.llvm_readobj, hsaco)
    manifest = manifest_for(args, hsaco, metadata)
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    verify_bundle(output, args.llvm_readobj)
    return command


def parser():
    result = argparse.ArgumentParser(
        description="Compile SparseWave MLIR into an HSACO bundle."
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
    result.add_argument("--operation", choices=("spmm", "spmv"), default="spmm")
    result.add_argument("--triple", default="amdgcn-amd-amdhsa")
    result.add_argument("--chip", required=False)
    result.add_argument("--features", default="")
    result.add_argument("--abi-version", default="600")
    result.add_argument("--wavefront-size", type=int, choices=(32, 64), default=64)
    result.add_argument("--rocm-path", default="")
    result.add_argument("--mapping")
    result.add_argument("--block-size", type=int, default=256)
    result.add_argument("--tile-size", type=int, default=4)
    return result


def main():
    args = parser().parse_args()
    try:
        if args.verify:
            verify_bundle(args.verify, args.llvm_readobj)
            return
        if args.input is None or args.output is None or not args.chip:
            parser().error("bundle generation requires INPUT, --output, and --chip")
        command = build_bundle(args)
        print("bundle compile: " + " ".join(map(str, command)), file=sys.stderr)
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as error:
        print(f"sparsewave-bundle: error: {error}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
