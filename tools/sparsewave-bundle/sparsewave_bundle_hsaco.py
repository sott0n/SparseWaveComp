import json
import re
import subprocess

from sparsewave_bundle_ir import fail


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
