import hashlib

from sparsewave_bundle_ir import (
    SCALAR_TYPE_SIZES,
    SPARSE_ATTENTION,
    SUPPORTED_OPERATION,
    fail,
)


MANIFEST_VERSION = 1
CODE_OBJECT = "kernels.hsaco"


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
        argument_type = ir_argument["type"]
        if argument_type == "ptr":
            if metadata_argument.get("value_kind") != "global_buffer":
                fail(
                    f"IR pointer argument '{ir_argument['name']}' does not "
                    "map to an HSACO global_buffer argument"
                )
            if metadata_argument.get("size") != 8:
                fail(
                    f"IR pointer argument '{ir_argument['name']}' must have "
                    "an 8-byte HSACO entry"
                )
        else:
            expected_size = SCALAR_TYPE_SIZES.get(argument_type)
            if expected_size is None:
                fail(f"unsupported scalar ABI type '{argument_type}'")
            if metadata_argument.get("value_kind") != "by_value":
                fail(
                    f"IR scalar argument '{ir_argument['name']}' does not "
                    "map to an HSACO by_value argument"
                )
            if metadata_argument.get("size") != expected_size:
                fail(
                    f"IR scalar argument '{ir_argument['name']}' has the "
                    "wrong HSACO size"
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


def linear_grid_expression(block_size):
    return f"ceil_div(n, {block_size}) * {block_size}"


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
                    "operation": SUPPORTED_OPERATION,
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


def sparse_attention_manifest_for(args, artifacts, ir):
    kernel_entries = []
    targets = set()
    for name in sorted(ir["kernels"]):
        artifact = artifacts[name]
        hsaco = artifact["path"]
        metadata = artifact["metadata"]
        targets.add(metadata["target"])
        if len(metadata["kernels"]) != 1:
            fail(
                f"expected one kernel in {hsaco}, "
                f"found {len(metadata['kernels'])}"
            )
        hsaco_kernel = metadata["kernels"][0]
        if hsaco_kernel["name"] != name:
            fail(
                f"stable kernel name '{name}' does not match HSACO .name "
                f"'{hsaco_kernel['name']}'"
            )
        if hsaco_kernel["wavefront_size"] != args.wavefront_size:
            fail(f"kernel '{name}' does not use the requested wavefront size")
        if hsaco_kernel["block"] != [args.block_size, 1, 1]:
            fail(f"kernel '{name}' does not use the requested block size")

        specification = ir["kernels"][name]
        digest = hashlib.sha256(hsaco.read_bytes()).hexdigest()
        kernel_entries.append(
            {
                "name": name,
                "symbol": hsaco_kernel["name"],
                "code_object": artifact["relative_path"],
                "args": merge_arguments(
                    specification["args"], hsaco_kernel["args"]
                ),
                "kernarg_size": hsaco_kernel["kernarg_size"],
                "block": hsaco_kernel["block"],
                "grid": [linear_grid_expression(args.block_size), 1, 1],
                "shared_memory_bytes": 0,
                "workspace_bytes": 0,
                "sparsewave": {
                    "application": SPARSE_ATTENTION,
                    "role": specification["role"],
                    "operation": specification["operation"],
                    "mapping": specification["mapping"],
                    "launch_n": specification["launch_n"],
                    "hsaco_sha256": digest,
                    "fixed_group_segment_bytes": hsaco_kernel[
                        "fixed_group_segment_bytes"
                    ],
                    "specialization": ir["specialization"],
                },
            }
        )

    if targets != {args.chip}:
        fail(
            f"requested target {args.chip} does not match HSACO targets "
            f"{sorted(targets)}"
        )
    return {
        "manifest_version": MANIFEST_VERSION,
        "target": args.chip,
        "kernels": kernel_entries,
        "sparsewave": {
            "application": SPARSE_ATTENTION,
            "buffers": [
                ir["buffers"][name] for name in sorted(ir["buffers"])
            ],
            "specialization": ir["specialization"],
        },
    }
