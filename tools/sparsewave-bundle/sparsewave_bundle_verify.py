import hashlib
import json
from pathlib import Path

from sparsewave_bundle_ir import (
    ATTENTION_BUFFER_NAMES,
    ATTENTION_KERNELS,
    SCALAR_TYPE_SIZES,
    SPMM_ARGUMENT_NAMES,
    SPARSE_ATTENTION,
    SUPPORTED_BLOCK_SIZE,
    SUPPORTED_MAPPING,
    SUPPORTED_OPERATION,
    SUPPORTED_TARGET,
    SUPPORTED_TILE_SIZE,
    SUPPORTED_WAVEFRONT_SIZE,
    fail,
)
from sparsewave_bundle_manifest import (
    CODE_OBJECT,
    MANIFEST_VERSION,
    grid_expression,
    linear_grid_expression,
)
from sparsewave_bundle_hsaco import inspect_hsaco


def validate_manifest_arguments(kernel, hsaco_kernel, expected_names):
    manifest_args = kernel.get("args")
    metadata_args = hsaco_kernel["args"]
    if not isinstance(manifest_args, list) or len(manifest_args) != len(
        metadata_args
    ):
        fail("manifest and HSACO kernel argument counts differ")
    if len(expected_names) != len(metadata_args):
        fail("manifest argument names do not match the kernel ABI contract")
    for index, (manifest_arg, metadata_arg) in enumerate(
        zip(manifest_args, metadata_args)
    ):
        if manifest_arg.get("name") != expected_names[index]:
            fail("manifest argument name does not match compiler IR contract")
        if manifest_arg.get("offset") != metadata_arg.get("offset"):
            fail("manifest argument offset does not match HSACO metadata")
        if manifest_arg.get("size") != metadata_arg.get("size"):
            fail("manifest argument size does not match HSACO metadata")
        argument_type = manifest_arg.get("type")
        if argument_type == "ptr":
            if metadata_arg.get("value_kind") != "global_buffer":
                fail("manifest ptr argument does not match HSACO global_buffer")
            if metadata_arg.get("size") != 8:
                fail("manifest ptr argument must occupy 8 bytes")
        else:
            if argument_type not in SCALAR_TYPE_SIZES:
                fail(f"unsupported manifest argument type '{argument_type}'")
            if metadata_arg.get("value_kind") != "by_value":
                fail("manifest scalar argument does not match HSACO by_value")
            if metadata_arg.get("size") != SCALAR_TYPE_SIZES[argument_type]:
                fail("manifest scalar argument has the wrong bit width")


def validate_common_kernel(kernel, hsaco_kernel, digest):
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
    producer = kernel.get("sparsewave")
    if not isinstance(producer, dict):
        fail("kernel is missing sparsewave producer metadata")
    if producer.get("hsaco_sha256") != digest:
        fail("invalid sparsewave producer field 'hsaco_sha256'")
    if producer.get("fixed_group_segment_bytes") != hsaco_kernel[
        "fixed_group_segment_bytes"
    ]:
        fail("invalid sparsewave producer field 'fixed_group_segment_bytes'")


def validate_spmm_manifest_kernel(kernel, hsaco_kernel, digest):
    if kernel.get("name") != SUPPORTED_OPERATION:
        fail("initial SparseWave logical kernel name must be 'spmm'")
    if kernel.get("code_object") != CODE_OBJECT:
        fail(f"kernel code_object must be '{CODE_OBJECT}'")
    validate_common_kernel(kernel, hsaco_kernel, digest)
    validate_manifest_arguments(kernel, hsaco_kernel, SPMM_ARGUMENT_NAMES)
    producer = kernel.get("sparsewave")
    expected_producer = {
        "operation": SUPPORTED_OPERATION,
        "mapping": SUPPORTED_MAPPING,
        "tile_size": SUPPORTED_TILE_SIZE,
        "wavefront_size": SUPPORTED_WAVEFRONT_SIZE,
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


def validate_attention_manifest(manifest, artifacts):
    producer = manifest.get("sparsewave")
    if (
        not isinstance(producer, dict)
        or producer.get("application") != SPARSE_ATTENTION
    ):
        fail("SparseAttention manifest is missing application metadata")
    if "execution_order" in manifest or "execution_order" in producer:
        fail("SparseAttention execution order belongs to the runtime Executor")
    buffers = producer.get("buffers")
    if not isinstance(buffers, list):
        fail("SparseAttention manifest is missing buffer ABI metadata")
    buffers_by_name = {buffer.get("name"): buffer for buffer in buffers}
    if len(buffers_by_name) != len(buffers):
        fail("SparseAttention buffer names must be unique")
    if set(buffers_by_name) != set(ATTENTION_BUFFER_NAMES):
        fail("SparseAttention manifest does not contain the stable buffer ABI")
    specialization = producer.get("specialization")
    required_specialization = {
        "output_rows",
        "key_value_rows",
        "head_dimension",
        "value_columns",
    }
    if (
        not isinstance(specialization, dict)
        or set(specialization) != required_specialization
    ):
        fail("SparseAttention specialization metadata is incomplete")
    if any(
        type(value) is not int or value <= 0
        for value in specialization.values()
    ):
        fail("SparseAttention specialization values must be positive integers")
    rows = specialization["output_rows"]
    key_rows = specialization["key_value_rows"]
    head_dimension = specialization["head_dimension"]
    value_columns = specialization["value_columns"]
    expected_buffers = {
        "rowOffsets": ("i32", [rows + 1], "input"),
        "columnIndices": ("i32", [None], "input"),
        "maskValues": ("fp32", [None], "input"),
        "query": ("fp32", [rows, head_dimension], "input"),
        "transposedKey": ("fp32", [head_dimension, key_rows], "input"),
        "value": ("fp32", [key_rows, value_columns], "input"),
        "scores": ("fp32", [None], "intermediate"),
        "rowMaximum": ("fp32", [rows], "intermediate"),
        "rowSum": ("fp32", [rows], "intermediate"),
        "output": ("fp32", [rows, value_columns], "output"),
    }
    for name, (dtype, shape, kind) in expected_buffers.items():
        buffer = buffers_by_name[name]
        tensor = buffer.get("tensor")
        if buffer.get("type") != "ptr" or buffer.get("kind") != kind:
            fail(f"SparseAttention buffer '{name}' has an invalid ABI role")
        if (
            not isinstance(tensor, dict)
            or tensor.get("dtype") != dtype
            or tensor.get("shape") != shape
            or tensor.get("rank") != len(shape)
            or tensor.get("specialized") != (None not in shape)
        ):
            fail(
                f"SparseAttention buffer '{name}' has invalid tensor metadata"
            )
        if None in shape and tensor.get("dynamic_dimensions") != [
            {"index": 0, "name": "nnz"}
        ]:
            fail(
                f"SparseAttention buffer '{name}' has invalid dynamic metadata"
            )

    kernels = manifest.get("kernels")
    if not isinstance(kernels, list):
        fail("manifest kernels must be an array")
    kernels_by_name = {kernel.get("name"): kernel for kernel in kernels}
    if set(kernels_by_name) != set(ATTENTION_KERNELS):
        fail("SparseAttention manifest does not contain the stable kernel set")
    if [kernel.get("name") for kernel in kernels] != sorted(ATTENTION_KERNELS):
        fail("SparseAttention kernels must be emitted deterministically")

    for name, specification in ATTENTION_KERNELS.items():
        kernel = kernels_by_name[name]
        expected_path = f"kernels/{name}.hsaco"
        if kernel.get("code_object") != expected_path:
            fail(f"kernel '{name}' has an unstable artifact path")
        artifact = artifacts[kernel["code_object"]]
        if len(artifact["metadata"]["kernels"]) != 1:
            fail(f"kernel '{name}' artifact must contain one entry point")
        hsaco_kernel = artifact["kernel"]
        validate_common_kernel(kernel, hsaco_kernel, artifact["digest"])
        validate_manifest_arguments(kernel, hsaco_kernel, specification["args"])
        if kernel.get("grid") != [
            linear_grid_expression(SUPPORTED_BLOCK_SIZE),
            1,
            1,
        ]:
            fail(f"kernel '{name}' has an invalid total-work-item grid")
        kernel_producer = kernel["sparsewave"]
        expected = {
            "application": SPARSE_ATTENTION,
            "role": specification["role"],
            "operation": specification["operation"],
            "mapping": specification["mapping"],
            "launch_n": specification["launch_n"],
            "specialization": specialization,
        }
        for field, value in expected.items():
            if kernel_producer.get(field) != value:
                fail(f"kernel '{name}' has invalid producer field '{field}'")
        for argument in kernel["args"]:
            if argument["name"] not in buffers_by_name:
                fail(f"kernel '{name}' refers to an unknown buffer")
            semantic = {
                key: argument[key]
                for key in ("name", "type", "tensor", "kind")
                if key in argument
            }
            if semantic != buffers_by_name[argument["name"]]:
                fail(f"kernel '{name}' buffer ABI metadata is inconsistent")


def verify_bundle(bundle, readobj, inspect=inspect_hsaco):
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        type(manifest.get("manifest_version")) is not int
        or manifest["manifest_version"] != MANIFEST_VERSION
    ):
        fail(
            "unsupported manifest_version: "
            f"{manifest.get('manifest_version')}"
        )

    if manifest.get("target") != SUPPORTED_TARGET:
        fail(f"initial SparseWave bundle target must be {SUPPORTED_TARGET}")
    manifest_kernels = manifest.get("kernels")
    if not isinstance(manifest_kernels, list) or not manifest_kernels:
        fail("manifest kernels must be a non-empty array")

    artifacts = {}
    logical_names = set()
    for kernel in manifest_kernels:
        if not isinstance(kernel, dict):
            fail("manifest kernel entries must be objects")
        logical_name = kernel.get("name")
        if not isinstance(logical_name, str) or not logical_name:
            fail("manifest kernel name must be a non-empty string")
        if logical_name in logical_names:
            fail(f"duplicate manifest kernel name '{logical_name}'")
        logical_names.add(logical_name)
        relative = kernel.get("code_object")
        if not isinstance(relative, str):
            fail("kernel code_object must be a relative path")
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            fail("kernel code_object must be relative to the bundle directory")
        if relative not in artifacts:
            hsaco = bundle / relative_path
            metadata = inspect(readobj, hsaco)
            if manifest["target"] != metadata["target"]:
                fail("manifest target does not match HSACO architecture")
            artifacts[relative] = {
                "metadata": metadata,
                "digest": hashlib.sha256(hsaco.read_bytes()).hexdigest(),
            }
        metadata = artifacts[relative]["metadata"]
        metadata_by_name = {
            item["name"]: item for item in metadata["kernels"]
        }
        symbol = kernel.get("symbol")
        if symbol not in metadata_by_name:
            fail(f"manifest kernel symbol '{symbol}' is missing from HSACO")
        artifacts[relative]["kernel"] = metadata_by_name[symbol]

    application = manifest.get("sparsewave", {}).get("application")
    if application == SPARSE_ATTENTION:
        validate_attention_manifest(manifest, artifacts)
    else:
        if len(manifest_kernels) != 1 or len(artifacts) != 1:
            fail("SpMM bundle must contain exactly one kernel artifact")
        kernel = manifest_kernels[0]
        artifact = artifacts[kernel["code_object"]]
        validate_spmm_manifest_kernel(
            kernel, artifact["kernel"], artifact["digest"]
        )
    return manifest
