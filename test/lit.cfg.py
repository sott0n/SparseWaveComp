import os
import re
import shutil
import subprocess

import lit.formats

from lit.llvm import llvm_config


config.name = "SPARSEWAVE"
config.test_format = lit.formats.ShTest(False)
config.suffixes = [".mlir", ".py"]
config.excludes = ["CMakeLists.txt", "Inputs", "README.txt", "lit.cfg.py"]
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.sparsewave_obj_root, "test")

llvm_config.with_system_environment(["HOME", "INCLUDE", "LIB", "TMP", "TEMP"])
llvm_config.use_default_substitutions()
llvm_config.with_environment("PATH", config.llvm_tools_dir, append_path=True)

tool_dirs = [config.sparsewave_tools_dir, config.llvm_tools_dir]
llvm_config.add_tool_substitutions(
    [
        "mlir-translate",
        "sparsewave-bundle",
        "sparsewave-opt",
        "sparsewave-pytorch-opt",
    ],
    tool_dirs,
)

try:
    import torch

    version = torch.__version__.split("+", maxsplit=1)[0].split(".")
    if tuple(int(component) for component in version[:2]) >= (2, 13):
        config.available_features.add("pytorch-2.13")
except (ImportError, ValueError):
    pass

rocm_path = next(
    (
        os.environ[name]
        for name in ("ROCM_PATH", "ROCM_ROOT", "ROCM_HOME")
        if os.environ.get(name)
    ),
    "/opt/rocm",
)
rocm_linker = os.path.join(rocm_path, "llvm", "bin", "ld.lld")
if os.path.isfile(rocm_linker) and os.access(rocm_linker, os.X_OK):
    config.available_features.add("rocm-toolkit")
    config.substitutions.append(("%rocm_path", rocm_path))

mlir_runner = os.path.join(config.llvm_tools_dir, "mlir-runner")
rocm_runtime = os.path.join(
    config.mlir_runtime_lib_dir, "libmlir_rocm_runtime.so"
)
runner_utils = os.path.join(
    config.mlir_runtime_lib_dir, "libmlir_runner_utils.so"
)
arch_tools = [
    os.path.join(rocm_path, "bin", name)
    for name in ("amdgpu-arch", "rocm_agent_enumerator")
]
arch_tool = next(
    (
        path
        for path in arch_tools
        if os.path.isfile(path) and os.access(path, os.X_OK)
    ),
    None,
)
arch = []
if arch_tool and os.path.exists("/dev/kfd"):
    arch = subprocess.run(
        [arch_tool],
        capture_output=True,
        check=False,
        text=True,
    ).stdout.splitlines()
elif os.path.exists("/dev/kfd"):
    rocminfo = shutil.which("rocminfo")
    if rocminfo:
        output = subprocess.run(
            [rocminfo],
            capture_output=True,
            check=False,
            text=True,
        ).stdout
        arch = re.findall(r"^\s*Name:\s+(gfx[0-9a-z]+)", output, re.MULTILINE)
if (
    all(
        os.path.isfile(path)
        for path in (mlir_runner, rocm_runtime, runner_utils)
    )
    and os.path.exists("/dev/kfd")
):
    if arch:
        config.available_features.add("amdgpu-runtime")
        config.substitutions.extend(
            [
                ("%amdgpu_chip", arch[0]),
                ("%mlir_rocm_runtime", rocm_runtime),
                ("%mlir_runner_utils", runner_utils),
            ]
        )
