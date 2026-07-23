import os

import lit.formats

from lit.llvm import llvm_config


config.name = "SPARSEWAVE"
config.test_format = lit.formats.ShTest(False)
config.suffixes = [".mlir"]
config.excludes = ["CMakeLists.txt", "Inputs", "README.txt"]
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.sparsewave_obj_root, "test")

llvm_config.with_system_environment(["HOME", "INCLUDE", "LIB", "TMP", "TEMP"])
llvm_config.use_default_substitutions()
llvm_config.with_environment("PATH", config.llvm_tools_dir, append_path=True)

tool_dirs = [config.sparsewave_tools_dir, config.llvm_tools_dir]
llvm_config.add_tool_substitutions(["sparsewave-opt"], tool_dirs)

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
