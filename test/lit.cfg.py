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
