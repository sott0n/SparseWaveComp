// RUN: not sparsewave-opt %s \
// RUN:   --pass-pipeline='builtin.module(sparsewave-amdgpu-pipeline{chip=gfx942 rocm-path=%S/not-a-rocm-install})' \
// RUN:   2>&1 | FileCheck %s --check-prefix=MISSING-TOOLKIT
// RUN: not sparsewave-opt %s \
// RUN:   --pass-pipeline='builtin.module(sparsewave-amdgpu-pipeline{chip=gfx942 rocm-path=%S})' \
// RUN:   2>&1 | FileCheck %s --check-prefix=MISSING-LINKER

// MISSING-TOOLKIT: ROCm toolkit path
// MISSING-TOOLKIT-SAME: does not exist or is not a directory
// MISSING-LINKER: ROCm linker
// MISSING-LINKER-SAME: does not exist or is not executable
gpu.module @kernels {
  gpu.func @kernel() kernel {
    gpu.return
  }
}
