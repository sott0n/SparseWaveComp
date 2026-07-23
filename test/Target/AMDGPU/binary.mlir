// REQUIRES: rocm-toolkit
// RUN: sparsewave-opt %s \
// RUN:   --pass-pipeline='builtin.module(sparsewave-amdgpu-pipeline{chip=gfx942 rocm-path=%rocm_path})' \
// RUN:   | FileCheck %s

// CHECK-NOT: gpu.module
// CHECK: gpu.binary @kernels
// CHECK-SAME: [#gpu.object<#rocdl.target<chip = "gfx942">, bin = "{{.*}}">]
gpu.module @kernels {
  gpu.func @kernel() kernel {
    gpu.return
  }
}
