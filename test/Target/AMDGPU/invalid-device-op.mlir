// RUN: not sparsewave-opt %s \
// RUN:   --pass-pipeline='builtin.module(sparsewave-amdgpu-pipeline{chip=gfx942 binary-format=none})' \
// RUN:   2>&1 | FileCheck %s

// CHECK: device lowering left illegal operation 'gpu.yield'
gpu.module @kernels {
  llvm.func @kernel() attributes {gpu.kernel} {
    gpu.yield
  }
}
