// RUN: not sparsewave-opt %s \
// RUN:   --pass-pipeline='builtin.module(sparsewave-amdgpu-pipeline{chip=gfx942 binary-format=none})' \
// RUN:   2>&1 | FileCheck %s

// CHECK: device lowering left an unrealized conversion cast
gpu.module @kernels {
  llvm.func @kernel(%arg: i32) -> f32 attributes {gpu.kernel} {
    %cast = builtin.unrealized_conversion_cast %arg : i32 to f32
    llvm.return %cast : f32
  }
}
