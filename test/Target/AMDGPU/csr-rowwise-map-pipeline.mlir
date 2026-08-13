// RUN: sparsewave-opt %s \
// RUN:   --pass-pipeline='builtin.module(sparsewave-to-amdgpu-pipeline{chip=gfx942 wavefront-size=64 binary-format=none rowwise-map-block-size=64})' \
// RUN:   | FileCheck %s

// CHECK-LABEL: func.func @subtract_exp(
// CHECK: gpu.launch_func @subtract_exp_kernel::@subtract_exp_kernel
// CHECK-NOT: sparsewave.csr_rowwise_map
// CHECK: gpu.module @subtract_exp_kernel
// CHECK: llvm.func @subtract_exp_kernel(
// CHECK: llvm.intr.exp
func.func @subtract_exp(
    %rowOffsets: memref<?xi32>, %columnIndices: memref<?xi32>,
    %values: memref<?xf32>, %rowValues: memref<?xf32>,
    %outputValues: memref<?xf32>) {
  sparsewave.csr_rowwise_map %rowOffsets, %columnIndices, %values, %rowValues,
      %outputValues {
    ^bb0(%value: f32, %rowValue: f32):
      %shifted = arith.subf %value, %rowValue : f32
      %mapped = math.exp %shifted : f32
      sparsewave.yield %mapped : f32
  } : memref<?xi32>, memref<?xi32>, memref<?xf32>, memref<?xf32>,
      memref<?xf32>
  return
}
