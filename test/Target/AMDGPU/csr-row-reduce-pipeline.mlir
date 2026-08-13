// RUN: sparsewave-opt %s \
// RUN:   --pass-pipeline='builtin.module(sparsewave-to-amdgpu-pipeline{chip=gfx942 wavefront-size=64 binary-format=none row-reduction-block-size=64})' \
// RUN:   | FileCheck %s

// CHECK-LABEL: func.func @sum(
// CHECK: gpu.launch_func @sum_kernel::@sum_kernel
// CHECK-NOT: sparsewave.csr_row_reduce
// CHECK: gpu.module @sum_kernel
// CHECK: llvm.func @sum_kernel(
func.func @sum(
    %rowOffsets: memref<?xi32>, %columnIndices: memref<?xi32>,
    %values: memref<?xf32>, %output: memref<?xf32>) {
  sparsewave.csr_row_reduce %rowOffsets, %columnIndices, %values, %output
      kind = "sum"
      : memref<?xi32>, memref<?xi32>, memref<?xf32>, memref<?xf32>
  return
}
