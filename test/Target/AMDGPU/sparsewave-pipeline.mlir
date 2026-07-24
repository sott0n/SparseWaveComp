// RUN: sparsewave-opt %s \
// RUN:   --pass-pipeline='builtin.module(sparsewave-to-amdgpu-pipeline{chip=gfx942 wavefront-size=64 binary-format=none})' \
// RUN:   | FileCheck %s

// CHECK-NOT: sparsewave.spmv
// CHECK-LABEL: func.func @spmv(
// CHECK: gpu.launch_func @spmv_kernel::@spmv_kernel
// CHECK-LABEL: gpu.module @spmv_kernel
// CHECK-SAME: [#rocdl.target<chip = "gfx942">]
// CHECK: llvm.func @spmv_kernel(
// CHECK-SAME: rocdl.kernel
// CHECK: rocdl.workgroup.id.x
// CHECK: rocdl.workitem.id.x
// CHECK-NOT: scf.
func.func @spmv(
    %rowOffsets: memref<?xi32>,
    %columnIndices: memref<?xi32>,
    %values: memref<?xf32>,
    %vector: memref<?xf32>,
    %output: memref<?xf32>) {
  sparsewave.spmv %rowOffsets, %columnIndices, %values, %vector, %output
      : memref<?xi32>, memref<?xi32>, memref<?xf32>, memref<?xf32>,
        memref<?xf32>
  return
}
