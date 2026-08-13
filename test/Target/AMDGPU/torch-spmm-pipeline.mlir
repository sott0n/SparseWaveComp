// RUN: sparsewave-opt --allow-unregistered-dialect %s \
// RUN:   --pass-pipeline='builtin.module(sparsewave-to-amdgpu-pipeline{chip=gfx942 wavefront-size=64 binary-format=none spmm-block-size=64})' \
// RUN:   | FileCheck %s

module {
  func.func @spmm(
      %arg0: !torch.vtensor<[2,3],f32,#sparse_tensor.encoding<{map=(d0,d1)->(d0:dense,d1:compressed),posWidth=32,crdWidth=32}>>,
      %arg1: !torch.vtensor<[3,2],f32>) -> !torch.vtensor<[2,2],f32> {
    %0 = "torch.operator"(%arg0, %arg1) <{name = "torch.aten._sparse_mm"}>
        : (!torch.vtensor<[2,3],f32,#sparse_tensor.encoding<{map=(d0,d1)->(d0:dense,d1:compressed),posWidth=32,crdWidth=32}>>,
           !torch.vtensor<[3,2],f32>) -> !torch.vtensor<[2,2],f32>
    return %0 : !torch.vtensor<[2,2],f32>
  }

  func.func @run(
      %rowOffsets: memref<3xi32>, %columnIndices: memref<?xi32>,
      %values: memref<?xf32>, %rhs: memref<3x2xf32>,
      %output: memref<2x2xf32>) {
    "sparsewave_runtime.call"(%rowOffsets, %columnIndices, %values, %rhs,
                              %output) <{callee = @spmm}>
        : (memref<3xi32>, memref<?xi32>, memref<?xf32>, memref<3x2xf32>,
           memref<2x2xf32>) -> ()
    return
  }
}

// CHECK-LABEL: func.func @spmm(
// CHECK: gpu.launch
// CHECK-NOT: torch.
// CHECK-NOT: sparsewave_runtime.call
// CHECK-NOT: sparsewave.spmm
// CHECK: gpu.module @spmm_kernel
// CHECK-LABEL: func.func @run(
// CHECK: call @spmm(
