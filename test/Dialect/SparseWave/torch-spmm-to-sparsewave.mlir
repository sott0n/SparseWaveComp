// RUN: sparsewave-opt --allow-unregistered-dialect %s \
// RUN:   --convert-torch-to-sparsewave | FileCheck %s

// CHECK-LABEL: func.func @spmm(
// CHECK-SAME: %[[POSITIONS:.*]]: memref<3xi32>,
// CHECK-SAME: %[[COORDINATES:.*]]: memref<?xi32>,
// CHECK-SAME: %[[VALUES:.*]]: memref<?xf32>,
// CHECK-SAME: %[[RHS:.*]]: memref<3x2xf32>,
// CHECK-SAME: %[[OUTPUT:.*]]: memref<2x2xf32>)
// CHECK: sparsewave.spmm %[[POSITIONS]], %[[COORDINATES]], %[[VALUES]], %[[RHS]], %[[OUTPUT]]
// CHECK-SAME: memref<3xi32>, memref<?xi32>, memref<?xf32>, memref<3x2xf32>, memref<2x2xf32>
// CHECK-NEXT: return
// CHECK-NOT: torch.
func.func @spmm(
    %matrix: !torch.vtensor<[2,3],f32,#sparse_tensor.encoding<{map=(d0,d1)->(d0:dense,d1:compressed),posWidth=32,crdWidth=32}>>,
    %rhs: !torch.vtensor<[3,2],f32>) -> !torch.vtensor<[2,2],f32> {
  %result = "torch.operator"(%matrix, %rhs) <{
    name = "torch.aten._sparse_mm"
  }> : (!torch.vtensor<[2,3],f32,#sparse_tensor.encoding<{map=(d0,d1)->(d0:dense,d1:compressed),posWidth=32,crdWidth=32}>>, !torch.vtensor<[3,2],f32>)
      -> !torch.vtensor<[2,2],f32>
  return %result : !torch.vtensor<[2,2],f32>
}

// CHECK-LABEL: func.func @run(
// CHECK: call @spmm(
// CHECK-NOT: sparsewave_runtime.call
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
