// RUN: not sparsewave-opt --allow-unregistered-dialect %s \
// RUN:   --convert-torch-to-sparsewave 2>&1 | FileCheck %s

// CHECK: error: expected an i32 CSR left-hand side and dense RHS/result tensors
func.func @dense_lhs(
    %matrix: !torch.vtensor<[2,3],f32>,
    %rhs: !torch.vtensor<[3,2],f32>) -> !torch.vtensor<[2,2],f32> {
  %result = "torch.operator"(%matrix, %rhs) <{
    name = "torch.aten._sparse_mm"
  }> : (!torch.vtensor<[2,3],f32>, !torch.vtensor<[3,2],f32>)
      -> !torch.vtensor<[2,2],f32>
  return %result : !torch.vtensor<[2,2],f32>
}
