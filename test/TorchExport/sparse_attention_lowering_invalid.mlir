// RUN: not sparsewave-pytorch-opt --allow-unregistered-dialect %s \
// RUN:   --convert-torch-sparse-attention-to-sparsewave 2>&1 | FileCheck %s

// CHECK: error: expected SparseAttention softmax along dimension 1
func.func @wrong_softmax_dimension(
    %mask: !torch.vtensor<[2,3],f32,#sparse_tensor.encoding<{map=(d0,d1)->(d0:dense,d1:compressed),posWidth=32,crdWidth=32}>>,
    %query: !torch.vtensor<[2,2],f32>,
    %key: !torch.vtensor<[3,2],f32>,
    %value: !torch.vtensor<[3,2],f32>) -> !torch.vtensor<[2,2],f32> {
  %c0 = "torch.constant.int"() <{value = 0 : i64}> : () -> !torch.int
  %c1 = "torch.constant.int"() <{value = 1 : i64}> : () -> !torch.int
  %key_transposed = "torch.aten.transpose.int"(%key, %c0, %c1)
      : (!torch.vtensor<[3,2],f32>, !torch.int, !torch.int)
      -> !torch.vtensor<[2,3],f32>
  %beta = "torch.constant.float"() <{value = 0.0 : f64}>
      : () -> !torch.float
  %alpha = "torch.constant.float"() <{value = 0.70710678118654746 : f64}>
      : () -> !torch.float
  %scores = "torch.operator"(%mask, %query, %key_transposed, %beta, %alpha) <{
    name = "torch.aten.sparse_sampled_addmm"
  }> : (!torch.vtensor<[2,3],f32,#sparse_tensor.encoding<{map=(d0,d1)->(d0:dense,d1:compressed),posWidth=32,crdWidth=32}>>,
        !torch.vtensor<[2,2],f32>, !torch.vtensor<[2,3],f32>, !torch.float,
        !torch.float)
      -> !torch.vtensor<[2,3],f32,#sparse_tensor.encoding<{map=(d0,d1)->(d0:dense,d1:compressed),posWidth=64,crdWidth=64}>>
  %none0 = "torch.constant.none"() : () -> !torch.none
  %none1 = "torch.constant.none"() : () -> !torch.none
  %none2 = "torch.constant.none"() : () -> !torch.none
  %coo = "torch.operator"(%scores, %none0, %none1, %none2) <{
    name = "torch.aten.to_sparse"
  }> : (!torch.vtensor<[2,3],f32,#sparse_tensor.encoding<{map=(d0,d1)->(d0:dense,d1:compressed),posWidth=64,crdWidth=64}>>,
        !torch.none, !torch.none, !torch.none)
      -> !torch.vtensor<[2,3],f32,#sparse_tensor.encoding<{map=(d0,d1)->(d0:compressed(nonunique),d1:singleton(soa)),posWidth=64,crdWidth=64}>>
  %wrong_dim = "torch.constant.int"() <{value = 0 : i64}> : () -> !torch.int
  %none3 = "torch.constant.none"() : () -> !torch.none
  %probabilities = "torch.operator"(%coo, %wrong_dim, %none3) <{
    name = "torch.aten._sparse_softmax.int"
  }> : (!torch.vtensor<[2,3],f32,#sparse_tensor.encoding<{map=(d0,d1)->(d0:compressed(nonunique),d1:singleton(soa)),posWidth=64,crdWidth=64}>>,
        !torch.int, !torch.none)
      -> !torch.vtensor<[2,3],f32,#sparse_tensor.encoding<{map=(d0,d1)->(d0:compressed(nonunique),d1:singleton(soa)),posWidth=64,crdWidth=64}>>
  %result = "torch.operator"(%probabilities, %value) <{
    name = "torch.aten._sparse_mm"
  }> : (!torch.vtensor<[2,3],f32,#sparse_tensor.encoding<{map=(d0,d1)->(d0:compressed(nonunique),d1:singleton(soa)),posWidth=64,crdWidth=64}>>,
        !torch.vtensor<[3,2],f32>) -> !torch.vtensor<[2,2],f32>
  return %result : !torch.vtensor<[2,2],f32>
}
