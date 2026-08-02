// RUN: sparsewave-opt %s --convert-linalg-sddmm-to-sparsewave | FileCheck %s

#csr = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : dense, d1 : compressed),
  posWidth = 32,
  crdWidth = 32
}>

#sddmm = {
  indexing_maps = [
    affine_map<(i, j, k) -> (i, k)>,
    affine_map<(i, j, k) -> (k, j)>,
    affine_map<(i, j, k) -> (i, j)>,
    affine_map<(i, j, k) -> (i, j)>
  ],
  iterator_types = ["parallel", "parallel", "reduction"]
}

// CHECK-LABEL: func.func @csr_sddmm(
// CHECK-SAME: %[[SAMPLE:[^,]+]]: tensor<?x?xf32, #[[$CSR:[a-zA-Z0-9_]+]]>
// CHECK-SAME: %[[LHS:[^,]+]]: tensor<?x?xf32>
// CHECK-SAME: %[[RHS:[^)]+]]: tensor<?x?xf32>
// CHECK: %[[POSITIONS:.*]] = sparse_tensor.positions %[[SAMPLE]] {level = 1 : index}
// CHECK: %[[COORDINATES:.*]] = sparse_tensor.coordinates %[[SAMPLE]] {level = 1 : index}
// CHECK: %[[VALUES:.*]] = sparse_tensor.values %[[SAMPLE]]
// CHECK: %[[LHS_BUFFER:.*]] = bufferization.to_buffer %[[LHS]] read_only
// CHECK: %[[RHS_BUFFER:.*]] = bufferization.to_buffer %[[RHS]] read_only
// CHECK: sparsewave.sddmm %[[POSITIONS]], %[[COORDINATES]], %[[VALUES]],
// CHECK-SAME: %[[LHS_BUFFER]], %[[RHS_BUFFER]], %[[VALUES]]
// CHECK: %[[RESULT:.*]] = sparse_tensor.load %[[SAMPLE]]
// CHECK-NOT: linalg.generic
// CHECK: return %[[RESULT]]
func.func @csr_sddmm(
    %sample: tensor<?x?xf32, #csr>,
    %lhs: tensor<?x?xf32>,
    %rhs: tensor<?x?xf32>) -> tensor<?x?xf32, #csr> {
  %zero = arith.constant 0.0 : f32
  %output = linalg.fill ins(%zero : f32)
      outs(%sample : tensor<?x?xf32, #csr>)
      -> tensor<?x?xf32, #csr>
  %result = linalg.generic #sddmm
      ins(%lhs, %rhs, %sample
          : tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32, #csr>)
      outs(%output : tensor<?x?xf32, #csr>) {
    ^bb0(%lhsValue: f32, %rhsValue: f32, %sampleValue: f32,
         %sum: f32):
      %denseProduct = arith.mulf %lhsValue, %rhsValue : f32
      %weightedProduct = arith.mulf %sampleValue, %denseProduct : f32
      %next = arith.addf %sum, %weightedProduct : f32
      linalg.yield %next : f32
  } -> tensor<?x?xf32, #csr>
  return %result : tensor<?x?xf32, #csr>
}

// CHECK-LABEL: func.func @plain_reduction_is_not_rewritten(
// CHECK: linalg.generic
// CHECK-NOT: sparsewave.sddmm
func.func @plain_reduction_is_not_rewritten(
    %sample: tensor<?x?xf32, #csr>,
    %lhs: tensor<?x?xf32>,
    %rhs: tensor<?x?xf32>) -> tensor<?x?xf32, #csr> {
  %zero = arith.constant 0.0 : f32
  %output = linalg.fill ins(%zero : f32)
      outs(%sample : tensor<?x?xf32, #csr>)
      -> tensor<?x?xf32, #csr>
  %result = linalg.generic #sddmm
      ins(%lhs, %rhs, %sample
          : tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32, #csr>)
      outs(%output : tensor<?x?xf32, #csr>) {
    ^bb0(%lhsValue: f32, %rhsValue: f32, %sampleValue: f32,
         %sum: f32):
      %product = arith.mulf %lhsValue, %rhsValue : f32
      %next = arith.addf %sum, %product : f32
      linalg.yield %next : f32
  } -> tensor<?x?xf32, #csr>
  return %result : tensor<?x?xf32, #csr>
}

// CHECK-LABEL: func.func @independent_output_is_not_rewritten(
// CHECK: linalg.generic
// CHECK-NOT: sparsewave.sddmm
func.func @independent_output_is_not_rewritten(
    %sample: tensor<?x?xf32, #csr>,
    %lhs: tensor<?x?xf32>,
    %rhs: tensor<?x?xf32>,
    %outputStorage: tensor<?x?xf32, #csr>) -> tensor<?x?xf32, #csr> {
  %zero = arith.constant 0.0 : f32
  %output = linalg.fill ins(%zero : f32)
      outs(%outputStorage : tensor<?x?xf32, #csr>)
      -> tensor<?x?xf32, #csr>
  %result = linalg.generic #sddmm
      ins(%lhs, %rhs, %sample
          : tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32, #csr>)
      outs(%output : tensor<?x?xf32, #csr>) {
    ^bb0(%lhsValue: f32, %rhsValue: f32, %sampleValue: f32,
         %sum: f32):
      %denseProduct = arith.mulf %lhsValue, %rhsValue : f32
      %weightedProduct = arith.mulf %sampleValue, %denseProduct : f32
      %next = arith.addf %sum, %weightedProduct : f32
      linalg.yield %next : f32
  } -> tensor<?x?xf32, #csr>
  return %result : tensor<?x?xf32, #csr>
}

// CHECK-LABEL: func.func @dense_sample_is_not_rewritten(
// CHECK: linalg.generic
// CHECK-NOT: sparsewave.sddmm
func.func @dense_sample_is_not_rewritten(
    %sample: tensor<?x?xf32>,
    %lhs: tensor<?x?xf32>,
    %rhs: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %zero = arith.constant 0.0 : f32
  %output = linalg.fill ins(%zero : f32)
      outs(%sample : tensor<?x?xf32>) -> tensor<?x?xf32>
  %result = linalg.generic #sddmm
      ins(%lhs, %rhs, %sample
          : tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%output : tensor<?x?xf32>) {
    ^bb0(%lhsValue: f32, %rhsValue: f32, %sampleValue: f32,
         %sum: f32):
      %denseProduct = arith.mulf %lhsValue, %rhsValue : f32
      %weightedProduct = arith.mulf %sampleValue, %denseProduct : f32
      %next = arith.addf %sum, %weightedProduct : f32
      linalg.yield %next : f32
  } -> tensor<?x?xf32>
  return %result : tensor<?x?xf32>
}
