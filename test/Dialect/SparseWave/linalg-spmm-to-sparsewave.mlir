// RUN: sparsewave-opt %s --convert-linalg-spmm-to-sparsewave | FileCheck %s

#csr = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : dense, d1 : compressed),
  posWidth = 32,
  crdWidth = 32
}>

#bsr = #sparse_tensor.encoding<{
  map = (d0, d1) -> (
    d0 floordiv 2 : dense,
    d1 floordiv 2 : compressed,
    d0 mod 2 : dense,
    d1 mod 2 : dense
  ),
  posWidth = 32,
  crdWidth = 32
}>

#rectangular_bsr = #sparse_tensor.encoding<{
  map = (d0, d1) -> (
    d0 floordiv 2 : dense,
    d1 floordiv 4 : compressed,
    d0 mod 2 : dense,
    d1 mod 4 : dense
  ),
  posWidth = 32,
  crdWidth = 32
}>

#spmm = {
  indexing_maps = [
    affine_map<(i, j, k) -> (i, k)>,
    affine_map<(i, j, k) -> (k, j)>,
    affine_map<(i, j, k) -> (i, j)>
  ],
  iterator_types = ["parallel", "parallel", "reduction"]
}

// CHECK-LABEL: func.func @csr_spmm(
// CHECK-SAME: %[[MATRIX:[^,]+]]: tensor<?x?xf32, #[[$CSR:[a-zA-Z0-9_]+]]>, %[[RHS:[^,]+]]: tensor<?x?xf32>
// CHECK: %[[EMPTY:.*]] = tensor.empty
// CHECK: %[[POSITIONS:.*]] = sparse_tensor.positions %[[MATRIX]] {level = 1 : index}
// CHECK: %[[COORDINATES:.*]] = sparse_tensor.coordinates %[[MATRIX]] {level = 1 : index}
// CHECK: %[[VALUES:.*]] = sparse_tensor.values %[[MATRIX]]
// CHECK: %[[RHS_BUFFER:.*]] = bufferization.to_buffer %[[RHS]] read_only
// CHECK: %[[OUTPUT_BUFFER:.*]] = bufferization.to_buffer %[[EMPTY]]
// CHECK: sparsewave.spmm %[[POSITIONS]], %[[COORDINATES]], %[[VALUES]], %[[RHS_BUFFER]], %[[OUTPUT_BUFFER]]
// CHECK: %[[RESULT:.*]] = bufferization.to_tensor %[[OUTPUT_BUFFER]]
// CHECK-NOT: linalg.generic
// CHECK: return %[[RESULT]]
func.func @csr_spmm(
    %matrix: tensor<?x?xf32, #csr>,
    %rhs: tensor<?x?xf32>,
    %rows: index,
    %columns: index) -> tensor<?x?xf32> {
  %empty = tensor.empty(%rows, %columns) : tensor<?x?xf32>
  %zero = arith.constant 0.0 : f32
  %output = linalg.fill ins(%zero : f32) outs(%empty : tensor<?x?xf32>)
      -> tensor<?x?xf32>
  %result = linalg.generic #spmm
      ins(%matrix, %rhs : tensor<?x?xf32, #csr>, tensor<?x?xf32>)
      outs(%output : tensor<?x?xf32>) {
    ^bb0(%matrixValue: f32, %rhsValue: f32, %sum: f32):
      %product = arith.mulf %matrixValue, %rhsValue : f32
      %next = arith.addf %sum, %product : f32
      linalg.yield %next : f32
  } -> tensor<?x?xf32>
  return %result : tensor<?x?xf32>
}

// CHECK-LABEL: func.func @named_csr_spmm(
// CHECK: sparsewave.spmm
// CHECK-NOT: linalg.matmul
func.func @named_csr_spmm(
    %matrix: tensor<?x?xf32, #csr>,
    %rhs: tensor<?x?xf32>,
    %outputStorage: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %zero = arith.constant 0.0 : f32
  %output = linalg.fill ins(%zero : f32)
      outs(%outputStorage : tensor<?x?xf32>) -> tensor<?x?xf32>
  %result = linalg.matmul
      ins(%matrix, %rhs : tensor<?x?xf32, #csr>, tensor<?x?xf32>)
      outs(%output : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %result : tensor<?x?xf32>
}

// CHECK-LABEL: func.func @named_bsr_spmm(
// CHECK-SAME: %[[MATRIX:[^,]+]]: tensor<?x?xf32, #[[$BSR:[a-zA-Z0-9_]+]]>, %[[RHS:[^,]+]]: tensor<?x?xf32>
// CHECK: %[[POSITIONS:.*]] = sparse_tensor.positions %[[MATRIX]] {level = 1 : index}
// CHECK: %[[COORDINATES:.*]] = sparse_tensor.coordinates %[[MATRIX]] {level = 1 : index}
// CHECK: %[[VALUES:.*]] = sparse_tensor.values %[[MATRIX]]
// CHECK: %[[RHS_BUFFER:.*]] = bufferization.to_buffer %[[RHS]] read_only
// CHECK: %[[OUTPUT_BUFFER:.*]] = bufferization.to_buffer %{{.*}}
// CHECK: sparsewave.bsr_spmm %[[POSITIONS]], %[[COORDINATES]], %[[VALUES]], %[[RHS_BUFFER]], %[[OUTPUT_BUFFER]] block_size = 2
// CHECK-NOT: linalg.matmul
func.func @named_bsr_spmm(
    %matrix: tensor<?x?xf32, #bsr>,
    %rhs: tensor<?x?xf32>,
    %outputStorage: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %zero = arith.constant 0.0 : f32
  %output = linalg.fill ins(%zero : f32)
      outs(%outputStorage : tensor<?x?xf32>) -> tensor<?x?xf32>
  %result = linalg.matmul
      ins(%matrix, %rhs : tensor<?x?xf32, #bsr>, tensor<?x?xf32>)
      outs(%output : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %result : tensor<?x?xf32>
}

// Rectangular blocks do not satisfy sparsewave.bsr_spmm's initial square-block
// contract and remain available to upstream SparseTensor lowering.
// CHECK-LABEL: func.func @rectangular_bsr_is_not_rewritten(
// CHECK: linalg.matmul
func.func @rectangular_bsr_is_not_rewritten(
    %matrix: tensor<?x?xf32, #rectangular_bsr>,
    %rhs: tensor<?x?xf32>,
    %outputStorage: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %zero = arith.constant 0.0 : f32
  %output = linalg.fill ins(%zero : f32)
      outs(%outputStorage : tensor<?x?xf32>) -> tensor<?x?xf32>
  %result = linalg.matmul
      ins(%matrix, %rhs : tensor<?x?xf32, #rectangular_bsr>, tensor<?x?xf32>)
      outs(%output : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %result : tensor<?x?xf32>
}

// CHECK-LABEL: func.func @dense_lhs_is_not_rewritten(
// CHECK: linalg.matmul
func.func @dense_lhs_is_not_rewritten(
    %matrix: tensor<?x?xf32>,
    %rhs: tensor<?x?xf32>,
    %outputStorage: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %zero = arith.constant 0.0 : f32
  %output = linalg.fill ins(%zero : f32)
      outs(%outputStorage : tensor<?x?xf32>) -> tensor<?x?xf32>
  %result = linalg.matmul
      ins(%matrix, %rhs : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%output : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %result : tensor<?x?xf32>
}
