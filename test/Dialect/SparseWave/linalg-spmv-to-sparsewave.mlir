// RUN: sparsewave-opt %s --convert-linalg-spmv-to-sparsewave | FileCheck %s

#csr = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : dense, d1 : compressed),
  posWidth = 32,
  crdWidth = 32
}>

#coo = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : compressed(nonunique), d1 : singleton),
  posWidth = 32,
  crdWidth = 32
}>

#spmv = {
  indexing_maps = [
    affine_map<(i, j) -> (i, j)>,
    affine_map<(i, j) -> (j)>,
    affine_map<(i, j) -> (i)>
  ],
  iterator_types = ["parallel", "reduction"]
}

// CHECK-LABEL: func.func @csr_spmv(
// CHECK-SAME: %[[MATRIX:[^,]+]]: tensor<?x?xf32, #[[$CSR:[a-zA-Z0-9_]+]]>, %[[VECTOR:[^,]+]]: tensor<?xf32>
// CHECK: %[[EMPTY:.*]] = tensor.empty
// CHECK: %[[POSITIONS:.*]] = sparse_tensor.positions %[[MATRIX]]{{.*}}level = 1
// CHECK: %[[COORDINATES:.*]] = sparse_tensor.coordinates %[[MATRIX]]{{.*}}level = 1
// CHECK: %[[VALUES:.*]] = sparse_tensor.values %[[MATRIX]]
// CHECK: %[[VECTOR_BUFFER:.*]] = bufferization.to_buffer %[[VECTOR]] read_only
// CHECK: %[[OUTPUT_BUFFER:.*]] = bufferization.to_buffer %[[EMPTY]]
// CHECK: sparsewave.spmv %[[POSITIONS]], %[[COORDINATES]], %[[VALUES]], %[[VECTOR_BUFFER]], %[[OUTPUT_BUFFER]]
// CHECK: %[[RESULT:.*]] = bufferization.to_tensor %[[OUTPUT_BUFFER]]
// CHECK-NOT: linalg.generic
// CHECK: return %[[RESULT]]
func.func @csr_spmv(
    %matrix: tensor<?x?xf32, #csr>,
    %vector: tensor<?xf32>,
    %rows: index) -> tensor<?xf32> {
  %empty = tensor.empty(%rows) : tensor<?xf32>
  %zero = arith.constant 0.0 : f32
  %output = linalg.fill ins(%zero : f32) outs(%empty : tensor<?xf32>)
      -> tensor<?xf32>
  %result = linalg.generic #spmv
      ins(%matrix, %vector : tensor<?x?xf32, #csr>, tensor<?xf32>)
      outs(%output : tensor<?xf32>) {
    ^bb0(%matrixValue: f32, %vectorValue: f32, %sum: f32):
      %product = arith.mulf %matrixValue, %vectorValue : f32
      %next = arith.addf %sum, %product : f32
      linalg.yield %next : f32
  } -> tensor<?xf32>
  return %result : tensor<?xf32>
}

// CHECK-LABEL: func.func @named_csr_spmv(
// CHECK: sparse_tensor.positions
// CHECK: sparse_tensor.coordinates
// CHECK: sparse_tensor.values
// CHECK: sparsewave.spmv
// CHECK-NOT: linalg.matvec
func.func @named_csr_spmv(
    %matrix: tensor<?x?xf32, #csr>,
    %vector: tensor<?xf32>,
    %rows: index) -> tensor<?xf32> {
  %empty = tensor.empty(%rows) : tensor<?xf32>
  %zero = arith.constant 0.0 : f32
  %output = linalg.fill ins(%zero : f32) outs(%empty : tensor<?xf32>)
      -> tensor<?xf32>
  %result = linalg.matvec
      ins(%matrix, %vector : tensor<?x?xf32, #csr>, tensor<?xf32>)
      outs(%output : tensor<?xf32>) -> tensor<?xf32>
  return %result : tensor<?xf32>
}

// CHECK-LABEL: func.func @coo_spmv(
// CHECK-SAME: %[[COO_MATRIX:[^,]+]]: tensor<?x?xf32, #[[$COO:[a-zA-Z0-9_]+]]>, %[[COO_VECTOR:[^,]+]]: tensor<?xf32>
// CHECK: %[[COO_EMPTY:.*]] = tensor.empty
// CHECK: %[[ROWS:.*]] = sparse_tensor.coordinates %[[COO_MATRIX]]{{.*}}level = 0
// CHECK: %[[COLUMNS:.*]] = sparse_tensor.coordinates %[[COO_MATRIX]]{{.*}}level = 1
// CHECK: %[[COO_VALUES:.*]] = sparse_tensor.values %[[COO_MATRIX]]
// CHECK: %[[COO_VECTOR_BUFFER:.*]] = bufferization.to_buffer %[[COO_VECTOR]] read_only
// CHECK: %[[COO_OUTPUT_BUFFER:.*]] = bufferization.to_buffer %[[COO_EMPTY]]
// CHECK: sparsewave.coo_spmv %[[ROWS]], %[[COLUMNS]], %[[COO_VALUES]], %[[COO_VECTOR_BUFFER]], %[[COO_OUTPUT_BUFFER]]
// CHECK-NOT: linalg.generic
func.func @coo_spmv(
    %matrix: tensor<?x?xf32, #coo>,
    %vector: tensor<?xf32>,
    %rows: index) -> tensor<?xf32> {
  %empty = tensor.empty(%rows) : tensor<?xf32>
  %zero = arith.constant 0.0 : f32
  %output = linalg.fill ins(%zero : f32) outs(%empty : tensor<?xf32>)
      -> tensor<?xf32>
  %result = linalg.generic #spmv
      ins(%matrix, %vector : tensor<?x?xf32, #coo>, tensor<?xf32>)
      outs(%output : tensor<?xf32>) {
    ^bb0(%matrixValue: f32, %vectorValue: f32, %sum: f32):
      %product = arith.mulf %matrixValue, %vectorValue : f32
      %next = arith.addf %sum, %product : f32
      linalg.yield %next : f32
  } -> tensor<?xf32>
  return %result : tensor<?xf32>
}

// CHECK-LABEL: func.func @csc_is_not_csr(
// CHECK: linalg.generic
func.func @csc_is_not_csr(
    %matrix: tensor<?x?xf32, #sparse_tensor.encoding<{
      map = (d0, d1) -> (d0 : compressed, d1 : dense),
      posWidth = 32,
      crdWidth = 32
    }>>,
    %vector: tensor<?xf32>,
    %rows: index) -> tensor<?xf32> {
  %empty = tensor.empty(%rows) : tensor<?xf32>
  %zero = arith.constant 0.0 : f32
  %output = linalg.fill ins(%zero : f32) outs(%empty : tensor<?xf32>)
      -> tensor<?xf32>
  %result = linalg.generic #spmv
      ins(%matrix, %vector : tensor<?x?xf32, #sparse_tensor.encoding<{
        map = (d0, d1) -> (d0 : compressed, d1 : dense),
        posWidth = 32,
        crdWidth = 32
      }>>, tensor<?xf32>)
      outs(%output : tensor<?xf32>) {
    ^bb0(%matrixValue: f32, %vectorValue: f32, %sum: f32):
      %product = arith.mulf %matrixValue, %vectorValue : f32
      %next = arith.addf %sum, %product : f32
      linalg.yield %next : f32
  } -> tensor<?xf32>
  return %result : tensor<?xf32>
}

// CHECK-LABEL: func.func @nonzero_init_is_not_overwritten(
// CHECK: linalg.generic
func.func @nonzero_init_is_not_overwritten(
    %matrix: tensor<?x?xf32, #csr>,
    %vector: tensor<?xf32>,
    %rows: index) -> tensor<?xf32> {
  %empty = tensor.empty(%rows) : tensor<?xf32>
  %one = arith.constant 1.0 : f32
  %output = linalg.fill ins(%one : f32) outs(%empty : tensor<?xf32>)
      -> tensor<?xf32>
  %result = linalg.generic #spmv
      ins(%matrix, %vector : tensor<?x?xf32, #csr>, tensor<?xf32>)
      outs(%output : tensor<?xf32>) {
    ^bb0(%matrixValue: f32, %vectorValue: f32, %sum: f32):
      %product = arith.mulf %matrixValue, %vectorValue : f32
      %next = arith.addf %sum, %product : f32
      linalg.yield %next : f32
  } -> tensor<?xf32>
  return %result : tensor<?xf32>
}
