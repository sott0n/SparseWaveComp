// RUN: sparsewave-opt %s \
// RUN:   --pass-pipeline='builtin.module(sparsewave-to-amdgpu-pipeline{chip=gfx942 wavefront-size=32 binary-format=none spmv-mapping=wave-per-row spmv-block-size=128})' \
// RUN:   | FileCheck %s

#csr = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : dense, d1 : compressed),
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

// CHECK-NOT: sparse_tensor
// CHECK-NOT: linalg.
// CHECK-NOT: sparsewave.
// CHECK-LABEL: func.func @csr_spmv(
// CHECK: gpu.launch_func @csr_spmv_kernel::@csr_spmv_kernel
// CHECK-NOT: sparse_tensor
// CHECK-NOT: linalg.
// CHECK-NOT: sparsewave.
// CHECK-LABEL: gpu.module @csr_spmv_kernel
// CHECK-SAME: [#rocdl.target<chip = "gfx942"
// CHECK-SAME: flags = {no_wave64}
// CHECK: llvm.func @csr_spmv_kernel(
// CHECK-SAME: rocdl.kernel
// CHECK: rocdl.ds_bpermute
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
