// RUN: sparsewave-opt %s | FileCheck %s

#csr = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : dense, d1 : compressed),
  posWidth = 32,
  crdWidth = 32
}>

// CHECK: #[[$CSR:.*]] = #sparse_tensor.encoding
// CHECK-LABEL: func.func private @consume_csr
// CHECK-SAME: tensor<?x?xf32, #[[$CSR]]>
func.func private @consume_csr(tensor<?x?xf32, #csr>)

// CHECK-LABEL: gpu.module @kernels
// CHECK: gpu.func @noop() kernel
gpu.module @kernels {
  gpu.func @noop() kernel {
    gpu.return
  }
}
