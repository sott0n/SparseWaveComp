// REQUIRES: pytorch-2.13
// RUN: %python %S/../../examples/pytorch/csr_spmm.py \
// RUN:   --runtime-mlir-output %t
// RUN: sparsewave-opt --allow-unregistered-dialect %t \
// RUN:   --pass-pipeline='builtin.module(sparsewave-to-amdgpu-pipeline{chip=gfx942 wavefront-size=64 binary-format=none spmm-block-size=64})' \
// RUN:   | FileCheck %s

// CHECK-LABEL: func.func @spmm(
// CHECK: gpu.launch
// CHECK-NOT: torch.
// CHECK-NOT: sparsewave_runtime.call
// CHECK-NOT: sparsewave.spmm
// CHECK: gpu.module @spmm_kernel
// CHECK-LABEL: func.func @main()
// CHECK: call @spmm(
// CHECK: call @printMemrefF32
