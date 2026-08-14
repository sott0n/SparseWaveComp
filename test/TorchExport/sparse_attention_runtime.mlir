// REQUIRES: pytorch-2.13
// RUN: %python %S/../../examples/pytorch/sparse_attention.py \
// RUN:   --runtime-mlir-output %t
// RUN: sparsewave-pytorch-opt --allow-unregistered-dialect %t \
// RUN:   --convert-torch-sparse-attention-to-sparsewave \
// RUN:   | sparsewave-opt --allow-unregistered-dialect \
// RUN:     --pass-pipeline='builtin.module(sparsewave-to-amdgpu-pipeline{chip=gfx942 wavefront-size=64 binary-format=none sddmm-block-size=64 row-reduction-block-size=64 rowwise-map-block-size=64 spmm-block-size=64})' \
// RUN:   | FileCheck %s

// CHECK-LABEL: func.func @sparse_attention(
// CHECK: gpu.launch
// CHECK-NOT: torch.
// CHECK-NOT: sparsewave_runtime.call
// CHECK-NOT: sparsewave.
// CHECK: gpu.module @sparse_attention_kernel
// CHECK: gpu.module @sparse_attention_kernel_0
// CHECK: gpu.module @sparse_attention_kernel_1
// CHECK: gpu.module @sparse_attention_kernel_2
// CHECK: gpu.module @sparse_attention_kernel_3
// CHECK: gpu.module @sparse_attention_kernel_4
// CHECK-LABEL: func.func @main()
// CHECK: call @sparse_attention(
// CHECK: call @printMemrefF32
