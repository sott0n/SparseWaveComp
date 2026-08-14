// REQUIRES: pytorch-2.13, amdgpu-runtime
// RUN: %python %S/../../examples/pytorch/sparse_attention.py \
// RUN:   --runtime-mlir-output %t
// RUN: sparsewave-pytorch-opt --allow-unregistered-dialect %t \
// RUN:   --convert-torch-sparse-attention-to-sparsewave \
// RUN:   | sparsewave-opt --allow-unregistered-dialect \
// RUN:     --pass-pipeline='builtin.module(sparsewave-to-amdgpu-pipeline{chip=%amdgpu_chip wavefront-size=32 rocm-path=%rocm_path sddmm-block-size=64 row-reduction-block-size=64 rowwise-map-block-size=64 spmm-block-size=64})' \
// RUN:   | mlir-runner \
// RUN:     --shared-libs=%mlir_rocm_runtime \
// RUN:     --shared-libs=%mlir_runner_utils \
// RUN:     --entry-point-result=void \
// RUN:   | FileCheck %s

// CHECK: [4.2177{{[0-9]*}}, 5.2177{{[0-9]*}}, 3, 4]
