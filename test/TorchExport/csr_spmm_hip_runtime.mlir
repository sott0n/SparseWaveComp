// REQUIRES: pytorch-2.13, amdgpu-runtime
// RUN: %python %S/../../examples/pytorch/csr_spmm.py \
// RUN:   --runtime-mlir-output %t
// RUN: sparsewave-opt --allow-unregistered-dialect %t \
// RUN:   --pass-pipeline='builtin.module(sparsewave-to-amdgpu-pipeline{chip=%amdgpu_chip wavefront-size=32 rocm-path=%rocm_path spmm-block-size=64})' \
// RUN:   | mlir-runner \
// RUN:     --shared-libs=%mlir_rocm_runtime \
// RUN:     --shared-libs=%mlir_runner_utils \
// RUN:     --entry-point-result=void \
// RUN:   | FileCheck %s

// CHECK: [8, 11, 6, 9]
