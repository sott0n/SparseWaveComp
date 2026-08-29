// RUN: sparsewave-opt %s \
// RUN:   --pass-pipeline='builtin.module(sparsewave-to-amdgpu-pipeline{chip=gfx942 wavefront-size=32 binary-format=none spmm-mapping=wave-per-position-tile spmm-block-size=64})' \
// RUN:   | FileCheck %s --check-prefixes=CHECK,WAVE32
// RUN: sparsewave-opt %s \
// RUN:   --pass-pipeline='builtin.module(sparsewave-to-amdgpu-pipeline{chip=gfx942 wavefront-size=64 binary-format=none spmm-mapping=wave-per-position-tile spmm-block-size=64})' \
// RUN:   | FileCheck %s --check-prefixes=CHECK,WAVE64

// CHECK-NOT: sparsewave.
// CHECK-LABEL: gpu.module @spmm_kernel
// CHECK-SAME: [#rocdl.target<chip = "gfx942"
// WAVE32-SAME: flags = {no_wave64}
// WAVE64-NOT: no_wave64
// CHECK: llvm.func @spmm_kernel(
// CHECK-SAME: rocdl.kernel
// CHECK: rocdl.ds_bpermute
// CHECK: llvm.atomicrmw fadd
// CHECK-NOT: scf.

func.func @spmm(
    %rowOffsets: memref<?xi32>,
    %columnIndices: memref<?xi32>,
    %values: memref<?xf32>,
    %rhs: memref<?x?xf32>,
    %output: memref<?x?xf32>) {
  sparsewave.spmm %rowOffsets, %columnIndices, %values, %rhs, %output
      : memref<?xi32>, memref<?xi32>, memref<?xf32>,
        memref<?x?xf32>, memref<?x?xf32>
  return
}
