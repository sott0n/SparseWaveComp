// RUN: sparsewave-opt %s \
// RUN:   --pass-pipeline='builtin.module(sparsewave-amdgpu-pipeline{chip=gfx942 wavefront-size=64})' \
// RUN:   | FileCheck %s --check-prefixes=CHECK,WAVE64
// RUN: sparsewave-opt %s \
// RUN:   --pass-pipeline='builtin.module(sparsewave-amdgpu-pipeline{chip=gfx1100 wavefront-size=32})' \
// RUN:   | FileCheck %s --check-prefixes=CHECK,WAVE32
// RUN: sparsewave-opt --help | FileCheck %s --check-prefix=HELP
// RUN: not sparsewave-opt %s \
// RUN:   --pass-pipeline='builtin.module(sparsewave-amdgpu-pipeline{wavefront-size=16})' \
// RUN:   2>&1 | FileCheck %s --check-prefix=INVALID-WAVE
// RUN: not sparsewave-opt %s \
// RUN:   --pass-pipeline='builtin.module(sparsewave-amdgpu-pipeline)' \
// RUN:   2>&1 | FileCheck %s --check-prefix=MISSING-CHIP
// RUN: not sparsewave-opt %s \
// RUN:   --pass-pipeline='builtin.module(sparsewave-amdgpu-pipeline{chip=invalid})' \
// RUN:   2>&1 | FileCheck %s --check-prefix=INVALID-CHIP

// HELP: --sparsewave-amdgpu-pipeline
// HELP-SAME: Lower SparseWave programs for an AMD GPU target.

// INVALID-WAVE: for the --wavefront-size option: Cannot find option named '16'!
// MISSING-CHIP: AMDGPU target chip must be specified
// INVALID-CHIP: Invalid chipset name: invalid

// WAVE64-LABEL: gpu.module @kernels
// WAVE64-SAME: [#rocdl.target<chip = "gfx942">]
// WAVE32-LABEL: gpu.module @kernels
// WAVE32-SAME: [#rocdl.target<chip = "gfx1100", flags = {no_wave64}>]
// CHECK: llvm.func @kernel(
// CHECK-SAME: rocdl.kernel
// CHECK: rocdl.workitem.id.x
// CHECK-NOT: scf.
gpu.module @kernels {
  gpu.func @kernel(%output: memref<1024xi32>) kernel {
    %thread_id = gpu.thread_id x
    %thread_id_i32 = arith.index_cast %thread_id : index to i32
    %limit = arith.constant 1024 : index
    %in_bounds = arith.cmpi ult, %thread_id, %limit : index
    scf.if %in_bounds {
      memref.store %thread_id_i32, %output[%thread_id] : memref<1024xi32>
    }
    gpu.return
  }

  // CHECK-LABEL: llvm.func @permute_lane(
  // CHECK: rocdl.update.dpp
  // CHECK-NOT: amdgpu.dpp
  gpu.func @permute_lane(%old: i32, %source: i32) -> i32 {
    %result = amdgpu.dpp %old %source row_shl ( 0x1 : i32 )
        {row_mask = 0xa : i32, bound_ctrl = false} : i32
    gpu.return %result : i32
  }
}

// CHECK-LABEL: func.func @identity
// CHECK-SAME: (%[[ARG:.*]]: i32) -> i32
// CHECK-NEXT: return %[[ARG]] : i32
func.func @identity(%arg: i32) -> i32 {
  return %arg : i32
}
