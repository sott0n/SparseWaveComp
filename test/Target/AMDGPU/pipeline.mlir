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

// HELP: --sparsewave-amdgpu-pipeline
// HELP-SAME: Lower SparseWave programs for an AMD GPU target.

// INVALID-WAVE: for the --wavefront-size option: Cannot find option named '16'!
// MISSING-CHIP: AMDGPU target chip must be specified

// WAVE64-LABEL: gpu.module @kernels
// WAVE64-SAME: [#rocdl.target<chip = "gfx942">]
// WAVE32-LABEL: gpu.module @kernels
// WAVE32-SAME: [#rocdl.target<chip = "gfx1100", flags = {no_wave64}>]
gpu.module @kernels {
  gpu.func @identity() kernel {
    gpu.return
  }
}

// CHECK-LABEL: func.func @identity
// CHECK-SAME: (%[[ARG:.*]]: i32) -> i32
// CHECK-NEXT: return %[[ARG]] : i32
func.func @identity(%arg: i32) -> i32 {
  return %arg : i32
}
