// RUN: sparsewave-opt %s \
// RUN:   --pass-pipeline='builtin.module(sparsewave-amdgpu-pipeline{chip=gfx942 wavefront-size=64})' \
// RUN:   | FileCheck %s
// RUN: sparsewave-opt --help | FileCheck %s --check-prefix=HELP
// RUN: not sparsewave-opt %s \
// RUN:   --pass-pipeline='builtin.module(sparsewave-amdgpu-pipeline{wavefront-size=16})' \
// RUN:   2>&1 | FileCheck %s --check-prefix=INVALID

// HELP: --sparsewave-amdgpu-pipeline
// HELP-SAME: Lower SparseWave programs for an AMD GPU target.

// INVALID: for the --wavefront-size option: Cannot find option named '16'!

// CHECK-LABEL: func.func @identity
// CHECK-SAME: (%[[ARG:.*]]: i32) -> i32
// CHECK-NEXT: return %[[ARG]] : i32
func.func @identity(%arg: i32) -> i32 {
  return %arg : i32
}
