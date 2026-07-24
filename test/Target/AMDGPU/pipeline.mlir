// RUN: sparsewave-opt %s \
// RUN:   --pass-pipeline='builtin.module(sparsewave-amdgpu-pipeline{chip=gfx942 wavefront-size=64 binary-format=none})' \
// RUN:   | FileCheck %s --check-prefixes=CHECK,WAVE64
// RUN: sparsewave-opt %s \
// RUN:   --pass-pipeline='builtin.module(sparsewave-amdgpu-pipeline{chip=gfx1100 wavefront-size=32 binary-format=none})' \
// RUN:   | FileCheck %s --check-prefixes=CHECK,WAVE32
// RUN: sparsewave-opt %s \
// RUN:   --pass-pipeline='builtin.module(sparsewave-amdgpu-pipeline{triple=amdgcn-amd-amdhsa chip=gfx942 features=+xnack abi-version=500 opt-level=3 index-bitwidth=32 kernel-bare-ptr-calling-convention=true binary-format=none})' \
// RUN:   | FileCheck %s --check-prefix=OPTIONS
// RUN: sparsewave-opt --help | FileCheck %s --check-prefix=HELP
// RUN: not sparsewave-opt %s \
// RUN:   --pass-pipeline='builtin.module(sparsewave-amdgpu-pipeline{wavefront-size=16 binary-format=none})' \
// RUN:   2>&1 | FileCheck %s --check-prefix=INVALID-WAVE
// RUN: not sparsewave-opt %s \
// RUN:   --pass-pipeline='builtin.module(sparsewave-amdgpu-pipeline{binary-format=none})' \
// RUN:   2>&1 | FileCheck %s --check-prefix=MISSING-CHIP
// RUN: not sparsewave-opt %s \
// RUN:   --pass-pipeline='builtin.module(sparsewave-amdgpu-pipeline{chip=invalid binary-format=none})' \
// RUN:   2>&1 | FileCheck %s --check-prefix=INVALID-CHIP
// RUN: not sparsewave-opt %s \
// RUN:   --pass-pipeline='builtin.module(sparsewave-amdgpu-pipeline{chip=gfx942 abi-version=700 binary-format=none})' \
// RUN:   2>&1 | FileCheck %s --check-prefix=INVALID-ABI
// RUN: not sparsewave-opt %s \
// RUN:   --pass-pipeline='builtin.module(sparsewave-amdgpu-pipeline{chip=gfx942 opt-level=4 binary-format=none})' \
// RUN:   2>&1 | FileCheck %s --check-prefix=INVALID-OPT
// RUN: not sparsewave-opt %s \
// RUN:   --pass-pipeline='builtin.module(sparsewave-amdgpu-pipeline{chip=gfx942 index-bitwidth=16 binary-format=none})' \
// RUN:   2>&1 | FileCheck %s --check-prefix=INVALID-INDEX
// RUN: not sparsewave-opt %s \
// RUN:   --pass-pipeline='builtin.module(sparsewave-amdgpu-pipeline{chip=gfx942 binary-format=none spmv-block-size=128})' \
// RUN:   2>&1 | FileCheck %s --check-prefix=BACKEND-SPMV-OPTION

// HELP: --sparsewave-amdgpu-pipeline
// HELP-SAME: Lower outlined GPU kernels for an AMD GPU target.
// HELP: --sparsewave-to-amdgpu-pipeline
// HELP-SAME: Compile SparseWave programs for an AMD GPU target.

// INVALID-WAVE: for the --wavefront-size option: Cannot find option named '16'!
// MISSING-CHIP: AMDGPU target chip must be specified
// INVALID-CHIP: Invalid chipset name: invalid
// INVALID-ABI: AMDHSA code object ABI version must be 400, 500, or 600
// INVALID-OPT: AMDGPU optimization level must be between 0 and 3
// INVALID-INDEX: AMDGPU index bit width must be 32 or 64
// BACKEND-SPMV-OPTION: no such option spmv-block-size

// WAVE64-LABEL: gpu.module @kernels
// WAVE64-SAME: [#rocdl.target<chip = "gfx942">]
// WAVE32-LABEL: gpu.module @kernels
// WAVE32-SAME: [#rocdl.target<chip = "gfx1100", flags = {no_wave64}>]
// OPTIONS-LABEL: gpu.module @kernels
// OPTIONS-SAME: [#rocdl.target<O = 3, chip = "gfx942", features = "+xnack", abi = "500">]
// OPTIONS: llvm.func @kernel(%{{.*}}: !llvm.ptr)
// OPTIONS: llvm.mlir.constant(1024 : index) : i32
// OPTIONS: rocdl.workitem.id.x : i32
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
