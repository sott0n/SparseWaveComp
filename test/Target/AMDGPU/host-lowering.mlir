// REQUIRES: rocm-toolkit
// RUN: sparsewave-opt %s \
// RUN:   --pass-pipeline='builtin.module(sparsewave-amdgpu-pipeline{chip=gfx942 rocm-path=%rocm_path})' \
// RUN:   | FileCheck %s
// RUN: sparsewave-opt %s \
// RUN:   --pass-pipeline='builtin.module(sparsewave-amdgpu-pipeline{chip=gfx942 rocm-path=%rocm_path})' \
// RUN:   | mlir-translate --mlir-to-llvmir \
// RUN:   | FileCheck %s --check-prefix=LLVM
// RUN: sparsewave-opt %s \
// RUN:   --pass-pipeline='builtin.module(sparsewave-amdgpu-pipeline{chip=gfx942 rocm-path=%rocm_path lower-host=false})' \
// RUN:   | FileCheck %s --check-prefix=NO-HOST

// CHECK: gpu.binary @kernels
// CHECK: llvm.func @launch()
// CHECK: gpu.launch_func @kernels::@kernel
// CHECK-SAME: blocks in
// CHECK-SAME: threads in
// CHECK-SAME: : i64
// LLVM-DAG: call ptr @mgpuModuleLoad(
// LLVM-DAG: call ptr @mgpuModuleGetFunction(
// LLVM-DAG: call void @mgpuLaunchKernel(
// LLVM-DAG: call void @mgpuStreamSynchronize(
// NO-HOST: gpu.binary @kernels
// NO-HOST: func.func @launch()
// NO-HOST: gpu.launch_func @kernels::@kernel
// NO-HOST-NOT: llvm.call @mgpu
module attributes {gpu.container_module} {
  gpu.module @kernels {
    gpu.func @kernel() kernel {
      gpu.return
    }
  }

  func.func @launch() {
    %c1 = arith.constant 1 : index
    gpu.launch_func @kernels::@kernel
        blocks in (%c1, %c1, %c1)
        threads in (%c1, %c1, %c1)
    return
  }
}
