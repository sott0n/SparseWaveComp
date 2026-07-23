// REQUIRES: amdgpu-runtime
// RUN: sparsewave-opt %s \
// RUN:   --pass-pipeline='builtin.module(convert-scf-to-cf,gpu-kernel-outlining,sparsewave-amdgpu-pipeline{chip=%amdgpu_chip rocm-path=%rocm_path})' \
// RUN:   | mlir-runner \
// RUN:     --shared-libs=%mlir_rocm_runtime \
// RUN:     --shared-libs=%mlir_runner_utils \
// RUN:     --entry-point-result=void \
// RUN:   | FileCheck %s

func.func @fill(%value: f32, %output: memref<?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %size = memref.dim %output, %c0 : memref<?xf32>
  gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %c1, %grid_y = %c1,
                                      %grid_z = %c1)
             threads(%tx, %ty, %tz) in (%block_x = %size, %block_y = %c1,
                                        %block_z = %c1) {
    memref.store %value, %output[%tx] : memref<?xf32>
    gpu.terminator
  }
  return
}

// CHECK: [3, 3, 3, 3, 3]
func.func @main() {
  %buffer = memref.alloc() : memref<5xf32>
  %dynamic = memref.cast %buffer : memref<5xf32> to memref<?xf32>
  %unranked = memref.cast %dynamic : memref<?xf32> to memref<*xf32>
  gpu.host_register %unranked : memref<*xf32>
  %device = call @mgpuMemGetDeviceMemRef1dFloat(%dynamic)
      : (memref<?xf32>) -> memref<?xf32>
  %value = arith.constant 3.0 : f32
  call @fill(%value, %device) : (f32, memref<?xf32>) -> ()
  call @printMemrefF32(%unranked) : (memref<*xf32>) -> ()
  return
}

func.func private @mgpuMemGetDeviceMemRef1dFloat(
    %buffer: memref<?xf32>) -> memref<?xf32>
func.func private @printMemrefF32(%buffer: memref<*xf32>)
