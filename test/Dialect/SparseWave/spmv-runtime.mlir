// REQUIRES: amdgpu-runtime
// RUN: sparsewave-opt %s \
// RUN:   --pass-pipeline='builtin.module(convert-sparsewave-to-gpu,gpu-kernel-outlining,sparsewave-amdgpu-pipeline{chip=%amdgpu_chip wavefront-size=32 rocm-path=%rocm_path})' \
// RUN:   | mlir-runner \
// RUN:     --shared-libs=%mlir_rocm_runtime \
// RUN:     --shared-libs=%mlir_runner_utils \
// RUN:     --entry-point-result=void \
// RUN:   | FileCheck %s

func.func @spmv(
    %rowOffsets: memref<?xi32>,
    %columnIndices: memref<?xi32>,
    %values: memref<?xf32>,
    %vector: memref<?xf32>,
    %output: memref<?xf32>) {
  sparsewave.spmv %rowOffsets, %columnIndices, %values, %vector, %output
      : memref<?xi32>, memref<?xi32>, memref<?xf32>, memref<?xf32>,
        memref<?xf32>
  return
}

// Matrix:
// [1 0 2 0]
// [0 3 0 0]
// [4 0 0 5]
// [0 0 6 0]
//
// CHECK: [70, 60, 240, 180]
func.func @main() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c5 = arith.constant 5 : index

  %i0 = arith.constant 0 : i32
  %i1 = arith.constant 1 : i32
  %i2 = arith.constant 2 : i32
  %i3 = arith.constant 3 : i32
  %i5 = arith.constant 5 : i32
  %i6 = arith.constant 6 : i32

  %f1 = arith.constant 1.0 : f32
  %f2 = arith.constant 2.0 : f32
  %f3 = arith.constant 3.0 : f32
  %f4 = arith.constant 4.0 : f32
  %f5 = arith.constant 5.0 : f32
  %f6 = arith.constant 6.0 : f32
  %f10 = arith.constant 10.0 : f32
  %f20 = arith.constant 20.0 : f32
  %f30 = arith.constant 30.0 : f32
  %f40 = arith.constant 40.0 : f32

  %rowOffsets = memref.alloc() : memref<5xi32>
  memref.store %i0, %rowOffsets[%c0] : memref<5xi32>
  memref.store %i2, %rowOffsets[%c1] : memref<5xi32>
  memref.store %i3, %rowOffsets[%c2] : memref<5xi32>
  memref.store %i5, %rowOffsets[%c3] : memref<5xi32>
  memref.store %i6, %rowOffsets[%c4] : memref<5xi32>

  %columnIndices = memref.alloc() : memref<6xi32>
  memref.store %i0, %columnIndices[%c0] : memref<6xi32>
  memref.store %i2, %columnIndices[%c1] : memref<6xi32>
  memref.store %i1, %columnIndices[%c2] : memref<6xi32>
  memref.store %i0, %columnIndices[%c3] : memref<6xi32>
  memref.store %i3, %columnIndices[%c4] : memref<6xi32>
  memref.store %i2, %columnIndices[%c5] : memref<6xi32>

  %values = memref.alloc() : memref<6xf32>
  memref.store %f1, %values[%c0] : memref<6xf32>
  memref.store %f2, %values[%c1] : memref<6xf32>
  memref.store %f3, %values[%c2] : memref<6xf32>
  memref.store %f4, %values[%c3] : memref<6xf32>
  memref.store %f5, %values[%c4] : memref<6xf32>
  memref.store %f6, %values[%c5] : memref<6xf32>

  %vector = memref.alloc() : memref<4xf32>
  memref.store %f10, %vector[%c0] : memref<4xf32>
  memref.store %f20, %vector[%c1] : memref<4xf32>
  memref.store %f30, %vector[%c2] : memref<4xf32>
  memref.store %f40, %vector[%c3] : memref<4xf32>

  %output = memref.alloc() : memref<4xf32>

  %rowOffsetsDynamic =
      memref.cast %rowOffsets : memref<5xi32> to memref<?xi32>
  %columnIndicesDynamic =
      memref.cast %columnIndices : memref<6xi32> to memref<?xi32>
  %valuesDynamic = memref.cast %values : memref<6xf32> to memref<?xf32>
  %vectorDynamic = memref.cast %vector : memref<4xf32> to memref<?xf32>
  %outputDynamic = memref.cast %output : memref<4xf32> to memref<?xf32>

  %rowOffsetsUnranked =
      memref.cast %rowOffsetsDynamic : memref<?xi32> to memref<*xi32>
  %columnIndicesUnranked =
      memref.cast %columnIndicesDynamic : memref<?xi32> to memref<*xi32>
  %valuesUnranked =
      memref.cast %valuesDynamic : memref<?xf32> to memref<*xf32>
  %vectorUnranked =
      memref.cast %vectorDynamic : memref<?xf32> to memref<*xf32>
  %outputUnranked =
      memref.cast %outputDynamic : memref<?xf32> to memref<*xf32>

  gpu.host_register %rowOffsetsUnranked : memref<*xi32>
  gpu.host_register %columnIndicesUnranked : memref<*xi32>
  gpu.host_register %valuesUnranked : memref<*xf32>
  gpu.host_register %vectorUnranked : memref<*xf32>
  gpu.host_register %outputUnranked : memref<*xf32>

  %rowOffsetsDevice = call @mgpuMemGetDeviceMemRef1dInt32(%rowOffsetsDynamic)
      : (memref<?xi32>) -> memref<?xi32>
  %columnIndicesDevice =
      call @mgpuMemGetDeviceMemRef1dInt32(%columnIndicesDynamic)
      : (memref<?xi32>) -> memref<?xi32>
  %valuesDevice = call @mgpuMemGetDeviceMemRef1dFloat(%valuesDynamic)
      : (memref<?xf32>) -> memref<?xf32>
  %vectorDevice = call @mgpuMemGetDeviceMemRef1dFloat(%vectorDynamic)
      : (memref<?xf32>) -> memref<?xf32>
  %outputDevice = call @mgpuMemGetDeviceMemRef1dFloat(%outputDynamic)
      : (memref<?xf32>) -> memref<?xf32>

  call @spmv(%rowOffsetsDevice, %columnIndicesDevice, %valuesDevice,
             %vectorDevice, %outputDevice)
      : (memref<?xi32>, memref<?xi32>, memref<?xf32>, memref<?xf32>,
         memref<?xf32>) -> ()
  call @printMemrefF32(%outputUnranked) : (memref<*xf32>) -> ()
  return
}

func.func private @mgpuMemGetDeviceMemRef1dInt32(
    %buffer: memref<?xi32>) -> memref<?xi32>
func.func private @mgpuMemGetDeviceMemRef1dFloat(
    %buffer: memref<?xf32>) -> memref<?xf32>
func.func private @printMemrefF32(%buffer: memref<*xf32>)
