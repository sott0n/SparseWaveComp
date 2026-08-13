// REQUIRES: amdgpu-runtime
// RUN: sparsewave-opt %s \
// RUN:   --pass-pipeline='builtin.module(sparsewave-to-amdgpu-pipeline{chip=%amdgpu_chip wavefront-size=32 rocm-path=%rocm_path row-reduction-block-size=64})' \
// RUN:   | mlir-runner \
// RUN:     --shared-libs=%mlir_rocm_runtime \
// RUN:     --shared-libs=%mlir_runner_utils \
// RUN:     --entry-point-result=void \
// RUN:   | FileCheck %s

func.func @sum(
    %rowOffsets: memref<?xi32>, %columnIndices: memref<?xi32>,
    %values: memref<?xf32>, %output: memref<?xf32>) {
  sparsewave.csr_row_reduce %rowOffsets, %columnIndices, %values, %output
      kind = "sum"
      : memref<?xi32>, memref<?xi32>, memref<?xf32>, memref<?xf32>
  return
}

func.func @max(
    %rowOffsets: memref<?xi32>, %columnIndices: memref<?xi32>,
    %values: memref<?xf32>, %output: memref<?xf32>) {
  sparsewave.csr_row_reduce %rowOffsets, %columnIndices, %values, %output
      kind = "max"
      : memref<?xi32>, memref<?xi32>, memref<?xf32>, memref<?xf32>
  return
}

// The middle CSR row is empty.
// CHECK: [3, 0, -4]
// CHECK: [2, -inf, -1]
func.func @main() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %i0 = arith.constant 0 : i32
  %i1 = arith.constant 1 : i32
  %i2 = arith.constant 2 : i32
  %i4 = arith.constant 4 : i32
  %f1 = arith.constant 1.0 : f32
  %f2 = arith.constant 2.0 : f32
  %fn3 = arith.constant -3.0 : f32
  %fn1 = arith.constant -1.0 : f32

  %rowOffsets = memref.alloc() : memref<4xi32>
  memref.store %i0, %rowOffsets[%c0] : memref<4xi32>
  memref.store %i2, %rowOffsets[%c1] : memref<4xi32>
  memref.store %i2, %rowOffsets[%c2] : memref<4xi32>
  memref.store %i4, %rowOffsets[%c3] : memref<4xi32>
  %columnIndices = memref.alloc() : memref<4xi32>
  memref.store %i0, %columnIndices[%c0] : memref<4xi32>
  memref.store %i1, %columnIndices[%c1] : memref<4xi32>
  memref.store %i0, %columnIndices[%c2] : memref<4xi32>
  memref.store %i1, %columnIndices[%c3] : memref<4xi32>
  %values = memref.alloc() : memref<4xf32>
  memref.store %f1, %values[%c0] : memref<4xf32>
  memref.store %f2, %values[%c1] : memref<4xf32>
  memref.store %fn3, %values[%c2] : memref<4xf32>
  memref.store %fn1, %values[%c3] : memref<4xf32>
  %sumOutput = memref.alloc() : memref<3xf32>
  %maxOutput = memref.alloc() : memref<3xf32>

  %offsets = memref.cast %rowOffsets : memref<4xi32> to memref<?xi32>
  %columns = memref.cast %columnIndices : memref<4xi32> to memref<?xi32>
  %input = memref.cast %values : memref<4xf32> to memref<?xf32>
  %sumResult = memref.cast %sumOutput : memref<3xf32> to memref<?xf32>
  %maxResult = memref.cast %maxOutput : memref<3xf32> to memref<?xf32>
  %offsetsUnranked = memref.cast %offsets : memref<?xi32> to memref<*xi32>
  %columnsUnranked = memref.cast %columns : memref<?xi32> to memref<*xi32>
  %inputUnranked = memref.cast %input : memref<?xf32> to memref<*xf32>
  %sumUnranked = memref.cast %sumResult : memref<?xf32> to memref<*xf32>
  %maxUnranked = memref.cast %maxResult : memref<?xf32> to memref<*xf32>

  gpu.host_register %offsetsUnranked : memref<*xi32>
  gpu.host_register %columnsUnranked : memref<*xi32>
  gpu.host_register %inputUnranked : memref<*xf32>
  gpu.host_register %sumUnranked : memref<*xf32>
  gpu.host_register %maxUnranked : memref<*xf32>
  %offsetsDevice = call @mgpuMemGetDeviceMemRef1dInt32(%offsets)
      : (memref<?xi32>) -> memref<?xi32>
  %columnsDevice = call @mgpuMemGetDeviceMemRef1dInt32(%columns)
      : (memref<?xi32>) -> memref<?xi32>
  %inputDevice = call @mgpuMemGetDeviceMemRef1dFloat(%input)
      : (memref<?xf32>) -> memref<?xf32>
  %sumDevice = call @mgpuMemGetDeviceMemRef1dFloat(%sumResult)
      : (memref<?xf32>) -> memref<?xf32>
  %maxDevice = call @mgpuMemGetDeviceMemRef1dFloat(%maxResult)
      : (memref<?xf32>) -> memref<?xf32>

  call @sum(%offsetsDevice, %columnsDevice, %inputDevice, %sumDevice)
      : (memref<?xi32>, memref<?xi32>, memref<?xf32>, memref<?xf32>) -> ()
  call @max(%offsetsDevice, %columnsDevice, %inputDevice, %maxDevice)
      : (memref<?xi32>, memref<?xi32>, memref<?xf32>, memref<?xf32>) -> ()
  call @printMemrefF32(%sumUnranked) : (memref<*xf32>) -> ()
  call @printMemrefF32(%maxUnranked) : (memref<*xf32>) -> ()
  return
}

func.func private @mgpuMemGetDeviceMemRef1dInt32(
    %buffer: memref<?xi32>) -> memref<?xi32>
func.func private @mgpuMemGetDeviceMemRef1dFloat(
    %buffer: memref<?xf32>) -> memref<?xf32>
func.func private @printMemrefF32(%buffer: memref<*xf32>)
