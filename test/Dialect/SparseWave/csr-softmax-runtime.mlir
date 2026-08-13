// REQUIRES: amdgpu-runtime
// RUN: sparsewave-opt %s \
// RUN:   --pass-pipeline='builtin.module(sparsewave-to-amdgpu-pipeline{chip=%amdgpu_chip wavefront-size=32 rocm-path=%rocm_path row-reduction-block-size=64 rowwise-map-block-size=64})' \
// RUN:   | mlir-runner \
// RUN:     --shared-libs=%mlir_rocm_runtime \
// RUN:     --shared-libs=%mlir_runner_utils \
// RUN:     --entry-point-result=void \
// RUN:   | FileCheck %s

func.func @row_max(
    %rowOffsets: memref<?xi32>, %columnIndices: memref<?xi32>,
    %values: memref<?xf32>, %output: memref<?xf32>) {
  sparsewave.csr_row_reduce %rowOffsets, %columnIndices, %values, %output
      kind = "max"
      : memref<?xi32>, memref<?xi32>, memref<?xf32>, memref<?xf32>
  return
}

func.func @subtract_exp(
    %rowOffsets: memref<?xi32>, %columnIndices: memref<?xi32>,
    %values: memref<?xf32>, %rowMax: memref<?xf32>,
    %outputValues: memref<?xf32>) {
  sparsewave.csr_rowwise_map %rowOffsets, %columnIndices, %values, %rowMax,
      %outputValues {
    ^bb0(%value: f32, %maximum: f32):
      %shifted = arith.subf %value, %maximum : f32
      %mapped = math.exp %shifted : f32
      sparsewave.yield %mapped : f32
  } : memref<?xi32>, memref<?xi32>, memref<?xf32>, memref<?xf32>,
      memref<?xf32>
  return
}

func.func @row_sum(
    %rowOffsets: memref<?xi32>, %columnIndices: memref<?xi32>,
    %values: memref<?xf32>, %output: memref<?xf32>) {
  sparsewave.csr_row_reduce %rowOffsets, %columnIndices, %values, %output
      kind = "sum"
      : memref<?xi32>, memref<?xi32>, memref<?xf32>, memref<?xf32>
  return
}

func.func @divide(
    %rowOffsets: memref<?xi32>, %columnIndices: memref<?xi32>,
    %values: memref<?xf32>, %rowSum: memref<?xf32>,
    %outputValues: memref<?xf32>) {
  sparsewave.csr_rowwise_map %rowOffsets, %columnIndices, %values, %rowSum,
      %outputValues {
    ^bb0(%value: f32, %sum: f32):
      %mapped = arith.divf %value, %sum : f32
      sparsewave.yield %mapped : f32
  } : memref<?xi32>, memref<?xi32>, memref<?xf32>, memref<?xf32>,
      memref<?xf32>
  return
}

// The middle CSR row is empty.
// CHECK: [0.268941, 0.731059, 0.119203, 0.880797]
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
  %maxValues = memref.alloc() : memref<3xf32>
  %expValues = memref.alloc() : memref<4xf32>
  %sumValues = memref.alloc() : memref<3xf32>
  %outputValues = memref.alloc() : memref<4xf32>

  %offsets = memref.cast %rowOffsets : memref<4xi32> to memref<?xi32>
  %columns = memref.cast %columnIndices : memref<4xi32> to memref<?xi32>
  %input = memref.cast %values : memref<4xf32> to memref<?xf32>
  %max = memref.cast %maxValues : memref<3xf32> to memref<?xf32>
  %exp = memref.cast %expValues : memref<4xf32> to memref<?xf32>
  %sum = memref.cast %sumValues : memref<3xf32> to memref<?xf32>
  %output = memref.cast %outputValues : memref<4xf32> to memref<?xf32>
  %offsetsUnranked = memref.cast %offsets : memref<?xi32> to memref<*xi32>
  %columnsUnranked = memref.cast %columns : memref<?xi32> to memref<*xi32>
  %inputUnranked = memref.cast %input : memref<?xf32> to memref<*xf32>
  %maxUnranked = memref.cast %max : memref<?xf32> to memref<*xf32>
  %expUnranked = memref.cast %exp : memref<?xf32> to memref<*xf32>
  %sumUnranked = memref.cast %sum : memref<?xf32> to memref<*xf32>
  %outputUnranked = memref.cast %output : memref<?xf32> to memref<*xf32>

  gpu.host_register %offsetsUnranked : memref<*xi32>
  gpu.host_register %columnsUnranked : memref<*xi32>
  gpu.host_register %inputUnranked : memref<*xf32>
  gpu.host_register %maxUnranked : memref<*xf32>
  gpu.host_register %expUnranked : memref<*xf32>
  gpu.host_register %sumUnranked : memref<*xf32>
  gpu.host_register %outputUnranked : memref<*xf32>
  %offsetsDevice = call @mgpuMemGetDeviceMemRef1dInt32(%offsets)
      : (memref<?xi32>) -> memref<?xi32>
  %columnsDevice = call @mgpuMemGetDeviceMemRef1dInt32(%columns)
      : (memref<?xi32>) -> memref<?xi32>
  %inputDevice = call @mgpuMemGetDeviceMemRef1dFloat(%input)
      : (memref<?xf32>) -> memref<?xf32>
  %maxDevice = call @mgpuMemGetDeviceMemRef1dFloat(%max)
      : (memref<?xf32>) -> memref<?xf32>
  %expDevice = call @mgpuMemGetDeviceMemRef1dFloat(%exp)
      : (memref<?xf32>) -> memref<?xf32>
  %sumDevice = call @mgpuMemGetDeviceMemRef1dFloat(%sum)
      : (memref<?xf32>) -> memref<?xf32>
  %outputDevice = call @mgpuMemGetDeviceMemRef1dFloat(%output)
      : (memref<?xf32>) -> memref<?xf32>

  call @row_max(%offsetsDevice, %columnsDevice, %inputDevice, %maxDevice)
      : (memref<?xi32>, memref<?xi32>, memref<?xf32>, memref<?xf32>) -> ()
  call @subtract_exp(%offsetsDevice, %columnsDevice, %inputDevice, %maxDevice,
      %expDevice)
      : (memref<?xi32>, memref<?xi32>, memref<?xf32>, memref<?xf32>,
         memref<?xf32>) -> ()
  call @row_sum(%offsetsDevice, %columnsDevice, %expDevice, %sumDevice)
      : (memref<?xi32>, memref<?xi32>, memref<?xf32>, memref<?xf32>) -> ()
  call @divide(%offsetsDevice, %columnsDevice, %expDevice, %sumDevice,
      %outputDevice)
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
