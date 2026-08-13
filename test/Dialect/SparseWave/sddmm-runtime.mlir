// REQUIRES: amdgpu-runtime
// RUN: sparsewave-opt %s \
// RUN:   --pass-pipeline='builtin.module(sparsewave-to-amdgpu-pipeline{chip=%amdgpu_chip wavefront-size=32 rocm-path=%rocm_path sddmm-block-size=64})' \
// RUN:   | mlir-runner \
// RUN:     --shared-libs=%mlir_rocm_runtime \
// RUN:     --shared-libs=%mlir_runner_utils \
// RUN:     --entry-point-result=void \
// RUN:   | FileCheck %s

func.func @sddmm(
    %rowOffsets: memref<?xi32>,
    %columnIndices: memref<?xi32>,
    %values: memref<?xf32>,
    %lhs: memref<2x2xf32, strided<[2, 1]>>,
    %rhs: memref<2x3xf32, strided<[3, 1]>>,
    %outputValues: memref<?xf32>) {
  sparsewave.sddmm %rowOffsets, %columnIndices, %values, %lhs, %rhs,
      %outputValues {
    ^bb0(%sample: f32, %dot: f32):
      %weighted = arith.mulf %sample, %dot : f32
      sparsewave.yield %weighted : f32
  }
      : memref<?xi32>, memref<?xi32>, memref<?xf32>,
        memref<2x2xf32, strided<[2, 1]>>,
        memref<2x3xf32, strided<[3, 1]>>, memref<?xf32>
  return
}

// Sparse samples:
// [2 0 3]
// [0 4 0]
//
// Dense left-hand side:
// [1 2]
// [3 4]
//
// Dense right-hand side:
// [5 6  7]
// [8 9 10]
//
// CHECK: [42, 81, 216]
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

  %f1 = arith.constant 1.0 : f32
  %f2 = arith.constant 2.0 : f32
  %f3 = arith.constant 3.0 : f32
  %f4 = arith.constant 4.0 : f32
  %f5 = arith.constant 5.0 : f32
  %f6 = arith.constant 6.0 : f32
  %f7 = arith.constant 7.0 : f32
  %f8 = arith.constant 8.0 : f32
  %f9 = arith.constant 9.0 : f32
  %f10 = arith.constant 10.0 : f32

  %rowOffsets = memref.alloc() : memref<3xi32>
  memref.store %i0, %rowOffsets[%c0] : memref<3xi32>
  memref.store %i2, %rowOffsets[%c1] : memref<3xi32>
  memref.store %i3, %rowOffsets[%c2] : memref<3xi32>

  %columnIndices = memref.alloc() : memref<3xi32>
  memref.store %i0, %columnIndices[%c0] : memref<3xi32>
  memref.store %i2, %columnIndices[%c1] : memref<3xi32>
  memref.store %i1, %columnIndices[%c2] : memref<3xi32>

  %values = memref.alloc() : memref<3xf32>
  memref.store %f2, %values[%c0] : memref<3xf32>
  memref.store %f3, %values[%c1] : memref<3xf32>
  memref.store %f4, %values[%c2] : memref<3xf32>

  %lhs = memref.alloc() : memref<4xf32>
  memref.store %f1, %lhs[%c0] : memref<4xf32>
  memref.store %f2, %lhs[%c1] : memref<4xf32>
  memref.store %f3, %lhs[%c2] : memref<4xf32>
  memref.store %f4, %lhs[%c3] : memref<4xf32>

  %rhs = memref.alloc() : memref<6xf32>
  memref.store %f5, %rhs[%c0] : memref<6xf32>
  memref.store %f6, %rhs[%c1] : memref<6xf32>
  memref.store %f7, %rhs[%c2] : memref<6xf32>
  memref.store %f8, %rhs[%c3] : memref<6xf32>
  memref.store %f9, %rhs[%c4] : memref<6xf32>
  memref.store %f10, %rhs[%c5] : memref<6xf32>

  %outputValues = memref.alloc() : memref<3xf32>

  %rowOffsetsDynamic =
      memref.cast %rowOffsets : memref<3xi32> to memref<?xi32>
  %columnIndicesDynamic =
      memref.cast %columnIndices : memref<3xi32> to memref<?xi32>
  %valuesDynamic = memref.cast %values : memref<3xf32> to memref<?xf32>
  %lhsDynamic = memref.cast %lhs : memref<4xf32> to memref<?xf32>
  %rhsDynamic = memref.cast %rhs : memref<6xf32> to memref<?xf32>
  %outputValuesDynamic =
      memref.cast %outputValues : memref<3xf32> to memref<?xf32>

  %rowOffsetsUnranked =
      memref.cast %rowOffsetsDynamic : memref<?xi32> to memref<*xi32>
  %columnIndicesUnranked =
      memref.cast %columnIndicesDynamic : memref<?xi32> to memref<*xi32>
  %valuesUnranked =
      memref.cast %valuesDynamic : memref<?xf32> to memref<*xf32>
  %lhsUnranked = memref.cast %lhsDynamic : memref<?xf32> to memref<*xf32>
  %rhsUnranked = memref.cast %rhsDynamic : memref<?xf32> to memref<*xf32>
  %outputValuesUnranked =
      memref.cast %outputValuesDynamic : memref<?xf32> to memref<*xf32>

  gpu.host_register %rowOffsetsUnranked : memref<*xi32>
  gpu.host_register %columnIndicesUnranked : memref<*xi32>
  gpu.host_register %valuesUnranked : memref<*xf32>
  gpu.host_register %lhsUnranked : memref<*xf32>
  gpu.host_register %rhsUnranked : memref<*xf32>
  gpu.host_register %outputValuesUnranked : memref<*xf32>

  %rowOffsetsDevice = call @mgpuMemGetDeviceMemRef1dInt32(%rowOffsetsDynamic)
      : (memref<?xi32>) -> memref<?xi32>
  %columnIndicesDevice =
      call @mgpuMemGetDeviceMemRef1dInt32(%columnIndicesDynamic)
      : (memref<?xi32>) -> memref<?xi32>
  %valuesDevice = call @mgpuMemGetDeviceMemRef1dFloat(%valuesDynamic)
      : (memref<?xf32>) -> memref<?xf32>
  %lhsDevice = call @mgpuMemGetDeviceMemRef1dFloat(%lhsDynamic)
      : (memref<?xf32>) -> memref<?xf32>
  %rhsDevice = call @mgpuMemGetDeviceMemRef1dFloat(%rhsDynamic)
      : (memref<?xf32>) -> memref<?xf32>
  %outputValuesDevice =
      call @mgpuMemGetDeviceMemRef1dFloat(%outputValuesDynamic)
      : (memref<?xf32>) -> memref<?xf32>

  %lhsDevice2d = memref.reinterpret_cast %lhsDevice to
      offset: [0], sizes: [2, 2], strides: [2, 1]
      : memref<?xf32> to memref<2x2xf32, strided<[2, 1]>>
  %rhsDevice2d = memref.reinterpret_cast %rhsDevice to
      offset: [0], sizes: [2, 3], strides: [3, 1]
      : memref<?xf32> to memref<2x3xf32, strided<[3, 1]>>

  call @sddmm(%rowOffsetsDevice, %columnIndicesDevice, %valuesDevice,
              %lhsDevice2d, %rhsDevice2d, %outputValuesDevice)
      : (memref<?xi32>, memref<?xi32>, memref<?xf32>,
         memref<2x2xf32, strided<[2, 1]>>,
         memref<2x3xf32, strided<[3, 1]>>, memref<?xf32>) -> ()
  call @printMemrefF32(%outputValuesUnranked) : (memref<*xf32>) -> ()
  return
}

func.func private @mgpuMemGetDeviceMemRef1dInt32(
    %buffer: memref<?xi32>) -> memref<?xi32>
func.func private @mgpuMemGetDeviceMemRef1dFloat(
    %buffer: memref<?xf32>) -> memref<?xf32>
func.func private @printMemrefF32(%buffer: memref<*xf32>)
