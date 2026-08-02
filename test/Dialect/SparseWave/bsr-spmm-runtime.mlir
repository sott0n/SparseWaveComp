// REQUIRES: amdgpu-runtime
// RUN: sparsewave-opt %s \
// RUN:   --pass-pipeline='builtin.module(sparsewave-to-amdgpu-pipeline{chip=%amdgpu_chip wavefront-size=32 rocm-path=%rocm_path spmm-block-size=64})' \
// RUN:   | mlir-runner \
// RUN:     --shared-libs=%mlir_rocm_runtime \
// RUN:     --shared-libs=%mlir_runner_utils \
// RUN:     --entry-point-result=void \
// RUN:   | FileCheck %s

func.func @bsr_spmm(
    %blockRowOffsets: memref<?xi32>,
    %blockColumnIndices: memref<?xi32>,
    %blockValues: memref<?xf32>,
    %rhs: memref<4x2xf32, strided<[2, 1]>>,
    %output: memref<4x2xf32, strided<[2, 1]>>) {
  sparsewave.bsr_spmm %blockRowOffsets, %blockColumnIndices, %blockValues,
      %rhs, %output block_size = 2
      : memref<?xi32>, memref<?xi32>, memref<?xf32>,
        memref<4x2xf32, strided<[2, 1]>>,
        memref<4x2xf32, strided<[2, 1]>>
  return
}

// BSR matrix with 2x2 blocks:
// [1 2 4 0]
// [0 3 5 6]
// [7 0 0 0]
// [0 8 0 0]
//
// CHECK: [17, 170, 45, 450, 7, 70, 16, 160]
func.func @main() {
  %blockRowOffsetsTensor = arith.constant dense<[0, 2, 3]> : tensor<3xi32>
  %blockColumnIndicesTensor = arith.constant dense<[0, 1, 0]> : tensor<3xi32>
  %blockValuesTensor = arith.constant dense<[
      1.0, 2.0, 0.0, 3.0,
      4.0, 0.0, 5.0, 6.0,
      7.0, 0.0, 0.0, 8.0]> : tensor<12xf32>
  %rhsTensor = arith.constant dense<[
      1.0, 10.0,
      2.0, 20.0,
      3.0, 30.0,
      4.0, 40.0]> : tensor<8xf32>

  %blockRowOffsets = bufferization.to_buffer %blockRowOffsetsTensor read_only
      : tensor<3xi32> to memref<3xi32>
  %blockColumnIndices =
      bufferization.to_buffer %blockColumnIndicesTensor read_only
      : tensor<3xi32> to memref<3xi32>
  %blockValues = bufferization.to_buffer %blockValuesTensor read_only
      : tensor<12xf32> to memref<12xf32>
  %rhs = bufferization.to_buffer %rhsTensor read_only
      : tensor<8xf32> to memref<8xf32>
  %output = memref.alloc() : memref<8xf32>

  %blockRowOffsetsDynamic =
      memref.cast %blockRowOffsets : memref<3xi32> to memref<?xi32>
  %blockColumnIndicesDynamic =
      memref.cast %blockColumnIndices : memref<3xi32> to memref<?xi32>
  %blockValuesDynamic =
      memref.cast %blockValues : memref<12xf32> to memref<?xf32>
  %rhsDynamic = memref.cast %rhs : memref<8xf32> to memref<?xf32>
  %outputDynamic = memref.cast %output : memref<8xf32> to memref<?xf32>

  %blockRowOffsetsUnranked =
      memref.cast %blockRowOffsetsDynamic : memref<?xi32> to memref<*xi32>
  %blockColumnIndicesUnranked =
      memref.cast %blockColumnIndicesDynamic : memref<?xi32> to memref<*xi32>
  %blockValuesUnranked =
      memref.cast %blockValuesDynamic : memref<?xf32> to memref<*xf32>
  %rhsUnranked = memref.cast %rhsDynamic : memref<?xf32> to memref<*xf32>
  %outputUnranked =
      memref.cast %outputDynamic : memref<?xf32> to memref<*xf32>

  gpu.host_register %blockRowOffsetsUnranked : memref<*xi32>
  gpu.host_register %blockColumnIndicesUnranked : memref<*xi32>
  gpu.host_register %blockValuesUnranked : memref<*xf32>
  gpu.host_register %rhsUnranked : memref<*xf32>
  gpu.host_register %outputUnranked : memref<*xf32>

  %blockRowOffsetsDevice =
      call @mgpuMemGetDeviceMemRef1dInt32(%blockRowOffsetsDynamic)
      : (memref<?xi32>) -> memref<?xi32>
  %blockColumnIndicesDevice =
      call @mgpuMemGetDeviceMemRef1dInt32(%blockColumnIndicesDynamic)
      : (memref<?xi32>) -> memref<?xi32>
  %blockValuesDevice =
      call @mgpuMemGetDeviceMemRef1dFloat(%blockValuesDynamic)
      : (memref<?xf32>) -> memref<?xf32>
  %rhsDevice = call @mgpuMemGetDeviceMemRef1dFloat(%rhsDynamic)
      : (memref<?xf32>) -> memref<?xf32>
  %outputDevice = call @mgpuMemGetDeviceMemRef1dFloat(%outputDynamic)
      : (memref<?xf32>) -> memref<?xf32>

  %rhsDevice2d = memref.reinterpret_cast %rhsDevice to
      offset: [0], sizes: [4, 2], strides: [2, 1]
      : memref<?xf32> to memref<4x2xf32, strided<[2, 1]>>
  %outputDevice2d = memref.reinterpret_cast %outputDevice to
      offset: [0], sizes: [4, 2], strides: [2, 1]
      : memref<?xf32> to memref<4x2xf32, strided<[2, 1]>>

  call @bsr_spmm(%blockRowOffsetsDevice, %blockColumnIndicesDevice,
                 %blockValuesDevice, %rhsDevice2d, %outputDevice2d)
      : (memref<?xi32>, memref<?xi32>, memref<?xf32>,
         memref<4x2xf32, strided<[2, 1]>>,
         memref<4x2xf32, strided<[2, 1]>>) -> ()
  call @printMemrefF32(%outputUnranked) : (memref<*xf32>) -> ()
  return
}

func.func private @mgpuMemGetDeviceMemRef1dInt32(
    %buffer: memref<?xi32>) -> memref<?xi32>
func.func private @mgpuMemGetDeviceMemRef1dFloat(
    %buffer: memref<?xf32>) -> memref<?xf32>
func.func private @printMemrefF32(%buffer: memref<*xf32>)
