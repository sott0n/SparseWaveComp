// REQUIRES: amdgpu-runtime
// RUN: sparsewave-opt %s \
// RUN:   --pass-pipeline='builtin.module(sparsewave-to-amdgpu-pipeline{chip=%amdgpu_chip wavefront-size=32 rocm-path=%rocm_path spmm-block-size=64})' \
// RUN:   | mlir-runner \
// RUN:     --shared-libs=%mlir_rocm_runtime \
// RUN:     --shared-libs=%mlir_runner_utils \
// RUN:     --entry-point-result=void \
// RUN:   | FileCheck %s

#bsr = #sparse_tensor.encoding<{
  map = (d0, d1) -> (
    d0 floordiv 2 : dense,
    d1 floordiv 2 : compressed,
    d0 mod 2 : dense,
    d1 mod 2 : dense
  ),
  posWidth = 32,
  crdWidth = 32
}>

func.func @spmm(
    %matrix: tensor<4x4xf32, #bsr>,
    %rhs: tensor<4x2xf32>,
    %output: tensor<4x2xf32>) -> tensor<4x2xf32> {
  %zero = arith.constant 0.0 : f32
  %initialized = linalg.fill ins(%zero : f32)
      outs(%output : tensor<4x2xf32>) -> tensor<4x2xf32>
  %result = linalg.matmul
      ins(%matrix, %rhs : tensor<4x4xf32, #bsr>, tensor<4x2xf32>)
      outs(%initialized : tensor<4x2xf32>) -> tensor<4x2xf32>
  return %result : tensor<4x2xf32>
}

// BSR matrix with 2x2 blocks:
// [1 2 4 0]
// [0 3 5 6]
// [7 0 0 0]
// [0 8 0 0]
//
// CHECK: [17, 170]
// CHECK-NEXT: [45, 450]
// CHECK-NEXT: [7, 70]
// CHECK-NEXT: [16, 160]
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

  %blockRowOffsetsDeviceTensor =
      bufferization.to_tensor %blockRowOffsetsDevice restrict
      : memref<?xi32> to tensor<?xi32>
  %blockColumnIndicesDeviceTensor =
      bufferization.to_tensor %blockColumnIndicesDevice restrict
      : memref<?xi32> to tensor<?xi32>
  %blockValuesDeviceTensor =
      bufferization.to_tensor %blockValuesDevice restrict
      : memref<?xf32> to tensor<?xf32>
  %rhsDeviceTensor = bufferization.to_tensor %rhsDevice2d restrict
      : memref<4x2xf32, strided<[2, 1]>> to tensor<4x2xf32>
  %outputDeviceTensor =
      bufferization.to_tensor %outputDevice2d restrict writable
      : memref<4x2xf32, strided<[2, 1]>> to tensor<4x2xf32>
  %matrix = sparse_tensor.assemble
      (%blockRowOffsetsDeviceTensor, %blockColumnIndicesDeviceTensor),
      %blockValuesDeviceTensor
      : (tensor<?xi32>, tensor<?xi32>), tensor<?xf32>
        to tensor<4x4xf32, #bsr>

  %result = call @spmm(%matrix, %rhsDeviceTensor, %outputDeviceTensor)
      : (tensor<4x4xf32, #bsr>, tensor<4x2xf32>, tensor<4x2xf32>)
        -> tensor<4x2xf32>
  %resultBuffer = bufferization.to_buffer %result
      : tensor<4x2xf32> to memref<4x2xf32>
  %resultUnranked =
      memref.cast %resultBuffer : memref<4x2xf32> to memref<*xf32>
  call @printMemrefF32(%resultUnranked) : (memref<*xf32>) -> ()
  return
}

func.func private @mgpuMemGetDeviceMemRef1dInt32(
    %buffer: memref<?xi32>) -> memref<?xi32>
func.func private @mgpuMemGetDeviceMemRef1dFloat(
    %buffer: memref<?xf32>) -> memref<?xf32>
func.func private @printMemrefF32(%buffer: memref<*xf32>)
