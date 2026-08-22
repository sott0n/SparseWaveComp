// REQUIRES: amdgpu-runtime
// RUN: sparsewave-opt %s \
// RUN:   --pass-pipeline='builtin.module(sparsewave-to-amdgpu-pipeline{chip=%amdgpu_chip wavefront-size=32 rocm-path=%rocm_path spmm-mapping=thread-per-output spmm-block-size=64})' \
// RUN:   | mlir-runner \
// RUN:     --shared-libs=%mlir_rocm_runtime \
// RUN:     --shared-libs=%mlir_runner_utils \
// RUN:     --entry-point-result=void \
// RUN:   | FileCheck %s
// RUN: sparsewave-opt %s \
// RUN:   --pass-pipeline='builtin.module(sparsewave-to-amdgpu-pipeline{chip=%amdgpu_chip wavefront-size=32 rocm-path=%rocm_path spmm-mapping=wave-per-row-tile spmm-block-size=64 spmm-tile-size=2})' \
// RUN:   | mlir-runner \
// RUN:     --shared-libs=%mlir_rocm_runtime \
// RUN:     --shared-libs=%mlir_runner_utils \
// RUN:     --entry-point-result=void \
// RUN:   | FileCheck %s
// RUN: sparsewave-opt %s \
// RUN:   --pass-pipeline='builtin.module(sparsewave-to-amdgpu-pipeline{chip=%amdgpu_chip wavefront-size=32 rocm-path=%rocm_path spmm-mapping=wave-per-row-tile spmm-block-size=64 spmm-tile-size=4})' \
// RUN:   | mlir-runner \
// RUN:     --shared-libs=%mlir_rocm_runtime \
// RUN:     --shared-libs=%mlir_runner_utils \
// RUN:     --entry-point-result=void \
// RUN:   | FileCheck %s
// RUN: sparsewave-opt %s \
// RUN:   --pass-pipeline='builtin.module(sparsewave-to-amdgpu-pipeline{chip=%amdgpu_chip wavefront-size=32 rocm-path=%rocm_path spmm-mapping=thread-per-position spmm-block-size=64 spmm-position-chunk-size=4})' \
// RUN:   | mlir-runner \
// RUN:     --shared-libs=%mlir_rocm_runtime \
// RUN:     --shared-libs=%mlir_runner_utils \
// RUN:     --entry-point-result=void \
// RUN:   | FileCheck %s

#csr = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : dense, d1 : compressed),
  posWidth = 32,
  crdWidth = 32
}>

func.func @spmm(
    %matrix: tensor<2x3xf32, #csr>,
    %rhs: tensor<?x?xf32>,
    %output: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %zero = arith.constant 0.0 : f32
  %initialized = linalg.fill ins(%zero : f32)
      outs(%output : tensor<?x?xf32>) -> tensor<?x?xf32>
  %result = linalg.matmul
      ins(%matrix, %rhs : tensor<2x3xf32, #csr>, tensor<?x?xf32>)
      outs(%initialized : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %result : tensor<?x?xf32>
}

// Sparse matrix:
// [1 0 2]
// [0 3 4]
//
// Dense right-hand side:
// [1 10]
// [2 20]
// [3 30]
//
// CHECK: [7, 70]
// CHECK-NEXT: [18, 180]
func.func @main() {
  %rowOffsetsTensor = arith.constant dense<[0, 2, 4]> : tensor<3xi32>
  %columnIndicesTensor = arith.constant dense<[0, 2, 1, 2]> : tensor<4xi32>
  %valuesTensor = arith.constant dense<[1.0, 2.0, 3.0, 4.0]>
      : tensor<4xf32>
  %rhsTensor = arith.constant dense<[1.0, 10.0, 2.0, 20.0, 3.0, 30.0]>
      : tensor<6xf32>

  %rowOffsets = bufferization.to_buffer %rowOffsetsTensor read_only
      : tensor<3xi32> to memref<3xi32>
  %columnIndices = bufferization.to_buffer %columnIndicesTensor read_only
      : tensor<4xi32> to memref<4xi32>
  %values = bufferization.to_buffer %valuesTensor read_only
      : tensor<4xf32> to memref<4xf32>
  %rhs = bufferization.to_buffer %rhsTensor read_only
      : tensor<6xf32> to memref<6xf32>
  %output = memref.alloc() : memref<4xf32>

  %rowOffsetsDynamic = memref.cast %rowOffsets : memref<3xi32> to memref<?xi32>
  %columnIndicesDynamic =
      memref.cast %columnIndices : memref<4xi32> to memref<?xi32>
  %valuesDynamic = memref.cast %values : memref<4xf32> to memref<?xf32>
  %rhsDynamic = memref.cast %rhs : memref<6xf32> to memref<?xf32>
  %outputDynamic = memref.cast %output : memref<4xf32> to memref<?xf32>

  %rowOffsetsUnranked =
      memref.cast %rowOffsetsDynamic : memref<?xi32> to memref<*xi32>
  %columnIndicesUnranked =
      memref.cast %columnIndicesDynamic : memref<?xi32> to memref<*xi32>
  %valuesUnranked =
      memref.cast %valuesDynamic : memref<?xf32> to memref<*xf32>
  %rhsUnranked = memref.cast %rhsDynamic : memref<?xf32> to memref<*xf32>
  %outputUnranked =
      memref.cast %outputDynamic : memref<?xf32> to memref<*xf32>

  gpu.host_register %rowOffsetsUnranked : memref<*xi32>
  gpu.host_register %columnIndicesUnranked : memref<*xi32>
  gpu.host_register %valuesUnranked : memref<*xf32>
  gpu.host_register %rhsUnranked : memref<*xf32>
  gpu.host_register %outputUnranked : memref<*xf32>

  %rowOffsetsDevice = call @mgpuMemGetDeviceMemRef1dInt32(%rowOffsetsDynamic)
      : (memref<?xi32>) -> memref<?xi32>
  %columnIndicesDevice =
      call @mgpuMemGetDeviceMemRef1dInt32(%columnIndicesDynamic)
      : (memref<?xi32>) -> memref<?xi32>
  %valuesDevice = call @mgpuMemGetDeviceMemRef1dFloat(%valuesDynamic)
      : (memref<?xf32>) -> memref<?xf32>
  %rhsDevice = call @mgpuMemGetDeviceMemRef1dFloat(%rhsDynamic)
      : (memref<?xf32>) -> memref<?xf32>
  %outputDevice = call @mgpuMemGetDeviceMemRef1dFloat(%outputDynamic)
      : (memref<?xf32>) -> memref<?xf32>

  %rhsDevice2d = memref.reinterpret_cast %rhsDevice to
      offset: [0], sizes: [3, 2], strides: [2, 1]
      : memref<?xf32> to memref<3x2xf32, strided<[2, 1]>>
  %outputDevice2d = memref.reinterpret_cast %outputDevice to
      offset: [0], sizes: [2, 2], strides: [2, 1]
      : memref<?xf32> to memref<2x2xf32, strided<[2, 1]>>

  %rowOffsetsDeviceTensor =
      bufferization.to_tensor %rowOffsetsDevice restrict
      : memref<?xi32> to tensor<?xi32>
  %columnIndicesDeviceTensor =
      bufferization.to_tensor %columnIndicesDevice restrict
      : memref<?xi32> to tensor<?xi32>
  %valuesDeviceTensor = bufferization.to_tensor %valuesDevice restrict
      : memref<?xf32> to tensor<?xf32>
  %rhsDeviceTensor = bufferization.to_tensor %rhsDevice2d restrict
      : memref<3x2xf32, strided<[2, 1]>> to tensor<3x2xf32>
  %outputDeviceTensor =
      bufferization.to_tensor %outputDevice2d restrict writable
      : memref<2x2xf32, strided<[2, 1]>> to tensor<2x2xf32>
  %matrix = sparse_tensor.assemble
      (%rowOffsetsDeviceTensor, %columnIndicesDeviceTensor), %valuesDeviceTensor
      : (tensor<?xi32>, tensor<?xi32>), tensor<?xf32>
        to tensor<2x3xf32, #csr>
  %rhsDynamicTensor = tensor.cast %rhsDeviceTensor
      : tensor<3x2xf32> to tensor<?x?xf32>
  %outputDynamicTensor = tensor.cast %outputDeviceTensor
      : tensor<2x2xf32> to tensor<?x?xf32>

  %result = call @spmm(%matrix, %rhsDynamicTensor, %outputDynamicTensor)
      : (tensor<2x3xf32, #csr>, tensor<?x?xf32>, tensor<?x?xf32>)
        -> tensor<?x?xf32>
  %resultBuffer = bufferization.to_buffer %result
      : tensor<?x?xf32> to memref<?x?xf32>
  %resultUnranked =
      memref.cast %resultBuffer : memref<?x?xf32> to memref<*xf32>
  call @printMemrefF32(%resultUnranked) : (memref<*xf32>) -> ()
  return
}

func.func private @mgpuMemGetDeviceMemRef1dInt32(
    %buffer: memref<?xi32>) -> memref<?xi32>
func.func private @mgpuMemGetDeviceMemRef1dFloat(
    %buffer: memref<?xf32>) -> memref<?xf32>
func.func private @printMemrefF32(%buffer: memref<*xf32>)
