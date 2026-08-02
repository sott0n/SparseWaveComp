// REQUIRES: amdgpu-runtime
// RUN: sparsewave-opt %s \
// RUN:   --pass-pipeline='builtin.module(sparsewave-to-amdgpu-pipeline{chip=%amdgpu_chip wavefront-size=32 rocm-path=%rocm_path sddmm-block-size=64})' \
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

#sddmm = {
  indexing_maps = [
    affine_map<(i, j, k) -> (i, k)>,
    affine_map<(i, j, k) -> (k, j)>,
    affine_map<(i, j, k) -> (i, j)>,
    affine_map<(i, j, k) -> (i, j)>
  ],
  iterator_types = ["parallel", "parallel", "reduction"]
}

func.func @sddmm(
    %sample: tensor<2x3xf32, #csr>,
    %lhs: tensor<?x?xf32>,
    %rhs: tensor<?x?xf32>) -> tensor<2x3xf32, #csr> {
  %zero = arith.constant 0.0 : f32
  %output = linalg.fill ins(%zero : f32)
      outs(%sample : tensor<2x3xf32, #csr>)
      -> tensor<2x3xf32, #csr>
  %result = linalg.generic #sddmm
      ins(%lhs, %rhs, %sample
          : tensor<?x?xf32>, tensor<?x?xf32>, tensor<2x3xf32, #csr>)
      outs(%output : tensor<2x3xf32, #csr>) {
    ^bb0(%lhsValue: f32, %rhsValue: f32, %sampleValue: f32,
         %sum: f32):
      %denseProduct = arith.mulf %lhsValue, %rhsValue : f32
      %weightedProduct = arith.mulf %sampleValue, %denseProduct : f32
      %next = arith.addf %sum, %weightedProduct : f32
      linalg.yield %next : f32
  } -> tensor<2x3xf32, #csr>
  return %result : tensor<2x3xf32, #csr>
}

// Sample:
// [2 0 3]
// [0 4 0]
//
// Dense left-hand side:
// [1 2]
// [3 4]
//
// Dense right-hand side:
// [5 6 7]
// [8 9 10]
//
// CHECK: [42, 81, 216]
func.func @main() {
  %rowOffsetsTensor = arith.constant dense<[0, 2, 3]> : tensor<3xi32>
  %columnIndicesTensor = arith.constant dense<[0, 2, 1]> : tensor<3xi32>
  %lhsTensor = arith.constant dense<[1.0, 2.0, 3.0, 4.0]>
      : tensor<4xf32>
  %rhsTensor = arith.constant dense<[5.0, 6.0, 7.0, 8.0, 9.0, 10.0]>
      : tensor<6xf32>

  %rowOffsets = bufferization.to_buffer %rowOffsetsTensor read_only
      : tensor<3xi32> to memref<3xi32>
  %columnIndices = bufferization.to_buffer %columnIndicesTensor read_only
      : tensor<3xi32> to memref<3xi32>
  %lhs = bufferization.to_buffer %lhsTensor read_only
      : tensor<4xf32> to memref<4xf32>
  %rhs = bufferization.to_buffer %rhsTensor read_only
      : tensor<6xf32> to memref<6xf32>
  %values = memref.alloc() : memref<3xf32>

  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %value0 = arith.constant 2.0 : f32
  %value1 = arith.constant 3.0 : f32
  %value2 = arith.constant 4.0 : f32
  memref.store %value0, %values[%c0] : memref<3xf32>
  memref.store %value1, %values[%c1] : memref<3xf32>
  memref.store %value2, %values[%c2] : memref<3xf32>

  %rowOffsetsDynamic = memref.cast %rowOffsets
      : memref<3xi32> to memref<?xi32>
  %columnIndicesDynamic = memref.cast %columnIndices
      : memref<3xi32> to memref<?xi32>
  %valuesDynamic = memref.cast %values : memref<3xf32> to memref<?xf32>
  %lhsDynamic = memref.cast %lhs : memref<4xf32> to memref<?xf32>
  %rhsDynamic = memref.cast %rhs : memref<6xf32> to memref<?xf32>

  %rowOffsetsUnranked = memref.cast %rowOffsetsDynamic
      : memref<?xi32> to memref<*xi32>
  %columnIndicesUnranked = memref.cast %columnIndicesDynamic
      : memref<?xi32> to memref<*xi32>
  %valuesUnranked = memref.cast %valuesDynamic
      : memref<?xf32> to memref<*xf32>
  %lhsUnranked = memref.cast %lhsDynamic : memref<?xf32> to memref<*xf32>
  %rhsUnranked = memref.cast %rhsDynamic : memref<?xf32> to memref<*xf32>

  gpu.host_register %rowOffsetsUnranked : memref<*xi32>
  gpu.host_register %columnIndicesUnranked : memref<*xi32>
  gpu.host_register %valuesUnranked : memref<*xf32>
  gpu.host_register %lhsUnranked : memref<*xf32>
  gpu.host_register %rhsUnranked : memref<*xf32>

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

  %lhsDevice2d = memref.reinterpret_cast %lhsDevice to
      offset: [0], sizes: [2, 2], strides: [2, 1]
      : memref<?xf32> to memref<2x2xf32, strided<[2, 1]>>
  %rhsDevice2d = memref.reinterpret_cast %rhsDevice to
      offset: [0], sizes: [2, 3], strides: [3, 1]
      : memref<?xf32> to memref<2x3xf32, strided<[3, 1]>>

  %rowOffsetsDeviceTensor =
      bufferization.to_tensor %rowOffsetsDevice restrict
      : memref<?xi32> to tensor<?xi32>
  %columnIndicesDeviceTensor =
      bufferization.to_tensor %columnIndicesDevice restrict
      : memref<?xi32> to tensor<?xi32>
  %valuesDeviceTensor =
      bufferization.to_tensor %valuesDevice restrict writable
      : memref<?xf32> to tensor<?xf32>
  %lhsDeviceTensor = bufferization.to_tensor %lhsDevice2d restrict
      : memref<2x2xf32, strided<[2, 1]>> to tensor<2x2xf32>
  %rhsDeviceTensor = bufferization.to_tensor %rhsDevice2d restrict
      : memref<2x3xf32, strided<[3, 1]>> to tensor<2x3xf32>

  %sample = sparse_tensor.assemble
      (%rowOffsetsDeviceTensor, %columnIndicesDeviceTensor),
      %valuesDeviceTensor
      : (tensor<?xi32>, tensor<?xi32>), tensor<?xf32>
        to tensor<2x3xf32, #csr>
  %lhsDynamicTensor = tensor.cast %lhsDeviceTensor
      : tensor<2x2xf32> to tensor<?x?xf32>
  %rhsDynamicTensor = tensor.cast %rhsDeviceTensor
      : tensor<2x3xf32> to tensor<?x?xf32>

  %result = call @sddmm(%sample, %lhsDynamicTensor, %rhsDynamicTensor)
      : (tensor<2x3xf32, #csr>, tensor<?x?xf32>, tensor<?x?xf32>)
        -> tensor<2x3xf32, #csr>
  %resultValues = sparse_tensor.values %result
      : tensor<2x3xf32, #csr> to memref<?xf32>
  %resultUnranked = memref.cast %resultValues
      : memref<?xf32> to memref<*xf32>
  call @printMemrefF32(%resultUnranked) : (memref<*xf32>) -> ()
  return
}

func.func private @mgpuMemGetDeviceMemRef1dInt32(
    %buffer: memref<?xi32>) -> memref<?xi32>
func.func private @mgpuMemGetDeviceMemRef1dFloat(
    %buffer: memref<?xf32>) -> memref<?xf32>
func.func private @printMemrefF32(%buffer: memref<*xf32>)
