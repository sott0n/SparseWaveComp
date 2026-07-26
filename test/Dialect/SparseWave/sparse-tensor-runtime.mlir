// REQUIRES: amdgpu-runtime
// RUN: sparsewave-opt %s \
// RUN:   --pass-pipeline='builtin.module(sparsewave-to-amdgpu-pipeline{chip=%amdgpu_chip wavefront-size=32 rocm-path=%rocm_path spmv-mapping=wave-per-row spmv-block-size=128})' \
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

func.func @spmv(
    %matrix: tensor<4x4xf32, #csr>,
    %vector: tensor<?xf32>,
    %output: tensor<?xf32>) -> tensor<?xf32> {
  %zero = arith.constant 0.0 : f32
  %initialized = linalg.fill ins(%zero : f32)
      outs(%output : tensor<?xf32>) -> tensor<?xf32>
  %result = linalg.matvec
      ins(%matrix, %vector : tensor<4x4xf32, #csr>, tensor<?xf32>)
      outs(%initialized : tensor<?xf32>) -> tensor<?xf32>
  return %result : tensor<?xf32>
}

// Matrix:
// [1 0 2 0]
// [0 3 0 0]
// [4 0 0 5]
// [0 0 6 0]
//
// CHECK: [70, 60, 240, 180]
func.func @main() {
  %rowOffsetsTensor = arith.constant dense<[0, 2, 3, 5, 6]>
      : tensor<5xi32>
  %columnIndicesTensor = arith.constant dense<[0, 2, 1, 0, 3, 2]>
      : tensor<6xi32>
  %valuesTensor = arith.constant dense<[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]>
      : tensor<6xf32>
  %vectorTensor = arith.constant dense<[10.0, 20.0, 30.0, 40.0]>
      : tensor<4xf32>

  %rowOffsets = bufferization.to_buffer %rowOffsetsTensor read_only
      : tensor<5xi32> to memref<5xi32>
  %columnIndices = bufferization.to_buffer %columnIndicesTensor read_only
      : tensor<6xi32> to memref<6xi32>
  %values = bufferization.to_buffer %valuesTensor read_only
      : tensor<6xf32> to memref<6xf32>
  %vector = bufferization.to_buffer %vectorTensor read_only
      : tensor<4xf32> to memref<4xf32>
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

  %rowOffsetsDeviceTensor =
      bufferization.to_tensor %rowOffsetsDevice restrict
      : memref<?xi32> to tensor<?xi32>
  %columnIndicesDeviceTensor =
      bufferization.to_tensor %columnIndicesDevice restrict
      : memref<?xi32> to tensor<?xi32>
  %valuesDeviceTensor = bufferization.to_tensor %valuesDevice restrict
      : memref<?xf32> to tensor<?xf32>
  %vectorDeviceTensor = bufferization.to_tensor %vectorDevice restrict
      : memref<?xf32> to tensor<?xf32>
  %outputDeviceTensor =
      bufferization.to_tensor %outputDevice restrict writable
      : memref<?xf32> to tensor<?xf32>
  %matrix = sparse_tensor.assemble
      (%rowOffsetsDeviceTensor, %columnIndicesDeviceTensor), %valuesDeviceTensor
      : (tensor<?xi32>, tensor<?xi32>), tensor<?xf32>
        to tensor<4x4xf32, #csr>

  %result = call @spmv(%matrix, %vectorDeviceTensor, %outputDeviceTensor)
      : (tensor<4x4xf32, #csr>, tensor<?xf32>, tensor<?xf32>)
        -> tensor<?xf32>
  %resultBuffer = bufferization.to_buffer %result
      : tensor<?xf32> to memref<?xf32>
  %resultUnranked =
      memref.cast %resultBuffer : memref<?xf32> to memref<*xf32>
  call @printMemrefF32(%resultUnranked) : (memref<*xf32>) -> ()
  return
}

func.func private @mgpuMemGetDeviceMemRef1dInt32(
    %buffer: memref<?xi32>) -> memref<?xi32>
func.func private @mgpuMemGetDeviceMemRef1dFloat(
    %buffer: memref<?xf32>) -> memref<?xf32>
func.func private @printMemrefF32(%buffer: memref<*xf32>)
