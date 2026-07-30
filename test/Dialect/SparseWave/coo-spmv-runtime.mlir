// REQUIRES: amdgpu-runtime
// RUN: sparsewave-opt %s \
// RUN:   --pass-pipeline='builtin.module(sparsewave-to-amdgpu-pipeline{chip=%amdgpu_chip wavefront-size=32 rocm-path=%rocm_path spmv-block-size=128})' \
// RUN:   | mlir-runner \
// RUN:     --shared-libs=%mlir_rocm_runtime \
// RUN:     --shared-libs=%mlir_runner_utils \
// RUN:     --entry-point-result=void \
// RUN:   | FileCheck %s

func.func @coo_spmv(
    %rowIndices: memref<?xi32>,
    %columnIndices: memref<?xi32>,
    %values: memref<?xf32>,
    %vector: memref<?xf32>,
    %output: memref<?xf32>) {
  sparsewave.coo_spmv %rowIndices, %columnIndices, %values, %vector, %output
      : memref<?xi32>, memref<?xi32>, memref<?xf32>, memref<?xf32>,
        memref<?xf32>
  return
}

// COO entries include the duplicate coordinate (0, 0).
//
// Matrix:
// [1 0 2 0]
// [0 3 0 0]
// [4 0 0 5]
// [0 0 6 0]
//
// CHECK: [70, 60, 240, 180]
func.func @main() {
  %rowIndicesTensor = arith.constant dense<[0, 0, 0, 1, 2, 2, 3]>
      : tensor<7xi32>
  %columnIndicesTensor = arith.constant dense<[0, 0, 2, 1, 0, 3, 2]>
      : tensor<7xi32>
  %valuesTensor =
      arith.constant dense<[0.25, 0.75, 2.0, 3.0, 4.0, 5.0, 6.0]>
      : tensor<7xf32>
  %vectorTensor = arith.constant dense<[10.0, 20.0, 30.0, 40.0]>
      : tensor<4xf32>

  %rowIndices = bufferization.to_buffer %rowIndicesTensor read_only
      : tensor<7xi32> to memref<7xi32>
  %columnIndices = bufferization.to_buffer %columnIndicesTensor read_only
      : tensor<7xi32> to memref<7xi32>
  %values = bufferization.to_buffer %valuesTensor read_only
      : tensor<7xf32> to memref<7xf32>
  %vector = bufferization.to_buffer %vectorTensor read_only
      : tensor<4xf32> to memref<4xf32>
  %output = memref.alloc() : memref<4xf32>

  %rowIndicesDynamic =
      memref.cast %rowIndices : memref<7xi32> to memref<?xi32>
  %columnIndicesDynamic =
      memref.cast %columnIndices : memref<7xi32> to memref<?xi32>
  %valuesDynamic = memref.cast %values : memref<7xf32> to memref<?xf32>
  %vectorDynamic = memref.cast %vector : memref<4xf32> to memref<?xf32>
  %outputDynamic = memref.cast %output : memref<4xf32> to memref<?xf32>

  %rowIndicesUnranked =
      memref.cast %rowIndicesDynamic : memref<?xi32> to memref<*xi32>
  %columnIndicesUnranked =
      memref.cast %columnIndicesDynamic : memref<?xi32> to memref<*xi32>
  %valuesUnranked =
      memref.cast %valuesDynamic : memref<?xf32> to memref<*xf32>
  %vectorUnranked =
      memref.cast %vectorDynamic : memref<?xf32> to memref<*xf32>
  %outputUnranked =
      memref.cast %outputDynamic : memref<?xf32> to memref<*xf32>

  gpu.host_register %rowIndicesUnranked : memref<*xi32>
  gpu.host_register %columnIndicesUnranked : memref<*xi32>
  gpu.host_register %valuesUnranked : memref<*xf32>
  gpu.host_register %vectorUnranked : memref<*xf32>
  gpu.host_register %outputUnranked : memref<*xf32>

  %rowIndicesDevice =
      call @mgpuMemGetDeviceMemRef1dInt32(%rowIndicesDynamic)
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

  call @coo_spmv(%rowIndicesDevice, %columnIndicesDevice, %valuesDevice,
                 %vectorDevice, %outputDevice)
      : (memref<?xi32>, memref<?xi32>, memref<?xf32>, memref<?xf32>,
         memref<?xf32>) -> ()
  %resultUnranked =
      memref.cast %outputDevice : memref<?xf32> to memref<*xf32>
  call @printMemrefF32(%resultUnranked) : (memref<*xf32>) -> ()
  return
}

func.func private @mgpuMemGetDeviceMemRef1dInt32(
    %buffer: memref<?xi32>) -> memref<?xi32>
func.func private @mgpuMemGetDeviceMemRef1dFloat(
    %buffer: memref<?xf32>) -> memref<?xf32>
func.func private @printMemrefF32(%buffer: memref<*xf32>)
