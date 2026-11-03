// REQUIRES: amdgpu-runtime
// RUN: sparsewave-opt %s \
// RUN:   --pass-pipeline='builtin.module(sparsewave-to-amdgpu-pipeline{chip=%amdgpu_chip wavefront-size=32 rocm-path=%rocm_path spmv-mapping=block-per-position spmv-block-size=128})' \
// RUN:   | mlir-runner \
// RUN:     --shared-libs=%mlir_rocm_runtime \
// RUN:     --shared-libs=%mlir_runner_utils \
// RUN:     --entry-point-result=void \
// RUN:   | FileCheck %s
// RUN: sparsewave-opt %s \
// RUN:   --pass-pipeline='builtin.module(sparsewave-to-amdgpu-pipeline{chip=%amdgpu_chip wavefront-size=32 rocm-path=%rocm_path spmv-mapping=block-per-position spmv-block-size=128 position-block-chunk-size=4 position-block-reduction=atomic})' \
// RUN:   | mlir-runner \
// RUN:     --shared-libs=%mlir_rocm_runtime \
// RUN:     --shared-libs=%mlir_runner_utils \
// RUN:     --entry-point-result=void \
// RUN:   | FileCheck %s
// RUN: sparsewave-opt %s \
// RUN:   --pass-pipeline='builtin.module(sparsewave-to-amdgpu-pipeline{chip=%amdgpu_chip wavefront-size=32 rocm-path=%rocm_path spmv-mapping=block-per-position spmv-block-size=128 position-block-chunk-size=4})' \
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

// One full 128-thread block processes 32 adjacent four-entry rows. This
// exercises every shuffle offset with many segment boundaries in each wave.
// CHECK: [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]
func.func @main() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c32 = arith.constant 32 : index
  %c33 = arith.constant 33 : index
  %c128 = arith.constant 128 : index
  %f1 = arith.constant 1.0 : f32

  %rowOffsets = memref.alloc() : memref<33xi32>
  %columnIndices = memref.alloc() : memref<128xi32>
  %values = memref.alloc() : memref<128xf32>
  %vector = memref.alloc() : memref<4xf32>
  %output = memref.alloc() : memref<32xf32>

  scf.for %row = %c0 to %c33 step %c1 {
    %offset = arith.muli %row, %c4 : index
    %offsetI32 = arith.index_cast %offset : index to i32
    memref.store %offsetI32, %rowOffsets[%row] : memref<33xi32>
  }
  scf.for %position = %c0 to %c128 step %c1 {
    %column = arith.remui %position, %c4 : index
    %columnI32 = arith.index_cast %column : index to i32
    memref.store %columnI32, %columnIndices[%position] : memref<128xi32>
    memref.store %f1, %values[%position] : memref<128xf32>
  }
  scf.for %column = %c0 to %c4 step %c1 {
    memref.store %f1, %vector[%column] : memref<4xf32>
  }

  %rowOffsetsDynamic =
      memref.cast %rowOffsets : memref<33xi32> to memref<?xi32>
  %columnIndicesDynamic =
      memref.cast %columnIndices : memref<128xi32> to memref<?xi32>
  %valuesDynamic = memref.cast %values : memref<128xf32> to memref<?xf32>
  %vectorDynamic = memref.cast %vector : memref<4xf32> to memref<?xf32>
  %outputDynamic = memref.cast %output : memref<32xf32> to memref<?xf32>
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
