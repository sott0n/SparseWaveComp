// REQUIRES: amdgpu-runtime
// RUN: sparsewave-opt %s \
// RUN:   --pass-pipeline='builtin.module(sparsewave-to-amdgpu-pipeline{chip=%amdgpu_chip wavefront-size=32 rocm-path=%rocm_path elementwise-block-size=64})' \
// RUN:   | mlir-runner \
// RUN:     --shared-libs=%mlir_rocm_runtime \
// RUN:     --shared-libs=%mlir_runner_utils \
// RUN:     --entry-point-result=void \
// RUN:   | FileCheck %s

func.func @csr_add(
    %lhsOffsets: memref<?xi32>, %lhsColumns: memref<?xi32>,
    %lhsValues: memref<?xf32>, %rhsOffsets: memref<?xi32>,
    %rhsColumns: memref<?xi32>, %rhsValues: memref<?xf32>,
    %outputOffsets: memref<?xi32>, %outputColumns: memref<?xi32>,
    %outputValues: memref<?xf32>, %outputNnz: memref<?xi32>) {
  sparsewave.csr_elementwise
      %lhsOffsets, %lhsColumns, %lhsValues,
      %rhsOffsets, %rhsColumns, %rhsValues,
      %outputOffsets, %outputColumns, %outputValues, %outputNnz kind = "add"
      : memref<?xi32>, memref<?xi32>, memref<?xf32>,
        memref<?xi32>, memref<?xi32>, memref<?xf32>,
        memref<?xi32>, memref<?xi32>, memref<?xf32>, memref<?xi32>
  return
}

func.func @csr_multiply(
    %lhsOffsets: memref<?xi32>, %lhsColumns: memref<?xi32>,
    %lhsValues: memref<?xf32>, %rhsOffsets: memref<?xi32>,
    %rhsColumns: memref<?xi32>, %rhsValues: memref<?xf32>,
    %outputOffsets: memref<?xi32>, %outputColumns: memref<?xi32>,
    %outputValues: memref<?xf32>, %outputNnz: memref<?xi32>) {
  sparsewave.csr_elementwise
      %lhsOffsets, %lhsColumns, %lhsValues,
      %rhsOffsets, %rhsColumns, %rhsValues,
      %outputOffsets, %outputColumns, %outputValues, %outputNnz
      kind = "multiply"
      : memref<?xi32>, memref<?xi32>, memref<?xf32>,
        memref<?xi32>, memref<?xi32>, memref<?xf32>,
        memref<?xi32>, memref<?xi32>, memref<?xf32>, memref<?xi32>
  return
}

// Left CSR matrix:
// [1 0  2 0]
// [0 3  0 0]
// [4 0  0 0]
//
// Right CSR matrix:
// [0 5 -2 0]
// [0 7  0 8]
// [0 0  9 0]
//
// Addition retains the structural zero at (0, 2).
// CHECK: [0, 3, 5, 7]
// CHECK: [0, 1, 2, 1, 3, 0, 2, 0, 0]
// CHECK: [1, 5, 0, 10, 8, 4, 9, 0, 0]
// CHECK: [7]
// CHECK: [0, 1, 2, 2]
// CHECK: [2, 1, 0, 0]
// CHECK: [-4, 21, 0, 0]
// CHECK: [2]
func.func @main() {
  %zeroI32 = arith.constant 0 : i32
  %zeroF32 = arith.constant 0.0 : f32
  %lhsOffsetsTensor = arith.constant dense<[0, 2, 3, 4]> : tensor<4xi32>
  %lhsColumnsTensor = arith.constant dense<[0, 2, 1, 0]> : tensor<4xi32>
  %lhsValuesTensor = arith.constant dense<[1.0, 2.0, 3.0, 4.0]>
      : tensor<4xf32>
  %rhsOffsetsTensor = arith.constant dense<[0, 2, 4, 5]> : tensor<4xi32>
  %rhsColumnsTensor = arith.constant dense<[1, 2, 1, 3, 2]> : tensor<5xi32>
  %rhsValuesTensor = arith.constant dense<[5.0, -2.0, 7.0, 8.0, 9.0]>
      : tensor<5xf32>

  %lhsOffsets = bufferization.to_buffer %lhsOffsetsTensor read_only
      : tensor<4xi32> to memref<4xi32>
  %lhsColumns = bufferization.to_buffer %lhsColumnsTensor read_only
      : tensor<4xi32> to memref<4xi32>
  %lhsValues = bufferization.to_buffer %lhsValuesTensor read_only
      : tensor<4xf32> to memref<4xf32>
  %rhsOffsets = bufferization.to_buffer %rhsOffsetsTensor read_only
      : tensor<4xi32> to memref<4xi32>
  %rhsColumns = bufferization.to_buffer %rhsColumnsTensor read_only
      : tensor<5xi32> to memref<5xi32>
  %rhsValues = bufferization.to_buffer %rhsValuesTensor read_only
      : tensor<5xf32> to memref<5xf32>

  %addOffsets = memref.alloc() : memref<4xi32>
  %addColumns = memref.alloc() : memref<9xi32>
  %addValues = memref.alloc() : memref<9xf32>
  %addNnz = memref.alloc() : memref<1xi32>
  linalg.fill ins(%zeroI32 : i32) outs(%addColumns : memref<9xi32>)
  linalg.fill ins(%zeroF32 : f32) outs(%addValues : memref<9xf32>)

  %multiplyOffsets = memref.alloc() : memref<4xi32>
  %multiplyColumns = memref.alloc() : memref<4xi32>
  %multiplyValues = memref.alloc() : memref<4xf32>
  %multiplyNnz = memref.alloc() : memref<1xi32>
  linalg.fill ins(%zeroI32 : i32) outs(%multiplyColumns : memref<4xi32>)
  linalg.fill ins(%zeroF32 : f32) outs(%multiplyValues : memref<4xf32>)

  %lhsOffsetsDynamic = memref.cast %lhsOffsets
      : memref<4xi32> to memref<?xi32>
  %lhsColumnsDynamic = memref.cast %lhsColumns
      : memref<4xi32> to memref<?xi32>
  %lhsValuesDynamic = memref.cast %lhsValues
      : memref<4xf32> to memref<?xf32>
  %rhsOffsetsDynamic = memref.cast %rhsOffsets
      : memref<4xi32> to memref<?xi32>
  %rhsColumnsDynamic = memref.cast %rhsColumns
      : memref<5xi32> to memref<?xi32>
  %rhsValuesDynamic = memref.cast %rhsValues
      : memref<5xf32> to memref<?xf32>
  %addOffsetsDynamic = memref.cast %addOffsets
      : memref<4xi32> to memref<?xi32>
  %addColumnsDynamic = memref.cast %addColumns
      : memref<9xi32> to memref<?xi32>
  %addValuesDynamic = memref.cast %addValues
      : memref<9xf32> to memref<?xf32>
  %addNnzDynamic = memref.cast %addNnz : memref<1xi32> to memref<?xi32>
  %multiplyOffsetsDynamic = memref.cast %multiplyOffsets
      : memref<4xi32> to memref<?xi32>
  %multiplyColumnsDynamic = memref.cast %multiplyColumns
      : memref<4xi32> to memref<?xi32>
  %multiplyValuesDynamic = memref.cast %multiplyValues
      : memref<4xf32> to memref<?xf32>
  %multiplyNnzDynamic = memref.cast %multiplyNnz
      : memref<1xi32> to memref<?xi32>

  %lhsOffsetsUnranked = memref.cast %lhsOffsetsDynamic
      : memref<?xi32> to memref<*xi32>
  %lhsColumnsUnranked = memref.cast %lhsColumnsDynamic
      : memref<?xi32> to memref<*xi32>
  %lhsValuesUnranked = memref.cast %lhsValuesDynamic
      : memref<?xf32> to memref<*xf32>
  %rhsOffsetsUnranked = memref.cast %rhsOffsetsDynamic
      : memref<?xi32> to memref<*xi32>
  %rhsColumnsUnranked = memref.cast %rhsColumnsDynamic
      : memref<?xi32> to memref<*xi32>
  %rhsValuesUnranked = memref.cast %rhsValuesDynamic
      : memref<?xf32> to memref<*xf32>
  %addOffsetsUnranked = memref.cast %addOffsetsDynamic
      : memref<?xi32> to memref<*xi32>
  %addColumnsUnranked = memref.cast %addColumnsDynamic
      : memref<?xi32> to memref<*xi32>
  %addValuesUnranked = memref.cast %addValuesDynamic
      : memref<?xf32> to memref<*xf32>
  %addNnzUnranked = memref.cast %addNnzDynamic
      : memref<?xi32> to memref<*xi32>
  %multiplyOffsetsUnranked = memref.cast %multiplyOffsetsDynamic
      : memref<?xi32> to memref<*xi32>
  %multiplyColumnsUnranked = memref.cast %multiplyColumnsDynamic
      : memref<?xi32> to memref<*xi32>
  %multiplyValuesUnranked = memref.cast %multiplyValuesDynamic
      : memref<?xf32> to memref<*xf32>
  %multiplyNnzUnranked = memref.cast %multiplyNnzDynamic
      : memref<?xi32> to memref<*xi32>

  gpu.host_register %lhsOffsetsUnranked : memref<*xi32>
  gpu.host_register %lhsColumnsUnranked : memref<*xi32>
  gpu.host_register %lhsValuesUnranked : memref<*xf32>
  gpu.host_register %rhsOffsetsUnranked : memref<*xi32>
  gpu.host_register %rhsColumnsUnranked : memref<*xi32>
  gpu.host_register %rhsValuesUnranked : memref<*xf32>
  gpu.host_register %addOffsetsUnranked : memref<*xi32>
  gpu.host_register %addColumnsUnranked : memref<*xi32>
  gpu.host_register %addValuesUnranked : memref<*xf32>
  gpu.host_register %addNnzUnranked : memref<*xi32>
  gpu.host_register %multiplyOffsetsUnranked : memref<*xi32>
  gpu.host_register %multiplyColumnsUnranked : memref<*xi32>
  gpu.host_register %multiplyValuesUnranked : memref<*xf32>
  gpu.host_register %multiplyNnzUnranked : memref<*xi32>

  %lhsOffsetsDevice = call @mgpuMemGetDeviceMemRef1dInt32(%lhsOffsetsDynamic)
      : (memref<?xi32>) -> memref<?xi32>
  %lhsColumnsDevice = call @mgpuMemGetDeviceMemRef1dInt32(%lhsColumnsDynamic)
      : (memref<?xi32>) -> memref<?xi32>
  %lhsValuesDevice = call @mgpuMemGetDeviceMemRef1dFloat(%lhsValuesDynamic)
      : (memref<?xf32>) -> memref<?xf32>
  %rhsOffsetsDevice = call @mgpuMemGetDeviceMemRef1dInt32(%rhsOffsetsDynamic)
      : (memref<?xi32>) -> memref<?xi32>
  %rhsColumnsDevice = call @mgpuMemGetDeviceMemRef1dInt32(%rhsColumnsDynamic)
      : (memref<?xi32>) -> memref<?xi32>
  %rhsValuesDevice = call @mgpuMemGetDeviceMemRef1dFloat(%rhsValuesDynamic)
      : (memref<?xf32>) -> memref<?xf32>
  %addOffsetsDevice = call @mgpuMemGetDeviceMemRef1dInt32(%addOffsetsDynamic)
      : (memref<?xi32>) -> memref<?xi32>
  %addColumnsDevice = call @mgpuMemGetDeviceMemRef1dInt32(%addColumnsDynamic)
      : (memref<?xi32>) -> memref<?xi32>
  %addValuesDevice = call @mgpuMemGetDeviceMemRef1dFloat(%addValuesDynamic)
      : (memref<?xf32>) -> memref<?xf32>
  %addNnzDevice = call @mgpuMemGetDeviceMemRef1dInt32(%addNnzDynamic)
      : (memref<?xi32>) -> memref<?xi32>
  %multiplyOffsetsDevice = call @mgpuMemGetDeviceMemRef1dInt32(
      %multiplyOffsetsDynamic) : (memref<?xi32>) -> memref<?xi32>
  %multiplyColumnsDevice = call @mgpuMemGetDeviceMemRef1dInt32(
      %multiplyColumnsDynamic) : (memref<?xi32>) -> memref<?xi32>
  %multiplyValuesDevice = call @mgpuMemGetDeviceMemRef1dFloat(
      %multiplyValuesDynamic) : (memref<?xf32>) -> memref<?xf32>
  %multiplyNnzDevice = call @mgpuMemGetDeviceMemRef1dInt32(
      %multiplyNnzDynamic) : (memref<?xi32>) -> memref<?xi32>

  call @csr_add(
      %lhsOffsetsDevice, %lhsColumnsDevice, %lhsValuesDevice,
      %rhsOffsetsDevice, %rhsColumnsDevice, %rhsValuesDevice,
      %addOffsetsDevice, %addColumnsDevice, %addValuesDevice, %addNnzDevice)
      : (memref<?xi32>, memref<?xi32>, memref<?xf32>,
         memref<?xi32>, memref<?xi32>, memref<?xf32>,
         memref<?xi32>, memref<?xi32>, memref<?xf32>, memref<?xi32>) -> ()
  call @csr_multiply(
      %lhsOffsetsDevice, %lhsColumnsDevice, %lhsValuesDevice,
      %rhsOffsetsDevice, %rhsColumnsDevice, %rhsValuesDevice,
      %multiplyOffsetsDevice, %multiplyColumnsDevice, %multiplyValuesDevice,
      %multiplyNnzDevice)
      : (memref<?xi32>, memref<?xi32>, memref<?xf32>,
         memref<?xi32>, memref<?xi32>, memref<?xf32>,
         memref<?xi32>, memref<?xi32>, memref<?xf32>, memref<?xi32>) -> ()

  call @printMemrefI32(%addOffsetsUnranked) : (memref<*xi32>) -> ()
  call @printMemrefI32(%addColumnsUnranked) : (memref<*xi32>) -> ()
  call @printMemrefF32(%addValuesUnranked) : (memref<*xf32>) -> ()
  call @printMemrefI32(%addNnzUnranked) : (memref<*xi32>) -> ()
  call @printMemrefI32(%multiplyOffsetsUnranked) : (memref<*xi32>) -> ()
  call @printMemrefI32(%multiplyColumnsUnranked) : (memref<*xi32>) -> ()
  call @printMemrefF32(%multiplyValuesUnranked) : (memref<*xf32>) -> ()
  call @printMemrefI32(%multiplyNnzUnranked) : (memref<*xi32>) -> ()
  return
}

func.func private @mgpuMemGetDeviceMemRef1dInt32(
    %buffer: memref<?xi32>) -> memref<?xi32>
func.func private @mgpuMemGetDeviceMemRef1dFloat(
    %buffer: memref<?xf32>) -> memref<?xf32>
func.func private @printMemrefI32(%buffer: memref<*xi32>)
func.func private @printMemrefF32(%buffer: memref<*xf32>)
