// RUN: sparsewave-opt %s --prepare-gpu-bare-ptr-abi | FileCheck %s

func.func @generic_kernel_input(%arg0: memref<?xi32>, %index: index) {
  %c1 = arith.constant 1 : index
  gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1)
      threads(%tx, %ty, %tz) in (%bx_size = %c1, %by_size = %c1,
                                 %bz_size = %c1) {
    %value = memref.load %arg0[%index] : memref<?xi32>
    memref.store %value, %arg0[%index] : memref<?xi32>
    gpu.terminator
  }
  return
}

// CHECK-LABEL: func.func @generic_kernel_input(
// CHECK: %[[VIEW:.*]] = memref.reinterpret_cast %arg0 to
// CHECK-SAME: offset: [0], sizes: [9223372036854775807], strides: [1]
// CHECK-SAME: memref<?xi32> to memref<9223372036854775807xi32>
// CHECK: gpu.launch
// CHECK: memref.load %[[VIEW]]
// CHECK-SAME: memref<9223372036854775807xi32>
// CHECK: memref.store {{.*}}, %[[VIEW]]
// CHECK-SAME: memref<9223372036854775807xi32>
