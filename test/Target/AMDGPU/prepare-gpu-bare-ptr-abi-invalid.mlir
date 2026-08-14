// RUN: not sparsewave-opt %s --prepare-gpu-bare-ptr-abi 2>&1 \
// RUN:   | FileCheck %s

func.func @descriptor_dependent_kernel(%arg0: memref<?xi32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1)
      threads(%tx, %ty, %tz) in (%bx_size = %c1, %by_size = %c1,
                                 %bz_size = %c1) {
    %size = memref.dim %arg0, %c0 : memref<?xi32>
    gpu.terminator
  }
  return
}

// CHECK: error: dynamic memref capture requires runtime descriptor metadata
// CHECK-SAME: and cannot use the bare-pointer kernel ABI
