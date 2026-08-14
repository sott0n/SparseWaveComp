// REQUIRES: rocm-toolkit
// RUN: rm -rf %t.bundle
// RUN: sparsewave-bundle %s --output %t.bundle --chip gfx1100 \
// RUN:   --rocm-path %rocm_path --operation spmm \
// RUN:   --mapping wave-per-row-tile --block-size 64 --tile-size 4 \
// RUN:   --wavefront-size 32
// RUN: sparsewave-bundle --verify %t.bundle
// RUN: FileCheck %s --input-file=%t.bundle/manifest.json

// CHECK: "block_size": 64
// CHECK: "kernel_bare_ptr_calling_convention": true
// CHECK: "mapping": "wave-per-row-tile"
// CHECK: "operation": "spmm"
// CHECK: "tile_size": 4
// CHECK: "file": "kernels.hsaco"
// CHECK: "block": [
// CHECK-NEXT: 64,
// CHECK: "kernarg": {
// CHECK: "shared_memory_bytes":
// CHECK: "symbol": "spmm_kernel.kd"
// CHECK: "wavefront_size":
// CHECK: "architecture": "amdgcn-amd-amdhsa-unknown-gfx1100"
// CHECK: "chip": "gfx1100"

func.func @spmm(
    %rowOffsets: memref<5xi32>,
    %columnIndices: memref<8xi32>,
    %values: memref<8xf32>,
    %rhs: memref<8x4xf32>,
    %output: memref<4x4xf32>) {
  sparsewave.spmm %rowOffsets, %columnIndices, %values, %rhs, %output
      : memref<5xi32>, memref<8xi32>, memref<8xf32>,
        memref<8x4xf32>, memref<4x4xf32>
  return
}
