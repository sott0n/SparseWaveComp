// REQUIRES: rocm-toolkit
// RUN: rm -rf %t.bundle
// RUN: sparsewave-bundle %s --output %t.bundle --chip gfx1101 \
// RUN:   --rocm-path %rocm_path --operation spmm \
// RUN:   --mapping wave-per-row-tile --block-size 64 --tile-size 4 \
// RUN:   --wavefront-size 32
// RUN: sparsewave-bundle --verify %t.bundle
// RUN: FileCheck %s --input-file=%t.bundle/manifest.json

// CHECK: "args": [
// CHECK: "name": "rowOffsets"
// CHECK: "offset": 0
// CHECK: "size": 8
// CHECK: "type": "ptr"
// CHECK: "name": "output"
// CHECK: "offset": 32
// CHECK: "block": [
// CHECK-NEXT: 64,
// CHECK: "code_object": "kernels.hsaco"
// CHECK: "grid": [
// CHECK-NEXT: "ceil_div(n, 2) * 64",
// CHECK: "kernarg_size": 40
// CHECK: "name": "spmm"
// CHECK: "shared_memory_bytes": 0
// CHECK: "fixed_group_segment_bytes":
// CHECK: "hsaco_sha256": "{{[0-9a-f]+}}"
// CHECK: "launch_n": "output_rows"
// CHECK: "mapping": "wave-per-row-tile"
// CHECK: "operation": "spmm"
// CHECK: "tile_size": 4
// CHECK: "wavefront_size": 32
// CHECK: "symbol": "spmm_kernel"
// CHECK: "workspace_bytes": 0
// CHECK: "manifest_version": 1
// CHECK: "target": "gfx1101"

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
