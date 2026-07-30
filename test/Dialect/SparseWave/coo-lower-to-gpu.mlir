// RUN: sparsewave-opt %s \
// RUN:   --convert-sparsewave-to-gpu='block-size=128' \
// RUN:   | FileCheck %s

// CHECK-LABEL: func.func @coo_spmv(
// CHECK: %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[C1:.*]] = arith.constant 1 : index
// CHECK: %[[BLOCK_SIZE:.*]] = arith.constant 128 : index
// CHECK: %[[OUTPUT_SIZE:.*]] = memref.dim %{{.*}}, %[[C0]]
// CHECK: %[[NNZ:.*]] = memref.dim %{{.*}}, %[[C0]]
// CHECK: %[[INIT_BLOCKS:.*]] = arith.ceildivui %[[OUTPUT_SIZE]], %[[BLOCK_SIZE]]
// CHECK: gpu.launch
// CHECK: scf.if
// CHECK: memref.store %[[ZERO]], %{{.*}}[%{{.*}}]
// CHECK: gpu.terminator
// CHECK: %[[NNZ_BLOCKS:.*]] = arith.ceildivui %[[NNZ]], %[[BLOCK_SIZE]]
// CHECK: gpu.launch
// CHECK: scf.if
// CHECK: %[[ROW_I32:.*]] = memref.load %{{.*}}[%{{.*}}]
// CHECK: %[[COLUMN_I32:.*]] = memref.load %{{.*}}[%{{.*}}]
// CHECK: %[[ROW:.*]] = arith.index_cast %[[ROW_I32]] : i32 to index
// CHECK: %[[COLUMN:.*]] = arith.index_cast %[[COLUMN_I32]] : i32 to index
// CHECK: %[[VALUE:.*]] = memref.load %{{.*}}[%{{.*}}]
// CHECK: %[[VECTOR_VALUE:.*]] = memref.load %{{.*}}[%[[COLUMN]]]
// CHECK: %[[PRODUCT:.*]] = arith.mulf %[[VALUE]], %[[VECTOR_VALUE]]
// CHECK: memref.atomic_rmw addf %[[PRODUCT]], %{{.*}}[%[[ROW]]]
// CHECK-NOT: sparsewave.coo_spmv

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
