// RUN: sparsewave-opt %s \
// RUN:   --decompose-position-spmv \
// RUN:   | FileCheck %s --check-prefix=DECOMPOSE
// RUN: sparsewave-opt %s \
// RUN:   --decompose-position-spmv='preserve-direct-mapping=true' \
// RUN:   | FileCheck %s --check-prefix=DIRECT
// RUN: sparsewave-opt %s \
// RUN:   --pass-pipeline='builtin.module(decompose-position-spmv,schedule-sparsewave-position{mapping=thread block-size=128},convert-sparsewave-to-gpu{position-block-size=128},lower-sparsewave-position-space)' \
// RUN:   | FileCheck %s --check-prefix=GPU

// DECOMPOSE-LABEL: func.func @coo_spmv(
// DECOMPOSE-NOT: gpu.launch
// DECOMPOSE: sparsewave.position_reduce lower(%{{.*}}) upper(%{{.*}}) axes = ["position"] order = [0] into %{{.*}} kind = "sum" {
// DECOMPOSE: ^bb0(%[[POSITION:.*]]: index):
// DECOMPOSE: %[[ROW_I32:.*]] = memref.load %{{.*}}[%[[POSITION]]]
// DECOMPOSE: %[[COLUMN_I32:.*]] = memref.load %{{.*}}[%[[POSITION]]]
// DECOMPOSE: %[[ROW:.*]] = arith.index_cast %[[ROW_I32]] : i32 to index
// DECOMPOSE: %[[COLUMN:.*]] = arith.index_cast %[[COLUMN_I32]] : i32 to index
// DECOMPOSE: %[[VALUE:.*]] = memref.load %{{.*}}[%[[POSITION]]]
// DECOMPOSE: %[[VECTOR_VALUE:.*]] = memref.load %{{.*}}[%[[COLUMN]]]
// DECOMPOSE: %[[PRODUCT:.*]] = arith.mulf %[[VALUE]], %[[VECTOR_VALUE]]
// DECOMPOSE: sparsewave.yield %[[ROW]], %[[PRODUCT]] : index, f32
// DECOMPOSE-NOT: sparsewave.coo_spmv

// DIRECT-LABEL: func.func @coo_spmv(
// DIRECT: sparsewave.position_reduce
// DIRECT-NOT: sparsewave.coo_spmv

// GPU-LABEL: func.func @coo_spmv(
// GPU: %[[BLOCK_SIZE:.*]] = arith.constant 128 : index
// GPU: %[[C1:.*]] = arith.constant 1 : index
// GPU: %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
// GPU: %[[C0:.*]] = arith.constant 0 : index
// GPU: %[[NNZ:.*]] = memref.dim %{{.*}}, %[[C0]]
// GPU: %[[OUTPUT_SIZE:.*]] = memref.dim %{{.*}}, %[[C0]]
// GPU: %[[INIT_BLOCKS:.*]] = arith.ceildivui %[[OUTPUT_SIZE]], %[[BLOCK_SIZE]]
// GPU: gpu.launch
// GPU: scf.if
// GPU: memref.store %[[ZERO]], %{{.*}}[%{{.*}}]
// GPU: gpu.terminator
// GPU: %[[NNZ_BLOCKS:.*]] = arith.ceildivui %[[NNZ]], %[[BLOCK_SIZE]]
// GPU: gpu.launch
// GPU: scf.if
// GPU: %[[ROW_I32_2:.*]] = memref.load %{{.*}}[%{{.*}}]
// GPU: %[[COLUMN_I32_2:.*]] = memref.load %{{.*}}[%{{.*}}]
// GPU: %[[ROW_2:.*]] = arith.index_cast %[[ROW_I32_2]] : i32 to index
// GPU: %[[COLUMN_2:.*]] = arith.index_cast %[[COLUMN_I32_2]] : i32 to index
// GPU: %[[VALUE_2:.*]] = memref.load %{{.*}}[%{{.*}}]
// GPU: %[[VECTOR_VALUE_2:.*]] = memref.load %{{.*}}[%[[COLUMN_2]]]
// GPU: %[[PRODUCT_2:.*]] = arith.mulf %[[VALUE_2]], %[[VECTOR_VALUE_2]]
// GPU: memref.atomic_rmw addf %[[PRODUCT_2]], %{{.*}}[%[[ROW_2]]]
// GPU-NOT: sparsewave.

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
