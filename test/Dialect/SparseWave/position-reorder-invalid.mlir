// RUN: not sparsewave-opt %s \
// RUN:   --reorder-sparsewave-position='order=position' 2>&1 \
// RUN:   | FileCheck %s --check-prefix=RANK
// RUN: not sparsewave-opt %s \
// RUN:   --reorder-sparsewave-position='order=position,position' 2>&1 \
// RUN:   | FileCheck %s --check-prefix=PERMUTATION
// RUN: not sparsewave-opt %s \
// RUN:   --reorder-sparsewave-position 2>&1 \
// RUN:   | FileCheck %s --check-prefix=EMPTY
// RUN: not sparsewave-opt %s \
// RUN:   --reorder-sparsewave-position='order=rhs,other' 2>&1 \
// RUN:   | FileCheck %s --check-prefix=UNKNOWN

// RANK: position reorder rank mismatch: got 1 axes for a rank-2 reduction
// PERMUTATION: position reorder must not repeat axis 'position'
// EMPTY: position reorder requires a non-empty axis permutation
// UNKNOWN: position reorder references unknown axis 'other'

func.func @rank_two(%values: memref<?x?xf32>, %output: memref<?xf32>) {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %dim0 = memref.dim %values, %zero : memref<?x?xf32>
  %dim1 = memref.dim %values, %one : memref<?x?xf32>
  sparsewave.position_reduce lower (%zero, %zero)
      upper (%dim0, %dim1) axes = ["position", "rhs"] order = [0, 1]
      into %output kind = "sum" {
  ^bb0(%i: index, %j: index):
    %key = arith.addi %i, %j : index
    %value = memref.load %values[%i, %j] : memref<?x?xf32>
    sparsewave.yield %key, %value : index, f32
  } : memref<?xf32>
  return
}
