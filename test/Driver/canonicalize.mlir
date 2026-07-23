// RUN: sparsewave-opt %s --canonicalize | FileCheck %s

// CHECK-LABEL: func.func @fold_add
// CHECK: %[[RESULT:.*]] = arith.constant 42 : i32
// CHECK-NEXT: return %[[RESULT]] : i32
func.func @fold_add() -> i32 {
  %lhs = arith.constant 40 : i32
  %rhs = arith.constant 2 : i32
  %result = arith.addi %lhs, %rhs : i32
  return %result : i32
}
