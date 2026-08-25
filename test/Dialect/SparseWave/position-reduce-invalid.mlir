// RUN: sparsewave-opt %s --split-input-file -verify-diagnostics

func.func @invalid_kind(%output: memref<?xf32>) {
  %zero = arith.constant 0 : index
  // expected-error @+1 {{kind must be 'sum', but got 'max'}}
  sparsewave.position_reduce lower (%zero) upper (%zero)
      axes = ["position"] order = [0]
      into %output kind = "max" {
  ^bb0(%position: index):
    %value = arith.constant 0.0 : f32
    sparsewave.yield %position, %value : index, f32
  } : memref<?xf32>
  return
}

// -----

func.func @invalid_output(%output: memref<?xi32>) {
  %zero = arith.constant 0 : index
  // expected-error @+1 {{output must have floating-point elements}}
  sparsewave.position_reduce lower (%zero) upper (%zero)
      axes = ["position"] order = [0]
      into %output kind = "sum" {
  ^bb0(%position: index):
    %value = arith.constant 0 : i32
    sparsewave.yield %position, %value : index, i32
  } : memref<?xi32>
  return
}

// -----

func.func @invalid_yield(%output: memref<?xf32>) {
  %zero = arith.constant 0 : index
  // expected-error @+1 {{body must yield an index key and one value of type 'f32'}}
  sparsewave.position_reduce lower (%zero) upper (%zero)
      axes = ["position"] order = [0]
      into %output kind = "sum" {
  ^bb0(%position: index):
    sparsewave.yield %position : index
  } : memref<?xf32>
  return
}

// -----

func.func @bound_count_mismatch(%output: memref<?xf32>) {
  %zero = arith.constant 0 : index
  // expected-error @+1 {{lower and upper bounds must have the same number of axes}}
  sparsewave.position_reduce lower (%zero, %zero) upper (%zero)
      axes = ["position", "rhs"] order = [0, 1]
      into %output kind = "sum" {
  ^bb0(%axis0: index, %axis1: index):
    %value = arith.constant 0.0 : f32
    sparsewave.yield %axis0, %value : index, f32
  } : memref<?xf32>
  return
}

// -----

func.func @duplicate_order_axis(%output: memref<?xf32>) {
  %zero = arith.constant 0 : index
  // expected-error @+1 {{order must be a permutation, but axis 0 appears more than once}}
  sparsewave.position_reduce lower (%zero, %zero)
      upper (%zero, %zero) axes = ["position", "rhs"] order = [0, 0]
      into %output kind = "sum" {
  ^bb0(%axis0: index, %axis1: index):
    %value = arith.constant 0.0 : f32
    sparsewave.yield %axis0, %value : index, f32
  } : memref<?xf32>
  return
}

// -----

func.func @body_rank_mismatch(%output: memref<?xf32>) {
  %zero = arith.constant 0 : index
  // expected-error @+1 {{body must have one argument per logical axis}}
  sparsewave.position_reduce lower (%zero, %zero)
      upper (%zero, %zero) axes = ["position", "rhs"] order = [0, 1]
      into %output kind = "sum" {
  ^bb0(%axis0: index):
    %value = arith.constant 0.0 : f32
    sparsewave.yield %axis0, %value : index, f32
  } : memref<?xf32>
  return
}

// -----

func.func @axis_name_count_mismatch(%output: memref<?xf32>) {
  %zero = arith.constant 0 : index
  // expected-error @+1 {{must have one axis name per logical axis}}
  sparsewave.position_reduce lower (%zero, %zero)
      upper (%zero, %zero) axes = ["position"] order = [0, 1]
      into %output kind = "sum" {
  ^bb0(%axis0: index, %axis1: index):
    %value = arith.constant 0.0 : f32
    sparsewave.yield %axis0, %value : index, f32
  } : memref<?xf32>
  return
}

// -----

func.func @duplicate_axis_name(%output: memref<?xf32>) {
  %zero = arith.constant 0 : index
  // expected-error @+1 {{axis name 'position' must not appear more than once}}
  sparsewave.position_reduce lower (%zero, %zero)
      upper (%zero, %zero) axes = ["position", "position"] order = [0, 1]
      into %output kind = "sum" {
  ^bb0(%axis0: index, %axis1: index):
    %value = arith.constant 0.0 : f32
    sparsewave.yield %axis0, %value : index, f32
  } : memref<?xf32>
  return
}
