// RUN: sparsewave-opt %s --split-input-file -verify-diagnostics

func.func @invalid_kind(%output: memref<?xf32>) {
  %zero = arith.constant 0 : index
  // expected-error @+1 {{kind must be 'sum', but got 'max'}}
  sparsewave.position_reduce %zero to %zero into %output kind = "max" {
  ^bb0(%position: index):
    %value = arith.constant 0.0 : f32
    sparsewave.yield %position, %value : index, f32
  } : index, memref<?xf32>
  return
}

// -----

func.func @invalid_output(%output: memref<?xi32>) {
  %zero = arith.constant 0 : index
  // expected-error @+1 {{output must have floating-point elements}}
  sparsewave.position_reduce %zero to %zero into %output kind = "sum" {
  ^bb0(%position: index):
    %value = arith.constant 0 : i32
    sparsewave.yield %position, %value : index, i32
  } : index, memref<?xi32>
  return
}

// -----

func.func @invalid_yield(%output: memref<?xf32>) {
  %zero = arith.constant 0 : index
  // expected-error @+1 {{body must yield an index key and one value of type 'f32'}}
  sparsewave.position_reduce %zero to %zero into %output kind = "sum" {
  ^bb0(%position: index):
    sparsewave.yield %position : index
  } : index, memref<?xf32>
  return
}
