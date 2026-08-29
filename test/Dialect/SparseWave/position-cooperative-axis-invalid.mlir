// RUN: not sparsewave-opt %s \
// RUN:   --schedule-sparsewave-position='mapping=thread cooperative-axis=column' \
// RUN:   2>&1 | FileCheck %s --check-prefix=THREAD
// RUN: not sparsewave-opt %s \
// RUN:   --schedule-sparsewave-position='mapping=wave cooperative-axis=missing' \
// RUN:   2>&1 | FileCheck %s --check-prefix=MISSING
// RUN: not sparsewave-opt %s \
// RUN:   --schedule-sparsewave-position='mapping=wave wave-size=16 cooperative-axis=column' \
// RUN:   2>&1 | FileCheck %s --check-prefix=WAVE-SIZE

// THREAD: cooperative position axis applies only to wave mapping
// MISSING: does not define cooperative axis 'missing'
// WAVE-SIZE: cooperative wave position mapping requires a wave size of 32 or 64, but got 16

func.func @invalid_cooperative_axis(%values: memref<?x?xf32>,
                                    %output: memref<?xf32>) {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %items = memref.dim %values, %zero : memref<?x?xf32>
  %columns = memref.dim %values, %one : memref<?x?xf32>
  sparsewave.position_reduce lower (%zero, %zero)
      upper (%items, %columns) axes = ["item", "column"] order = [0, 1]
      into %output kind = "sum" {
  ^bb0(%item: index, %column: index):
    %value = memref.load %values[%item, %column] : memref<?x?xf32>
    sparsewave.yield %item, %value : index, f32
  } : memref<?xf32>
  return
}
