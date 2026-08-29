// RUN: sparsewave-opt %s \
// RUN:   --schedule-sparsewave-position='mapping=wave block-size=64 wave-size=32 cooperative-axis=column' \
// RUN:   | FileCheck %s

// This operator-independent reduction verifies that a named logical axis can
// be assigned to wave lanes without relying on SpMM operation semantics.

// CHECK-LABEL: func.func @cooperative_axis(
// CHECK: %[[GROUPS:.*]] = arith.ceildivui %{{.*}}, %{{.*}} : index
// CHECK: %[[WORKERS:.*]] = arith.muli %[[GROUPS]], %{{.*}} : index
// CHECK: sparsewave.position_parallel %[[WORKERS]] mapping = "wave" block_size = 64 {
// CHECK: ^bb0(%[[WORKER:.*]]: index, %[[LANE:.*]]: index, %[[LANES:.*]]: index):
// CHECK: %[[COLUMN_GROUP:.*]] = arith.remui %[[WORKER]], %[[GROUPS]] : index
// CHECK: %[[COLUMN_BASE:.*]] = arith.muli %[[COLUMN_GROUP]], %[[LANES]] : index
// CHECK: %[[COLUMN:.*]] = arith.addi %[[COLUMN_BASE]], %[[LANE]] : index
// CHECK: %[[ACTIVE:.*]] = arith.cmpi ult, %[[COLUMN]], %{{.*}} : index
// CHECK: %[[LANE_ZERO:.*]] = arith.cmpi eq, %[[LANE]], %{{.*}} : index
// CHECK: %[[SHARED:.*]] = scf.if %[[LANE_ZERO]] -> (f32) {
// CHECK:   %[[LOADED:.*]] = memref.load %{{.*}}[%{{.*}}] : memref<?xf32>
// CHECK:   scf.yield %[[LOADED]] : f32
// CHECK: }
// CHECK: %[[BROADCAST:.*]], %{{.*}} = gpu.shuffle idx %[[SHARED]], %{{.*}}, %{{.*}} : f32
// CHECK: scf.if %[[ACTIVE]] {
// CHECK:   %[[PER_COLUMN:.*]] = memref.load %{{.*}}[%{{.*}}, %[[COLUMN]]] : memref<?x?xf32>
// CHECK:   %[[PRODUCT:.*]] = arith.mulf %[[BROADCAST]], %[[PER_COLUMN]] : f32
// CHECK:   memref.atomic_rmw addf %[[PRODUCT]],
// CHECK-NOT: sparsewave.position_reduce

func.func @cooperative_axis(%shared: memref<?xf32>,
                            %perColumn: memref<?x?xf32>,
                            %output: memref<?xf32>) {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %items = memref.dim %perColumn, %zero : memref<?x?xf32>
  %columns = memref.dim %perColumn, %one : memref<?x?xf32>
  sparsewave.position_reduce lower (%zero, %zero)
      upper (%items, %columns) axes = ["item", "column"] order = [0, 1]
      into %output kind = "sum" {
  ^bb0(%item: index, %column: index):
    %sharedValue = memref.load %shared[%item] : memref<?xf32>
    %columnValue = memref.load %perColumn[%item, %column] : memref<?x?xf32>
    %product = arith.mulf %sharedValue, %columnValue : f32
    %base = arith.muli %item, %columns : index
    %key = arith.addi %base, %column : index
    sparsewave.yield %key, %product : index, f32
  } : memref<?xf32>
  return
}
