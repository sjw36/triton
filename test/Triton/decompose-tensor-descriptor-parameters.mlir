// RUN: triton-opt %s --triton-decompose-tensor-descriptor-parameters --canonicalize --cse --mlir-print-debuginfo --split-input-file | FileCheck %s

#loc2 = loc("rewrite-tensor-descriptor-to-pointer.mlir":147:28)
module {
  tt.func public @callee(%tensordesc: !tt.tensordesc<128x128xf32> loc("tensordesc"(#loc2))) -> !tt.tensordesc<128x128xf32> {
    tt.return %tensordesc : !tt.tensordesc<128x128xf32>
  }

  tt.func public @caller(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    %c1_i64 = arith.constant 1 : i64
    %c256_i32 = arith.constant 256 : i32
    %c256_i64 = arith.constant 256 : i64
    %0 = tt.make_tensor_descriptor %arg0, [%c256_i32, %c256_i32], [%c256_i64, %c1_i64] {order = array<i32: 0>} : !tt.ptr<f32>, <128x128xf32>
    %1 = tt.call @callee(%0) : (!tt.tensordesc<128x128xf32>) -> !tt.tensordesc<128x128xf32>
    tt.return
  }
}

// CHECK-LABEL: @callee
// CHECK-SAME: %[[PTR:[^:]*]]
// CHECK-SAME: loc("tensordesc.ptr"(#loc{{[^,]*}}))
// CHECK-SAME: %[[SHAPE0:[^:]*]]
// CHECK-SAME: loc("tensordesc.shape.0"(#loc{{[^,]*}}))
// CHECK-SAME: %[[SHAPE1:[^:]*]]
// CHECK-SAME: loc("tensordesc.shape.1"(#loc{{[^,]*}}))
// CHECK-SAME: %[[STRIDE0:[^:]*]]
// CHECK-SAME: loc("tensordesc.stride.0"(#loc{{[^,]*}}))
// CHECK-SAME: %[[STRIDE1:[^:]*]]
// CHECK-SAME: loc("tensordesc.stride.1"(#loc{{[^,]*}}))
// CHECK-NEXT: tt.return %[[PTR]], %[[SHAPE0]], %[[SHAPE1]], %[[STRIDE0]], %[[STRIDE1]]

// CHECK-LABEL: @caller
// CHECK-SAME: %[[PTR:[^:]*]]
// CHECK-DAG: %[[c1:.*]] = arith.constant 1 : i64
// CHECK-DAG: %[[c256_i32:.*]] = arith.constant 256 : i32
// CHECK-DAG: %[[c256_i64:.*]] = arith.constant 256 : i64
// CHECK: %{{.*}}:5 = tt.call @callee(%[[PTR]], %[[c256_i32]], %[[c256_i32]], %[[c256_i64]], %[[c1]])
// CHECK-SAME: -> (!tt.ptr<f32>, i32, i32, i64, i64)

// -----

#loc2 = loc("rewrite-tensor-descriptor-to-pointer.mlir":147:28)
module {
  tt.func public @callee_passthru(%tensordesc: !tt.tensordesc<128x128xf32> loc("tensordesc"(#loc2))) -> !tt.tensordesc<128x128xf32> {
    tt.return %tensordesc : !tt.tensordesc<128x128xf32>
  }

  tt.func public @caller_passthru(%arg0: !tt.tensordesc<128x128xf32> {tt.divisibility = 16 : i32}) {
    %1 = tt.call @callee_passthru(%arg0) : (!tt.tensordesc<128x128xf32>) -> !tt.tensordesc<128x128xf32>
    tt.return
  }
}

// CHECK-LABEL: @callee_passthru
// CHECK-SAME: %[[PTR:[^:]*]]
// CHECK-SAME: loc("tensordesc.ptr"(#loc{{[^,]*}}))
// CHECK-SAME: %[[SHAPE0:[^:]*]]
// CHECK-SAME: loc("tensordesc.shape.0"(#loc{{[^,]*}}))
// CHECK-SAME: %[[SHAPE1:[^:]*]]
// CHECK-SAME: loc("tensordesc.shape.1"(#loc{{[^,]*}}))
// CHECK-SAME: %[[STRIDE0:[^:]*]]
// CHECK-SAME: loc("tensordesc.stride.0"(#loc{{[^,]*}}))
// CHECK-SAME: %[[STRIDE1:[^:]*]]
// CHECK-SAME: loc("tensordesc.stride.1"(#loc{{[^,]*}}))
// CHECK-NEXT: tt.return %[[PTR]], %[[SHAPE0]], %[[SHAPE1]], %[[STRIDE0]], %[[STRIDE1]]

// CHECK-NOT: make_tensor_descriptor

// -----

module {
  tt.func public @arg_attr(%arg0: !tt.tensordesc<128x128xf32>, %arg1: i32 {tt.divisibility = 16 : i32}) {
    tt.return
  }
}

// CHECK-LABEL: @arg_attr
// CHECK-SAME: %arg5: i32 {tt.divisibility = 16 : i32, tt.user_index = 1 : i32} loc({{.*}})) {
