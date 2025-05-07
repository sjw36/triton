#ifndef TRITON_DIALECT_TRITONGEN_IR_DIALECT_H_
#define TRITON_DIALECT_TRITONGEN_IR_DIALECT_H_

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"

// TritonGEN depends on Triton
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "Dialect/TritonGEN/IR/Attributes.h"
#include "Dialect/TritonGEN/IR/Dialect.h.inc"
#include "Dialect/TritonGEN/IR/Types.h"

#define GET_OP_CLASSES
#include "Dialect/TritonGEN/IR/Ops.h.inc"

#endif // TRITON_DIALECT_TRITONGEN_IR_DIALECT_H_
