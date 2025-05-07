#ifndef TRITON_CONVERSION_TRITONGEN_TO_LLVM_UTILITY_H
#define TRITON_CONVERSION_TRITONGEN_TO_LLVM_UTILITY_H

#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

namespace mlir::triton::gen {

Value getProgramId(mlir::FunctionOpInterface funcOp, int axis);
Value getNumPrograms(mlir::FunctionOpInterface funcOp, int axis);

} // namespace mlir::triton::gen

#endif
