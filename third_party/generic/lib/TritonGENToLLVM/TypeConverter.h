#ifndef TRITON_CONVERSION_TRITONGEN_TO_LLVM_TYPECONVERTER_H
#define TRITON_CONVERSION_TRITONGEN_TO_LLVM_TYPECONVERTER_H

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "triton/Conversion/MLIRTypes.h"
#include "triton/Dialect/Triton/IR/Types.h"

using namespace mlir;
using namespace mlir::triton;

class TritonGENToLLVMTypeConverter : public LLVMTypeConverter {
public:
  using TypeConverter::convertType;

  TritonGENToLLVMTypeConverter(MLIRContext *ctx, LowerToLLVMOptions &option,
                               const DataLayoutAnalysis *analysis = nullptr);

  Type convertTritonPointerType(triton::PointerType type);
  Type convertTritonTensorType(RankedTensorType type);
};

#endif
