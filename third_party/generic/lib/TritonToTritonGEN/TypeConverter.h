#ifndef TRITON_CONVERSION_TRITON_TO_TRITONGEN_TYPECONVERTER_H
#define TRITON_CONVERSION_TRITON_TO_TRITONGEN_TYPECONVERTER_H

#include "mlir/Transforms/DialectConversion.h"
#include "triton/Dialect/Triton/IR/Types.h"


namespace mlir {

class TritonToTritonGENTypeConverter : public TypeConverter {
public:
  using TypeConverter::convertType;

  TritonToTritonGENTypeConverter();

  Type convertTritonPointerType(triton::PointerType type);
};

}

#endif
