#ifndef TRITONGEN_DIALECT
#define TRITONGEN_DIALECT

include "mlir/IR/OpBase.td"

def TritonGEN_Dialect : Dialect {
  let name = "triton_gen";

  let cppNamespace = "::mlir::triton::gen";

  let hasOperationAttrVerify = 1;

  let description = [{
    Triton GEN Dialect.
  }];

  let dependentDialects = [
    "triton::TritonDialect",
    "tensor::TensorDialect",
    "mlir::memref::MemRefDialect",
  ];

  let extraClassDeclaration = [{
    void registerTypes();
  }];

  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
  let usePropertiesForAttributes = 1;
}

#endif
